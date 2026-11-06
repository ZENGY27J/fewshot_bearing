#!/usr/bin/env python3
"""Validated small A-pilot for CWRU LOOCV.

This runner keeps the previous pilot outputs untouched and fixes the main
reproducibility/comparability problems:
  * CWT caches are keyed and validated by source-array hash and transform version.
  * The candidate uses the same EMA-PCL training recipe as the v3 baseline.
  * Validation loss is accumulated per sample, not as an unweighted batch mean.
  * GPU use is required by default and is recorded in every result.

Default pilot: fold 0, shots 1/3/5, ratios 0/5/10, seeds 0/1/2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import run_pcl_eval as R


WORKSPACE = Path(__file__).resolve().parent
SOURCE_ROOT = WORKSPACE
OUT_ROOT = WORKSPACE / "results/a_small_eval_v2"
CACHE_ROOT = OUT_ROOT / "cache"
TRANSFORM_VERSION = "cwt_batch_v1_source_hash"
NUM_CLASSES = 10
WINDOW_SIZE = 1024
BATCH_SIZE = 32
LR = 5e-4
WEIGHT_DECAY = 0.01
EPOCHS = 100
T_MAX = 80
PATIENCE = 20
PCL_LAMBDA = 0.5
PCL_TAU = 0.07
DEFAULT_SEEDS = [0, 1, 2]
DEFAULT_SHOTS = [1, 3, 5]
DEFAULT_RATIOS = [0.0, 5.0, 10.0]


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this pilot, but torch.cuda.is_available() is False. "
            "Launch with the CUDA-enabled pytorch_env environment."
        )


def array_sha256(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("ascii"))
    h.update(repr(arr.shape).encode("ascii"))
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _raw_views(X: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(X, dtype=np.float32)
    local = (x - x.mean(axis=1, keepdims=True)) / (
        x.std(axis=1, keepdims=True) + 1e-6
    )
    global_view = (x - mean) / std
    return np.stack([local, global_view], axis=1).astype(np.float32)


def _cache_metadata(npz: Any) -> tuple[str, str, tuple[int, ...]]:
    source = str(np.asarray(npz["source_sha256"]).item())
    version = str(np.asarray(npz["transform_version"]).item())
    shape = tuple(int(v) for v in np.asarray(npz["source_shape"]).tolist())
    return source, version, shape


def precompute_cwt_verified(X: np.ndarray, cache_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    source_hash = array_sha256(X)
    source_shape = tuple(int(v) for v in X.shape)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            with np.load(cache_path, allow_pickle=False) as archive:
                cached_source, cached_version, cached_shape = _cache_metadata(archive)
                cwt = np.asarray(archive["cwt"], dtype=np.float32)
            if (
                cached_source == source_hash
                and cached_version == TRANSFORM_VERSION
                and cached_shape == source_shape
                and cwt.shape[0] == source_shape[0]
            ):
                return cwt, {
                    "path": str(cache_path),
                    "source_sha256": source_hash,
                    "transform_version": TRANSFORM_VERSION,
                    "cache_hit": True,
                    "shape": list(cwt.shape),
                }
        except (OSError, KeyError, ValueError):
            pass

    x_t = torch.from_numpy(np.asarray(X, dtype=np.float32)).to(R.DEVICE)
    pieces = []
    with torch.no_grad():
        for start in range(0, len(x_t), 64):
            pieces.append(R.cwt_batch(x_t[start : start + 64]).cpu().numpy())
    cwt = np.concatenate(pieces, axis=0).astype(np.float32)
    if cwt.shape[0] != source_shape[0]:
        raise RuntimeError(f"CWT row count mismatch: {cwt.shape[0]} != {source_shape[0]}")

    tmp_path = cache_path.with_name(cache_path.name + ".tmp.npz")
    np.savez_compressed(
        tmp_path,
        cwt=cwt,
        source_sha256=np.asarray(source_hash),
        source_shape=np.asarray(source_shape, dtype=np.int64),
        source_dtype=np.asarray(str(np.asarray(X).dtype)),
        transform_version=np.asarray(TRANSFORM_VERSION),
    )
    tmp_path.replace(cache_path)
    return cwt, {
        "path": str(cache_path),
        "source_sha256": source_hash,
        "transform_version": TRANSFORM_VERSION,
        "cache_hit": False,
        "shape": list(cwt.shape),
    }


def _resolve_fake_dir(fold: int, shot: int, ratio: float) -> Path:
    ratio_tag = f"{ratio:g}"
    base = (
        SOURCE_ROOT
        / "results/cwru_generated_loocv"
        / f"fold{fold}"
        / "diffusion"
        / f"{shot}shot"
        / "generated"
    )
    candidates = [base / f"ratio{ratio_tag}", base / f"ratio{ratio_tag}.0"]
    for candidate in candidates:
        if (candidate / "fake_X.npy").exists() and (candidate / "fake_y.npy").exists():
            return candidate
    raise FileNotFoundError(
        f"Missing generated data for fold={fold}, shot={shot}, ratio={ratio}; "
        f"checked {[str(p) for p in candidates]}"
    )


def _load_arrays(fold: int, shot: int, ratio: float) -> tuple[dict[str, Any], dict[str, Any]]:
    split = SOURCE_ROOT / "data/splits_loocv_v2/cwru" / f"fold{fold}" / f"{shot}shot"
    required = [
        split / "train_X.npy",
        split / "train_y.npy",
        split / "val_X.npy",
        split / "val_y.npy",
        split / "test_X.npy",
        split / "test_y.npy",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing v2 split files: {missing}")

    real_X = np.load(split / "train_X.npy").astype(np.float32, copy=False)
    real_y = np.load(split / "train_y.npy").astype(np.int64, copy=False)
    val_X = np.load(split / "val_X.npy").astype(np.float32, copy=False)
    val_y = np.load(split / "val_y.npy").astype(np.int64, copy=False)
    test_X = np.load(split / "test_X.npy").astype(np.float32, copy=False)
    test_y = np.load(split / "test_y.npy").astype(np.int64, copy=False)

    fake_dir: Path | None = None
    fake_X = np.empty((0, real_X.shape[1]), dtype=np.float32)
    fake_y = np.empty((0,), dtype=np.int64)
    if ratio > 0:
        fake_dir = _resolve_fake_dir(fold, shot, ratio)
        fake_X = np.load(fake_dir / "fake_X.npy").astype(np.float32, copy=False)
        fake_y = np.load(fake_dir / "fake_y.npy").astype(np.int64, copy=False)
        if len(fake_X) != len(fake_y):
            raise ValueError(f"Fake X/y row mismatch in {fake_dir}")
        if fake_X.ndim != 2 or fake_X.shape[1] != WINDOW_SIZE:
            raise ValueError(f"Unexpected fake_X shape {fake_X.shape} in {fake_dir}")

    train_X = np.concatenate([real_X, fake_X], axis=0)
    train_y = np.concatenate([real_y, fake_y], axis=0)
    if not (len(train_X) == len(train_y) and len(val_X) == len(val_y) and len(test_X) == len(test_y)):
        raise ValueError("X/y row mismatch in one of the split arrays")

    real_mean = float(real_X.mean())
    real_std = max(float(real_X.std()), 1e-6)
    arrays = {
        "train_X": train_X,
        "train_y": train_y,
        "val_X": val_X,
        "val_y": val_y,
        "test_X": test_X,
        "test_y": test_y,
        "real_count": int(len(real_X)),
        "fake_count": int(len(fake_X)),
        "split_dir": str(split),
        "fake_dir": str(fake_dir) if fake_dir else None,
        "real_mean": real_mean,
        "real_std": real_std,
    }
    sources = {
        "train_source_sha256": array_sha256(train_X),
        "real_source_sha256": array_sha256(real_X),
        "val_source_sha256": array_sha256(val_X),
        "test_source_sha256": array_sha256(test_X),
        "fake_source_sha256": array_sha256(fake_X) if len(fake_X) else None,
        "split_metadata_sha256": file_sha256(split / "metadata.json"),
        "fake_config_sha256": (
            file_sha256(fake_dir.parent.parent / "config.json") if fake_dir else None
        ),
    }
    return arrays, sources


class DualNormRawCWT(nn.Module):
    """Lightweight raw/CWT fusion candidate for the A pilot."""

    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.raw = nn.Sequential(
            nn.Conv1d(2, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.cwt = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def extract_features(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raw, cwt = inputs
        return torch.cat(
            [self.raw(raw).flatten(1), self.cwt(cwt).flatten(1)], dim=1
        )

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


def _make_loaders(
    raw_train: np.ndarray,
    cwt_train: np.ndarray,
    train_y: np.ndarray,
    raw_val: np.ndarray,
    cwt_val: np.ndarray,
    val_y: np.ndarray,
    n_real: int,
    seed: int,
) -> tuple[DataLoader, DataLoader]:
    is_real = np.zeros(len(train_y), dtype=np.float32)
    is_real[:n_real] = 1.0
    generator = torch.Generator()
    generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(raw_train),
            torch.from_numpy(cwt_train),
            torch.from_numpy(train_y).long(),
            torch.from_numpy(is_real),
        ),
        batch_size=min(BATCH_SIZE, len(train_y)),
        shuffle=True,
        drop_last=False,
        generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(raw_val),
            torch.from_numpy(cwt_val),
            torch.from_numpy(val_y).long(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    return train_loader, val_loader


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    return_predictions: bool = False,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    predictions: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for raw, cwt, y in loader:
        raw = raw.to(R.DEVICE)
        cwt = cwt.to(R.DEVICE)
        y = y.to(R.DEVICE)
        logits = model((raw, cwt))
        batch_size = int(y.shape[0])
        total_loss += float(criterion(logits, y).item()) * batch_size
        total_count += batch_size
        if return_predictions:
            predictions.append(logits.argmax(dim=1).cpu().numpy())
            labels.append(y.cpu().numpy())
    result: dict[str, Any] = {"loss": total_loss / max(total_count, 1)}
    if return_predictions:
        pred = np.concatenate(predictions)
        true = np.concatenate(labels)
        result.update(
            {
                "pred": pred,
                "true": true,
                "accuracy": float(accuracy_score(true, pred)),
                "f1_macro": float(
                    f1_score(true, pred, average="macro", zero_division=0)
                ),
            }
        )
    return result


def _train(
    model: DualNormRawCWT,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_real: int,
) -> tuple[DualNormRawCWT, dict[str, list[float]], int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_MAX)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        dummy_raw = torch.zeros((1, 2, WINDOW_SIZE), device=R.DEVICE)
        dummy_cwt = torch.zeros((1, 1, 128, 128), device=R.DEVICE)
        feat_dim = model.extract_features((dummy_raw, dummy_cwt)).shape[1]
    bank = R.EMAPrototypeBank(NUM_CLASSES, feat_dim, tau=PCL_TAU).to(R.DEVICE)

    model.eval()
    # The loader is shuffled; initialize prototypes from the explicitly ordered
    # real prefix so that the EMA anchors are independent of shuffle order.
    real_raw = train_loader.dataset.tensors[0][:n_real].to(R.DEVICE)
    real_cwt = train_loader.dataset.tensors[1][:n_real].to(R.DEVICE)
    real_labels = train_loader.dataset.tensors[2][:n_real].to(R.DEVICE)
    with torch.no_grad():
        for start in range(0, n_real, min(BATCH_SIZE, n_real)):
            stop = min(start + BATCH_SIZE, n_real)
            features = model.extract_features((real_raw[start:stop], real_cwt[start:stop]))
            bank.update(
                features,
                real_labels[start:stop],
                torch.ones(stop - start, dtype=torch.bool, device=R.DEVICE),
            )
    model.train()

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_ce": [],
        "train_pcl": [],
        "val_loss": [],
        "val_acc": [],
    }
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    patience = 0

    for _epoch in range(EPOCHS):
        model.train()
        total_loss = total_ce = total_pcl = 0.0
        total_count = 0
        for raw, cwt, y, is_real in train_loader:
            raw = raw.to(R.DEVICE)
            cwt = cwt.to(R.DEVICE)
            y = y.to(R.DEVICE)
            is_real = is_real.to(R.DEVICE).bool()

            logits = model((raw, cwt))
            features = model.extract_features((raw, cwt))
            ce_loss = criterion(logits, y)
            bank.update(features.detach(), y, is_real)
            pcl_loss = bank(features, y)
            loss = ce_loss + PCL_LAMBDA * pcl_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = int(y.shape[0])
            total_loss += float(loss.item()) * batch_size
            total_ce += float(ce_loss.item()) * batch_size
            total_pcl += float(pcl_loss.item()) * batch_size
            total_count += batch_size

        val = _evaluate(model, val_loader, criterion, return_predictions=True)
        scheduler.step()
        history["train_loss"].append(total_loss / max(total_count, 1))
        history["train_ce"].append(total_ce / max(total_count, 1))
        history["train_pcl"].append(total_pcl / max(total_count, 1))
        history["val_loss"].append(float(val["loss"]))
        history["val_acc"].append(float(val["accuracy"]))

        if float(val["loss"]) < best_loss:
            best_loss = float(val["loss"])
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            patience = 0
        else:
            patience += 1
        if patience >= PATIENCE:
            break

    if best_state is None:
        raise RuntimeError("Training ended without a validation checkpoint")
    model.load_state_dict(best_state)
    return model, history, len(history["train_loss"])


def _cache_path(fold: int, shot: int, ratio: float, split_name: str, source_hash: str) -> Path:
    return CACHE_ROOT / (
        f"fold{fold}_shot{shot}_ratio{ratio:g}_{split_name}_"
        f"{source_hash[:16]}_{TRANSFORM_VERSION}.npz"
    )


def run_one(fold: int, shot: int, ratio: float, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    arrays, sources = _load_arrays(fold, shot, ratio)
    real_mean = arrays["real_mean"]
    real_std = arrays["real_std"]
    raw_train = _raw_views(arrays["train_X"], real_mean, real_std)
    raw_val = _raw_views(arrays["val_X"], real_mean, real_std)
    raw_test = _raw_views(arrays["test_X"], real_mean, real_std)

    train_hash = sources["train_source_sha256"]
    val_hash = sources["val_source_sha256"]
    test_hash = sources["test_source_sha256"]
    cwt_train, train_cache = precompute_cwt_verified(
        arrays["train_X"], _cache_path(fold, shot, ratio, "train", train_hash)
    )
    cwt_val, val_cache = precompute_cwt_verified(
        arrays["val_X"], _cache_path(fold, shot, ratio, "val", val_hash)
    )
    cwt_test, test_cache = precompute_cwt_verified(
        arrays["test_X"], _cache_path(fold, shot, ratio, "test", test_hash)
    )

    seed_everything(seed)
    train_loader, val_loader = _make_loaders(
        raw_train,
        cwt_train,
        arrays["train_y"],
        raw_val,
        cwt_val,
        arrays["val_y"],
        arrays["real_count"],
        seed,
    )
    model = DualNormRawCWT(NUM_CLASSES).to(R.DEVICE)
    model, history, epochs_run = _train(
        model,
        train_loader,
        val_loader,
        arrays["real_count"],
    )

    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(raw_test),
            torch.from_numpy(cwt_test),
            torch.from_numpy(arrays["test_y"]).long(),
        ),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test = _evaluate(model, test_loader, nn.CrossEntropyLoss(), return_predictions=True)
    elapsed = time.perf_counter() - started
    result = {
        "fold": fold,
        "shot": shot,
        "ratio": float(ratio),
        "seed": seed,
        "accuracy": float(test["accuracy"]),
        "f1_macro": float(test["f1_macro"]),
        "epochs": epochs_run,
        "elapsed_sec": float(elapsed),
        "model_params": int(sum(p.numel() for p in model.parameters())),
        "data": {
            "split_dir": arrays["split_dir"],
            "fake_dir": arrays["fake_dir"],
            "n_real": arrays["real_count"],
            "n_fake": arrays["fake_count"],
            "n_train": len(arrays["train_y"]),
            "real_mean": real_mean,
            "real_std": real_std,
            "raw_view_max_abs_difference": float(
                np.max(np.abs(raw_train[:, 0] - raw_train[:, 1]))
            ),
            "sources": sources,
        },
        "cache": {"train": train_cache, "val": val_cache, "test": test_cache},
        "training": {
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "epochs_max": EPOCHS,
            "t_max": T_MAX,
            "patience": PATIENCE,
            "batch_size": min(BATCH_SIZE, len(arrays["train_y"])),
            "pcl_lambda": PCL_LAMBDA,
            "pcl_tau": PCL_TAU,
            "proto_mode": "ema",
        },
    }
    # Keep the full predictions for later error analysis without serializing
    # model weights in this small pilot.
    pred_path = (
        OUT_ROOT
        / "predictions"
        / f"fold{fold}_shot{shot}_ratio{ratio:g}_seed{seed}.npz"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(pred_path, pred=test["pred"], true=test["true"])
    result["prediction_path"] = str(pred_path)
    return result


def _config_path(fold: int, shot: int, ratio: float, seed: int) -> Path:
    return (
        OUT_ROOT
        / "runs"
        / f"fold{fold}"
        / f"{shot}shot"
        / f"ratio{ratio:g}"
        / f"seed{seed}.json"
    )


def main() -> None:
    global OUT_ROOT, CACHE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, nargs="*", default=[0])
    parser.add_argument("--shots", type=int, nargs="*", default=DEFAULT_SHOTS)
    parser.add_argument("--ratios", type=float, nargs="*", default=DEFAULT_RATIOS)
    parser.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=OUT_ROOT,
        help="独立输出目录；用于避免不同折次覆盖汇总文件。",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    OUT_ROOT = args.output_root
    CACHE_ROOT = OUT_ROOT / "cache"
    _require_cuda()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "workspace": str(WORKSPACE),
        "source_root": str(SOURCE_ROOT),
        "device": str(R.DEVICE),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "platform": platform.platform(),
        "transform_version": TRANSFORM_VERSION,
        "folds": args.folds,
        "shots": args.shots,
        "ratios": args.ratios,
        "seeds": args.seeds,
    }
    (OUT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_rows: list[dict[str, Any]] = []
    total = len(args.folds) * len(args.shots) * len(args.ratios) * len(args.seeds)
    completed = 0
    print(json.dumps(metadata, ensure_ascii=False), flush=True)
    for fold in args.folds:
        for shot in args.shots:
            for ratio in args.ratios:
                for seed in args.seeds:
                    completed += 1
                    output_path = _config_path(fold, shot, ratio, seed)
                    if output_path.exists() and not args.force:
                        row = json.loads(output_path.read_text(encoding="utf-8"))
                        summary_rows.append(row)
                        print(
                            f"[{completed}/{total}] reuse fold={fold} shot={shot} "
                            f"ratio={ratio:g} seed={seed}",
                            flush=True,
                        )
                        continue
                    print(
                        f"[{completed}/{total}] run fold={fold} shot={shot} "
                        f"ratio={ratio:g} seed={seed}",
                        flush=True,
                    )
                    row = run_one(fold, shot, ratio, seed)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(
                        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    summary_rows.append(row)
                    print(
                        f"  acc={row['accuracy']:.4f} f1={row['f1_macro']:.4f} "
                        f"epochs={row['epochs']} elapsed={row['elapsed_sec']:.1f}s",
                        flush=True,
                    )

    aggregate: list[dict[str, Any]] = []
    for fold in args.folds:
        for shot in args.shots:
            for ratio in args.ratios:
                rows = [
                    row
                    for row in summary_rows
                    if row["fold"] == fold
                    and row["shot"] == shot
                    and float(row["ratio"]) == float(ratio)
                ]
                if not rows:
                    continue
                accs = np.asarray([row["accuracy"] for row in rows], dtype=float)
                f1s = np.asarray([row["f1_macro"] for row in rows], dtype=float)
                aggregate.append(
                    {
                        "fold": fold,
                        "shot": shot,
                        "ratio": float(ratio),
                        "n_seeds": len(rows),
                        "seeds": [int(row["seed"]) for row in rows],
                        "mean_accuracy": float(accs.mean()),
                        "std_accuracy": float(accs.std(ddof=1)) if len(accs) > 1 else 0.0,
                        "mean_f1_macro": float(f1s.mean()),
                        "std_f1_macro": float(f1s.std(ddof=1)) if len(f1s) > 1 else 0.0,
                        "repeats": rows,
                    }
                )
    summary_path = OUT_ROOT / "summary.json"
    summary_path.write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metadata["finished_utc"] = datetime.now(timezone.utc).isoformat()
    (OUT_ROOT / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"saved: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
