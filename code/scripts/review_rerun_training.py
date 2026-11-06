#!/usr/bin/env python3
"""Corrected, fixed-update classifier training for the October review reruns.

The trainer receives only support/train and validation tensors. Test-set access
belongs to the caller, after the validation-selected model has been restored.
"""
from __future__ import annotations

import copy
import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset


TRAINER_VERSION = "review_fixed_updates_v1"


def seed_everything(seed: int) -> None:
    """Seed training randomness; callers must also seed before model creation."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class RawOnly(nn.Module):
    """The original two-normalization raw branch, without the CWT branch."""

    def __init__(self, num_classes: int = 10):
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
        self.classifier = nn.Linear(128, num_classes)

    def extract_features(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        raw, _cwt = inputs
        return self.raw(raw).flatten(1)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        return self.classifier(self.extract_features(inputs))


class RealSupportPrototypeBank(nn.Module):
    """EMA anchors initialized from *all* unique labeled support examples.

Initialization aggregates feature sums and counts before normalization, so a
class spanning several encoding batches is not initialized from a partial mean.
Later updates preserve the previous real-anchor EMA rule (momentum 0.99).
"""

    def __init__(self, num_classes: int, feat_dim: int, momentum: float = 0.99,
                 tau: float = 0.07):
        super().__init__()
        if not 0 <= momentum < 1 or tau <= 0:
            raise ValueError("Require 0 <= momentum < 1 and tau > 0")
        self.momentum = float(momentum)
        self.tau = float(tau)
        self.register_buffer("prototypes", torch.zeros(num_classes, feat_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def initialize(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        self._validate_labels(labels)
        if features.ndim != 2 or features.shape != (len(labels), self.prototypes.shape[1]):
            raise ValueError("Support feature dimensions do not match prototype bank")
        # Float64 accumulation suppresses order-dependent float32 reduction error.
        # Every class must occur in support; silently retaining a zero anchor is invalid.
        class_means = []
        for cls in range(len(self.prototypes)):
            selected = features[labels == cls]
            if not len(selected):
                raise ValueError(f"Unique support is missing class {cls}")
            class_means.append(selected.double().mean(dim=0).to(features.dtype))
        means = torch.stack(class_means)
        if not torch.isfinite(means).all() or (means.norm(dim=1) == 0).any():
            raise ValueError("Support class means must be finite and nonzero")
        self.prototypes.copy_(F.normalize(means, dim=1))
        self.initialized.fill_(True)

    def _validate_labels(self, labels: torch.Tensor) -> None:
        if labels.ndim != 1 or labels.dtype != torch.long or not len(labels):
            raise ValueError("Labels must be a nonempty one-dimensional long tensor")
        if (labels < 0).any() or (labels >= len(self.prototypes)).any():
            raise ValueError("Class labels are outside the prototype bank")

    @torch.no_grad()
    def update(self, features: torch.Tensor, labels: torch.Tensor,
               is_real: torch.Tensor) -> None:
        if not bool(self.initialized.all()):
            raise RuntimeError("Initialize all class anchors before EMA updates")
        if is_real.shape != labels.shape:
            raise ValueError("Real-support mask must have the same shape as labels")
        is_real = is_real.bool()
        if not bool(is_real.any()):
            return
        real_features, real_labels = features[is_real], labels[is_real]
        self._validate_labels(real_labels)
        for cls in real_labels.unique():
            mean = F.normalize(real_features[real_labels == cls].mean(dim=0), dim=0)
            self.prototypes[cls].mul_(self.momentum).add_(mean, alpha=1 - self.momentum)

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not bool(self.initialized.all()):
            raise RuntimeError("Initialize all class anchors before computing PCL")
        similarities = F.normalize(features, dim=1) @ F.normalize(self.prototypes, dim=1).T
        return F.cross_entropy(similarities / self.tau, labels)


def _validate_dataset(dataset: TensorDataset, n_tensors: int, name: str) -> None:
    if not isinstance(dataset, TensorDataset) or len(dataset.tensors) != n_tensors:
        raise ValueError(f"{name} must be TensorDataset with {n_tensors} tensors")
    if len(dataset) == 0:
        raise ValueError(f"{name} cannot be empty")
    labels = dataset.tensors[2]
    if labels.ndim != 1 or labels.dtype != torch.long:
        raise ValueError(f"{name} labels must be one-dimensional torch.long")


@torch.no_grad()
def initialize_prototypes(model: nn.Module, support_dataset: TensorDataset,
                          bank: RealSupportPrototypeBank, *,
                          device: str | torch.device, batch_size: int = 64) -> None:
    """Encode complete support in eval mode, then initialize class means once."""
    _validate_dataset(support_dataset, 3, "support_dataset")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    features, labels = [], []
    try:
        loader = DataLoader(support_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        for raw, cwt, y in loader:
            features.append(model.extract_features((raw.to(device), cwt.to(device))))
            labels.append(y.to(device))
        bank.initialize(torch.cat(features), torch.cat(labels))
    finally:
        model.train(was_training)


@torch.no_grad()
def evaluate_model(model: nn.Module, dataset: TensorDataset, *,
                   device: str | torch.device, batch_size: int = 64,
                   return_predictions: bool = True) -> dict[str, Any]:
    """Evaluate sample-weighted CE and, optionally, predictions in input order."""
    _validate_dataset(dataset, 3, "evaluation dataset")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    was_training = model.training
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    predictions, true_labels, probabilities = [], [], []
    try:
        for raw, cwt, y in DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0):
            raw, cwt, y = raw.to(device), cwt.to(device), y.to(device)
            logits = model((raw, cwt))
            total_loss += float(F.cross_entropy(logits, y, reduction="sum").item())
            pred = logits.argmax(dim=1)
            total_correct += int((pred == y).sum().item())
            total_count += len(y)
            if return_predictions:
                predictions.append(pred.cpu().numpy())
                true_labels.append(y.cpu().numpy())
                probabilities.append(logits.softmax(dim=1).cpu().numpy())
        result: dict[str, Any] = {
            "loss": total_loss / total_count,
            "accuracy": total_correct / total_count,
        }
        if return_predictions:
            pred, true = np.concatenate(predictions), np.concatenate(true_labels)
            result.update(pred=pred, true=true, probabilities=np.concatenate(probabilities),
                          f1_macro=float(f1_score(true, pred, labels=range(model.classifier.out_features),
                                                  average="macro", zero_division=0)))
        return result
    finally:
        model.train(was_training)


def _cpu_state(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def train_fixed_steps(
    model: nn.Module, train_dataset: TensorDataset, val_dataset: TensorDataset,
    support_dataset: TensorDataset, *, device: str | torch.device, seed: int,
    max_steps: int = 1000, val_every: int = 50, batch_size: int = 32,
    eval_batch_size: int = 64, lr: float = 5e-4, weight_decay: float = 0.01,
    pcl_lambda: float = 0.5, pcl_tau: float = 0.07, prototype_momentum: float = 0.99,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    progress_every: int = 25,
) -> dict[str, Any]:
    """Run exactly ``max_steps`` AdamW updates and select minimum validation CE.

    ``mask`` identifies the original unique support prefix in the augmented
    training set. Augmented or copied rows cannot update anchors. The optimizer
    budget and validation schedule are independent of augmented dataset length.
    A fresh shuffled traversal starts whenever a training loader is exhausted.
    Model initialization is the caller's responsibility: seed before constructing
    the model if initialization must be paired across augmentation methods.
    """
    for value, name in [(max_steps, "max_steps"), (val_every, "val_every"),
                        (batch_size, "batch_size"), (eval_batch_size, "eval_batch_size"),
                        (progress_every, "progress_every")]:
        if value < 1 or int(value) != value:
            raise ValueError(f"{name} must be a positive integer")
    if lr <= 0 or weight_decay < 0 or pcl_lambda < 0:
        raise ValueError("Require lr > 0, weight_decay >= 0, pcl_lambda >= 0")
    _validate_dataset(train_dataset, 4, "train_dataset")
    _validate_dataset(val_dataset, 3, "val_dataset")
    _validate_dataset(support_dataset, 3, "support_dataset")
    mask = train_dataset.tensors[3]
    if mask.ndim != 1 or not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Training mask must be one-dimensional and binary")
    # Identity is checked in tensors, not inferred from augmented labels. This
    # rejects accidental use of synthetic rows as real-anchor support.
    for supplied, reference in zip(support_dataset.tensors, train_dataset.tensors[:3]):
        if not torch.equal(supplied, reference[mask.bool()]):
            raise ValueError("Support tensors must exactly match mask-selected original training rows")
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear):
        raise TypeError("Model must expose a Linear classifier and extract_features(inputs)")
    seed_everything(seed)
    model.to(device)
    bank = RealSupportPrototypeBank(classifier.out_features, classifier.in_features,
                                    momentum=prototype_momentum, tau=pcl_tau).to(device)
    initialize_prototypes(model, support_dataset, bank, device=device, batch_size=eval_batch_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, len(train_dataset)),
                              shuffle=True, drop_last=False, num_workers=0, generator=generator)
    iterator = iter(train_loader)
    history: list[dict[str, Any]] = []
    best_loss, best_step = float("inf"), 0
    best_state: dict[str, torch.Tensor] | None = None
    interval_loss = interval_ce = interval_pcl = 0.0
    interval_count = 0
    cumulative_samples = 0
    traversals = 1
    last_val: dict[str, Any] | None = None
    for step in range(1, max_steps + 1):
        try:
            raw, cwt, y, is_real = next(iterator)
        except StopIteration:
            traversals += 1
            iterator = iter(train_loader)
            raw, cwt, y, is_real = next(iterator)
        raw, cwt, y, is_real = raw.to(device), cwt.to(device), y.to(device), is_real.to(device).bool()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Exactly one train-mode feature encoding, shared by CE and PCL.
        features = model.extract_features((raw, cwt))
        logits = model.classifier(features)
        ce_loss = F.cross_entropy(logits, y)
        bank.update(features.detach(), y, is_real)
        pcl_loss = bank(features, y)
        loss = ce_loss + pcl_lambda * pcl_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at optimizer step {step}")
        loss.backward()
        step_lr = float(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
        count = len(y)
        cumulative_samples += count
        interval_count += count
        interval_loss += float(loss.item()) * count
        interval_ce += float(ce_loss.item()) * count
        interval_pcl += float(pcl_loss.item()) * count
        if step % val_every == 0 or step == max_steps:
            val = evaluate_model(model, val_dataset, device=device, batch_size=eval_batch_size,
                                 return_predictions=False)
            if not math.isfinite(val["loss"]):
                raise FloatingPointError(f"Nonfinite validation CE at optimizer step {step}")
            last_val = {"step": step, "train_loss": interval_loss / interval_count,
                        "train_ce": interval_ce / interval_count,
                        "train_pcl": interval_pcl / interval_count,
                        "val_loss": float(val["loss"]), "val_accuracy": float(val["accuracy"]),
                        "lr_used": step_lr, "lr_next": float(optimizer.param_groups[0]["lr"]),
                        "samples_seen": cumulative_samples, "loader_traversals": traversals}
            history.append(last_val)
            if val["loss"] < best_loss:
                best_loss, best_step, best_state = float(val["loss"]), step, _cpu_state(model)
            interval_loss = interval_ce = interval_pcl = 0.0
            interval_count = 0
        if progress_callback and (step % progress_every == 0 or step == max_steps):
            progress_callback({"completed_steps": step, "max_steps": max_steps,
                               "best_step": best_step, "best_val_loss": best_loss if best_state else None,
                               "samples_seen": cumulative_samples,
                               "last_validation": dict(last_val) if last_val else None})
    if best_state is None:
        raise RuntimeError("No validation-selected checkpoint was produced")
    checkpoint = {
        "trainer_version": TRAINER_VERSION,
        "selection": {"criterion": "minimum_validation_cross_entropy", "step": best_step,
                      "val_loss": best_loss},
        "model_state_dict": best_state,
        # These are a *final-step* audit snapshot, not an optimizer state for the
        # selected checkpoint and not a claimed resumable minibatch iterator.
        "final_training_state": {
            "completed_steps": max_steps, "model_state_dict": _cpu_state(model),
            "prototype_bank_state_dict": _cpu_state(bank),
            "optimizer_state_dict": _cpu_tree(optimizer.state_dict()),
            "scheduler_state_dict": _cpu_tree(scheduler.state_dict()),
        },
        "config": {"seed": seed, "max_steps": max_steps, "val_every": val_every,
                   "batch_size": batch_size, "eval_batch_size": eval_batch_size,
                   "lr": lr, "weight_decay": weight_decay, "pcl_lambda": pcl_lambda,
                   "pcl_tau": pcl_tau, "prototype_momentum": prototype_momentum,
                   "sampling": "shuffled_without_replacement_passes_keep_partial_batches",
                   "real_anchor_policy": "mask_selected_original_support_rows_only"},
    }
    model.load_state_dict(best_state)
    model.eval()
    return {"model": model, "history": history, "best_step": best_step,
            "completed_steps": max_steps, "best_val_loss": best_loss,
            "samples_seen": cumulative_samples, "checkpoint": checkpoint}
