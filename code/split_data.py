#!/usr/bin/env python3
"""Dataset splitting script for few-shot bearing fault diagnosis.

Splits CWRU, PU, and JNU datasets into 1/3/5-shot configurations.
Settings: window_size=1024, overlap=0, per-sample z-score normalization.

Split strategy (from task/split_task.md):
  CWRU (10-class): load-wise — train=load0, val=load1, test=load2+load3
  PU   (3-class):  bearing-wise split under N15_M07_F10
  JNU  (4-class):  speed-wise — train=600rpm, val=800rpm, test=1000rpm

Output: data/splits/{cwru,pu,jnu}/{1shot,3shot,5shot}/
"""

import json
import os
import numpy as np
from scipy.io import loadmat
from pathlib import Path

# ============================================================
# Global config
# ============================================================
WINDOW_SIZE = 1024
OVERLAP = 0
VAL_WINDOWS_PER_CLASS = 20
FEW_SHOT_SETTINGS = [1, 3, 5]
DATA_DIR = Path("data")
OUTPUT_DIR = Path("data/splits")

# ============================================================
# Utilities
# ============================================================

def segment_signal(signal: np.ndarray, window_size: int, overlap: int) -> np.ndarray:
    """Segment a 1D signal into non-overlapping windows."""
    step = window_size - overlap
    n_windows = max(0, (len(signal) - window_size) // step + 1)
    windows = np.zeros((n_windows, window_size), dtype=np.float32)
    for i in range(n_windows):
        start = i * step
        windows[i] = signal[start:start + window_size]
    return windows


def normalize_windows(windows: np.ndarray) -> np.ndarray:
    """Per-sample z-score normalization."""
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True)
    std[std == 0] = 1.0
    return ((windows - mean) / std).astype(np.float32)


# ============================================================
# CWRU
# ============================================================

CWRU_CLASSES = [
    "Normal", "B007", "B014", "B021",
    "IR007", "IR014", "IR021",
    "OR007", "OR014", "OR021",
]

def _cwru_class_name(diameter: int, suffix: int) -> str:
    """Map CWRU (diameter, suffix) -> class name.

    diameter: 0, 7, 14, 21
    suffix: 0=Normal, 1=IR, 2=B, 3=OR
    """
    if diameter == 0:
        return "Normal"
    diam_map = {7: "007", 14: "014", 21: "021"}
    fault_map = {1: "IR", 2: "B", 3: "OR"}
    return f"{fault_map[suffix]}{diam_map[diameter]}"


def load_cwru(hp_dirs: list[str]) -> dict:
    """Load CWRU windows from hp directories. Returns {class: ndarray}."""
    windows_by_class = {cls: [] for cls in CWRU_CLASSES}

    for hp in hp_dirs:
        hp_path = DATA_DIR / "cwru" / hp
        if not hp_path.is_dir():
            continue
        for mat_file in sorted(hp_path.glob("*.mat")):
            diameter_str, suffix_str = mat_file.stem.split("_")
            diameter = int(diameter_str)
            suffix = int(suffix_str)
            cls_name = _cwru_class_name(diameter, suffix)

            mat = loadmat(str(mat_file))
            de_key = next(k for k in mat if "DE_time" in k and not k.startswith("__"))
            signal = mat[de_key].ravel().astype(np.float64)
            windows = segment_signal(signal, WINDOW_SIZE, OVERLAP)
            windows_by_class[cls_name].append(windows)

    result = {}
    for cls in CWRU_CLASSES:
        arrs = windows_by_class[cls]
        result[cls] = np.concatenate(arrs, axis=0) if arrs else np.empty((0, WINDOW_SIZE), dtype=np.float32)
    return result


# ============================================================
# PU
# ============================================================

PU_CLASSES = ["ROLL", "IR", "OR"]
PU_CONDITION = "N15_M07_F10"

# Bearing -> class
PU_BEARING_TO_CLASS = {}
for _b in ["KA04", "KA15", "KA16", "KA22", "KA30"]:
    PU_BEARING_TO_CLASS[_b] = "ROLL"
for _b in ["KI14", "KI16", "KI17", "KI18", "KI21"]:
    PU_BEARING_TO_CLASS[_b] = "IR"
for _b in ["KB23", "KB24", "KB27"]:
    PU_BEARING_TO_CLASS[_b] = "OR"

# Bearing-wise split (adapted: KA25 -> KA22 since KA25 not in dataset)
PU_TRAIN_BEARINGS = ["KA04", "KI14", "KB23"]
PU_VAL_BEARINGS = ["KA16", "KI16", "KB24"]
PU_TEST_BEARINGS = ["KA22", "KI18", "KB27"]  # KA25 unavailable, using KA22


def load_pu(bearings: list[str], conditions: list[str] | None = None) -> dict:
    """Load PU windows for given bearings. Returns {class: ndarray}.

    Args:
        bearings: list of bearing IDs (e.g. ["KA04", "KI14"]).
        conditions: list of operating-condition folder names to read. Defaults to
            [PU_CONDITION] (single condition, original behaviour). Pass all three
            condition folders to POOL them per bearing (used by bearing-level LOOCV
            so that condition is present in both train and test -> the only novel
            axis at test time is the bearing identity).
    """
    if conditions is None:
        conditions = [PU_CONDITION]
    windows_by_class = {cls: [] for cls in PU_CLASSES}

    for cond in conditions:
        pu_path = DATA_DIR / "pu" / cond
        if not pu_path.is_dir():
            continue
        for bearing in bearings:
            for mat_file in sorted(pu_path.glob(f"{cond}_{bearing}_*.mat")):
                mat = loadmat(str(mat_file))
                key = next(k for k in mat if mat_file.stem.startswith(k) and not k.startswith("__"))
                signal = mat[key][0][0][2][0][6][2].ravel().astype(np.float64)
                cls_name = PU_BEARING_TO_CLASS[bearing]
                windows = segment_signal(signal, WINDOW_SIZE, OVERLAP)
                windows_by_class[cls_name].append(windows)

    result = {}
    for cls in PU_CLASSES:
        arrs = windows_by_class[cls]
        result[cls] = np.concatenate(arrs, axis=0) if arrs else np.empty((0, WINDOW_SIZE), dtype=np.float32)
    return result


# ============================================================
# JNU
# ============================================================

JNU_CLASSES = ["Normal", "Inner", "Outer", "Ball"]

JNU_PREFIX_MAP = {"n": "Normal", "ib": "Inner", "ob": "Outer", "tb": "Ball"}

JNU_TRAIN_SPEED = "600"
JNU_VAL_SPEED = "800"
JNU_TEST_SPEED = "1000"


def _jnu_prefix(filename: str) -> str | None:
    """Extract fault-type prefix from JNU filename."""
    for pfx in ["ib", "ob", "tb", "n"]:
        if filename.startswith(pfx):
            return pfx
    return None


def load_jnu(speed: str) -> dict:
    """Load JNU windows for a given speed. Returns {class: ndarray}."""
    windows_by_class = {cls: [] for cls in JNU_CLASSES}
    jnu_path = DATA_DIR / "jnu"

    for csv_file in sorted(jnu_path.glob("*.csv")):
        fname = csv_file.stem
        pfx = _jnu_prefix(fname)
        if pfx is None:
            continue
        if speed not in fname:
            continue
        cls_name = JNU_PREFIX_MAP[pfx]
        signal = np.loadtxt(str(csv_file), delimiter=None, dtype=np.float64).ravel()
        signal = signal[np.isfinite(signal)]
        windows = segment_signal(signal, WINDOW_SIZE, OVERLAP)
        windows_by_class[cls_name].append(windows)

    result = {}
    for cls in JNU_CLASSES:
        arrs = windows_by_class[cls]
        result[cls] = np.concatenate(arrs, axis=0) if arrs else np.empty((0, WINDOW_SIZE), dtype=np.float32)
    return result


# ============================================================
# Split builder
# ============================================================

# Percentile-based few-shot sampling (percentile positions within the training signal)
FEWSHOT_PERCENTILES: dict[int, list[float]] = {
    1: [0.50],
    3: [0.25, 0.50, 0.75],
    5: [0.10, 0.30, 0.50, 0.70, 0.90],
}


def _sample_at_percentiles(arr: np.ndarray, percentiles: list[float]) -> np.ndarray:
    """Sample windows at given percentile positions from the signal.

    Args:
        arr: (n_windows, window_size) — all windows from a training file.
        percentiles: e.g. [0.10, 0.30, 0.50, 0.70, 0.90]

    Returns:
        (len(percentiles), window_size) — selected windows.
    """
    n = arr.shape[0]
    indices = [int(np.floor(p * n)) for p in percentiles]
    # Clip to valid range
    indices = [max(0, min(n - 1, i)) for i in indices]
    return arr[indices]


def _sample_uniform(arr: np.ndarray, n_samples: int) -> np.ndarray:
    """Uniform interval sampling (evenly spaced) over the signal.

    Args:
        arr: (n_windows, window_size)
        n_samples: number of windows to sample (e.g. 20 for val)

    Returns:
        (n_samples, window_size)
    """
    n = arr.shape[0]
    if n <= n_samples:
        return arr.copy()
    indices = np.linspace(0, n - 1, n_samples, dtype=int)
    return arr[indices]


def build_fewshot_split(
    train_data: dict,
    val_data: dict,
    test_data: dict,
    n_shot: int,
    classes: list[str],
) -> tuple:
    """Build (train_X, train_y, val_X, val_y, test_X, test_y) for n-shot.

    Train: percentile-based sampling (e.g. 5-shot at 10/30/50/70/90%).
    Val: uniform interval sampling (20 windows/class).
    Test: class-balanced uniform sampling (all classes have equal samples).
    """
    pcts = FEWSHOT_PERCENTILES[n_shot]

    # ---- Train ----
    train_X_parts, train_y_parts = [], []
    for cls_idx, cls in enumerate(classes):
        arr = train_data[cls]
        selected = _sample_at_percentiles(arr, pcts)
        selected = normalize_windows(selected)
        train_X_parts.append(selected)
        train_y_parts.append(np.full(len(pcts), cls_idx, dtype=np.int32))

    train_X = np.concatenate(train_X_parts, axis=0)
    train_y = np.concatenate(train_y_parts, axis=0)

    # ---- Val ----
    val_X_parts, val_y_parts = [], []
    for cls_idx, cls in enumerate(classes):
        arr = val_data[cls]
        selected = _sample_uniform(arr, VAL_WINDOWS_PER_CLASS)
        selected = normalize_windows(selected)
        val_X_parts.append(selected)
        val_y_parts.append(np.full(selected.shape[0], cls_idx, dtype=np.int32))

    val_X = np.concatenate(val_X_parts, axis=0)
    val_y = np.concatenate(val_y_parts, axis=0)

    # ---- Test (class-balanced) ----
    n_per_class = min(arr.shape[0] for cls in classes if (arr := test_data[cls]).shape[0] > 0)
    n_per_class = max(n_per_class, 1)

    test_X_parts, test_y_parts = [], []
    for cls_idx, cls in enumerate(classes):
        arr = test_data[cls]
        selected = _sample_uniform(arr, n_per_class)
        selected = normalize_windows(selected)
        test_X_parts.append(selected)
        test_y_parts.append(np.full(n_per_class, cls_idx, dtype=np.int32))

    test_X = np.concatenate(test_X_parts, axis=0) if test_X_parts else np.empty((0, WINDOW_SIZE), dtype=np.float32)
    test_y = np.concatenate(test_y_parts, axis=0) if test_y_parts else np.empty((0,), dtype=np.int32)

    # Shuffle train set deterministically
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(train_X))
    train_X, train_y = train_X[perm], train_y[perm]

    return train_X, train_y, val_X, val_y, test_X, test_y


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("Few-Shot Bearing Fault Dataset Split")
    print(f"Window: {WINDOW_SIZE}  Overlap: {OVERLAP}  Val/class: {VAL_WINDOWS_PER_CLASS}")
    print("=" * 60)

    # ---- CWRU ----
    print("\n[CWRU] Loading 10-class data ...")
    cwru_train = load_cwru(["hp0"])
    cwru_val = load_cwru(["hp1"])
    cwru_test_hp2 = load_cwru(["hp2"])
    cwru_test_hp3 = load_cwru(["hp3"])
    cwru_test_combined = load_cwru(["hp2", "hp3"])

    def _build_balanced_test(data_dict, classes, n_per_class):
        """Build a class-balanced test set from a data dict."""
        parts_x, parts_y = [], []
        for cls_idx, cls in enumerate(classes):
            arr = data_dict[cls]
            selected = _sample_uniform(arr, n_per_class)
            selected = normalize_windows(selected)
            parts_x.append(selected)
            parts_y.append(np.full(n_per_class, cls_idx, dtype=np.int32))
        return (np.concatenate(parts_x, axis=0), np.concatenate(parts_y, axis=0))

    for n in FEW_SHOT_SETTINGS:
        tx, ty, vx, vy, tstx, tsty = build_fewshot_split(
            cwru_train, cwru_val, cwru_test_combined, n, CWRU_CLASSES,
        )
        out = OUTPUT_DIR / "cwru" / f"{n}shot"
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in [("train_X", tx), ("train_y", ty),
                          ("val_X", vx), ("val_y", vy),
                          ("test_X", tstx), ("test_y", tsty)]:
            np.save(out / f"{name}.npy", arr)

        # Per-condition test splits (class-balanced, hp2 and hp3 separately)
        n_per_class = int(tsty.shape[0] // len(CWRU_CLASSES))
        tst_hp2_X, tst_hp2_y = _build_balanced_test(cwru_test_hp2, CWRU_CLASSES, n_per_class)
        tst_hp3_X, tst_hp3_y = _build_balanced_test(cwru_test_hp3, CWRU_CLASSES, n_per_class)
        np.save(out / "test_hp2_X.npy", tst_hp2_X)
        np.save(out / "test_hp2_y.npy", tst_hp2_y)
        np.save(out / "test_hp3_X.npy", tst_hp3_X)
        np.save(out / "test_hp3_y.npy", tst_hp3_y)

        meta = {
            "dataset": "CWRU", "n_shot": n,
            "window_size": WINDOW_SIZE, "overlap": OVERLAP,
            "classes": CWRU_CLASSES, "num_classes": len(CWRU_CLASSES),
            "train_files": "hp0 (load 0)", "val_files": "hp1 (load 1)",
            "test_files": "hp2+hp3 (load 2+3)",
            "train_samples": int(tx.shape[0]), "val_samples": int(vx.shape[0]),
            "test_samples": int(tstx.shape[0]),
            "test_per_class": n_per_class,
            "test_hp2_samples": int(tst_hp2_X.shape[0]),
            "test_hp3_samples": int(tst_hp3_X.shape[0]),
            "split_type": "load-wise (balanced test, per-condition)",
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  {n}-shot: train {tx.shape}  val {vx.shape}  test {tstx.shape}  "
              f"(per class: {n_per_class})")
        print(f"    test_hp2: {tst_hp2_X.shape}  test_hp3: {tst_hp3_X.shape}")

    # ---- PU ----
    print("\n[PU] Loading 3-class data (condition: {}) ...".format(PU_CONDITION))
    pu_train = load_pu(PU_TRAIN_BEARINGS)
    pu_val = load_pu(PU_VAL_BEARINGS)
    pu_test = load_pu(PU_TEST_BEARINGS)

    for n in FEW_SHOT_SETTINGS:
        tx, ty, vx, vy, tstx, tsty = build_fewshot_split(
            pu_train, pu_val, pu_test, n, PU_CLASSES
        )
        out = OUTPUT_DIR / "pu" / f"{n}shot"
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in [("train_X", tx), ("train_y", ty),
                          ("val_X", vx), ("val_y", vy),
                          ("test_X", tstx), ("test_y", tsty)]:
            np.save(out / f"{name}.npy", arr)
        meta = {
            "dataset": "PU", "condition": PU_CONDITION,
            "n_shot": n, "window_size": WINDOW_SIZE, "overlap": OVERLAP,
            "classes": PU_CLASSES, "num_classes": len(PU_CLASSES),
            "train_bearings": PU_TRAIN_BEARINGS,
            "val_bearings": PU_VAL_BEARINGS,
            "test_bearings": PU_TEST_BEARINGS,
            "bearing_class_mapping": {
                "ROLL": ["KA04", "KA15", "KA16", "KA22", "KA30"],
                "IR": ["KI14", "KI16", "KI17", "KI18", "KI21"],
                "OR": ["KB23", "KB24", "KB27"],
            },
            "note": "KA25 not in dataset; substituted KA22 for test ROLL",
            "train_samples": int(tx.shape[0]), "val_samples": int(vx.shape[0]),
            "test_samples": int(tstx.shape[0]),
            "split_type": "bearing-wise",
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  {n}-shot: train {tx.shape}  val {vx.shape}  test {tstx.shape}")

    # ---- JNU ----
    print("\n[JNU] Loading 4-class data ...")
    jnu_train = load_jnu(JNU_TRAIN_SPEED)
    jnu_val = load_jnu(JNU_VAL_SPEED)
    jnu_test = load_jnu(JNU_TEST_SPEED)

    for n in FEW_SHOT_SETTINGS:
        tx, ty, vx, vy, tstx, tsty = build_fewshot_split(
            jnu_train, jnu_val, jnu_test, n, JNU_CLASSES
        )
        out = OUTPUT_DIR / "jnu" / f"{n}shot"
        out.mkdir(parents=True, exist_ok=True)
        for name, arr in [("train_X", tx), ("train_y", ty),
                          ("val_X", vx), ("val_y", vy),
                          ("test_X", tstx), ("test_y", tsty)]:
            np.save(out / f"{name}.npy", arr)
        meta = {
            "dataset": "JNU", "n_shot": n,
            "window_size": WINDOW_SIZE, "overlap": OVERLAP,
            "classes": JNU_CLASSES, "num_classes": len(JNU_CLASSES),
            "train_speed": "600 rpm", "val_speed": "800 rpm",
            "test_speed": "1000 rpm",
            "train_samples": int(tx.shape[0]), "val_samples": int(vx.shape[0]),
            "test_samples": int(tstx.shape[0]),
            "split_type": "speed-wise",
        }
        with open(out / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  {n}-shot: train {tx.shape}  val {vx.shape}  test {tstx.shape}")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print(f"Splits saved to: {OUTPUT_DIR.resolve()}")
    print("\nSummary:")
    for ds in ["cwru", "pu", "jnu"]:
        print(f"  {ds}:")
        for n in FEW_SHOT_SETTINGS:
            with open(OUTPUT_DIR / ds / f"{n}shot" / "metadata.json") as f:
                m = json.load(f)
            print(f"    {n}-shot  "
                  f"train={m['train_samples']:>5d}  "
                  f"val={m['val_samples']:>5d}  "
                  f"test={m['test_samples']:>6d}")
    print("=" * 60)


if __name__ == "__main__":
    main()
