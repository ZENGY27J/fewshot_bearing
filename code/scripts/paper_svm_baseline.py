"""Frozen, validation-selected signal-feature SVM reference for 36 CWRU settings."""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import review_rerun_protocol as P

ROOT = P.WORKSPACE
OUTPUT = ROOT / 'results/paper_evidence_20261019'
BUNDLES = {'legacy': ROOT / 'results/review_rerun_20261017',
           'draw1': ROOT / 'results/review_rerun_20261017_draw1',
           'draw2': ROOT / 'results/review_rerun_20261017_draw2'}
FEATURES = ['mean_abs', 'abs_peak', 'peak_to_peak', 'skewness', 'pearson_kurtosis',
            'crest_factor', 'shape_factor', 'impulse_factor', 'clearance_factor',
            'zero_crossing_rate', 'spectral_centroid', 'spectral_std',
            'spectral_entropy', 'spectral_flatness'] + [f'band_{i}' for i in range(8)]
GRID = list(itertools.product((0.1, 1.0, 10.0), ('scale', 0.01, 0.1)))


def normalized_psd(x: np.ndarray) -> np.ndarray:
    """Hann-windowed, mean-removed one-sided PSD, excluding DC, per-window unit sum."""
    a = np.asarray(x, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] < 16 or not np.isfinite(a).all():
        raise ValueError('Expected finite signal windows')
    power = abs(np.fft.rfft((a - a.mean(axis=1, keepdims=True)) * np.hanning(a.shape[1]), axis=1)) ** 2
    power[:, 1:-1 if a.shape[1] % 2 == 0 else None] *= 2
    power = power[:, 1:]
    total = power.sum(axis=1, keepdims=True)
    return np.divide(power, total, out=np.full_like(power, 1 / power.shape[1]), where=total > 0)


def features(x: np.ndarray) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64)
    psd = normalized_psd(a)
    eps = 1e-12
    centered = a - a.mean(axis=1, keepdims=True)
    sigma = np.sqrt(np.mean(centered ** 2, axis=1))
    standardized = centered / np.maximum(sigma[:, None], eps)
    mean_abs = np.mean(abs(a), axis=1)
    rms = np.sqrt(np.mean(a ** 2, axis=1))
    peak = np.max(abs(a), axis=1)
    freq = np.fft.rfftfreq(a.shape[1])[1:]
    centroid = psd @ freq
    entropy = -np.sum(psd * np.log(np.maximum(psd, eps)), axis=1) / np.log(psd.shape[1])
    flatness = np.exp(np.log(np.maximum(psd, eps)).mean(axis=1)) / psd.mean(axis=1)
    columns = [mean_abs, peak, np.ptp(a, axis=1), np.mean(standardized ** 3, axis=1),
               np.mean(standardized ** 4, axis=1), peak / np.maximum(rms, eps),
               rms / np.maximum(mean_abs, eps), peak / np.maximum(mean_abs, eps),
               peak / np.maximum(np.mean(np.sqrt(abs(a)), axis=1) ** 2, eps),
               np.mean(np.signbit(a[:, 1:]) != np.signbit(a[:, :-1]), axis=1),
               centroid, np.sqrt(np.sum(psd * (freq - centroid[:, None]) ** 2, axis=1)), entropy, flatness]
    columns.extend(psd[:, band].sum(axis=1) for band in np.array_split(np.arange(psd.shape[1]), 8))
    result = np.column_stack(columns)
    if result.shape[1] != len(FEATURES) or not np.isfinite(result).all():
        raise ValueError('Invalid fixed features')
    return result


def select_model(train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray,
                 val_y: np.ndarray) -> tuple:
    """No test argument; train-only scaling and stable, validation-only grid selection."""
    scaler = StandardScaler().fit(train_x)
    train, val = scaler.transform(train_x), scaler.transform(val_x)
    best_score, selected, records = -1.0, None, []
    for c, gamma in GRID:
        model = SVC(C=c, gamma=gamma, kernel='rbf', probability=False).fit(train, train_y)
        score = float(accuracy_score(val_y, model.predict(val)))
        records.append(dict(C=c, gamma=gamma, validation_accuracy=score))
        if score > best_score:
            best_score, selected = score, model
    return scaler, selected, records


def load_setting(bundle: str, fold: int, shot: int) -> tuple:
    manifest = json.loads((BUNDLES[bundle] / 'manifest.json').read_text(encoding='utf-8'))
    if P.object_hash(manifest['payload']) != manifest['fingerprint']:
        raise ValueError('Corrupt frozen experiment manifest')
    split = Path(manifest['payload']['paths']['split_base']) / f'fold{fold}/{shot}shot'
    expected = manifest['payload']['sources'][f'{fold}/{shot}/none']['sources']
    arrays = {f'{part}_{v}': np.load(split / f'{part}_{v}.npy', allow_pickle=False).astype(
        np.float32 if v == 'X' else np.int64) for part in ('train', 'val', 'test') for v in ('X', 'y')}
    for name, value in arrays.items():
        if P.array_hash(value) != expected[name]:
            raise ValueError(f'Changed split: {split}/{name}')
    if len(arrays['train_y']) != 10 * shot or len(arrays['val_y']) != 200:
        raise ValueError('Unexpected label budget')
    return arrays, {name: expected[name] for name in arrays}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output / 'svm'
    out.mkdir(parents=True, exist_ok=True)
    payload = dict(features=FEATURES, grid=GRID, selection='validation accuracy; first tie',
                   code_sha256=P.file_hash(Path(__file__)),
                   protocol_sha256=P.file_hash(ROOT / 'docs/paper_revision_20261019/analysis_protocol.md'),
                   manifests={k: json.loads((v / 'manifest.json').read_text())['fingerprint'] for k, v in BUNDLES.items()},
                   n_settings=36, fit_labels_per_class='K', validation_labels_per_class=20,
                   numpy_version=np.__version__, sklearn_version=sklearn.__version__)
    fingerprint = P.object_hash(payload)
    frozen = out / 'manifest.json'
    if frozen.exists() and json.loads(frozen.read_text())['fingerprint'] != fingerprint:
        raise ValueError('Protocol/code changed: use another output path')
    if not frozen.exists():
        P.dump(frozen, dict(created_utc=P.now(), fingerprint=fingerprint, payload=payload))
        (out / 'source.py').write_bytes(Path(__file__).read_bytes())
    for bundle, fold, shot in itertools.product(BUNDLES, range(4), (1, 3, 5)):
        tag = f'{bundle}_f{fold}_k{shot}'
        dest = out / tag
        result_path = dest / 'result.json'
        if result_path.exists():
            saved = json.loads(result_path.read_text())
            if saved['fingerprint'] != fingerprint:
                raise ValueError('Stale SVM result')
            for name, digest in saved['files'].items():
                if P.file_hash(dest / name) != digest:
                    raise ValueError('Changed SVM artifact')
            continue
        started = time.perf_counter()
        arrays, sources = load_setting(bundle, fold, shot)
        feature_start = time.perf_counter()
        train, val = features(arrays['train_X']), features(arrays['val_X'])
        feature_sec = time.perf_counter() - feature_start
        fit_start = time.perf_counter()
        scaler, model, grid = select_model(train, arrays['train_y'], val, arrays['val_y'])
        fit_sec = time.perf_counter() - fit_start
        # Test is first transformed/evaluated only after model selection is complete.
        inference_start = time.perf_counter()
        pred = model.predict(scaler.transform(features(arrays['test_X'])))
        inference_sec = time.perf_counter() - inference_start
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / 'predictions.npz', true=arrays['test_y'], pred=pred)
        joblib.dump(dict(scaler=scaler, classifier=model, features=FEATURES), dest / 'model.joblib')
        row = dict(bundle=bundle, fold=fold, shot=shot, method='feature_svm',
            accuracy=float(accuracy_score(arrays['test_y'], pred)),
            f1_macro=float(f1_score(arrays['test_y'], pred, average='macro', zero_division=0)),
            selected_C=model.C, selected_gamma=model.gamma, grid=grid,
            scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
            scaler_fit_samples=int(scaler.n_samples_seen_), train_val_features_sec=feature_sec,
            grid_fit_predict_sec=fit_sec, test_features_predict_sec=inference_sec,
            elapsed_sec=time.perf_counter() - started, sources=sources, fingerprint=fingerprint,
            platform=platform.platform(), processor=platform.processor(), finished_utc=P.now(),
            files={p.name: P.file_hash(p) for p in (dest / 'predictions.npz', dest / 'model.joblib')})
        P.dump(result_path, row)
        print(f'{tag}: accuracy={row["accuracy"]:.6f} C={model.C} gamma={model.gamma}', flush=True)
    P.dump(out / 'status.json', dict(complete=True, completed=36, fingerprint=fingerprint, finished_utc=P.now()))


if __name__ == '__main__':
    main()
