"""Frozen inputs and fail-closed result validation for the October review reruns."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / 'results/review_rerun_20261017'
MAIN = ('none', 'diffusion', 'jitter', 'mix')
CONTROLS = ('repeat', 'diffusion_zscore', 'mix_zscore')
METHODS = MAIN + CONTROLS
STAGES = ('core', 'controls', 'backbone', 'seed_extension')
CODE_FILES = ('run_a_small_eval_v2.py', 'run_pcl_eval.py',
              'scripts/review_rerun_protocol.py', 'scripts/review_rerun_training.py',
              'scripts/run_review_rerun.py')


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(array)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode('ascii'))
    h.update(repr(arr.shape).encode('ascii'))
    h.update(arr.tobytes(order='C'))
    return h.hexdigest()


def object_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def tag(key: dict) -> str:
    return f"f{key['fold']}_k{key['shot']}_{key['backbone']}_{key['method']}_s{key['seed']}"


def build_matrix(stages: tuple[str, ...] = STAGES, smoke: bool = False) -> list[dict]:
    if len(set(stages)) != len(stages) or not set(stages).issubset(STAGES):
        raise ValueError('Unknown or duplicated stage')
    matrix = []
    for stage in stages:
        methods = CONTROLS if stage == 'controls' else MAIN
        seeds = range(3, 10) if stage == 'seed_extension' else range(3)
        backbone = 'raw' if stage == 'backbone' else 'dual'
        if smoke:
            seeds = (3,) if stage == 'seed_extension' else (0,)
        for shot in (1, 3, 5):
            for fold in ((0,) if smoke else range(4)):
                for seed in seeds:
                    for method in methods:
                        matrix.append(dict(stage=stage, fold=fold, shot=shot, seed=seed,
                                           method=method, backbone=backbone))
    return matrix


def augment(real: np.ndarray, labels: np.ndarray, fake: np.ndarray,
            fake_labels: np.ndarray, fold: int, shot: int, method: str) -> tuple:
    if method not in METHODS:
        raise ValueError(f'Unknown method: {method}')
    for y, expected in ((labels, shot), (fake_labels, 10 * shot)):
        if not np.array_equal(np.bincount(y, minlength=10), np.full(10, expected)):
            raise ValueError('Unbalanced or incorrect support/pool class labels')
    seed = 20260914 + 100 * fold + shot
    parent = np.tile(np.arange(len(real)), 10)
    jitter = real[parent] + np.random.default_rng(seed).normal(
        0, .05, (len(parent), real.shape[1])).astype(np.float32)
    base = method.removesuffix('_zscore')
    pool = fake
    if method.endswith('_zscore'):
        pool = ((fake - fake.mean(axis=1, keepdims=True)) /
                (fake.std(axis=1, keepdims=True) + 1e-6)).astype(np.float32)
    selected = np.empty(0, dtype=np.int64)
    jitter_parent = np.empty(0, dtype=np.int64)
    repeat_parent = np.empty(0, dtype=np.int64)
    if base == 'none':
        added, ay = np.empty((0, real.shape[1]), np.float32), np.empty(0, np.int64)
    elif base == 'repeat':
        repeat_parent = parent
        added, ay = real[parent], labels[parent]
    elif base == 'diffusion':
        selected = np.arange(len(pool))
        added, ay = pool, fake_labels
    elif base == 'jitter':
        jitter_parent = parent
        added, ay = jitter, labels[parent]
    else:
        rng = np.random.default_rng(seed + 1_000_000)
        selected = np.concatenate([np.sort(rng.choice(np.flatnonzero(fake_labels == c),
                                                     5 * shot, replace=False)) for c in range(10)])
        jitter_parent = parent[:5 * len(real)]
        added = np.concatenate((pool[selected], jitter[:len(jitter_parent)]))
        ay = np.concatenate((fake_labels[selected], labels[jitter_parent]))
    x, y = np.concatenate((real, added)), np.concatenate((labels, ay))
    mask = np.arange(len(y)) < len(real)
    info = dict(seed=seed, diffusion_indices=selected.tolist(),
                jitter_parent_indices=jitter_parent.tolist(), repeat_parent_indices=repeat_parent.tolist(),
                generated_window_zscore=method.endswith('_zscore'), n_unique_real=len(real),
                n_augmented=len(added), train_X_sha256=array_hash(x), train_y_sha256=array_hash(y),
                anchor_mask_sha256=array_hash(mask))
    return x, y, mask, info


def load_arrays(source: Path, fold: int, shot: int, method: str,
                split_base: Path | None = None, generated_base: Path | None = None,
                generation_layout: str = 'legacy') -> tuple[dict, dict]:
    split_base = split_base or source / 'data/splits_loocv_v2/cwru'
    generated_base = generated_base or source / 'results/cwru_generated_loocv'
    split = split_base / f'fold{fold}/{shot}shot'
    gen = (generated_base / f'fold{fold}/diffusion/{shot}shot' if generation_layout == 'legacy'
           else generated_base / f'fold{fold}/{shot}shot')
    fake_dir = next((gen / 'generated' / ratio for ratio in ('ratio10', 'ratio10.0')
                     if (gen / 'generated' / ratio / 'fake_X.npy').is_file()), None)
    if fake_dir is None:
        raise FileNotFoundError(f'No ratio10 pool: {gen}')
    a = {f'{part}_{v}': np.load(split / f'{part}_{v}.npy', allow_pickle=False).astype(
            np.float32 if v == 'X' else np.int64)
         for part in ('train', 'val', 'test') for v in ('X', 'y')}
    fake = np.load(fake_dir / 'fake_X.npy', allow_pickle=False).astype(np.float32)
    fy = np.load(fake_dir / 'fake_y.npy', allow_pickle=False).astype(np.int64)
    for part, n in (('train', 10 * shot), ('val', 200), ('test', 1180)):
        if a[f'{part}_X'].shape != (n, 1024) or a[f'{part}_y'].shape != (n,):
            raise ValueError(f'Unexpected {part} shape in {split}')
        if not np.isfinite(a[f'{part}_X']).all():
            raise ValueError(f'Nonfinite {part}')
        if not np.array_equal(np.bincount(a[f'{part}_y'], minlength=10), np.full(10, n // 10)):
            raise ValueError(f'Unexpected {part} class counts')
    if fake.shape != (100 * shot, 1024) or not np.isfinite(fake).all():
        raise ValueError('Invalid generated pool')
    meta = json.loads((split / 'metadata.json').read_text(encoding='utf-8'))
    groups = [set(meta['train_loads']), {meta['val_load']}, {meta['test_load']}]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError('Metadata assigns a load to multiple roles')
    if meta['test_load'] != f'hp{fold}' or meta['n_shot'] != shot:
        raise ValueError('Split metadata identity mismatch')
    sources = {name: array_hash(value) for name, value in a.items()}
    sources.update(fake_X=array_hash(fake), fake_y=array_hash(fy),
                   metadata=file_hash(split / 'metadata.json'), config=file_hash(gen / 'config.json'),
                   generator_checkpoint=file_hash(gen / 'models/diffusion_model.pt'))
    real, labels = a['train_X'], a['train_y']
    if generation_layout != 'legacy':
        config = json.loads((gen / 'config.json').read_text(encoding='utf-8'))
        # New pools must carry the binding that is absent from historical pools.
        if config.get('support_X_sha256') != array_hash(real) or config.get('support_y_sha256') != array_hash(labels):
            raise ValueError('New generator pool does not match this support set')
    x, y, mask, aug = augment(real, labels, fake, fy, fold, shot, method)
    a.update(train_X=x, train_y=y, anchor_mask=mask, real_X=real, real_y=labels,
             real_mean=float(real.mean()), real_std=max(float(real.std()), 1e-6))
    info = dict(sources=sources, augmentation=aug, split_dir=str(split), pool_dir=str(fake_dir),
                real_mean=a['real_mean'], real_std=a['real_std'], generation_layout=generation_layout)
    return a, info


def preflight(root: Path, source: Path, *, stages: tuple[str, ...] = STAGES,
              smoke: bool = False, max_steps: int = 1000, val_every: int = 50,
              split_base: Path | None = None, generated_base: Path | None = None,
              generation_layout: str = 'legacy') -> dict:
    import subprocess
    if not smoke and (max_steps, val_every) != (1000, 50):
        raise ValueError('Formal protocol fixes max_steps=1000 and val_every=50')
    if not 1 <= val_every <= max_steps:
        raise ValueError('Require 1 <= val_every <= max_steps')
    matrix = build_matrix(stages, smoke)
    if not matrix:
        raise ValueError('Empty matrix')
    paths = dict(source=str(source.resolve()),
                 split_base=str((split_base or source / 'data/splits_loocv_v2/cwru').resolve()),
                 generated_base=str((generated_base or source / 'results/cwru_generated_loocv').resolve()),
                 generation_layout=generation_layout)
    settings = {}
    for key in matrix:
        setting = f"{key['fold']}/{key['shot']}/{key['method']}"
        if setting not in settings:
            _, settings[setting] = load_arrays(source, key['fold'], key['shot'], key['method'],
                Path(paths['split_base']), Path(paths['generated_base']), generation_layout)
    training = dict(max_steps=max_steps, val_every=val_every, batch_size=32, eval_batch_size=32,
                    lr=0.0005, weight_decay=0.01, pcl_lambda=0.5, pcl_tau=0.07,
                    prototype_momentum=0.99, scheduler='CosineAnnealingLR', early_stopping=False)
    code = {name: file_hash(WORKSPACE / name) for name in CODE_FILES}
    payload = dict(protocol=dict(version='review-rerun-v1', smoke_only=smoke,
                    matrix=matrix, training=training, primary_metric='accuracy',
                    scope='fixed support sets; independent draws require separate bound generators',
                    baseline_commit='66f24f2'), paths=paths, sources=settings, code=code)
    fingerprint = object_hash(payload)
    path = root / 'manifest.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['fingerprint'] != fingerprint or object_hash(saved['payload']) != fingerprint:
            raise ValueError('Inputs, code or protocol changed. Use a new output directory.')
        for name, expected in code.items():
            if file_hash(root / 'source_snapshot' / name) != expected:
                raise ValueError('Source snapshot changed')
        return saved
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=WORKSPACE,
                            capture_output=True, text=True, check=True).stdout.strip()
    status = subprocess.run(['git', 'status', '--porcelain'], cwd=WORKSPACE,
                            capture_output=True, text=True, check=True).stdout
    manifest = dict(fingerprint=fingerprint, created_utc=now(), git_commit=commit,
                    git_status=status, payload=payload)
    for name in code:
        dest = root / 'source_snapshot' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((WORKSPACE / name).read_bytes())
    dump(path, manifest)
    return manifest


def read_valid_rows(root: Path, manifest: dict, only_tags: set[str] | None = None) -> list[dict]:
    if object_hash(manifest['payload']) != manifest['fingerprint']:
        raise ValueError('Corrupt manifest fingerprint')
    expected = {tag(k): k for k in manifest['payload']['protocol']['matrix']}
    rows = []
    for path in sorted((root / 'runs').glob('*.json')):
        if only_tags is not None and path.stem not in only_tags:
            continue
        row = json.loads(path.read_text(encoding='utf-8'))
        key = expected.get(path.stem)
        if key is None or any(row.get(k) != v for k, v in key.items()):
            raise ValueError(f'Unexpected run identity: {path}')
        if row['manifest_fingerprint'] != manifest['fingerprint']:
            raise ValueError('Result belongs to a different protocol')
        setting = f"{key['fold']}/{key['shot']}/{key['method']}"
        info = manifest['payload']['sources'][setting]
        if row['data_hash'] != object_hash(info):
            raise ValueError('Result data signature differs from manifest')
        for artifact in row['artifacts'].values():
            ap = (root / artifact['path']).resolve()
            if not ap.is_relative_to(root.resolve()) or file_hash(ap) != artifact['sha256']:
                raise ValueError('Missing, modified or out-of-root artifact')
        with np.load(root / row['artifacts']['predictions']['path'], allow_pickle=False) as z:
            true, pred, probabilities = z['true'], z['pred'], z['probabilities']
        if (array_hash(true.astype(np.int64)) != info['sources']['test_y'] or
                pred.shape != true.shape or not np.isin(pred, range(10)).all() or
                probabilities.shape != (len(true), 10) or not np.isfinite(probabilities).all() or
                np.any(probabilities < 0) or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5) or
                not np.array_equal(pred, probabilities.argmax(axis=1))):
            raise ValueError('Invalid prediction arrays')
        accuracy = float(np.mean(true == pred))
        f1 = float(f1_score(true, pred, average='macro', zero_division=0))
        if (not np.isfinite([row['accuracy'], row['f1_macro'], row['elapsed_sec']]).all() or
                abs(row['accuracy'] - accuracy) > 1e-12 or abs(row['f1_macro'] - f1) > 1e-12):
            raise ValueError('Result metrics differ from saved predictions')
        training = manifest['payload']['protocol']['training']
        if (row['completed_steps'] != training['max_steps'] or
                not 1 <= row['best_step'] <= row['completed_steps'] or row['elapsed_sec'] <= 0):
            raise ValueError('Invalid optimizer budget or runtime')
        rows.append(row)
    return rows
