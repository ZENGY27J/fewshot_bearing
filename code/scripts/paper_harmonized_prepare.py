"""Prepare explicitly selected, 12 kHz CWRU records and support-bound DDPM pools.

Preparation is CPU-only by default. Generation requires --generate; historical
splits and generation artifacts are never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.io import loadmat
from scipy.signal import resample_poly

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_rerun_protocol as P
from prepare_review_support_draws import validate_generator_resume

VERSION = 'paper_cwru_12k_v1'
CLASSES = ['Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021']
SHOTS = (1, 3, 5)
BASE_SEED = 20261019
WINDOW_SIZE = 1024
TARGET_FS = 12000
DEFAULT_DATA_ROOT = REPO_ROOT/'data/paper_cwru_12k_20261019'
DEFAULT_GENERATOR_ROOT = REPO_ROOT/'results/paper_cwru_12k_20261019/generators'
RECORDS = {
    'Normal': ('0_0.mat', 97), 'B007': ('7_2.mat', 118),
    'B014': ('14_2.mat', 185), 'B021': ('21_2.mat', 222),
    'IR007': ('7_1.mat', 105), 'IR014': ('14_1.mat', 169),
    'IR021': ('21_1.mat', 209), 'OR007': ('7_3.mat', 130),
    'OR014': ('14_3.mat', 197), 'OR021': ('21_3.mat', 234),
}
ARRAY_NAMES = ('train_X', 'train_y', 'val_X', 'val_y', 'test_X', 'test_y')
PREP_CODE_FILES = ('scripts/paper_harmonized_prepare.py', 'scripts/review_rerun_protocol.py')
GEN_CODE_FILES = PREP_CODE_FILES + ('run_generation.py', 'scripts/prepare_review_support_draws.py', 'split_data.py')


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def support_path(draw: int, fold: int, shot: int, data_root: Path = DEFAULT_DATA_ROOT) -> Path:
    return Path(data_root)/f'draw{draw}/cwru/fold{fold}/{shot}shot'


def generator_path(draw: int, fold: int, shot: int,
                   generator_root: Path = DEFAULT_GENERATOR_ROOT) -> Path:
    return Path(generator_root)/f'draw{draw}/fold{fold}/{shot}shot'


def _choices(draws, folds, shots) -> None:
    for values, allowed in ((draws, (0, 1, 2)), (folds, range(4)), (shots, SHOTS)):
        if not values or len(set(values)) != len(values) or not set(values).issubset(allowed):
            raise ValueError('Require distinct draws 0/1/2, folds 0..3, shots 1/3/5')


def _protect_destination(destination: Path, source_root: Path) -> None:
    target = destination.resolve()
    if target == source_root.resolve() or target in source_root.resolve().parents:
        raise ValueError('Output may not be the source repository or an ancestor')
    protected_names = ('data/cwru', 'data/splits', 'data/splits_loocv', 'data/splits_loocv_v2',
        'data/splits_loocv_pu', 'data/review_support_draws_20261017',
        'results/cwru_generated_loocv', 'results/review_support_generators_20261017')
    for relative in protected_names:
        protected = (source_root/relative).resolve()
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError(f'Output overlaps historical assets: {target}')


def _code_hashes(generation: bool = False) -> dict:
    return {name: P.file_hash(REPO_ROOT/name) for name in
            (GEN_CODE_FILES if generation else PREP_CODE_FILES)}


def _freeze_code(directory: Path, hashes: dict) -> None:
    for relative, digest in hashes.items():
        source = REPO_ROOT/relative
        destination = directory/relative
        if P.file_hash(source) != digest:
            raise ValueError('Code changed during preparation')
        if destination.exists():
            if P.file_hash(destination) != digest:
                raise ValueError('Frozen source snapshot changed')
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name+'.tmp')
            shutil.copyfile(source, temporary)
            temporary.replace(destination)


def _checked_json(path: Path, content: dict) -> None:
    if path.exists():
        if _json(path) != content:
            raise ValueError(f'Existing manifest differs: {path}')
    else:
        P.dump(path, content)


def load_record(source_root: Path, load: int, class_name: str) -> tuple[np.ndarray, dict]:
    """Resample the complete selected recording, then segment and normalize."""
    filename, base_id = RECORDS[class_name]
    path = source_root/f'data/cwru/hp{load}'/filename
    expected_field = f'X{base_id + load:03d}_DE_time'
    source_hash = P.file_hash(path)
    mat = loadmat(path)
    fields = [name for name in mat if name.endswith('_DE_time')]
    if expected_field not in mat:
        raise ValueError(f'Missing explicit channel {expected_field}: {path}; available={fields}')
    raw = np.asarray(mat[expected_field], dtype=np.float64)
    if raw.ndim not in (1, 2) or (raw.ndim == 2 and 1 not in raw.shape):
        raise ValueError(f'Expected a vector channel: {path}/{expected_field}')
    raw = raw.reshape(-1)
    if not np.isfinite(raw).all() or not len(raw):
        raise ValueError('Empty or nonfinite source signal')
    source_fs = 48000 if class_name == 'Normal' else TARGET_FS
    down = source_fs//TARGET_FS
    # Whole-record antialias filtering precedes every window boundary.
    signal = resample_poly(raw, 1, down, window=('kaiser', 5.0), padtype='constant') if down > 1 else raw
    count = len(signal)//WINDOW_SIZE
    if count < 20:
        raise ValueError('Each record must contain at least 20 complete 12 kHz windows')
    windows = signal[:count*WINDOW_SIZE].reshape(count, WINDOW_SIZE)
    mean, std = windows.mean(axis=1, keepdims=True), windows.std(axis=1, keepdims=True)
    if (std == 0).any():
        raise ValueError('Constant raw window: inspect input before preparing data')
    windows = ((windows-mean)/std).astype(np.float32)
    if P.file_hash(path) != source_hash:
        raise ValueError('Source changed while reading')
    record = dict(path=path.relative_to(source_root).as_posix(), sha256=source_hash,
        load=f'hp{load}', class_name=class_name, class_index=CLASSES.index(class_name),
        channel=expected_field, available_de_fields=fields,
        channel_selection='Explicit official record id for class and load; never first field',
        source_sampling_rate_hz=source_fs, target_sampling_rate_hz=TARGET_FS,
        source_sample_count=len(raw), resampled_sample_count=len(signal),
        full_window_count=count, resample_up=1, resample_down=down,
        resampling='scipy.signal.resample_poly whole record, Kaiser beta=5, constant padding' if down > 1 else 'identity',
        time_origin='selected recording sample 0; acquisition wall time unavailable',
        raw_channel_array_sha256=P.array_hash(raw),
        sampling_rate_basis='CWRU normal baseline DE 48 kHz; selected 12 kHz drive-end fault records')
    return windows, record


def collect_records(source_root: Path = REPO_ROOT) -> tuple[dict, dict, int]:
    source_root = Path(source_root)
    arrays, records = {}, {}
    for load in range(4):
        expected = {v[0] for v in RECORDS.values()}
        actual = {p.name for p in (source_root/f'data/cwru/hp{load}').glob('*.mat')}
        if actual != expected:
            raise ValueError(f'Expected exactly ten known records in hp{load}')
        for cls in CLASSES:
            key = f'hp{load}/{cls}'
            arrays[key], records[key] = load_record(source_root, load, cls)
    count = min(len(value) for value in arrays.values())
    # Equal record duration, determined once across all forty records.
    for key in arrays:
        arrays[key] = arrays[key][:count]
        records[key]['retained_window_count'] = count
    return arrays, records, count


def _split_relation(source_root: Path, fold: int) -> tuple[dict, dict]:
    metas, digests = {}, {}
    for shot in SHOTS:
        path = source_root/f'data/splits_loocv_v2/cwru/fold{fold}/{shot}shot/metadata.json'
        meta = _json(path)
        train, val, test = meta.get('train_loads', []), meta.get('val_load'), meta.get('test_load')
        if (meta.get('fold') != fold or meta.get('n_shot') != shot or meta.get('classes') != CLASSES
                or meta.get('num_classes') != 10 or meta.get('window_size') != WINDOW_SIZE
                or meta.get('overlap') != 0 or len(train) != 2
                or len(set(train+[val, test])) != 4
                or set(train+[val, test]) != {f'hp{i}' for i in range(4)} or test != f'hp{fold}'):
            raise ValueError(f'Invalid load-role metadata: {path}')
        metas[shot] = {key: meta[key] for key in ('train_loads', 'val_load', 'test_load')}
        digests[path.relative_to(source_root).as_posix()] = P.file_hash(path)
    if any(value != metas[1] for value in metas.values()):
        raise ValueError('Load roles differ across K')
    return metas[1], digests


def _row(record: dict, index: int) -> dict:
    start, end = index*WINDOW_SIZE, (index+1)*WINDOW_SIZE
    factor = record['resample_down']
    return dict(class_name=record['class_name'], class_index=record['class_index'], load=record['load'],
        record_path=record['path'], record_sha256=record['sha256'], channel=record['channel'],
        record_window_index=index, target_sample_start=start, target_sample_stop_exclusive=end,
        source_nominal_sample_start=start*factor, source_nominal_sample_stop_exclusive=end*factor,
        time_start_seconds=start/TARGET_FS, time_stop_seconds=end/TARGET_FS,
        source_interval_note='nominal interval; whole-record antialias FIR has neighboring-sample context')


def _build_setting(draw: int, fold: int, shot: int, pools: dict, records: dict,
                   count: int, relation: dict, relation_hashes: dict, audit_hash: str) -> tuple[dict, dict]:
    seed = BASE_SEED+1000*draw+100*fold
    rng = np.random.RandomState(seed)
    maximum, train_parts, val_parts, test_parts = {}, [], [], []
    support_rows, val_rows, test_rows = [], [], []
    val_indices = np.linspace(0, count-1, 20, dtype=np.int64)
    for cls in CLASSES:
        keys = [f'{load}/{cls}' for load in relation['train_loads']]
        candidates = np.concatenate([pools[key] for key in keys])
        selected = rng.choice(2*count, 5, replace=False)
        maximum[cls] = [_row(records[keys[int(index)//count]], int(index)%count) for index in selected]
        train_parts.append(candidates[selected[:shot]])
        support_rows.extend(maximum[cls][:shot])
        for role, indices, parts, rows in (('val', val_indices, val_parts, val_rows),
                ('test', range(count), test_parts, test_rows)):
            key = f"{relation[role+'_load']}/{cls}"
            parts.append(pools[key][indices])
            rows.extend(_row(records[key], int(index)) for index in indices)
    arrays = dict(train_X=np.concatenate(train_parts), train_y=np.repeat(np.arange(10, dtype=np.int64), shot),
        val_X=np.concatenate(val_parts), val_y=np.repeat(np.arange(10, dtype=np.int64), 20),
        test_X=np.concatenate(test_parts), test_y=np.repeat(np.arange(10, dtype=np.int64), count))
    metadata = dict(scheme=VERSION, draw=draw, fold=fold, n_shot=shot, support_seed=seed,
        classes=CLASSES, num_classes=10, window_size=WINDOW_SIZE, overlap=0,
        target_sampling_rate_hz=TARGET_FS, common_windows_per_record=count,
        train_samples=10*shot, val_samples=200, test_samples=10*count, **relation,
        support_sampling='RandomState(seed), class-order choice(2*N,5,replace=False), nested first K',
        validation_sampling='20 deterministic linspace integer indices spanning first N windows',
        validation_indices=val_indices.tolist(), test_sampling='all first N windows of held-out record',
        max_support_rows=maximum, selected_windows=support_rows, val_rows=val_rows, test_rows=test_rows,
        raw_records=list(records.values()), source_metadata_sha256=relation_hashes,
        source_audit_sha256=audit_hash, source_code_sha256=_code_hashes(),
        array_sha256={name: P.array_hash(array) for name, array in arrays.items()})
    return arrays, metadata


def _validate_files(directory: Path, meta: dict) -> None:
    for name in ARRAY_NAMES:
        path = directory/f'{name}.npy'
        if P.file_hash(path) != meta['file_sha256'][path.name]:
            raise ValueError(f'Prepared file changed: {path}')
        value = np.load(path, allow_pickle=False)
        if P.array_hash(value) != meta['array_sha256'][name] or not np.isfinite(value).all():
            raise ValueError(f'Prepared array changed: {path}')


def prepare_all(data_root: Path = DEFAULT_DATA_ROOT, draws=(0, 1, 2), folds=(0, 1, 2, 3),
                shots=SHOTS, source_root: Path = REPO_ROOT) -> list[Path]:
    """Prepare selected settings after validating all forty authoritative sources."""
    draws, folds, shots = tuple(draws), tuple(folds), tuple(shots)
    _choices(draws, folds, shots)
    source_root, data_root = Path(source_root), Path(data_root)
    _protect_destination(data_root, source_root)
    pools, records, count = collect_records(source_root)
    relations = {fold: _split_relation(source_root, fold) for fold in range(4)}
    audit = dict(scheme=VERSION, source_root=str(source_root.resolve()), records=list(records.values()),
        common_windows_per_record=count, source_code_sha256=_code_hashes(),
        source_metadata_sha256={k:v for _, digests in relations.values() for k,v in digests.items()},
        numpy_version=np.__version__, scipy_version=scipy.__version__)
    _checked_json(data_root/'source_audit.json', audit)
    _freeze_code(data_root/'code_snapshot', audit['source_code_sha256'])
    audit_hash = P.file_hash(data_root/'source_audit.json')
    outputs = []
    for draw in draws:
        for fold in folds:
            relation, digests = relations[fold]
            for shot in shots:
                arrays, metadata = _build_setting(draw, fold, shot, pools, records, count, relation, digests, audit_hash)
                destination = support_path(draw, fold, shot, data_root)
                if destination.exists():
                    existing = _json(destination/'metadata.json')
                    if {k:v for k,v in existing.items() if k != 'file_sha256'} != metadata:
                        raise ValueError(f'Existing support manifest differs: {destination}')
                    _validate_files(destination, existing)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    staging = Path(tempfile.mkdtemp(prefix=f'.{shot}shot-', dir=destination.parent))
                    try:
                        for name, array in arrays.items():
                            np.save(staging/f'{name}.npy', array, allow_pickle=False)
                        metadata['file_sha256'] = {f'{name}.npy':P.file_hash(staging/f'{name}.npy') for name in arrays}
                        P.dump(staging/'metadata.json', metadata)
                        staging.rename(destination)
                    finally:
                        # staging is a verified local temporary directory created above.
                        if staging.exists():
                            shutil.rmtree(staging)
                outputs.append(destination)
                print(f'[prepared] draw={draw} fold={fold} K={shot} N={count}', flush=True)
    return outputs


def verify_support(draw: int, fold: int, shot: int, data_root: Path = DEFAULT_DATA_ROOT,
                   source_root: Path = REPO_ROOT) -> dict:
    """Fail closed on code, raw source, roles, provenance, or array changes."""
    _choices((draw,), (fold,), (shot,))
    data_root, source_root = Path(data_root), Path(source_root)
    directory = support_path(draw, fold, shot, data_root)
    meta = _json(directory/'metadata.json')
    audit = _json(data_root/'source_audit.json')
    if audit['source_root'] != str(source_root.resolve()) or audit['source_code_sha256'] != _code_hashes():
        raise ValueError('Preparation source root or code changed')
    _freeze_code(data_root/'code_snapshot', audit['source_code_sha256'])
    if meta.get('source_audit_sha256') != P.file_hash(data_root/'source_audit.json'):
        raise ValueError('Source audit changed')
    pools, records, count = collect_records(source_root)
    if list(records.values()) != audit['records'] or count != audit['common_windows_per_record']:
        raise ValueError('Raw source records changed')
    relation, digests = _split_relation(source_root, fold)
    if any(audit['source_metadata_sha256'].get(name) != digest for name, digest in digests.items()):
        raise ValueError('Source split metadata changed')
    arrays, expected = _build_setting(draw, fold, shot, pools, records, count, relation, digests,
        P.file_hash(data_root/'source_audit.json'))
    if {k:v for k,v in meta.items() if k != 'file_sha256'} != expected:
        raise ValueError('Prepared provenance manifest differs from reconstructed sources')
    _validate_files(directory, meta)
    for name, value in arrays.items():
        if not np.array_equal(value, np.load(directory/f'{name}.npy', allow_pickle=False)):
            raise ValueError('Prepared arrays do not reproduce from sources')
    return meta


def _seed_torch(seed: int, torch) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_one(draw: int, fold: int, shot: int, data_root: Path = DEFAULT_DATA_ROOT,
                 generator_root: Path = DEFAULT_GENERATOR_ROOT, source_root: Path = REPO_ROOT,
                 smoke: bool = False) -> Path:
    _choices((draw,), (fold,), (shot,))
    generator_root = Path(generator_root)
    _protect_destination(generator_root, Path(source_root))
    if smoke and (generator_root.resolve() == DEFAULT_GENERATOR_ROOT.resolve()
            or DEFAULT_GENERATOR_ROOT.resolve() in generator_root.resolve().parents):
        raise ValueError('Smoke requires an independent generator root outside the formal root')
    verify_support(draw, fold, shot, data_root, source_root)
    import torch
    import run_generation as G
    if not torch.cuda.is_available() or G.DEVICE.type != 'cuda':
        raise RuntimeError('CUDA required for generation; preparation alone needs no GPU')
    if G.DIFF_T != 1000 or G.DIFF_BATCH != 16 or G.DIFF_EPOCHS != {1:2000, 3:1200, 5:1000}:
        raise ValueError('Historical DDPM defaults changed')
    torch.set_num_threads(4)
    support = support_path(draw, fold, shot, data_root)
    x = np.load(support/'train_X.npy', allow_pickle=False)
    y = np.load(support/'train_y.npy', allow_pickle=False)
    train_seed = BASE_SEED+100000+10000*draw+100*fold+shot
    sample_seed = BASE_SEED+200000+10000*draw+100*fold+shot
    epochs, per_class = (1, 1) if smoke else (G.DIFF_EPOCHS[shot], 10*shot)
    pool_tag = 'smoke_1_per_class' if smoke else 'ratio10.0'
    config = dict(scheme='paper_cwru_12k_diffusion_v1', draw=draw, fold=fold, shot=shot, smoke=smoke,
        pool_tag=pool_tag, support_path=str(support.resolve()),
        support_manifest_sha256=P.file_hash(support/'metadata.json'),
        support_X_sha256=P.array_hash(x), support_y_sha256=P.array_hash(y),
        source_code_sha256=_code_hashes(generation=True), training_seed=train_seed, sampling_seed=sample_seed,
        class_sampling_seeds={cls:sample_seed+1000*i for i,cls in enumerate(CLASSES)},
        epochs=epochs, training_batch=G.DIFF_BATCH, sampling_batch=16,
        training_updates=epochs*math.ceil(len(x)/min(G.DIFF_BATCH,len(x))),
        num_classes=10, classes=CLASSES, ratio=None if smoke else 10, samples_per_class=per_class,
        T=G.DIFF_T, beta_start=G.DIFF_BETA_START, beta_end=G.DIFF_BETA_END,
        lr=G.DIFF_LR, weight_decay=G.DIFF_WD, ema_decay=G.DIFF_EMA,
        unet_channels=G.DIFF_UNET_CH, time_dim=G.DIFF_TIME_DIM, use_diameter=False,
        torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        device='cuda', gpu_name=torch.cuda.get_device_name(0))
    output = generator_path(draw, fold, shot, generator_root)
    state = validate_generator_resume(output, config)
    if state == 'complete':
        _freeze_code(output/'code_snapshot', config['source_code_sha256'])
        print(f'[generator] verified complete: {output}', flush=True)
        return output
    output.mkdir(parents=True, exist_ok=True)
    _freeze_code(output/'code_snapshot', config['source_code_sha256'])
    def progress(stage: str, **fields) -> None:
        P.dump(output/'status.json', dict(stage=stage, draw=draw, fold=fold, shot=shot,
            smoke=smoke, updated_utc=P.now(), **fields))
    try:
        if state == 'fresh':
            P.dump(output/'config.json', config)
            # G writes into a staging model directory; publish completed files atomically.
            model_stage = output/'model_staging'
            model_stage.mkdir(exist_ok=True)
            _seed_torch(train_seed, torch)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started, clock_start = P.now(), time.perf_counter()
            progress('training')
            model, betas, alphas, cumulative = G.train_diffusion(x,y,epochs,model_stage,10,use_diameter=False)
            torch.cuda.synchronize()
            history = _json(model_stage/'diffusion_history.json')['loss']
            if len(history) != epochs or not np.isfinite(history).all():
                raise ValueError('Incomplete or nonfinite diffusion training history')
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError('Nonfinite trained generator weights')
            model_dir = output/'models'
            model_dir.mkdir(exist_ok=True)
            for name in ('diffusion_model.pt', 'diffusion_history.json'):
                (model_stage/name).replace(model_dir/name)
            model_stage.rmdir()
            P.dump(output/'training_complete.json', dict(config_sha256=P.file_hash(output/'config.json'),
                started_utc=started, finished_utc=P.now(), elapsed_seconds=time.perf_counter()-clock_start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(), optimizer_updates=config['training_updates'],
                files={f'models/{name}':P.file_hash(model_dir/name) for name in
                    ('diffusion_model.pt','diffusion_history.json')}))
        else:
            model = G.CondUNet1D(num_classes=10, use_diameter=False).to(G.DEVICE)
            model.load_state_dict(torch.load(output/'models/diffusion_model.pt', map_location=G.DEVICE, weights_only=True))
            betas, alphas, cumulative = G.make_schedule(G.DIFF_T,G.DIFF_BETA_START,G.DIFF_BETA_END)
        fake_parts = []
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started, clock_start = P.now(), time.perf_counter()
        for class_id, cls in enumerate(CLASSES):
            progress('sampling', class_id=class_id, class_name=cls)
            _seed_torch(config['class_sampling_seeds'][cls], torch)
            part = G.diffusion_generate(model,per_class,class_id,betas,alphas,cumulative)
            if part.shape != (per_class,1024) or not np.isfinite(part).all():
                raise ValueError('Invalid generated windows')
            fake_parts.append(part.astype(np.float32,copy=False))
            print(f'[generation] draw={draw} fold={fold} K={shot} class={cls}', flush=True)
        torch.cuda.synchronize()
        pool = output/'generated'/pool_tag
        pool.mkdir(parents=True, exist_ok=True)
        fake_x = np.concatenate(fake_parts)
        fake_y = np.repeat(np.arange(10,dtype=np.int64),per_class)
        for name,array in (('fake_X',fake_x),('fake_y',fake_y)):
            temporary = pool/f'{name}.tmp.npy'
            np.save(temporary,array,allow_pickle=False)
            temporary.replace(pool/f'{name}.npy')
        P.dump(output/'generation_complete.json',dict(config_sha256=P.file_hash(output/'config.json'),
            training_receipt_sha256=P.file_hash(output/'training_complete.json'),
            started_utc=started,finished_utc=P.now(),elapsed_seconds=time.perf_counter()-clock_start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            fake_X_sha256=P.array_hash(fake_x),fake_y_sha256=P.array_hash(fake_y),
            files={f'generated/{pool_tag}/{name}.npy':P.file_hash(pool/f'{name}.npy') for name in ('fake_X','fake_y')}))
        validate_generator_resume(output, config)
        progress('complete')
        return output
    except BaseException as exc:
        progress('failed', error=f'{type(exc).__name__}: {exc}')
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=REPO_ROOT)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument('--generator-root', type=Path, default=DEFAULT_GENERATOR_ROOT)
    parser.add_argument('--draws', nargs='+', type=int, choices=(0,1,2), default=[0,1,2])
    parser.add_argument('--folds', nargs='+', type=int, choices=range(4), default=list(range(4)))
    parser.add_argument('--shots', nargs='+', type=int, choices=SHOTS, default=list(SHOTS))
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if args.smoke and (not args.generate or args.generator_root.resolve() == DEFAULT_GENERATOR_ROOT.resolve()
            or len(args.draws) != 1 or len(args.folds) != 1 or len(args.shots) != 1):
        parser.error('Smoke requires --generate, separate generator root, exactly one draw/fold/shot')
    prepare_all(args.data_root,args.draws,args.folds,args.shots,args.source_root)
    if args.generate:
        from run_review_rerun import gpu_lock
        with gpu_lock():
            for draw in args.draws:
                for fold in args.folds:
                    for shot in args.shots:
                        generate_one(draw,fold,shot,args.data_root,args.generator_root,args.source_root,args.smoke)


if __name__ == '__main__':
    main()
