#!/usr/bin/env python3
"""Nested support draws and explicit support-only diffusion retraining."""
from __future__ import annotations
import argparse
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from scipy.io import loadmat

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
import split_data as SD
import review_rerun_protocol as P

VERSION = 'review_nested_support_v1'
CLASSES = list(SD.CWRU_CLASSES)
SHOTS = (1, 3, 5)
BASE_SEED = 20261017
DATA_NAME = 'review_support_draws_20261017'
GEN_NAME = 'review_support_generators_20261017'
EVAL_FILES = ('val_X.npy', 'val_y.npy', 'test_X.npy', 'test_y.npy')
file_sha256 = P.file_hash
_write_json = P.dump


def _json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def support_path(root, draw, fold, shot, split_root=None):
    return (split_root or root/'data'/DATA_NAME)/f'draw{draw}/cwru/fold{fold}/{shot}shot'


def generator_path(root, draw, fold, shot, generator_root=None):
    return (generator_root or root/'results'/GEN_NAME)/f'draw{draw}/fold{fold}/{shot}shot'


def _validate_choices(draw, fold, shots):
    if draw not in (1,2) or fold not in range(4) or not shots or any(k not in SHOTS for k in shots):
        raise ValueError('Require draw 1/2, fold 0..3 and shots from 1/3/5')


def _protect_destination(root, destination):
    target = destination.resolve()
    for relative in ('data/cwru','data/splits','data/splits_loocv','data/splits_loocv_v2',
                     'data/splits_loocv_pu','results/cwru_generated_loocv','results/pu_generated_loocv_bearing'):
        protected = (root/relative).resolve()
        if target == protected or protected in target.parents or target in protected.parents:
            raise ValueError(f'Output overlaps historical assets: {target}')


def _legacy_metadata(root, fold):
    paths = {k: root/f'data/splits_loocv_v2/cwru/fold{fold}/{k}shot' for k in SHOTS}
    metas = {k:_json(path/'metadata.json') for k,path in paths.items()}
    reference = metas[1]
    for k, meta in metas.items():
        train = meta.get('train_loads', [])
        val, test = meta.get('val_load'), meta.get('test_load')
        if (meta.get('fold') != fold or meta.get('n_shot') != k or meta.get('classes') != CLASSES
            or meta.get('num_classes') != 10 or meta.get('window_size') != 1024 or meta.get('overlap') != 0
            or len(train) != 2 or len(set(train+[val,test])) != 4
            or set(train+[val,test]) != {f'hp{i}' for i in range(4)} or test != f'hp{fold}'):
            raise ValueError(f'Invalid legacy split metadata: {paths[k]}')
        for field in ('train_loads','val_load','test_load','classes'):
            if meta[field] != reference[field]:
                raise ValueError(f'Legacy metadata differ across K: {field}')
        for subset in ('val','test'):
            x = np.load(paths[k]/f'{subset}_X.npy', allow_pickle=False)
            y = np.load(paths[k]/f'{subset}_y.npy', allow_pickle=False)
            if (x.ndim != 2 or x.shape[1] != 1024 or y.shape != (len(x),) or not np.isfinite(x).all()
                or not np.issubdtype(y.dtype,np.integer) or set(y.tolist()) != set(range(10))
                or meta.get(f'{subset}_samples') != len(x)):
                raise ValueError('Invalid legacy evaluation arrays')
            if subset == 'val' and not np.all(np.bincount(y,minlength=10) == 20):
                raise ValueError('Expected 20 validation labels per class')
            if k != 1:
                for suffix, arr in (('X',x),('y',y)):
                    ref = np.load(paths[1]/f'{subset}_{suffix}.npy', allow_pickle=False)
                    if not np.array_equal(arr,ref) or arr.dtype != ref.dtype:
                        raise ValueError('Historical evaluation arrays differ across K')
    return metas, paths


def _raw_training_pool(root, loads):
    arrays = {c:[] for c in CLASSES}
    rows = {c:[] for c in CLASSES}
    records = []
    for load in loads:
        seen = set()
        for path in sorted((root/'data/cwru'/load).glob('*.mat')):
            diameter, suffix = map(int,path.stem.split('_'))
            cls = SD._cwru_class_name(diameter,suffix)
            if cls in seen or cls not in CLASSES:
                raise ValueError('Invalid or multiple records for a load/class')
            seen.add(cls)
            mat = loadmat(path)
            keys = [k for k in mat if 'DE_time' in k and not k.startswith('__')]
            if not keys:
                raise ValueError(f'Missing DE channel: {path}')
            # Match SD.load_cwru exactly: use the first DE field in MAT order.
            # Some source containers have additional fields; record the choice
            # explicitly instead of changing the historical signal selection.
            signal = mat[keys[0]].ravel().astype(np.float64)
            if not np.isfinite(signal).all():
                raise ValueError('Nonfinite raw signal')
            windows = SD.segment_signal(signal,1024,0)
            relative, digest = path.relative_to(root).as_posix(), file_sha256(path)
            records.append(dict(path=relative,sha256=digest,channel=keys[0],available_de_fields=keys,
                channel_selection='first DE field in MAT order, matching split_data.load_cwru',load=load,
                class_name=cls,class_index=CLASSES.index(cls),signal_length=len(signal),window_count=len(windows)))
            arrays[cls].append(windows)
            offset = len(rows[cls])
            rows[cls].extend(dict(class_name=cls,class_index=CLASSES.index(cls),load=load,
                record_path=relative,record_sha256=digest,channel=keys[0],record_window_index=i,
                sample_start=i*1024,pool_index=offset+i) for i in range(len(windows)))
        if seen != set(CLASSES):
            raise ValueError(f'Missing classes in {load}')
    pooled = {c:np.concatenate(arrays[c]) for c in CLASSES}
    if any(len(a)<5 for a in pooled.values()):
        raise ValueError('Require at least 5 windows per class')
    return pooled,rows,records


def verify_prepared_support(root,draw,fold,shot,split_root=None,include_eval=True):
    directory = support_path(root,draw,fold,shot,split_root)
    meta = _json(directory/'metadata.json')
    if (meta.get('scheme')!=VERSION or meta.get('draw')!=draw or meta.get('fold')!=fold
        or meta.get('n_shot')!=shot or meta.get('classes')!=CLASSES):
        raise ValueError('Support metadata identity mismatch')
    names = ('train_X','train_y','val_X','val_y','test_X','test_y') if include_eval else ('train_X','train_y')
    for name in names:
        if P.array_hash(np.load(directory/f'{name}.npy',allow_pickle=False)) != meta['array_sha256'][name]:
            raise ValueError('Changed prepared array')
    for record in meta['raw_records']:
        if file_sha256(root/record['path']) != record['sha256']:
            raise ValueError('Changed original source')
    if include_eval:
        for name,expected in meta['copied_eval_file_sha256'].items():
            if file_sha256(directory/name)!=expected:
                raise ValueError('Copied evaluation file changed')
    return meta


def prepare_fold(root,draw,fold,shots=SHOTS,split_root=None):
    _validate_choices(draw,fold,shots)
    _protect_destination(root,split_root or root/'data'/DATA_NAME)
    metas, old_paths = _legacy_metadata(root,fold)
    ref = metas[1]
    pools,source_rows,records = _raw_training_pool(root,ref['train_loads'])
    seed = BASE_SEED+1000*draw+100*fold
    rng = np.random.RandomState(seed)
    selected = {c:rng.choice(len(pools[c]),5,replace=False) for c in CLASSES}
    maximum = {c:[source_rows[c][int(i)] for i in selected[c]] for c in CLASSES}
    out_paths = []
    for k in shots:
        source = old_paths[k]
        output = support_path(root,draw,fold,k,split_root)
        x = np.concatenate([SD.normalize_windows(pools[c][selected[c][:k]]) for c in CLASSES]).astype(np.float32)
        y = np.repeat(np.arange(10,dtype=np.int64),k)
        ev = {name[:-4]:np.load(source/name,allow_pickle=False) for name in EVAL_FILES}
        arrays = dict(train_X=x,train_y=y,**ev)
        metadata = dict(scheme=VERSION,draw=draw,fold=fold,n_shot=k,support_seed=seed,
            window_size=1024,overlap=0,classes=CLASSES,num_classes=10,train_loads=ref['train_loads'],
            val_load=ref['val_load'],test_load=ref['test_load'],train_samples=len(x),
            val_samples=len(ev['val_X']),test_samples=len(ev['test_X']),
            sampling='RandomState per draw/fold; class-order choice(5, replace=False); first K per class',
            raw_records=records,max_support_rows=maximum,selected_windows=[r for c in CLASSES for r in maximum[c][:k]],
            source_split=source.relative_to(root).as_posix(),source_metadata_sha256=file_sha256(source/'metadata.json'),
            source_code_sha256={Path(p).name:file_sha256(Path(p)) for p in (__file__,SD.__file__,P.__file__)},
            copied_eval_file_sha256={n:file_sha256(source/n) for n in EVAL_FILES},
            array_sha256={n:P.array_hash(a) for n,a in arrays.items()})
        if output.exists():
            if not (output/'metadata.json').is_file() or _json(output/'metadata.json')!=metadata:
                raise ValueError(f'Existing support manifest differs: {output}')
        else:
            output.mkdir(parents=True)
            np.save(output/'train_X.npy',x)
            np.save(output/'train_y.npy',y)
            for name in EVAL_FILES:
                shutil.copyfile(source/name,output/name)
            _write_json(output/'metadata.json',metadata)
        verify_prepared_support(root,draw,fold,k,split_root)
        out_paths.append(output)
        print(f'[support] verified draw={draw} fold={fold} K={k}',flush=True)
    return out_paths


def _seed_torch(seed,torch):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False


def _check_file_map(directory,expected):
    for relative,digest in expected.items():
        if not (directory/relative).is_file() or file_sha256(directory/relative)!=digest:
            raise ValueError('Incomplete or changed generator artifact')


def validate_generator_resume(output,expected_config):
    if not output.exists() or not any(output.iterdir()):
        return 'fresh'
    if not (output/'config.json').is_file() or _json(output/'config.json')!=expected_config:
        raise ValueError('Generator configuration or support hashes changed')
    if not (output/'training_complete.json').is_file():
        # Never load an unverified partial model. Restart the complete seeded
        # training job, writing a receipt only after model and history exist.
        return 'fresh'
    training = _json(output/'training_complete.json')
    if (training.get('config_sha256')!=file_sha256(output/'config.json')
        or set(training.get('files',{}))!={'models/diffusion_model.pt','models/diffusion_history.json'}):
        raise ValueError('Invalid training receipt')
    _check_file_map(output,training['files'])
    pool_tag = expected_config['pool_tag']
    if (output/'generation_complete.json').is_file():
        generation = _json(output/'generation_complete.json')
        if (generation.get('config_sha256')!=file_sha256(output/'config.json')
            or generation.get('training_receipt_sha256')!=file_sha256(output/'training_complete.json')
            or set(generation.get('files',{}))!={f'generated/{pool_tag}/fake_X.npy',f'generated/{pool_tag}/fake_y.npy'}):
            raise ValueError('Invalid generation receipt')
        _check_file_map(output,generation['files'])
        return 'complete'
    return 'trained'


def generate_one(root,draw,fold,shot,split_root=None,generator_root=None,smoke=False):
    _validate_choices(draw,fold,(shot,))
    output_base = generator_root or root/'results'/GEN_NAME
    _protect_destination(root,output_base)
    if smoke and (generator_root is None or generator_root.resolve()==(root/'results'/GEN_NAME).resolve()):
        raise ValueError('Smoke requires a separate explicit generator root')
    import torch
    import run_generation as G
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    support = support_path(root,draw,fold,shot,split_root)
    verify_prepared_support(root,draw,fold,shot,split_root,include_eval=False)
    x = np.load(support/'train_X.npy',allow_pickle=False).astype(np.float32,copy=False)
    y = np.load(support/'train_y.npy',allow_pickle=False).astype(np.int64,copy=False)
    if (x.shape!=(10*shot,1024) or y.shape!=(10*shot,) or not np.isfinite(x).all()
        or not np.array_equal(np.bincount(y,minlength=10),np.full(10,shot))):
        raise ValueError('Invalid generator support')
    train_seed = BASE_SEED+100000+10000*draw+100*fold+shot
    sample_seed = BASE_SEED+200000+10000*draw+100*fold+shot
    epochs = 1 if smoke else {1:2000,3:1200,5:1000}[shot]
    per_class = 1 if smoke else 10*shot
    pool_tag = 'smoke_1_per_class' if smoke else 'ratio10.0'
    config = dict(scheme='review_support_diffusion_v1',draw=draw,fold=fold,shot=shot,smoke=smoke,
        pool_tag=pool_tag,support_path=str(support.resolve()),support_manifest_sha256=file_sha256(support/'metadata.json'),
        support_X_sha256=P.array_hash(x),support_y_sha256=P.array_hash(y),
        source_code_sha256={Path(p).name:file_sha256(Path(p)) for p in (G.__file__,__file__,P.__file__)},
        training_seed=train_seed,sampling_seed=sample_seed,
        class_sampling_seeds={c:sample_seed+1000*i for i,c in enumerate(CLASSES)},epochs=epochs,
        training_batch=G.DIFF_BATCH,sampling_batch=16,training_updates=epochs*math.ceil(len(x)/min(G.DIFF_BATCH,len(x))),
        num_classes=10,classes=CLASSES,ratio=None if smoke else 10,samples_per_class=per_class,T=G.DIFF_T,
        beta_start=G.DIFF_BETA_START,beta_end=G.DIFF_BETA_END,lr=G.DIFF_LR,weight_decay=G.DIFF_WD,
        ema_decay=G.DIFF_EMA,unet_channels=G.DIFF_UNET_CH,time_dim=G.DIFF_TIME_DIM,use_diameter=False,
        torch_version=str(torch.__version__),cuda_version=torch.version.cuda,device='cuda',gpu_name=torch.cuda.get_device_name(0))
    if G.DIFF_T!=1000 or G.DIFF_BATCH!=16:
        raise ValueError('Generator defaults changed')
    output = generator_path(root,draw,fold,shot,output_base)
    state = validate_generator_resume(output,config)
    if state=='complete':
        print(f'[generation] verified complete: {output}',flush=True)
        return output
    G.DEVICE=torch.device('cuda')
    output.mkdir(parents=True,exist_ok=True)
    def progress(stage,**fields):
        _write_json(output/'status.json',dict(stage=stage,draw=draw,fold=fold,shot=shot,updated_utc=P.now(),**fields))
    try:
        if state=='fresh':
            _write_json(output/'config.json',config)
            model_dir=output/'models'
            model_dir.mkdir(exist_ok=True)
            _seed_torch(train_seed,torch)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started,clock_start=P.now(),time.perf_counter()
            progress('training')
            model,betas,alphas,cumulative=G.train_diffusion(x,y,epochs,model_dir,10,use_diameter=False)
            torch.cuda.synchronize()
            history=_json(model_dir/'diffusion_history.json')['loss']
            if len(history)!=epochs or not np.isfinite(history).all():
                raise ValueError('Incomplete or nonfinite generator training history')
            if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                raise ValueError('Nonfinite trained generator weights')
            _write_json(output/'training_complete.json',dict(config_sha256=file_sha256(output/'config.json'),
                started_utc=started,finished_utc=P.now(),elapsed_seconds=time.perf_counter()-clock_start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),optimizer_updates=config['training_updates'],
                files={p.relative_to(output).as_posix():file_sha256(p) for p in
                       (model_dir/'diffusion_model.pt',model_dir/'diffusion_history.json')}))
        else:
            model=G.CondUNet1D(num_classes=10,use_diameter=False).to(G.DEVICE)
            model.load_state_dict(torch.load(output/'models/diffusion_model.pt',map_location=G.DEVICE,weights_only=True))
            betas,alphas,cumulative=G.make_schedule(G.DIFF_T,G.DIFF_BETA_START,G.DIFF_BETA_END)
        fake_parts=[]
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started,clock_start=P.now(),time.perf_counter()
        for class_id,cls in enumerate(CLASSES):
            progress('sampling',class_id=class_id,class_name=cls)
            _seed_torch(config['class_sampling_seeds'][cls],torch)
            part=G.diffusion_generate(model,per_class,class_id,betas,alphas,cumulative)
            if part.shape!=(per_class,1024) or not np.isfinite(part).all():
                raise ValueError('Invalid generated windows')
            fake_parts.append(part.astype(np.float32,copy=False))
            print(f'[generation] draw={draw} fold={fold} K={shot} class={cls} complete',flush=True)
        torch.cuda.synchronize()
        pool=output/'generated'/pool_tag
        pool.mkdir(parents=True,exist_ok=True)
        fake_x=np.concatenate(fake_parts)
        fake_y=np.repeat(np.arange(10,dtype=np.int64),per_class)
        for name,array in (('fake_X',fake_x),('fake_y',fake_y)):
            temp=pool/f'{name}.tmp.npy'
            np.save(temp,array)
            temp.replace(pool/f'{name}.npy')
        _write_json(output/'generation_complete.json',dict(config_sha256=file_sha256(output/'config.json'),
            training_receipt_sha256=file_sha256(output/'training_complete.json'),started_utc=started,finished_utc=P.now(),
            elapsed_seconds=time.perf_counter()-clock_start,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
            fake_X_sha256=P.array_hash(fake_x),fake_y_sha256=P.array_hash(fake_y),
            files={p.relative_to(output).as_posix():file_sha256(p) for p in (pool/'fake_X.npy',pool/'fake_y.npy')}))
        validate_generator_resume(output,config)
        progress('complete')
        return output
    except BaseException as exc:
        progress('failed',error=f'{type(exc).__name__}: {exc}')
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=REPO_ROOT)
    parser.add_argument('--split-root',type=Path)
    parser.add_argument('--generator-root',type=Path)
    parser.add_argument('--draws',nargs='+',type=int,choices=(1,2),default=[1,2])
    parser.add_argument('--folds',nargs='+',type=int,choices=range(4),default=list(range(4)))
    parser.add_argument('--shots',nargs='+',type=int,choices=SHOTS,default=list(SHOTS))
    parser.add_argument('--generate',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.smoke:
        if not args.generate or args.generator_root is None:
            parser.error('Smoke requires --generate and separate --generator-root')
        if len(args.draws)!=1 or len(args.folds)!=1 or len(args.shots)!=1:
            parser.error('Smoke requires exactly one draw, fold and shot')
    for draw in args.draws:
        for fold in args.folds:
            prepare_fold(args.source_root,draw,fold,tuple(args.shots),args.split_root)
    if args.generate:
        from run_review_rerun import gpu_lock
        with gpu_lock():
            for draw in args.draws:
                for fold in args.folds:
                    for shot in args.shots:
                        generate_one(args.source_root,draw,fold,shot,args.split_root,args.generator_root,args.smoke)


if __name__=='__main__':
    main()
