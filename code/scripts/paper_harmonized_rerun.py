"""Fixed-update CWRU reruns with explicit records and harmonized 12 kHz inputs."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import f1_score

import review_rerun_protocol as P
from prepare_review_support_draws import validate_generator_resume

WORKSPACE=P.WORKSPACE
DATA_ROOT=WORKSPACE/'data/paper_cwru_12k_20261019'
ROOT=WORKSPACE/'results/paper_cwru_12k_20261019'
GEN_ROOT=ROOT/'generators'
DEFAULT_OUTPUT=ROOT/'classification'
CODE_FILES=('scripts/paper_harmonized_rerun.py','scripts/paper_harmonized_prepare.py',
    'scripts/review_rerun_training.py','scripts/review_rerun_protocol.py',
    'scripts/prepare_review_support_draws.py','scripts/run_review_rerun.py',
    'run_generation.py','split_data.py','run_a_small_eval_v2.py','run_pcl_eval.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tag(key):
    return f"d{key['draw']}_f{key['fold']}_k{key['shot']}_{key['backbone']}_{key['method']}_s{key['seed']}"


def source_key(key):
    return f"{key['draw']}/{key['fold']}/{key['shot']}/{key['method']}"


def matrix(smoke=False):
    rows=[]
    for draw in ((0,) if smoke else (0,1,2)):
        for fold in ((0,) if smoke else range(4)):
            for shot in ((1,) if smoke else (1,3,5)):
                configurations=[('main','dual',m) for m in P.MAIN]
                if draw==0:
                    configurations += [('repeat','dual','repeat'),('raw','raw','none'),('raw','raw','mix')]
                    if shot==1:
                        configurations += [('standardization','dual',m) for m in ('diffusion_zscore','mix_zscore')]
                for seed in ((0,) if smoke else (0,1,2)):
                    rows.extend(dict(stage=stage,draw=draw,fold=fold,shot=shot,backbone=backbone,method=method,seed=seed)
                                for stage,backbone,method in configurations)
    return rows


def load_arrays(key, data_root=DATA_ROOT, generator_root=GEN_ROOT):
    d,f,k=key['draw'],key['fold'],key['shot']
    split=Path(data_root)/f'draw{d}/cwru/fold{f}/{k}shot'
    gen=Path(generator_root)/f'draw{d}/fold{f}/{k}shot'
    meta=read(split/'metadata.json')
    a={}
    for part in ('train','val','test'):
        for v in ('X','y'):
            value=np.load(split/f'{part}_{v}.npy',allow_pickle=False)
            if v=='y' and not np.issubdtype(value.dtype,np.integer):raise ValueError('Labels must be integers')
            a[f'{part}_{v}']=value.astype(np.float32 if v=='X' else np.int64)
    n_test=len(a['test_y'])//10
    for part,n in [('train',10*k),('val',200),('test',10*n_test)]:
        x,y=a[f'{part}_X'],a[f'{part}_y']
        if x.shape!=(n,1024) or y.shape!=(n,) or not np.isfinite(x).all():
            raise ValueError('Invalid harmonized split shape/values')
        if not np.array_equal(np.bincount(y,minlength=10),np.full(10,n//10)):
            raise ValueError('Invalid harmonized class balance')
    if n_test < 20 or meta['draw']!=d or meta['fold']!=f or meta['n_shot']!=k:
        raise ValueError('Invalid harmonized split identity')
    config=read(gen/'config.json')
    if config.get('smoke') or config.get('T')!=1000 or config.get('epochs')!={1:2000,3:1200,5:1000}[k]:
        raise ValueError('Smoke or wrong generator budget cannot enter classifier comparison')
    if (config.get('support_X_sha256')!=P.array_hash(a['train_X']) or
        config.get('support_y_sha256')!=P.array_hash(a['train_y']) or
        config.get('support_manifest_sha256')!=P.file_hash(split/'metadata.json')):
        raise ValueError('Generator not bound to harmonized support')
    if validate_generator_resume(gen,config)!='complete':
        raise ValueError('Generator/pool incomplete')
    pool=gen/'generated'/config['pool_tag']
    fake=np.load(pool/'fake_X.npy',allow_pickle=False).astype(np.float32)
    fy=np.load(pool/'fake_y.npy',allow_pickle=False)
    if not np.issubdtype(fy.dtype,np.integer):raise ValueError('Pool labels must be integers')
    fy=fy.astype(np.int64)
    if fake.shape!=(100*k,1024) or not np.isfinite(fake).all() or not np.array_equal(np.bincount(fy,minlength=10),np.full(10,10*k)):
        raise ValueError('Invalid formal generation pool')
    file_paths={f'{part}_{v}':split/f'{part}_{v}.npy' for part in ('train','val','test') for v in ('X','y')}
    file_paths.update(metadata=split/'metadata.json',config=gen/'config.json',
        training_receipt=gen/'training_complete.json',generation_receipt=gen/'generation_complete.json',
        generator_checkpoint=gen/'models/diffusion_model.pt',fake_X=pool/'fake_X.npy',fake_y=pool/'fake_y.npy')
    sources={n:P.array_hash(v) for n,v in a.items()}
    sources.update(fake_X=P.array_hash(fake),fake_y=P.array_hash(fy))
    real,labels=a['train_X'],a['train_y']
    x,y,mask,aug=P.augment(real,labels,fake,fy,f,k,key['method'])
    a.update(real_X=real,real_y=labels,real_mean=float(real.mean()),real_std=max(float(real.std()),1e-6),
             train_X=x,train_y=y,anchor_mask=mask)
    info=dict(sources=sources,augmentation=aug,real_mean=a['real_mean'],real_std=a['real_std'],
        split_dir=str(split),pool_dir=str(pool),n_test_per_class=n_test,
        files={n:dict(path=str(p),sha256=P.file_hash(p)) for n,p in file_paths.items()})
    return a,info


def preflight(output=DEFAULT_OUTPUT, data_root=DATA_ROOT, generator_root=GEN_ROOT, *, smoke=False,max_steps=1000):
    import paper_harmonized_prepare as D
    output=Path(output).resolve();data_root=Path(data_root).resolve();generator_root=Path(generator_root).resolve()
    if (not smoke and max_steps!=1000) or max_steps<1:
        raise ValueError('Formal classifier budget is 1000 steps')
    if smoke and output==DEFAULT_OUTPUT.resolve():
        raise ValueError('Smoke requires a separate output')
    tasks=matrix(smoke);sources={};verified=set()
    for key in tasks:
        split=(key['draw'],key['fold'],key['shot'])
        if split not in verified:
            D.verify_support(*split,data_root=data_root,source_root=WORKSPACE)
            verified.add(split)
        ident=source_key(key)
        if ident not in sources:
            _,sources[ident]=load_arrays(key,data_root,generator_root)
    protocol=dict(version='paper-cwru-12k-classifier-v1',smoke_only=smoke,matrix=tasks,
        training=dict(max_steps=max_steps,val_every=min(50,max_steps),batch_size=32,eval_batch_size=32,
            lr=.0005,weight_decay=.01,pcl_lambda=.5,pcl_tau=.07,prototype_momentum=.99),
        selection='minimum_validation_CE_after_full_budget',target_sampling_rate=12000,
        test_role='reanalysis_of_previously_explored_source_records_not_independent_confirmation',
        primary_metric='accuracy',secondary_metrics=['macro_f1','fault_conditional_accuracy'],
        baseline_commit='e95cb98')
    payload=dict(protocol=protocol,paths=dict(data_root=str(data_root),generator_root=str(generator_root)),
        sources=sources,code={n:P.file_hash(WORKSPACE/n) for n in CODE_FILES})
    fp=P.object_hash(payload);path=output/'manifest.json'
    if path.exists():
        existing=read(path)
        if existing['fingerprint']!=fp or P.object_hash(existing['payload'])!=fp:
            raise ValueError('Frozen harmonized classifier inputs/code changed')
        for name,digest in payload['code'].items():
            if P.file_hash(output/'source_snapshot'/name)!=digest:
                raise ValueError('Classifier source snapshot changed')
        return existing
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=WORKSPACE,capture_output=True,text=True,check=True).stdout.strip()
    manifest=dict(payload=payload,fingerprint=fp,git_commit=commit,created_utc=P.now())
    for name in payload['code']:
        dest=output/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes((WORKSPACE/name).read_bytes())
    P.dump(path,manifest)
    return manifest


def read_valid_rows(output, manifest, only_tags=None):
    output=Path(output)
    if P.object_hash(manifest['payload'])!=manifest['fingerprint']:
        raise ValueError('Corrupt manifest')
    for name,digest in manifest['payload']['code'].items():
        if P.file_hash(output/'source_snapshot'/name)!=digest:
            raise ValueError('Changed source snapshot')
    expected={tag(k):k for k in manifest['payload']['protocol']['matrix']}
    rows=[]
    for path in sorted((output/'runs').glob('*.json')):
        if only_tags is not None and path.stem not in only_tags:continue
        row=read(path);key=expected.get(path.stem)
        if key is None or any(row.get(k)!=v for k,v in key.items()):
            raise ValueError('Unexpected classifier run')
        if row['manifest_fingerprint']!=manifest['fingerprint']:
            raise ValueError('Classifier run fingerprint mismatch')
        info=manifest['payload']['sources'][source_key(row)]
        if row['data_hash']!=P.object_hash(info):raise ValueError('Run input identity mismatch')
        for artifact in row['artifacts'].values():
            p=(output/artifact['path']).resolve()
            if not p.is_relative_to(output.resolve()) or P.file_hash(p)!=artifact['sha256']:
                raise ValueError('Run artifact checksum/path mismatch')
        with np.load(output/row['artifacts']['predictions']['path'],allow_pickle=False) as z:
            true,pred,probs=z['true'],z['pred'],z['probabilities']
        if (P.array_hash(true)!=info['sources']['test_y'] or probs.shape!=(len(true),10) or
            not np.isfinite(probs).all() or (probs<0).any() or
            not np.allclose(probs.sum(1),1,atol=1e-5) or not np.array_equal(pred,probs.argmax(1))):
            raise ValueError('Invalid prediction arrays')
        metrics=dict(accuracy=float(np.mean(true==pred)),
            f1_macro=float(f1_score(true,pred,labels=range(10),average='macro',zero_division=0)),
            fault_conditional_accuracy=float(np.mean(true[true!=0]==pred[true!=0])))
        for name,value in metrics.items():
            if not np.isclose(row[name],value,rtol=0,atol=1e-12):raise ValueError('Metric mismatch')
        cfg=manifest['payload']['protocol']['training'];steps=cfg['max_steps']
        history=read(output/row['artifacts']['history']['path'])
        schedule=sorted(set(range(cfg['val_every'],steps+1,cfg['val_every']))|{steps})
        if [r['step'] for r in history]!=schedule or any(not np.isfinite(r['val_loss']) for r in history):
            raise ValueError('Incomplete validation history')
        best=min(history,key=lambda r:r['val_loss'])
        if row['best_step']!=best['step'] or row['best_val_loss']!=best['val_loss']:
            raise ValueError('Wrong validation checkpoint selection')
        if row['completed_steps']!=steps or set(row['final_bn_counts'].values())!={steps}:
            raise ValueError('Wrong optimizer/BN budget')
        if row['samples_seen']!=history[-1]['samples_seen']:raise ValueError('Wrong sample budget')
        import torch
        checkpoint=torch.load(output/row['artifacts']['checkpoint']['path'],map_location='cpu',weights_only=True)
        final=checkpoint['final_training_state']
        counts={n:int(v.item()) for n,v in final['model_state_dict'].items() if n.endswith('num_batches_tracked')}
        if (checkpoint['key']!=key or checkpoint['manifest_fingerprint']!=manifest['fingerprint'] or
            checkpoint['data_hash']!=row['data_hash'] or checkpoint['selection']['step']!=row['best_step'] or
            final['completed_steps']!=steps or counts!=row['final_bn_counts']):
            raise ValueError('Checkpoint identity/selection/update mismatch')
        rows.append(row)
    return rows


def run_one(output,manifest,key,progress=lambda **kw:None):
    import torch
    from torch.utils.data import TensorDataset
    sys.path.insert(0,str(WORKSPACE))
    import run_a_small_eval_v2 as A
    import review_rerun_training as T
    start=time.perf_counter();paths=manifest['payload']['paths']
    a,info=load_arrays(key,Path(paths['data_root']),Path(paths['generator_root']))
    if info!=manifest['payload']['sources'][source_key(key)]:raise ValueError('Inputs changed after preflight')
    tensors={};caches={}
    for part in ('train','val','test'):
        x=a[f'{part}_X'];raw=A._raw_views(x,a['real_mean'],a['real_std'])
        if key['backbone']=='raw':
            cwt=np.zeros((len(x),1,1,1),dtype=np.float32)
        else:
            cwt,caches[part]=A.precompute_cwt_verified(x,ROOT/'cache'/f'{A.TRANSFORM_VERSION}_{P.array_hash(x)}.npz')
            if cwt.shape!=(len(x),1,128,128) or not np.isfinite(cwt).all():raise ValueError('Invalid CWT')
        tensors[part]=(torch.from_numpy(raw),torch.from_numpy(cwt),torch.from_numpy(a[f'{part}_y']).long())
    train=TensorDataset(*tensors['train'],torch.from_numpy(a['anchor_mask']))
    support=TensorDataset(*(v[:len(a['real_y'])] for v in tensors['train']))
    val=TensorDataset(*tensors['val']);T.seed_everything(key['seed'])
    model=A.DualNormRawCWT(10) if key['backbone']=='dual' else T.RawOnly(10)
    preproc_sec=time.perf_counter()-start;train_start=time.perf_counter()
    trained=T.train_fixed_steps(model,train,val,support,device='cuda',seed=key['seed'],
        progress_callback=lambda s:progress(run_progress=s),**manifest['payload']['protocol']['training'])
    torch.cuda.synchronize();training_sec=time.perf_counter()-train_start
    bn={n:int(v.item()) for n,v in trained['checkpoint']['final_training_state']['model_state_dict'].items()
        if n.endswith('num_batches_tracked')}
    if not bn or set(bn.values())!={trained['completed_steps']}:raise ValueError('Wrong BN update count')
    test_start=time.perf_counter()
    test=T.evaluate_model(model,TensorDataset(*tensors['test']),device='cuda',batch_size=32)
    torch.cuda.synchronize();test_sec=time.perf_counter()-test_start
    dest=Path(output)/'artifacts'/tag(key);dest.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(dest/'predictions.tmp.npz',true=test['true'],pred=test['pred'],probabilities=test['probabilities'])
    (dest/'predictions.tmp.npz').replace(dest/'predictions.npz')
    P.dump(dest/'history.json',trained['history'])
    checkpoint=dict(trained['checkpoint'],key=key,manifest_fingerprint=manifest['fingerprint'],data_hash=P.object_hash(info))
    torch.save(checkpoint,dest/'checkpoint.tmp.pt');(dest/'checkpoint.tmp.pt').replace(dest/'checkpoint.pt')
    artifacts={n:dict(path=str((dest/p).relative_to(output)),sha256=P.file_hash(dest/p)) for n,p in
        [('predictions','predictions.npz'),('history','history.json'),('checkpoint','checkpoint.pt')]}
    mask=test['true']!=0
    row=dict(**key,accuracy=test['accuracy'],f1_macro=test['f1_macro'],test_ce=test['loss'],
        fault_conditional_accuracy=float(np.mean(test['pred'][mask]==test['true'][mask])),
        completed_steps=trained['completed_steps'],best_step=trained['best_step'],best_val_loss=trained['best_val_loss'],
        samples_seen=trained['samples_seen'],final_bn_counts=bn,n_test=len(test['true']),
        elapsed_sec=time.perf_counter()-start,preprocessing_sec=preproc_sec,training_sec=training_sec,
        test_evaluation_sec=test_sec,model_params=sum(p.numel() for p in model.parameters()),
        artifacts=artifacts,data_hash=P.object_hash(info),cache=caches,manifest_fingerprint=manifest['fingerprint'],
        gpu=torch.cuda.get_device_name(0),torch_version=str(torch.__version__),finished_utc=P.now())
    P.dump(Path(output)/'runs'/f'{tag(key)}.json',row)
    del model,trained,checkpoint,train,support,val,tensors,cwt;gc.collect();torch.cuda.empty_cache()
    return row


def execute(output,manifest,limit=None):
    import torch
    from run_review_rerun import gpu_lock
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    torch.set_num_threads(4)
    complete={tag(r) for r in read_valid_rows(output,manifest)}
    tasks=manifest['payload']['protocol']['matrix']
    state=dict(pid=os.getpid(),total=len(tasks),completed=len(complete),stage='starting',started_utc=P.now(),
        smoke_only=manifest['payload']['protocol']['smoke_only'],manifest_fingerprint=manifest['fingerprint'])
    def progress(**fields):
        state.update(fields,updated_utc=P.now());P.dump(Path(output)/'status.json',state)
    try:
        with gpu_lock():
            count=0
            for key in tasks:
                if tag(key) in complete:continue
                if limit is not None and count>=limit:break
                progress(stage='training',current=key,run_progress=None)
                print(f'START {tag(key)} {len(complete)}/{len(tasks)}',flush=True)
                row=run_one(output,manifest,key,progress)
                if len(read_valid_rows(output,manifest,{tag(key)}))!=1:raise ValueError('New result verification failed')
                complete.add(tag(key));count+=1
                progress(completed=len(complete),last_elapsed_sec=row['elapsed_sec'])
                print(f'SAVED {tag(key)} accuracy={row["accuracy"]:.6f} sec={row["elapsed_sec"]:.1f}',flush=True)
        progress(stage='complete' if len(complete)==len(tasks) else 'limited_run_complete',current=None,
                 run_progress=None,finished_utc=P.now())
    except BaseException as exc:
        progress(stage='failed',error=f'{type(exc).__name__}: {exc}');raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,default=DATA_ROOT)
    parser.add_argument('--generator-root',type=Path,default=GEN_ROOT)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--max-steps',type=int,default=1000)
    parser.add_argument('--limit',type=int)
    args=parser.parse_args()
    if args.smoke and args.output_root is None:parser.error('Smoke requires explicit separate output-root')
    if args.limit is not None and args.limit<1:parser.error('limit must be positive')
    output=(args.output_root or DEFAULT_OUTPUT).resolve()
    manifest=preflight(output,args.data_root,args.generator_root,smoke=args.smoke,max_steps=args.max_steps)
    print(json.dumps(dict(preflight='passed',tasks=len(matrix(args.smoke)),fingerprint=manifest['fingerprint'],
        valid_completed=len(read_valid_rows(output,manifest)))),flush=True)
    if args.execute:execute(output,manifest,args.limit)


if __name__=='__main__':main()
