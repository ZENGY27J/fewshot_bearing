#!/usr/bin/env python3
"""Run the frozen review matrix; resume verified completed runs only."""
from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from pathlib import Path

import numpy as np

import review_rerun_protocol as P


@contextmanager
def gpu_lock():
    """A process-owned Windows lock shared by this campaign's GPU jobs."""
    import msvcrt
    path = P.WORKSPACE / 'results/review_rerun_gpu.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.seek(0, os.SEEK_END)
        if not stream.tell():
            stream.write(b'0')
            stream.flush()
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def run_one(root: Path, manifest: dict, key: dict, progress) -> dict:
    sys.path.insert(0, str(P.WORKSPACE))
    import torch
    from torch.utils.data import TensorDataset
    import run_a_small_eval_v2 as A
    import review_rerun_training as T

    start = time.perf_counter()
    paths = manifest['payload']['paths']
    a, info = P.load_arrays(Path(paths['source']), key['fold'], key['shot'], key['method'],
            Path(paths['split_base']), Path(paths['generated_base']), paths['generation_layout'])
    expected = manifest['payload']['sources'][f"{key['fold']}/{key['shot']}/{key['method']}"]
    if info != expected:
        raise ValueError('Data changed after preflight')
    raw = {part: A._raw_views(a[f'{part}_X'], a['real_mean'], a['real_std'])
           for part in ('train', 'val', 'test')}
    cwt, cache = {}, {}
    for part in ('train', 'val', 'test'):
        if key['backbone'] == 'raw':
            cwt[part] = np.zeros((len(raw[part]), 1, 1, 1), dtype=np.float32)
            continue
        x = a[f'{part}_X']
        cache_path = P.WORKSPACE / 'results/review_rerun_cache_20261017' / (
            f'{A.TRANSFORM_VERSION}_{P.array_hash(x)}.npz')
        cwt[part], cache[part] = A.precompute_cwt_verified(x, cache_path)
        if cwt[part].shape != (len(x), 1, 128, 128) or not np.isfinite(cwt[part]).all():
            raise ValueError('Invalid CWT cache')
        cache[part]['array_sha256'] = P.array_hash(cwt[part])
    tensors = {part: (torch.from_numpy(raw[part]), torch.from_numpy(cwt[part]),
                      torch.from_numpy(a[f'{part}_y']).long()) for part in raw}
    train = TensorDataset(*tensors['train'], torch.from_numpy(a['anchor_mask']))
    support = TensorDataset(*(t[:len(a['real_y'])] for t in tensors['train']))
    val = TensorDataset(*tensors['val'])
    T.seed_everything(key['seed'])
    model = A.DualNormRawCWT(10) if key['backbone'] == 'dual' else T.RawOnly(10)
    config = {k: v for k, v in manifest['payload']['protocol']['training'].items()
              if k not in ('scheduler', 'early_stopping')}
    trained = T.train_fixed_steps(model, train, val, support, device='cuda', seed=key['seed'],
                                  progress_callback=lambda state: progress(run_progress=state), **config)
    final_state = trained['checkpoint']['final_training_state']['model_state_dict']
    bn_counts = {k: int(v.item()) for k, v in final_state.items() if k.endswith('num_batches_tracked')}
    if not bn_counts or set(bn_counts.values()) != {config['max_steps']}:
        raise ValueError(f'Unexpected BatchNorm update counts: {bn_counts}')
    test = T.evaluate_model(model, TensorDataset(*tensors['test']), device='cuda', batch_size=32)
    run_tag = P.tag(key)
    dest = root / 'artifacts' / run_tag
    dest.mkdir(parents=True, exist_ok=True)
    prediction = dest / 'predictions.npz'
    temp = dest / 'predictions.tmp.npz'
    np.savez_compressed(temp, true=test['true'], pred=test['pred'], probabilities=test['probabilities'])
    temp.replace(prediction)
    P.dump(dest / 'history.json', trained['history'])
    checkpoint = dict(trained['checkpoint'], key=key, manifest_fingerprint=manifest['fingerprint'],
                      data_hash=P.object_hash(info))
    temp = dest / 'checkpoint.tmp.pt'
    torch.save(checkpoint, temp)
    temp.replace(dest / 'checkpoint.pt')
    artifacts = {name: dict(path=str(p.relative_to(root)), sha256=P.file_hash(p))
                 for name, p in (('predictions', prediction), ('history', dest / 'history.json'),
                                 ('checkpoint', dest / 'checkpoint.pt'))}
    row = dict(**key, accuracy=test['accuracy'], f1_macro=test['f1_macro'], test_ce=test['loss'],
               completed_steps=trained['completed_steps'], best_step=trained['best_step'],
               best_val_loss=trained['best_val_loss'], samples_seen=trained['samples_seen'],
               elapsed_sec=time.perf_counter() - start, model_params=sum(p.numel() for p in model.parameters()),
               final_bn_counts=bn_counts, data_hash=P.object_hash(info), cache=cache,
               artifacts=artifacts, manifest_fingerprint=manifest['fingerprint'],
               gpu=torch.cuda.get_device_name(0), torch_version=str(torch.__version__),
               cuda_version=torch.version.cuda, numpy_version=np.__version__, finished_utc=P.now())
    P.dump(root / 'runs' / f'{run_tag}.json', row)
    del trained, checkpoint, model, train, val, support, tensors, cwt, raw, a
    gc.collect()
    torch.cuda.empty_cache()
    return row


def execute(root: Path, manifest: dict, limit: int | None = None,
            only_stages: list[str] | None = None) -> None:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This matrix requires CUDA; use the verified miniconda Python runtime')
    torch.set_num_threads(4)
    matrix = manifest['payload']['protocol']['matrix']
    rows = P.read_valid_rows(root, manifest)
    complete = {P.tag(r) for r in rows}
    state = dict(pid=os.getpid(), started_utc=P.now(), stage='starting', total=len(matrix),
                 completed=len(complete), manifest_fingerprint=manifest['fingerprint'],
                 smoke_only=manifest['payload']['protocol']['smoke_only'])

    def progress(**changes):
        state.update(changes)
        state['updated_utc'] = P.now()
        P.dump(root / 'status.json', state)

    awake = False
    try:
        with gpu_lock():
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            count = 0
            for key in matrix:
                if only_stages is not None and key['stage'] not in only_stages:
                    continue
                if P.tag(key) in complete:
                    continue
                if limit is not None and count >= limit:
                    break
                progress(stage='training', current=key, run_progress=None)
                print(f"START {P.tag(key)} ({len(complete)}/{len(matrix)})", flush=True)
                row = run_one(root, manifest, key, progress)
                # Existing runs were fully checked on entry. Validate the newly
                # written run now; check the full bundle again during analysis.
                checked = P.read_valid_rows(root, manifest, only_tags={P.tag(key)})
                if len(checked) != 1:
                    raise ValueError('New run did not pass artifact verification')
                complete.add(P.tag(key))
                count += 1
                progress(completed=len(complete), last_elapsed_sec=row['elapsed_sec'])
                print(f"SAVED {P.tag(key)} steps={row['completed_steps']} seconds={row['elapsed_sec']:.1f}", flush=True)
            done = len(complete) == len(matrix)
            progress(stage='analysis' if done else 'limited_run_complete', current=None, run_progress=None)
        if done:
            from analyze_review_rerun import analyze
            analyze(root, require_complete=True)
            progress(stage='complete', finished_utc=P.now())
    except BaseException as exc:
        progress(stage='failed', error=f'{type(exc).__name__}: {exc}', current=state.get('current'))
        traceback.print_exc()
        raise
    finally:
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true', help='Default is input verification only')
    parser.add_argument('--smoke', action='store_true', help='Separate non-paper matrix, fold0/K1,3,5/one seed')
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--source-root', type=Path, default=P.WORKSPACE)
    parser.add_argument('--split-base', type=Path)
    parser.add_argument('--generated-base', type=Path)
    parser.add_argument('--generation-layout', choices=('legacy', 'review'), default='legacy')
    parser.add_argument('--stages', nargs='+', choices=P.STAGES, default=list(P.STAGES))
    parser.add_argument('--only-stages', nargs='+', choices=P.STAGES,
                        help='Execute a subset of an already frozen matrix; leave the manifest intact')
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--val-every', type=int, default=50)
    parser.add_argument('--limit', type=int, help='Stop after N new runs; completed runs remain verified')
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.only_stages is not None and not set(args.only_stages).issubset(args.stages):
        parser.error('--only-stages must be a subset of the frozen --stages matrix')
    if args.smoke and args.output_root is None:
        parser.error('--smoke requires an explicit separate --output-root')
    root = (args.output_root or P.DEFAULT_ROOT).resolve()
    if args.smoke and root == P.DEFAULT_ROOT.resolve():
        parser.error('Smoke results cannot use the formal output root')
    manifest = P.preflight(root, args.source_root.resolve(), stages=tuple(args.stages), smoke=args.smoke,
               max_steps=args.max_steps, val_every=args.val_every, split_base=args.split_base,
               generated_base=args.generated_base, generation_layout=args.generation_layout)
    rows = P.read_valid_rows(root, manifest)
    print(json.dumps(dict(preflight='passed', expected=len(manifest['payload']['protocol']['matrix']),
                         valid_completed=len(rows), fingerprint=manifest['fingerprint']), ensure_ascii=False), flush=True)
    if args.execute:
        execute(root, manifest, args.limit, args.only_stages)


if __name__ == '__main__':
    main()
