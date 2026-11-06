"""Measure frozen classifiers and the actual online transform separately.

Data already reside in device memory; disk I/O, host/device transfer and data
acquisition are excluded. No training, test scoring or model selection occurs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import run_a_small_eval_v2 as A
import review_rerun_training as T
from run_review_rerun import gpu_lock


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(values):
    x = np.array(values) * 1000
    return dict(median_ms=float(np.median(x)), p25_ms=float(np.percentile(x, 25)),
                p75_ms=float(np.percentile(x, 75)), mean_ms=float(x.mean()),
                repetitions=len(x), samples_ms=x.tolist())


def measure(fn, device, warmup=20, repetitions=100):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        if device == 'cuda':
            torch.cuda.synchronize()
        times = []
        for _ in range(repetitions):
            start = time.perf_counter()
            fn()
            if device == 'cuda':
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
    return summarize(times)


def raw_views(x, mean, std):
    # NumPy's existing raw view uses population std, unlike default torch.std.
    local = (x - x.mean(1, keepdim=True)) / (x.std(1, correction=0, keepdim=True) + 1e-6)
    return torch.stack((local, (x - mean) / std), dim=1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=ROOT / 'results/paper_evidence_20261019/inference_cost.json')
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    T.seed_everything(0)
    source = ROOT / 'data/splits_loocv_v2/cwru/fold0/3shot/train_X.npy'
    support = np.load(source, allow_pickle=False).astype(np.float32)
    mean, std = float(support.mean()), max(float(support.std()), 1e-6)
    output = dict(device=args.device, gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  processor=platform.processor(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  cpu_threads=4, input_length=1024, precision='float32', warmup=20,
                  timed_repetitions=100, clock='perf_counter plus device synchronization',
                  scope='Device-resident inputs; no acquisition, disk I/O or host/device transfer.',
                  input_path=str(source), input_sha256=sha(source), script_sha256=sha(__file__), rows=[])
    with gpu_lock():
        for name in ('raw', 'dual'):
            cp = ROOT / f'results/review_rerun_20261017/artifacts/f0_k3_{name}_none_s0/checkpoint.pt'
            checkpoint = torch.load(cp, map_location='cpu', weights_only=False)
            model = (T.RawOnly(10) if name == 'raw' else A.DualNormRawCWT(10)).to(args.device).eval()
            model.load_state_dict(checkpoint['model_state_dict'])
            for batch in (1, 32):
                x = torch.from_numpy(support[np.arange(batch) % len(support)]).to(args.device)
                raw = raw_views(x, mean, std)
                assert np.allclose(raw.cpu().numpy(), A._raw_views(x.cpu().numpy(), mean, std), atol=1e-5)
                with torch.inference_mode():
                    cwt = A.R.cwt_batch(x) if name == 'dual' else torch.zeros((batch, 1, 1, 1), device=args.device)
                def preprocess():
                    r = raw_views(x, mean, std)
                    c = A.R.cwt_batch(x) if name == 'dual' else cwt
                    return r, c
                row = dict(backbone=name, batch=batch, parameters=sum(v.numel() for v in model.parameters()),
                           checkpoint=str(cp), checkpoint_sha256=sha(cp),
                           model_only=measure(lambda: model((raw, cwt)), args.device),
                           preprocessing=measure(preprocess, args.device),
                           combined=measure(lambda: model(preprocess()), args.device))
                output['rows'].append(row)
                print(json.dumps({k: v for k, v in row.items() if k not in ('checkpoint', 'checkpoint_sha256')},
                                 ensure_ascii=False), flush=True)
            del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
