"""Additional prediction, pool, cost and fixed-feature reference evidence; CPU only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import analyze_review_rerun as A
import paper_svm_baseline as S
import review_rerun_protocol as P

METHODS = ('none', 'diffusion', 'jitter', 'mix')
CLASS_NAMES = ('Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021')
CONTRASTS = (('diffusion', 'none'), ('mix', 'jitter'))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def frame_out(out: Path, name: str, records) -> pd.DataFrame:
    frame = records if isinstance(records, pd.DataFrame) else pd.DataFrame(records)
    frame.to_csv(out / name, index=False, encoding='utf-8-sig')
    return frame


def verified_predictions(root: Path, manifest: dict, row: dict) -> tuple:
    if row['manifest_fingerprint'] != manifest['fingerprint']:
        raise ValueError('Mismatched result identity')
    info = manifest['payload']['sources'][f'{row["fold"]}/{row["shot"]}/{row["method"]}']
    if row['data_hash'] != P.object_hash(info):
        raise ValueError('Mismatched data identity')
    artifact = row['artifacts']['predictions']
    path = (root / artifact['path']).resolve()
    if not path.is_relative_to(root.resolve()) or P.file_hash(path) != artifact['sha256']:
        raise ValueError('Changed prediction file')
    with np.load(path, allow_pickle=False) as data:
        true, pred, proba = data['true'], data['pred'], data['probabilities']
    if (P.array_hash(true.astype(np.int64)) != info['sources']['test_y'] or
        pred.shape != true.shape or proba.shape != (len(true), 10) or
        not np.isfinite(proba).all() or not np.array_equal(pred, proba.argmax(axis=1))):
        raise ValueError('Invalid prediction content')
    if (abs(row['accuracy'] - np.mean(true == pred)) > 1e-12 or
        abs(row['f1_macro'] - f1_score(true, pred, average='macro')) > 1e-12):
        raise ValueError('Invalid saved metric')
    return true, pred


def pool_metrics(real: np.ndarray, fake: np.ndarray) -> dict:
    real, fake = np.asarray(real, dtype=np.float64), np.asarray(fake, dtype=np.float64)
    if not np.isfinite(real).all() or not np.isfinite(fake).all():
        raise ValueError('Nonfinite pool')
    real_rms = np.sqrt(np.mean(real ** 2, axis=1))
    fake_rms = np.sqrt(np.mean(fake ** 2, axis=1))
    rp, fp = S.normalized_psd(real).mean(axis=0), S.normalized_psd(fake).mean(axis=0)
    corr = float(np.corrcoef(rp, fp)[0, 1]) if min(rp.std(), fp.std()) > 1e-15 else None
    return dict(n_real=len(real), n_fake=len(fake), real_rms_mean=float(real_rms.mean()),
                fake_rms_mean=float(fake_rms.mean()), fake_rms_sd=float(fake_rms.std(ddof=1)),
                rms_ratio=float(fake_rms.mean() / max(real_rms.mean(), 1e-12)),
                real_window_std_mean=float(real.std(axis=1).mean()),
                fake_window_std_mean=float(fake.std(axis=1).mean()),
                real_window_mean_abs=float(abs(real.mean(axis=1)).mean()),
                fake_window_mean_abs=float(abs(fake.mean(axis=1)).mean()),
                psd_js_distance=float(jensenshannon(rp, fp, base=2)), psd_correlation=corr)


def analyze_existing(out: Path) -> tuple:
    classes, counts, runs, generation, costs, provenance = [], [], [], [], [], []
    for bundle, root in S.BUNDLES.items():
        manifest = read_json(root / 'manifest.json')
        if P.object_hash(manifest['payload']) != manifest['fingerprint']:
            raise ValueError('Corrupt manifest')
        planned = {P.tag(k): k for k in manifest['payload']['protocol']['matrix']}
        paths = sorted((root / 'runs').glob('*.json'))
        if {p.stem for p in paths} != set(planned):
            raise ValueError('Incomplete or extra matrix')
        for path in paths:
            row = read_json(path)
            if any(row.get(k) != v for k, v in planned[path.stem].items()):
                raise ValueError('Incorrect run key')
            true, pred = verified_predictions(root, manifest, row)
            key = dict(bundle=bundle, fold=row['fold'], shot=row['shot'], backbone=row['backbone'],
                       method=row['method'], seed=row['seed'])
            precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=range(10), zero_division=0)
            for c in range(10):
                classes.append(dict(key, class_id=c, class_name=CLASS_NAMES[c], precision_pct=100 * precision[c],
                                    recall_pct=100 * recall[c], f1_pct=100 * f1[c], n_test=int(support[c])))
            cm = confusion_matrix(true, pred, labels=range(10))
            for c in range(10):
                for j in range(10):
                    counts.append(dict(key, true_class=CLASS_NAMES[c], pred_class=CLASS_NAMES[j], count=int(cm[c, j])))
            cache = row.get('cache', {})
            runs.append(dict(key, accuracy_pct=100 * row['accuracy'], f1_macro_pct=100 * row['f1_macro'],
                elapsed_sec=row['elapsed_sec'], model_params=row['model_params'], samples_seen=row['samples_seen'],
                cwt_cache_all_hit=all(v['cache_hit'] for v in cache.values()) if cache else None,
                gpu=row['gpu'], n_test=len(true)))
        for fold in range(4):
            for shot in (1, 3, 5):
                info = manifest['payload']['sources'][f'{fold}/{shot}/none']
                split, pool = Path(info['split_dir']), Path(info['pool_dir'])
                meta = read_json(split / 'metadata.json')
                if tuple(meta['classes']) != CLASS_NAMES:
                    raise ValueError('Class mapping differs')
                a = {key: np.load(path, allow_pickle=False).astype(dtype) for key, path, dtype in (
                    ('train_X', split / 'train_X.npy', np.float32), ('train_y', split / 'train_y.npy', np.int64),
                    ('fake_X', pool / 'fake_X.npy', np.float32), ('fake_y', pool / 'fake_y.npy', np.int64))}
                if any(P.array_hash(value) != info['sources'][key] for key, value in a.items()):
                    raise ValueError('Changed source pool')
                for c in range(10):
                    generation.append(dict(bundle=bundle, fold=fold, shot=shot, class_id=c, class_name=CLASS_NAMES[c],
                        **pool_metrics(a['train_X'][a['train_y'] == c], a['fake_X'][a['fake_y'] == c])))
                gen = pool.parents[1]
                for stage, filename in (('generator_training', 'training_complete.json'),
                                        ('generator_sampling_and_pool_save', 'generation_complete.json')):
                    receipt = gen / filename
                    if receipt.is_file():
                        record = read_json(receipt)
                        if record['config_sha256'] != P.file_hash(gen / 'config.json'):
                            raise ValueError('Changed generator cost source')
                        seconds = float(record['elapsed_seconds'])
                        if not np.isfinite(seconds) or seconds <= 0:
                            raise ValueError('Invalid generation elapsed time')
                        costs.append(dict(bundle=bundle, fold=fold, shot=shot, stage=stage, elapsed_sec=seconds,
                            peak_cuda_allocated_bytes=record['peak_cuda_allocated_bytes'],
                            optimizer_updates=record.get('optimizer_updates'), receipt=str(receipt), missing=False))
                    else:
                        costs.append(dict(bundle=bundle, fold=fold, shot=shot, stage=stage, missing=True))
        provenance.append(dict(bundle=bundle, root=str(root), fingerprint=manifest['fingerprint'],
                               verified_predictions=len(paths), verified_pools=12))
        print(f'Verified {bundle}: {len(paths)} predictions, 12 pools', flush=True)
    return (frame_out(out, 'class-metrics.csv', classes), frame_out(out, 'confusion-counts.csv', counts),
            frame_out(out, 'run-metrics-cost.csv', runs), frame_out(out, 'generation-diagnostics.csv', generation),
            frame_out(out, 'generator-cost.csv', costs), provenance)


def class_summaries(out: Path, classes: pd.DataFrame, runs: pd.DataFrame) -> tuple:
    fields = ['bundle', 'backbone', 'method', 'shot', 'fold', 'class_id', 'class_name']
    frame_out(out, 'class-fold-summary.csv', classes.groupby(fields).agg(
        recall_mean_pct=('recall_pct', 'mean'), recall_seed_sd_pp=('recall_pct', 'std'),
        precision_mean_pct=('precision_pct', 'mean'), f1_mean_pct=('f1_pct', 'mean'),
        n_seeds=('seed', 'size')).reset_index())
    frame_out(out, 'fold-metrics.csv', runs.groupby(['bundle', 'backbone', 'method', 'shot', 'fold']).agg(
        accuracy_mean_pct=('accuracy_pct', 'mean'), accuracy_seed_sd_pp=('accuracy_pct', 'std'),
        f1_mean_pct=('f1_macro_pct', 'mean'), n_seeds=('seed', 'size')).reset_index())
    common = classes[(classes.backbone == 'dual') & (classes.seed < 3) & classes.method.isin(METHODS)]
    differences = []
    for bundle in S.BUNDLES:
        for shot in (1, 3, 5):
            for method, baseline in CONTRASTS:
                for c in range(10):
                    part = common[(common.bundle == bundle) & (common.shot == shot) & (common.class_id == c)]
                    left = part[part.method == method].pivot(index='fold', columns='seed', values='recall_pct')
                    right = part[part.method == baseline].pivot(index='fold', columns='seed', values='recall_pct')
                    if left.shape != (4, 3) or right.shape != (4, 3):
                        raise ValueError('Missing common class comparison')
                    delta = (left - right).to_numpy()
                    low, high = A.bootstrap(delta, f'{bundle}/{shot}/{method}/{baseline}/{c}')
                    differences.append(dict(bundle=bundle, shot=shot, contrast=f'{method}-{baseline}',
                        class_id=c, class_name=CLASS_NAMES[c], n_folds=4, n_seeds=3,
                        recall_difference_pp=float(delta.mean()), fold_sd_pp=float(delta.mean(axis=1).std(ddof=1)),
                        conditional_seed_ci_low=low, conditional_seed_ci_high=high,
                        positive_folds=int(np.sum(delta.mean(axis=1) > 1e-12)),
                        negative_folds=int(np.sum(delta.mean(axis=1) < -1e-12))))
    return frame_out(out, 'class-contrasts-common3.csv', differences), common


def read_svm(out: Path, runs: pd.DataFrame) -> tuple:
    root = out / 'svm'
    manifest = read_json(root / 'manifest.json')
    if P.object_hash(manifest['payload']) != manifest['fingerprint']:
        raise ValueError('Corrupt SVM manifest')
    if not read_json(root / 'status.json')['complete']:
        raise ValueError('SVM not complete')
    records = []
    for bundle in S.BUNDLES:
        for fold in range(4):
            for shot in (1, 3, 5):
                dest = root / f'{bundle}_f{fold}_k{shot}'
                row = read_json(dest / 'result.json')
                if (row['bundle'], row['fold'], row['shot']) != (bundle, fold, shot):
                    raise ValueError('Incorrect SVM configuration identity')
                if row['fingerprint'] != manifest['fingerprint'] or row['scaler_fit_samples'] != 10 * shot:
                    raise ValueError('Incorrect SVM training identity')
                if P.file_hash(dest / 'predictions.npz') != row['files']['predictions.npz']:
                    raise ValueError('Changed SVM predictions')
                with np.load(dest / 'predictions.npz') as data:
                    if (P.array_hash(data['true']) != row['sources']['test_y'] or
                        abs(np.mean(data['true'] == data['pred']) - row['accuracy']) > 1e-12 or
                        abs(f1_score(data['true'], data['pred'], average='macro') - row['f1_macro']) > 1e-12):
                        raise ValueError('SVM metrics mismatch')
                records.append({k: v for k, v in row.items() if k in ('bundle', 'fold', 'shot', 'selected_C',
                    'selected_gamma', 'elapsed_sec', 'grid_fit_predict_sec', 'train_val_features_sec',
                    'test_features_predict_sec')} | dict(accuracy_pct=100 * row['accuracy'], f1_macro_pct=100 * row['f1_macro']))
    svm = frame_out(out, 'svm-results.csv', records)
    frame_out(out, 'svm-summary.csv', svm.groupby(['bundle', 'shot']).agg(
        accuracy_pct=('accuracy_pct', 'mean'), fold_sd_pp=('accuracy_pct', 'std'),
        fold_min_pct=('accuracy_pct', 'min'), fold_max_pct=('accuracy_pct', 'max'),
        f1_macro_pct=('f1_macro_pct', 'mean'), n_folds=('fold', 'size')).reset_index())
    deep = runs[(runs.backbone == 'dual') & (runs.seed < 3) & runs.method.isin(METHODS)]
    comparison = deep.groupby(['bundle', 'shot', 'method', 'fold']).agg(
        accuracy_pct=('accuracy_pct', 'mean'), f1_macro_pct=('f1_macro_pct', 'mean')).reset_index()
    comparison = pd.concat([comparison, svm[['bundle', 'shot', 'fold', 'accuracy_pct', 'f1_macro_pct']].assign(method='feature_svm')])
    frame_out(out, 'svm-deep-comparison-fold.csv', comparison)
    summary = comparison.groupby(['bundle', 'shot', 'method']).agg(
        accuracy_pct=('accuracy_pct', 'mean'), fold_sd_pp=('accuracy_pct', 'std'),
        fold_min_pct=('accuracy_pct', 'min'), fold_max_pct=('accuracy_pct', 'max'),
        f1_macro_pct=('f1_macro_pct', 'mean')).reset_index()
    frame_out(out, 'svm-deep-comparison-summary.csv', summary)
    return svm, summary


def cost_summaries(out: Path, runs: pd.DataFrame, costs: pd.DataFrame) -> tuple:
    grouped = runs.groupby(['bundle', 'backbone', 'shot', 'method'])
    summary = grouped.agg(n_runs=('elapsed_sec', 'size'), median_sec=('elapsed_sec', 'median'),
        q25_sec=('elapsed_sec', lambda a: a.quantile(.25)), q75_sec=('elapsed_sec', lambda a: a.quantile(.75)),
        min_sec=('elapsed_sec', 'min'), max_sec=('elapsed_sec', 'max'),
        mean_sec=('elapsed_sec', 'mean'), model_params=('model_params', 'first'),
        cache_all_hit_count=('cwt_cache_all_hit', 'sum')).reset_index()
    frame_out(out, 'classifier-cost-summary.csv', summary)
    gen = costs[~costs.missing].groupby(['shot', 'stage']).agg(n=('elapsed_sec', 'size'),
        median_sec=('elapsed_sec', 'median'), q25_sec=('elapsed_sec', lambda a: a.quantile(.25)),
        q75_sec=('elapsed_sec', lambda a: a.quantile(.75)), min_sec=('elapsed_sec', 'min'),
        max_sec=('elapsed_sec', 'max'), sum_sec=('elapsed_sec', 'sum')).reset_index()
    frame_out(out, 'generator-cost-summary.csv', gen)
    return summary, gen


def figures(out: Path, contrast: pd.DataFrame, generation: pd.DataFrame, svm: pd.DataFrame) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'pdf.fonttype': 42})
    directory = out / 'figures'
    directory.mkdir(exist_ok=True)
    stems = []

    def save(fig, stem):
        for ext in ('png', 'pdf'):
            fig.savefig(directory / f'{stem}.{ext}', dpi=250, bbox_inches='tight')
        plt.close(fig)
        stems.append(stem)

    fig, axes = plt.subplots(3, 3, figsize=(12.2, 7.2), sharex=True, sharey=True, layout='constrained')
    limit = np.ceil(contrast.recall_difference_pp.abs().max() / 5) * 5
    for i, bundle in enumerate(S.BUNDLES):
        for j, shot in enumerate((1, 3, 5)):
            part = contrast[(contrast.bundle == bundle) & (contrast.shot == shot)]
            matrix = part.pivot(index='contrast', columns='class_id', values='recall_difference_pp').reindex(
                [f'{a}-{b}' for a, b in CONTRASTS]).to_numpy()
            im = axes[i, j].imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto')
            axes[i, j].set_xticks(range(10), CLASS_NAMES, rotation=60, ha='right')
            axes[i, j].set_yticks([0, 1], ['Diffusion - none', 'Mix - jitter'])
            axes[i, j].set_title(f'{bundle}, K={shot}', fontsize=10)
    fig.colorbar(im, ax=axes, shrink=.7, label='Class recall difference (percentage points)')
    save(fig, 'figure-03-class-errors')

    fig, axes = plt.subplots(3, 3, figsize=(10.5, 8), sharex=True, layout='constrained')
    metric_labels = [('rms_ratio', 'Generated/support RMS ratio'), ('psd_js_distance', 'PSD Jensen-Shannon distance'),
                     ('psd_correlation', 'PSD correlation')]
    for i, bundle in enumerate(S.BUNDLES):
        for j, (metric, label) in enumerate(metric_labels):
            ax = axes[i, j]
            part = generation[generation.bundle == bundle]
            for xi, shot in enumerate((1, 3, 5)):
                values = part[part.shot == shot].sort_values(['fold', 'class_id'])[metric].to_numpy()
                ax.scatter(xi + np.linspace(-.18, .18, len(values)), values, s=10, alpha=.4, color='#426C94')
                ax.plot([xi-.2, xi+.2], [np.mean(values)] * 2, color='black', lw=2)
            if metric == 'rms_ratio':
                ax.axhline(1, color='grey', lw=.7, ls='--')
                ax.set_yscale('log')
                ax.set_ylim(.65, 6.5)
                ax.set_yticks([.7, 1, 2, 4, 6], ['0.7', '1', '2', '4', '6'])
                ax.minorticks_off()
            elif metric == 'psd_js_distance':
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(-.1, 1.02)
            ax.set_xticks(range(3), ['1', '3', '5'])
            ax.set_xlabel('K (support windows per class)')
            ax.set_ylabel(label)
            ax.set_title(bundle, fontsize=10)
            ax.grid(axis='y', alpha=.2)
    save(fig, 'figure-04-generation-diagnostics')

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.3), sharey=True, layout='constrained')
    labels = {'none': 'Dual: none', 'diffusion': 'Dual: diffusion', 'jitter': 'Dual: jitter',
              'mix': 'Dual: mix', 'feature_svm': 'Features + SVM'}
    for ax, bundle in zip(axes, S.BUNDLES):
        for index, method in enumerate((*METHODS, 'feature_svm')):
            rows = svm[(svm.bundle == bundle) & (svm.method == method)].sort_values('shot')
            ax.plot(rows.shot, rows.accuracy_pct, marker='s' if method == 'feature_svm' else 'o',
                    color='black' if method == 'feature_svm' else plt.get_cmap('tab10')(index),
                    linestyle='--' if method == 'feature_svm' else '-', label=labels[method], lw=1.4)
        ax.set_title(bundle, fontsize=10)
        ax.set_xticks([1, 3, 5])
        ax.set_xlabel('K (support windows per class)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean accuracy across four loads (%)')
    axes[-1].legend(fontsize=7, loc='lower right')
    save(fig, 'figure-05-svm-reference')
    return stems


def inference_table(out: Path) -> tuple:
    path = out / 'inference_cost.json'
    if not path.is_file():
        return [], None
    saved = read_json(path)
    if P.file_hash(P.WORKSPACE / 'scripts/paper_inference_cost.py') != saved['script_sha256']:
        raise ValueError('Inference benchmark source changed')
    if P.file_hash(Path(saved['input_path'])) != saved['input_sha256']:
        raise ValueError('Inference benchmark input changed')
    result = []
    for row in saved['rows']:
        for phase in ('model_only', 'preprocessing', 'combined'):
            record = row[phase]
            samples = np.asarray(record['samples_ms'])
            if len(samples) != saved['timed_repetitions'] or not np.isfinite(samples).all() or min(samples) <= 0:
                raise ValueError('Invalid latency samples')
            for field, quantile in (('median_ms', .5), ('p25_ms', .25), ('p75_ms', .75)):
                if abs(record[field] - np.quantile(samples, quantile)) > 1e-9:
                    raise ValueError('Latency summary does not match observations')
            result.append(dict(backbone=row['backbone'], batch=row['batch'], parameters=row['parameters'],
                phase=phase, median_ms=record['median_ms'], q25_ms=record['p25_ms'], q75_ms=record['p75_ms'],
                repetitions=len(samples), warmup=saved['warmup'], gpu=saved['gpu'], scope=saved['scope']))
    frame_out(out, 'inference-cost-summary.csv', result)
    return result, P.file_hash(path)


def reports(out: Path, classes, contrasts, generation, runs, cost, gcost, svm, svmsummary, provenance, stems) -> None:
    inference, inference_hash = inference_table(out)
    grouped = generation.groupby(['bundle', 'shot']).agg(
        rms_ratio=('rms_ratio', 'mean'), rms_min=('rms_ratio', 'min'), rms_max=('rms_ratio', 'max'),
        psd_js_distance=('psd_js_distance', 'mean'), psd_correlation=('psd_correlation', 'mean'),
        real_rms_mean=('real_rms_mean', 'mean'), n_class_fold_cells=('class_id', 'size')).reset_index()
    frame_out(out, 'generation-summary.csv', grouped)
    lines = ['# 小论文补充证据分析', '', '问题：固定信号特征基线是否有竞争力；增强收益和退化来自哪些类别；生成信号与有限真实支持的幅值/频谱有多大差异；可从日志恢复哪些成本。', '',
        f'核验 {len(runs)} 份分类预测、36 个生成池、36 个确定性 SVM 模型。未重跑历史分类器，未用测试结果筛选生成池。', '',
        '## 简单基线', '', '下表为四折等权平均准确率（%）。神经网络使用所有支持配置共有的 seeds=0/1/2；SVM 每配置仅一个确定性模型，无伪重复种子。均使用 K 个训练标签及每类 20 个验证标签。', '',
        '| 支持配置 | K | 特征 SVM | Dual none | Dual diffusion | Dual jitter | Dual mix |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for bundle in S.BUNDLES:
        for shot in (1, 3, 5):
            part = svmsummary[(svmsummary.bundle == bundle) & (svmsummary.shot == shot)].set_index('method')
            values = [part.loc[m, 'accuracy_pct'] for m in ('feature_svm', *METHODS)]
            lines.append(f'| {bundle} | {shot} | ' + ' | '.join(f'{v:.3f}' for v in values) + ' |')
    lines += ['', 'SVM 仅作为预先固定的低复杂度参照，没有穷尽工程特征或核函数。折间差异详见 svm-summary.csv。若 SVM 在某条件更高，必须保留，不将复杂后端视作必要条件。', '',
        '## 全类别错误', '', '完整类别对比均采用共同三个分类器种子，先平均种子后等权平均四折。以下逐条件列出 recall 差值最大下降与最大上升类别，仅作完整热图的索引，非择优案例。', '',
        '| 配置 | K | 对比 | 最大下降/最小增益 | 最大上升/最小退化 |', '|---|---:|---|---|---|']
    for (bundle, shot, comparison), part in contrasts.groupby(['bundle', 'shot', 'contrast'], sort=False):
        worst, best = part.loc[part.recall_difference_pp.idxmin()], part.loc[part.recall_difference_pp.idxmax()]
        lines.append(f'| {bundle} | {shot} | {comparison} | {worst.class_name}: {worst.recall_difference_pp:+.3f} pp | {best.class_name}: {best.recall_difference_pp:+.3f} pp |')
    lines += ['', '逐类变化为事后描述，不新增显著性检验家族。类别分别有固定 118 个测试窗口，重复训练不产生新的测试事件。条件种子区间见 class-contrasts-common3.csv；全部 1020 次逐类指标及混淆计数均已保存。', '',
        '## 生成诊断', '', '| 配置 | K | 平均 RMS 比 | 类×折范围 | 平均 PSD JS 距离 | 平均 PSD 相关 |', '|---|---:|---:|---|---:|---:|']
    for r in grouped.itertuples():
        lines.append(f'| {r.bundle} | {r.shot} | {r.rms_ratio:.4f} | {r.rms_min:.4f}–{r.rms_max:.4f} | {r.psd_js_distance:.4f} | {r.psd_correlation:.4f} |')
    lines += ['', '每格为 4 折×10 类的描述均值；完整点分布见图。真实支持已逐窗标准化，RMS 比反映模型输入尺度，不能解释为物理振动强度。PSD 比较去直流、加 Hann 窗后逐窗单位功率谱的类别均值；JS 越低、相关越高仅表示与有限支持的频谱形状更接近，不能证明生成样本包含真实故障机理或对分类有益。',
        '', 'K 个真实支持不足以刻画类总体；这些诊断是事后解释，不能据此选池后重报测试。新配置同时改变支持与生成随机性，不能把收益改变单独归因为支持样本。', '',
        '## 已记录成本', '', '分类器 elapsed_sec 包含数据核验、CWT 缓存访问/生成、训练/验证/测试及保存，是完整运行墙钟时间。神经网络硬件为结果日志中的 NVIDIA GeForce GTX 1050；不是纯训练或在线推理时长。以下旧支持主组仅作实际运行记录。', '',
        '| K | 方法 | 次数 | 完整运行中位数 s | IQR s | 全 CWT 缓存命中次数 |', '|---|---|---:|---:|---|---:|']
    for r in cost[(cost.bundle == 'legacy') & (cost.backbone == 'dual') & cost.method.isin(METHODS)].itertuples():
        lines.append(f'| {r.shot} | {r.method} | {r.n_runs} | {r.median_sec:.2f} | {r.q25_sec:.2f}–{r.q75_sec:.2f} | {int(r.cache_all_hit_count)} |')
    lines += ['', '| K | 生成阶段 | 个数 | 中位数 s | IQR s | 总计 s |', '|---|---|---:|---:|---|---:|']
    for r in gcost.itertuples():
        lines.append(f'| {r.shot} | {r.stage} | {r.n} | {r.median_sec:.2f} | {r.q25_sec:.2f}–{r.q75_sec:.2f} | {r.sum_sec:.2f} |')
    lines += ['', f'SVM 的 9 组参数训练及验证选择耗时中位数为 {svm.grid_fit_predict_sec.median():.4f} s；1180 个测试窗特征计算及预测合计中位数 {svm.test_features_predict_sec.median():.4f} s（CPU 单次运行记录，非专门延迟基准）。', '',
        '生成耗时只覆盖新支持集 24 个生成器；旧 12 个生成器缺少同口径 receipt。采样阶段计时包含池文件保存。离线生成成本不得当作在线诊断成本。', '']
    if inference:
        lines += ['## 同硬件设备内推理计时', '',
            'NVIDIA GeForce GTX 1050，float32，1024 点窗口，输入已驻留设备。每项预热 20 次后独立计时 100 次，墙钟计时并同步 GPU。预处理包含实际 raw 双视图及（dual 时）实际时频变换。排除采集、磁盘、CPU/GPU 传输，不称端到端现场响应延迟。combined 独立实测，不是两个中位数相加。', '',
            '| 后端 | Batch | 参数量 | 阶段 | 中位数 ms/batch | IQR ms/batch |', '|---|---:|---:|---|---:|---|']
        for r in inference:
            lines.append(f'| {r["backbone"]} | {r["batch"]} | {r["parameters"]} | {r["phase"]} | {r["median_ms"]:.3f} | {r["q25_ms"]:.3f}–{r["q75_ms"]:.3f} |')
        lines += ['', '100 次重复反映同一设备短时间计时波动，不是独立设备重复或真实生产吞吐。SVM 的 CPU 全测试集单次计时与此批量 GPU 基准口径不同，不据此计算跨设备加速比。', '']
    else:
        lines += ['在线推理与独立 CWT 预处理延迟尚缺同硬件实测，不从参数量反推。', '']
    lines += ['## 解释边界', '', '本分析补充类别错误、相对支持的生成诊断和低复杂度参照。它不改变既有显著性结论，不新增“失效阈值”，不证明增强具有普遍优势。对比必须保留方法不占优的配置，论文主线应围绕条件性收益、简单对照及工程成本。']
    (out / 'analysis-report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (out / 'stats-appendix.md').write_text('''# 统计口径与限制

- 主指标准确率，次指标 macro-F1。4 个固定负载折等权平均，种子先在折内平均。主分析原有 10 seeds 的结果保留在 review assessment，本补充跨支持配置图统一为 3 seeds。
- 逐类 recall 差为百分点效应量。class-contrasts-common3.csv 的 95% 区间沿用原分析：固定 4 折，对 3 个配对 seed 列整体重抽 10000 次。仅表示条件于数据与生成池的分类器波动；不能表示总体泛化或支持抽样不确定性。4 折训练集合重叠，无独立总体假设。
- 新诊断为探索性描述，不针对 180 个类别对比追加显著性声明，无新 p 值家族。正式既定主对比的参考符号翻转/Holm 结果仍以原 assessment 为准，不将类别后验发现用于规避不显著的主结论。
- SVM 每支持配置/折/K 运行一次，9 组预定参数只用 200 个验证标签选参，未重新拟合验证标签。报告 4 折样本 SD、极值，不造训练 seed 方差或 p 值。与深度模型比较限定相同数据/标签预算，不声称优化资源或模型容量等价。
- 生成统计为 36 池×10 类；每类 K 个真实支持、10K 个生成窗。类×折散点是完整描述分布，不能当作 40 个独立重复试验。池、类、折存在依赖，不提供其假独立 CI。Jensen-Shannon 使用以 2 为底距离（散度平方根），范围 0~1；频谱相关只是形状关联。
- 完整运行计时报告中位数、IQR、最小最大。未随机交错的硬件计时不是严格方法成本因果对照，缓存命中等需同时说明。缺失旧生成器计时不填零。
- 本分析重新校验所需预测、真值、指标与池哈希；此前已完成的 1020 检查点全文件核验不重复。所有输出均可追溯 manifest。
''', encoding='utf-8')
    (out / 'figure-catalog.md').write_text('''# 图件目录与解读

1. figure-03-class-errors（PNG/PDF）。来源 class-contrasts-common3.csv；9 个面板覆盖全部配置×K，两个对比为 diffusion-none 与 mix-jitter，颜色表示逐类召回差（百分点），正值蓝色。固定四折和共同三个分类器种子均值，无图中误差条，条件区间在 CSV。目的：识别平均准确率背后的改善/退化类别。解读时逐行关注正负混合和支持改变后的反转，不将单类改善外推到全部故障或机理。

2. figure-04-generation-diagnostics（PNG/PDF）。来源 generation-diagnostics.csv；各组每个 K 的 40 个小点为全部类×折，黑横线为算术均值，不是置信区间。分别画 RMS 比、PSD JS 距离、PSD 相关；RMS 比使用三个支持配置共同的对数轴，完整保留极值。目的：区分幅值偏离与频谱形状偏离。需说明支持只有 K 个、真实输入逐窗标准化、诊断不验证故障机理；更相似也不保证分类收益，不能因图而筛除生成池。图不使用未知采样率或虚构 Hz。

3. figure-05-svm-reference（PNG/PDF）。来源 svm-deep-comparison-summary.csv；四负载均值，深度方法为共同三个 seed，SVM 为每配置确定性一次，不画伪 seed 误差条。逐折 SD/极值及全部值见配套 CSV。目的：判断复杂后端在既定标签预算下是否必要；若 SVM 领先必须保留，若低于混合也不推出所有简单特征都劣于深度方法。折重叠且测试集合固定，不宣称统计显著优劣。
''', encoding='utf-8')
    P.dump(out / 'verification.json', dict(created_utc=P.now(), complete=True, runs=len(runs), pools=36,
        svm_models=36, class_metric_rows=len(classes), generator_class_rows=len(generation),
        provenance=provenance, analysis_code_sha256=P.file_hash(Path(__file__)),
        protocol_sha256=P.file_hash(P.WORKSPACE / 'docs/paper_revision_20261019/analysis_protocol.md'),
        inference_cost_sha256=inference_hash, figures=stems))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=S.OUTPUT)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    classes, counts, runs, generation, costs, provenance = analyze_existing(out)
    contrasts, common = class_summaries(out, classes, runs)
    svm, svmsummary = read_svm(out, runs)
    cost, gcost = cost_summaries(out, runs, costs)
    stems = figures(out, contrasts, generation, svmsummary)
    reports(out, classes, contrasts, generation, runs, cost, gcost, svm, svmsummary, provenance, stems)
    print('Complete evidence bundle: ' + str(out), flush=True)


if __name__ == '__main__':
    main()
