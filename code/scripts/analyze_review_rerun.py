"""Strict paired analysis of complete scopes in the frozen review matrix."""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
import numpy as np
import review_rerun_protocol as P

FIELDS = ('fold', 'shot', 'backbone', 'method', 'seed')
METRICS = ('accuracy', 'f1_macro')
METHODS = ('none', 'diffusion', 'jitter', 'mix')
PRIMARY = (('diffusion', 'none'), ('diffusion', 'jitter'), ('mix', 'jitter'))
CONTROLS = (('repeat', 'none'), ('diffusion', 'repeat'), ('jitter', 'repeat'),
            ('mix', 'repeat'), ('diffusion_zscore', 'diffusion'), ('mix_zscore', 'mix'))
VERSION = 'review-analysis-v1'


def key(row):
    return tuple(row[n] for n in FIELDS)


def csv_out(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def sd(a):
    return float(np.std(a, ddof=1)) if len(a) > 1 else None


def sign_flip(a):
    a = np.asarray(a, dtype=float)
    null = [abs((a * s).mean()) for s in itertools.product((-1, 1), repeat=len(a))]
    return float(np.mean(np.asarray(null) >= abs(a.mean()) - 1e-12))


def bootstrap(a, salt):
    if a.shape[1] < 2:
        return None, None
    entropy = int.from_bytes(hashlib.sha256(salt.encode()).digest()[:8], 'little')
    rng = np.random.default_rng(entropy)
    indices = rng.integers(0, a.shape[1], size=(10000, a.shape[1]))
    lo, hi = np.quantile(a[:, indices].mean(axis=(0, 2)), (0.025, 0.975))
    return float(lo), float(hi)


def scopes_from_matrix(matrix):
    scopes = []
    for backbone, shot in sorted({(r['backbone'], r['shot']) for r in matrix}):
        part = [r for r in matrix if (r['backbone'], r['shot']) == (backbone, shot)]
        folds = sorted({r['fold'] for r in part})
        available = {r['method'] for r in part}
        definitions = [('main', METHODS, PRIMARY)]
        if any(m in available for m in ('repeat', 'diffusion_zscore', 'mix_zscore')):
            definitions.append(('controls', METHODS + ('repeat', 'diffusion_zscore', 'mix_zscore'), CONTROLS))
        for kind, methods, comparisons in definitions:
            if not set(methods).issubset(available):
                continue
            sets = [{r['seed'] for r in part if r['fold'] == f and r['method'] == m}
                    for f in folds for m in methods]
            seeds = sorted(set.intersection(*sets))
            if not seeds or (kind == 'main' and any(s != set(seeds) for s in sets)):
                raise ValueError('Nonrectangular or empty planned comparison')
            if kind == 'controls':
                for f in folds:
                    for m in ('repeat', 'diffusion_zscore', 'mix_zscore'):
                        if {r['seed'] for r in part if r['fold'] == f and r['method'] == m} != set(seeds):
                            raise ValueError('Nonrectangular control matrix')
            scopes.append(dict(name=f'{backbone}_{kind}_k{shot}', kind=kind, backbone=backbone,
                shot=shot, folds=folds, seeds=seeds, methods=methods, comparisons=comparisons,
                expected=[(f, shot, backbone, m, s) for f in folds for m in methods for s in seeds]))
    return scopes


def calculate(scopes, indexed, smoke):
    summaries, folds, contrasts, paired = [], [], [], []
    for scope in scopes:
        scope['completed'] = sum(k in indexed for k in scope['expected'])
        scope['complete'] = scope['completed'] == len(scope['expected'])
        if not scope['complete']:
            continue
        common = dict(scope=scope['name'], kind=scope['kind'], backbone=scope['backbone'], shot=scope['shot'])
        for metric in METRICS:
            values = {m: np.asarray([[100 * indexed[(f, scope['shot'], scope['backbone'], m, s)][metric]
                    for s in scope['seeds']] for f in scope['folds']]) for m in scope['methods']}
            for method, a in values.items():
                low, high = bootstrap(a, f'{scope["name"]}/{metric}/{method}') if not smoke else (None, None)
                summaries.append(dict(common, method=method, metric=metric, n_folds=a.shape[0],
                    n_seeds=a.shape[1], n_runs=a.size, mean_pct=float(a.mean()),
                    seed_sd_pp=sd(a.mean(axis=0)), fold_sd_pp=sd(a.mean(axis=1)),
                    conditional_seed_ci_low=low, conditional_seed_ci_high=high))
                for i, fold in enumerate(scope['folds']):
                    folds.append(dict(common, fold=fold, method=method, metric=metric,
                        n_seeds=a.shape[1], mean_pct=float(a[i].mean()), seed_sd_pp=sd(a[i])))
            comparisons = [(a, b, False) for a, b in scope['comparisons']]
            if scope['kind'] == 'main':
                comparisons.append(('mix', 'diffusion', True))
            for method, baseline, descriptive in comparisons:
                d = values[method] - values[baseline]
                df = d.mean(axis=1)
                low, high = bootstrap(d, f'{scope["name"]}/{metric}/{method}-{baseline}') if not smoke else (None, None)
                infer = len(df) == 4 and d.shape[1] > 1 and not descriptive and not smoke
                contrasts.append(dict(common, metric=metric, contrast=f'{method}-{baseline}',
                    method=method, baseline=baseline, descriptive_only=descriptive,
                    n_folds=len(df), n_seeds=d.shape[1], n_pairs=d.size, mean_difference_pp=float(df.mean()),
                    seed_sd_pp=sd(d.mean(axis=0)), fold_sd_pp=sd(df),
                    conditional_seed_ci_low=low, conditional_seed_ci_high=high,
                    positive_folds=int(np.sum(df > 1e-12)), zero_folds=int(np.sum(abs(df) <= 1e-12)),
                    negative_folds=int(np.sum(df < -1e-12)), p_reference=sign_flip(df) if infer else None,
                    holm_reference=None, family=f'{scope["backbone"]}/{scope["kind"]}/{metric}' if not descriptive else ''))
                for fi, f in enumerate(scope['folds']):
                    for si, s in enumerate(scope['seeds']):
                        paired.append(dict(common, metric=metric, contrast=f'{method}-{baseline}', fold=f, seed=s,
                            method_pct=float(values[method][fi, si]), baseline_pct=float(values[baseline][fi, si]),
                            difference_pp=float(d[fi, si])))
    for family in sorted({r['family'] for r in contrasts if r['family']}):
        backbone, kind, metric = family.split('/')
        members = [s for s in scopes if s['backbone'] == backbone and s['kind'] == kind]
        records = [r for r in contrasts if r['family'] == family]
        if all(s['complete'] for s in members) and all(r['p_reference'] is not None for r in records):
            previous = 0.0
            for rank, row in enumerate(sorted(records, key=lambda r: r['p_reference'])):
                previous = max(previous, (len(records) - rank) * row['p_reference'])
                row['holm_reference'] = min(1.0, previous)
    return summaries, folds, contrasts, paired


def make_figures(out, summaries, paired, complete, smoke):
    names = sorted({r['scope'] for r in summaries if r['kind'] == 'main'})
    if not names:
        return []
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'pdf.fonttype': 42})
    dest = out / 'figures'
    dest.mkdir(exist_ok=True)
    cols = min(3, len(names))
    nrows = math.ceil(len(names) / cols)
    suffix = ' | SMOKE ONLY' if smoke else (' | INCOMPLETE MATRIX' if not complete else '')
    outputs = []
    for figure_id in (1, 2):
        fig, axes = plt.subplots(nrows, cols, figsize=(4.6 * cols, 4.4 * nrows), squeeze=False)
        for ax, name in zip(axes.flat, names):
            if figure_id == 1:
                records = {r['method']: r for r in summaries if r['scope'] == name and r['metric'] == 'accuracy'}
                for x, method in enumerate(METHODS):
                    row = records[method]
                    ax.scatter(x, row['mean_pct'], color=plt.get_cmap('tab10')(x), zorder=3)
                    if row['conditional_seed_ci_low'] is not None:
                        ax.vlines(x, row['conditional_seed_ci_low'], row['conditional_seed_ci_high'])
                ax.set_xticks(range(4), METHODS, rotation=25, ha='right')
                ax.set_ylabel('Accuracy (%)')
            else:
                for x, (method, baseline) in enumerate(PRIMARY):
                    records = [r for r in paired if r['scope'] == name and r['metric'] == 'accuracy'
                               and r['contrast'] == f'{method}-{baseline}']
                    ids = sorted({r['fold'] for r in records})
                    offsets = np.linspace(-.2, .2, len(ids)) if len(ids) > 1 else [0]
                    for offset, fold in zip(offsets, ids):
                        effect = np.mean([r['difference_pp'] for r in records if r['fold'] == fold])
                        ax.scatter(x + offset, effect, color=plt.get_cmap('tab10')(fold),
                                   label=f'Fold {fold}' if x == 0 else None)
                ax.axhline(0, color='black', lw=.8)
                ax.set_xticks(range(3), [f'{a}\n- {b}' for a, b in PRIMARY], rotation=15, ha='right')
                ax.set_ylabel('Paired difference (pp)')
                ax.legend(fontsize=8)
            ax.set_title(name)
            ax.grid(axis='y', alpha=.2)
        for ax in list(axes.flat)[len(names):]:
            ax.set_visible(False)
        title = 'Main comparisons' if figure_id == 1 else 'Effects by held-out load'
        fig.suptitle(title + suffix)
        fig.text(.5, .01, 'Fixed data; intervals describe seed variability only. Smoke/single seed: no inference.', ha='center', fontsize=8)
        fig.tight_layout(rect=(0, .045, 1, .94))
        stem = 'figure-01-main-comparison' if figure_id == 1 else 'figure-02-fold-differences'
        for ext in ('png', 'pdf'):
            fig.savefig(dest / f'{stem}.{ext}', dpi=220, bbox_inches='tight')
        plt.close(fig)
        outputs.append(stem)
    return outputs


def reports(out, state, coverage, summaries, contrasts, figures):
    status = '完整' if state['complete'] else '未完成；不得据已完成部分推导总体结论'
    if state['smoke_only']:
        status += '；仅程序冒烟验证，不可用于论文'
    def num(v):
        return '—' if v is None else f'{v:.3f}'
    lines = ['# 补实验严格分析', '', f'状态：**{status}**。已核验 {state["verified_runs"]}/{state["planned_runs"]} 次。',
        '', '主要问题：扩散相对不增强和扰动有多少增量，混合相对扰动是否稳定；以 accuracy 为主、macro-F1 为次要指标。',
        '', '| 阶段 | 完成 | 预定 |', '|---|---:|---:|']
    lines += [f'| {r["stage"]} | {r["completed"]} | {r["planned"]} |' for r in coverage]
    lines += ['', '每个 scope 的所有预定方法、折及种子配齐后才汇总。不同支持集单独报告；分类器种子不是独立支持集或测试工况。', '',
        '| 完整范围 | 方法 | 指标 | 均值 % | 种子 SD pp | 条件种子 95% CI | 折×种子 |', '|---|---|---|---:|---:|---|---|']
    lines += [f'| {r["scope"]} | {r["method"]} | {r["metric"]} | {r["mean_pct"]:.3f} | {num(r["seed_sd_pp"])} | [{num(r["conditional_seed_ci_low"])}, {num(r["conditional_seed_ci_high"])}] | {r["n_folds"]}×{r["n_seeds"]} |' for r in summaries]
    if not summaries:
        lines += ['尚无完整 scope，不生成性能汇总或性能图。缺失项见 missing-runs.csv。']
    lines += ['', '种子 SD：先对固定折求均值，再计算种子样本 SD。逐折种子 SD 见 fold-summary.csv。条件区间只描述固定数据上的训练随机性，不是总体泛化区间。', '',
        '| 完整范围 | 指标 | 配对比较 | 差值 pp | 正/零/负折 | 参考 p / Holm |', '|---|---|---|---:|---|---|']
    lines += [f'| {r["scope"]} | {r["metric"]} | {r["contrast"]} | {r["mean_difference_pp"]:+.3f} | {r["positive_folds"]}/{r["zero_folds"]}/{r["negative_folds"]} | {num(r["p_reference"])} / {num(r["holm_reference"])} |' for r in contrasts]
    lines += ['', '保留全部预定比较及负增益。mix-diffusion 为描述性核对；mix-jitter 为正不足以说明混合优于最强单一增强。配对差值的条件区间见 paired-contrasts.csv。', '',
        '4 个负载折训练集重叠。双侧符号翻转参考检验最小 p=0.125，且需要设计未保证的符号可交换假设；不能由这些结果宣称总体显著提升。', '',
        '新实现和调度同时改变，不能将新旧差值全部归因于某个 bug。none 与增强组的批次组成不同；相同步数不等于相同样本呈现量或墙钟时间。raw 与 dual 只可在共同种子上比较增强效应。']
    (out/'analysis-report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    (out/'stats-appendix.md').write_text('\n'.join([
        '# 统计方法与边界', '', f'状态：{status}。', '',
        '- 配对单元为 fold/K/backbone/seed；效应量为准确率或 macro-F1 差的百分点，先对每折的配对种子求均值，再等权汇总折。',
        '- 条件 bootstrap 固定所有折，对全部折共同重抽相同的配对 seed 列 10,000 次，取 2.5/97.5 百分位。它不描述工况总体泛化；只有 3 seeds 时区间估计也有限。',
        '- 仅完整非 smoke、4 折且多于一个种子的预定比较枚举 16 个符号向量。该双侧参考检验依赖符号可交换性，LOOCV 训练集重叠限制解释。4 折不足以验证分布假设，不以低功效正态性检验筛选显著检验。',
        '- Holm 家族为 backbone×问题组(main/controls)×指标，跨所有预定 K；3 个 K 的主比较 9 项、控制比较 18 项。家族未完成不提供调整 p。',
        '- mix-diffusion 仅描述。单 seed 或 smoke 不提供 CI/p。空值不是零。保留逐折及逐 seed 数值，不能把测试窗口或分类器种子视为独立工况。',
        '- 主 dual 使用全部预定主种子；控制比较使用控制组预定种子；raw 和独立 draw 使用其 manifest 预定种子，不混合 10-seed/3-seed 均值。',
        '', '精确数值见 method-summary.csv、fold-summary.csv、paired-contrasts.csv、paired-seed-differences.csv。'])+'\n', encoding='utf-8')
    text = ['# 图件目录', '', f'状态：{status}。', '']
    if figures:
        text += ['1. figure-01-main-comparison（PNG/PDF）：来源 method-summary.csv。点为固定折等权均值，竖线为条件种子 bootstrap 95% 区间。逐一比较三个预定主效应，不能由误差条是否重叠判显著。结论需结合第二张图的逐折一致性；smoke 不作科研解释。', '',
            '2. figure-02-fold-differences（PNG/PDF）：来源 paired-seed-differences.csv。每点是一个负载折上的配对种子均值差，零线表示无增量。核对正负折数与最大反向差；若方向不一致，论文需限定到相应负载/K，不能仅写总体均值提升。4 折即使同向也不能宣称总体显著。']
    else:
        text += ['预定主比较尚未完整，不使用未配齐或合成分数生成性能图。']
    (out/'figure-catalog.md').write_text('\n'.join(text)+'\n', encoding='utf-8')


class IncompleteMatrixError(RuntimeError):
    pass


def analyze(root: Path, require_complete: bool = False):
    root = Path(root).resolve()
    manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
    spec = manifest['payload']['protocol']
    matrix = spec['matrix']
    planned = {key(r) for r in matrix}
    if not matrix or len(planned) != len(matrix):
        raise ValueError('Empty or duplicate matrix')
    indexed, truths = {}, {}
    for row in P.read_valid_rows(root, manifest):
        k = key(row)
        if k not in planned or k in indexed:
            raise ValueError('Unexpected or duplicated run')
        with np.load(root/row['artifacts']['predictions']['path'], allow_pickle=False) as z:
            truth = z['true']
        if row['fold'] in truths and not np.array_equal(truths[row['fold']], truth):
            raise ValueError('Test label order differs within fold')
        truths[row['fold']] = truth
        indexed[k] = row
    smoke = spec.get('smoke_only', False)
    scopes = scopes_from_matrix(matrix)
    summaries, folds, contrasts, paired = calculate(scopes, indexed, smoke)
    missing = sorted(planned - set(indexed))
    out = root/'analysis'
    out.mkdir(exist_ok=True)
    totals = Counter(r['stage'] for r in matrix)
    done = Counter(r['stage'] for r in matrix if key(r) in indexed)
    coverage = [dict(stage=s, completed=done[s], planned=n) for s, n in totals.items()]
    csv_out(out/'completion.csv', coverage, ['stage','completed','planned'])
    csv_out(out/'missing-runs.csv', [dict(zip(FIELDS,k)) for k in missing], list(FIELDS))
    csv_out(out/'verified-runs.csv', list(indexed.values()), list(FIELDS)+['stage','accuracy','f1_macro','elapsed_sec','completed_steps','best_step'])
    shared = ['scope','kind','backbone','shot']
    csv_out(out/'method-summary.csv', summaries, shared+['method','metric','n_folds','n_seeds','n_runs','mean_pct','seed_sd_pp','fold_sd_pp','conditional_seed_ci_low','conditional_seed_ci_high'])
    csv_out(out/'fold-summary.csv', folds, shared+['fold','method','metric','n_seeds','mean_pct','seed_sd_pp'])
    csv_out(out/'paired-contrasts.csv', contrasts, shared+['metric','contrast','method','baseline','descriptive_only','n_folds','n_seeds','n_pairs','mean_difference_pp','seed_sd_pp','fold_sd_pp','conditional_seed_ci_low','conditional_seed_ci_high','positive_folds','zero_folds','negative_folds','p_reference','holm_reference','family'])
    csv_out(out/'paired-seed-differences.csv', paired, shared+['metric','contrast','fold','seed','method_pct','baseline_pct','difference_pp'])
    figures = make_figures(out, summaries, paired, not missing, smoke)
    state = dict(analysis=str(out), analysis_version=VERSION, manifest_fingerprint=manifest['fingerprint'],
        script_sha256=P.file_hash(Path(__file__)), created_utc=P.now(), complete=not missing, smoke_only=smoke,
        verified_runs=len(indexed), planned_runs=len(matrix), figures=figures, scopes=scopes)
    reports(out, state, coverage, summaries, contrasts, figures)
    P.dump(out/'analysis-state.json', state)
    if require_complete and missing:
        raise IncompleteMatrixError(f'Matrix incomplete: {len(indexed)}/{len(matrix)}; report: {out}')
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    try:
        result = analyze(args.root, args.require_complete)
    except IncompleteMatrixError as exc:
        parser.exit(2, str(exc)+'\n')
    print(json.dumps({k:result[k] for k in ('analysis','complete','smoke_only','verified_runs','planned_runs')}))
