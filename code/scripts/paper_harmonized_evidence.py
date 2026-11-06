"""Complete-only evidence for explicitly selected, harmonized 12 kHz CWRU data.

CPU-only. `--phase svm` may run once support preparation is complete; the main
analysis is gated on all 564 classifier results and 36 support-bound pools.
"""
from __future__ import annotations

import argparse
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support

import analyze_review_rerun as A
import paper_evidence_analysis as E
import paper_harmonized_prepare as D
import paper_harmonized_rerun as H
import paper_svm_baseline as S
import review_rerun_protocol as P

ROOT = P.WORKSPACE / 'results/paper_cwru_12k_20261019'
OUT = ROOT / 'analysis'
DATA = P.WORKSPACE / 'data/paper_cwru_12k_20261019'
METRICS = ('accuracy', 'f1_macro', 'fault_conditional_accuracy')
METHODS = ('none', 'diffusion', 'jitter', 'mix')
SCOPES = ('draw0', 'draw1', 'draw2')


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def csv(out: Path, name: str, rows) -> pd.DataFrame:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(out / name, index=False, encoding='utf-8-sig')
    return frame


def metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    if true.ndim != 1 or pred.shape != true.shape or not np.isin(pred, range(10)).all():
        raise ValueError('Incorrect prediction dimensions or class ids')
    fault = true != 0
    if not fault.any():
        raise ValueError('Fault-condition metric requires fault cases')
    return dict(accuracy=float(np.mean(true == pred)),
        f1_macro=float(f1_score(true, pred, labels=range(10), average='macro', zero_division=0)),
        fault_conditional_accuracy=float(np.mean(true[fault] == pred[fault])))


def load_split(draw: int, fold: int, shot: int, data: Path = DATA) -> tuple:
    path = D.support_path(draw, fold, shot, data)
    meta = read(path / 'metadata.json')
    if (meta.get('scheme') != D.VERSION or meta.get('target_sampling_rate_hz') != 12000 or
        (meta['draw'], meta['fold'], meta['n_shot']) != (draw, fold, shot) or
        meta['source_audit_sha256'] != P.file_hash(data / 'source_audit.json')):
        raise ValueError('Not this harmonized data protocol')
    arrays = {}
    for name in D.ARRAY_NAMES:
        file = path / f'{name}.npy'
        value = np.load(file, allow_pickle=False)
        if (P.file_hash(file) != meta['file_sha256'][file.name] or
            P.array_hash(value) != meta['array_sha256'][name] or not np.isfinite(value).all()):
            raise ValueError('Changed prepared support/validation/test')
        arrays[name] = value
    n_test = len(arrays['test_y']) // 10
    for part, n in (('train', shot), ('val', 20), ('test', n_test)):
        if (arrays[f'{part}_X'].shape != (10*n, 1024) or
            not np.array_equal(np.bincount(arrays[f'{part}_y'], minlength=10), np.full(10, n))):
            raise ValueError('Invalid balanced split shape or labels')
    if n_test != meta['common_windows_per_record']:
        raise ValueError('Test count does not match explicit source preparation')
    return arrays, meta, path


def run_svm(out: Path = OUT, data: Path = DATA) -> None:
    dest = out / 'svm'
    dest.mkdir(parents=True, exist_ok=True)
    audit = read(data / 'source_audit.json')
    if audit['scheme'] != D.VERSION:
        raise ValueError('Cannot use historical CWRU splits')
    support_meta = {}
    # Source reconstruction is independent of generator and classifier artifacts.
    for draw, fold, shot in itertools.product(range(3), range(4), (1, 3, 5)):
        D.verify_support(draw, fold, shot, data_root=data, source_root=P.WORKSPACE)
        path = D.support_path(draw, fold, shot, data) / 'metadata.json'
        support_meta[f'{draw}/{fold}/{shot}'] = P.file_hash(path)
    code_files = [Path(__file__), Path(S.__file__)]
    payload = dict(version='harmonized-12k-feature-svm-v1', feature_names=S.FEATURES, grid=S.GRID,
        selection='first maximum validation accuracy; train-only StandardScaler; no refit with validation',
        train_labels_per_class='K', validation_labels_per_class=20, settings=36,
        source_audit_sha256=P.file_hash(data / 'source_audit.json'), split_metadata=support_meta,
        code_sha256={p.name: P.file_hash(p) for p in code_files}, numpy=np.__version__, sklearn=S.sklearn.__version__)
    fingerprint = P.object_hash(payload)
    manifest_path = dest / 'manifest.json'
    if manifest_path.exists() and read(manifest_path)['fingerprint'] != fingerprint:
        raise ValueError('SVM protocol/code/data changed; use a new output directory')
    if not manifest_path.exists():
        P.dump(manifest_path, dict(fingerprint=fingerprint, payload=payload, created_utc=P.now()))
        for p in code_files:
            (dest / p.name).write_bytes(p.read_bytes())
    for draw, fold, shot in itertools.product(range(3), range(4), (1, 3, 5)):
        key = f'd{draw}_f{fold}_k{shot}'
        folder = dest / key
        result_path = folder / 'result.json'
        if result_path.exists():
            saved = read(result_path)
            if saved['fingerprint'] != fingerprint:
                raise ValueError('SVM resume fingerprint mismatch')
            for name, digest in saved['files'].items():
                if P.file_hash(folder / name) != digest:
                    raise ValueError('Changed SVM result artifact')
            continue
        start = time.perf_counter()
        arrays, meta, split = load_split(draw, fold, shot, data)
        feature_start = time.perf_counter()
        train, val = S.features(arrays['train_X']), S.features(arrays['val_X'])
        feature_sec = time.perf_counter() - feature_start
        fit_start = time.perf_counter()
        scaler, model, grid = S.select_model(train, arrays['train_y'], val, arrays['val_y'])
        fit_sec = time.perf_counter() - fit_start
        test_start = time.perf_counter()
        pred = model.predict(scaler.transform(S.features(arrays['test_X'])))
        test_sec = time.perf_counter() - test_start
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions.npz', true=arrays['test_y'], pred=pred)
        joblib.dump(dict(scaler=scaler, classifier=model, features=S.FEATURES), folder / 'model.joblib')
        row = dict(draw=draw, bundle=f'draw{draw}', fold=fold, shot=shot, method='feature_svm',
            **metrics(arrays['test_y'], pred), selected_C=model.C, selected_gamma=model.gamma, grid=grid,
            scaler_fit_samples=int(scaler.n_samples_seen_), scaler_mean=scaler.mean_.tolist(), scaler_scale=scaler.scale_.tolist(),
            train_val_features_sec=feature_sec, grid_fit_predict_sec=fit_sec, test_features_predict_sec=test_sec,
            elapsed_sec=time.perf_counter() - start, fingerprint=fingerprint, n_test=len(pred),
            split_metadata_sha256=P.file_hash(split / 'metadata.json'), sources=meta['array_sha256'],
            processor=platform.processor(), finished_utc=P.now(),
            files={p.name: P.file_hash(p) for p in (folder / 'predictions.npz', folder / 'model.joblib')})
        P.dump(result_path, row)
        print(f'SVM {key} acc={row["accuracy"]:.6f} fault_acc={row["fault_conditional_accuracy"]:.6f}', flush=True)
    P.dump(dest / 'status.json', dict(complete=True, completed=36, fingerprint=fingerprint, finished_utc=P.now()))


def require_complete(classification: Path, out: Path) -> dict:
    manifest_path = classification / 'manifest.json'
    if not manifest_path.exists():
        P.dump(out / 'status.json', dict(complete=False, reason='classification_manifest_missing'))
        raise RuntimeError('Main analysis blocked until the harmonized classification campaign is complete')
    manifest = read(manifest_path)
    payload = manifest['payload']
    expected = {H.tag(k) for k in H.matrix(False)}
    actual_plan = {H.tag(k) for k in payload['protocol']['matrix']}
    actual = {p.stem for p in (classification / 'runs').glob('*.json')}
    if (P.object_hash(payload) != manifest['fingerprint'] or payload['protocol']['smoke_only'] or
        payload['protocol']['target_sampling_rate'] != 12000 or actual_plan != expected):
        raise ValueError('Wrong or corrupt harmonized classifier protocol')
    if actual != expected:
        P.dump(out / 'status.json', dict(complete=False, completed=len(actual & expected), planned=len(expected),
            missing=sorted(expected - actual), unexpected=sorted(actual - expected)))
        raise RuntimeError(f'No main results: {len(actual & expected)}/{len(expected)} classification records')
    return manifest


def scopes(draw: int) -> list[dict]:
    definitions = []
    for k in (1, 3, 5):
        definitions.append(('main', 'dual', k, METHODS, A.PRIMARY))
        if draw == 0:
            definitions.append(('repeat', 'dual', k, METHODS + ('repeat',),
                (('repeat', 'none'), ('diffusion', 'repeat'), ('jitter', 'repeat'), ('mix', 'repeat'))))
            definitions.append(('raw', 'raw', k, ('none', 'mix'), (('mix', 'none'),)))
    if draw == 0:
        definitions.append(('standardization', 'dual', 1,
            ('diffusion', 'mix', 'diffusion_zscore', 'mix_zscore'),
            (('diffusion_zscore', 'diffusion'), ('mix_zscore', 'mix'))))
    return [dict(name=f'{backbone}_{kind}_k{k}', kind=kind, backbone=backbone, shot=k,
        folds=list(range(4)), seeds=list(range(3)), methods=methods, comparisons=comparisons,
        expected=[(f, k, backbone, m, s) for f, m, s in itertools.product(range(4), methods, range(3))])
        for kind, backbone, k, methods, comparisons in definitions]


def summary_statistics(rows: list[dict], out: Path) -> tuple:
    summaries, folds, contrasts, paired = [], [], [], []
    previous = A.METRICS
    try:
        A.METRICS = METRICS
        for draw in range(3):
            indexed = {A.key(r): r for r in rows if r['draw'] == draw}
            calculated = A.calculate(scopes(draw), indexed, False)
            for target, records in zip((summaries, folds, contrasts, paired), calculated):
                target.extend(dict(draw=draw, bundle=f'draw{draw}', **r) for r in records)
    finally:
        A.METRICS = previous
    return (csv(out, 'method-summary.csv', summaries), csv(out, 'fold-summary.csv', folds),
            csv(out, 'paired-contrasts.csv', contrasts), csv(out, 'paired-seed-differences.csv', paired))


def prediction_details(rows: list[dict], classification: Path, out: Path) -> tuple:
    classes, counts, runs = [], [], []
    for row in rows:
        with np.load(classification / row['artifacts']['predictions']['path']) as data:
            true, pred = data['true'], data['pred']
        recalculated = metrics(true, pred)
        if any(abs(row[m] - recalculated[m]) > 1e-12 for m in METRICS):
            raise ValueError('Unexpected metric after validation')
        key = dict(bundle=f'draw{row["draw"]}', fold=row['fold'], shot=row['shot'],
                   backbone=row['backbone'], method=row['method'], seed=row['seed'])
        precision, recall, f1, support = precision_recall_fscore_support(true, pred, labels=range(10), zero_division=0)
        for c in range(10):
            classes.append(dict(key, class_id=c, class_name=D.CLASSES[c], precision_pct=100*precision[c],
                recall_pct=100*recall[c], f1_pct=100*f1[c], n_test=int(support[c])))
        cm = confusion_matrix(true, pred, labels=range(10))
        for c, j in itertools.product(range(10), range(10)):
            counts.append(dict(key, true_class=D.CLASSES[c], pred_class=D.CLASSES[j], count=int(cm[c,j])))
        runs.append(dict(key, accuracy_pct=100*row['accuracy'], f1_macro_pct=100*row['f1_macro'],
            fault_conditional_accuracy_pct=100*row['fault_conditional_accuracy'], n_test=len(true),
            elapsed_sec=row['elapsed_sec'], preprocessing_sec=row['preprocessing_sec'], training_sec=row['training_sec'],
            test_evaluation_sec=row['test_evaluation_sec'], model_params=row['model_params'], gpu=row['gpu'],
            samples_seen=row['samples_seen'],
            cwt_cache_all_hit=all(c['cache_hit'] for c in row['cache'].values()) if row['cache'] else None))
    return csv(out, 'class-metrics.csv', classes), csv(out, 'confusion-counts.csv', counts), csv(out, 'run-metrics-cost.csv', runs)


def class_contrasts(classes: pd.DataFrame, out: Path) -> pd.DataFrame:
    all_rows = []
    for bundle, k, (method, baseline), c in itertools.product(SCOPES, (1, 3, 5), E.CONTRASTS, range(10)):
        part = classes[(classes.bundle == bundle) & (classes.backbone == 'dual') & (classes.shot == k) & (classes.class_id == c)]
        a = part[part.method == method].pivot(index='fold', columns='seed', values='recall_pct')
        b = part[part.method == baseline].pivot(index='fold', columns='seed', values='recall_pct')
        if a.shape != (4, 3) or b.shape != (4, 3):
            raise ValueError('Missing full class contrast')
        delta = (a-b).to_numpy()
        low, high = A.bootstrap(delta, f'12k/{bundle}/{k}/{method}/{baseline}/{c}')
        all_rows.append(dict(bundle=bundle, shot=k, contrast=f'{method}-{baseline}', class_id=c, class_name=D.CLASSES[c],
            n_folds=4, n_seeds=3, recall_difference_pp=float(delta.mean()),
            conditional_seed_ci_low=low, conditional_seed_ci_high=high,
            positive_folds=int(np.sum(delta.mean(1)>1e-12)), negative_folds=int(np.sum(delta.mean(1)<-1e-12))))
    return csv(out, 'class-contrasts-common3.csv', all_rows)


def generation_details(manifest: dict, out: Path) -> tuple:
    rows, costs = [], []
    paths = manifest['payload']['paths']
    for draw, fold, k in itertools.product(range(3), range(4), (1, 3, 5)):
        key = dict(draw=draw, fold=fold, shot=k, method='none')
        arrays, info = H.load_arrays(key, Path(paths['data_root']), Path(paths['generator_root']))
        if info != manifest['payload']['sources'][H.source_key(key)]:
            raise ValueError('Generator/source changed after classification freeze')
        pool = Path(info['pool_dir'])
        fake = np.load(pool / 'fake_X.npy', allow_pickle=False)
        fake_y = np.load(pool / 'fake_y.npy', allow_pickle=False)
        for c in range(10):
            rows.append(dict(bundle=f'draw{draw}', fold=fold, shot=k, class_id=c, class_name=D.CLASSES[c],
                **E.pool_metrics(arrays['real_X'][arrays['real_y']==c], fake[fake_y==c])))
        gen = pool.parents[1]
        for stage, filename in (('generator_training', 'training_complete.json'), ('generator_sampling_and_pool_save', 'generation_complete.json')):
            receipt = read(gen / filename)
            if receipt['config_sha256'] != P.file_hash(gen / 'config.json'):
                raise ValueError('Wrong generator timing source')
            costs.append(dict(bundle=f'draw{draw}', fold=fold, shot=k, stage=stage,
                elapsed_sec=receipt['elapsed_seconds'], peak_cuda_allocated_bytes=receipt['peak_cuda_allocated_bytes'],
                optimizer_updates=receipt.get('optimizer_updates'), source_sha256=P.file_hash(gen/filename)))
    return csv(out, 'generation-diagnostics.csv', rows), csv(out, 'generator-cost.csv', costs)


def svm_details(out: Path) -> pd.DataFrame:
    path = out / 'svm'
    manifest = read(path / 'manifest.json')
    if P.object_hash(manifest['payload']) != manifest['fingerprint'] or not read(path/'status.json')['complete']:
        raise ValueError('SVM results incomplete or corrupt')
    rows = []
    for draw, fold, k in itertools.product(range(3), range(4), (1, 3, 5)):
        folder = path / f'd{draw}_f{fold}_k{k}'
        row = read(folder / 'result.json')
        if (row['draw'], row['fold'], row['shot']) != (draw,fold,k) or row['fingerprint'] != manifest['fingerprint']:
            raise ValueError('Wrong SVM source')
        if row['scaler_fit_samples'] != 10*k:
            raise ValueError('SVM scaler used more than support')
        for name, digest in row['files'].items():
            if P.file_hash(folder/name) != digest:
                raise ValueError('Changed SVM artifact')
        with np.load(folder/'predictions.npz') as data:
            if P.array_hash(data['true']) != row['sources']['test_y']:
                raise ValueError('Changed SVM target')
            measured = metrics(data['true'], data['pred'])
        if any(abs(row[m]-measured[m]) > 1e-12 for m in METRICS):
            raise ValueError('Changed SVM metrics')
        rows.append({n: row[n] for n in ('draw','bundle','fold','shot',*METRICS,'n_test','selected_C',
            'selected_gamma','train_val_features_sec','grid_fit_predict_sec','test_features_predict_sec','elapsed_sec')})
    return csv(out, 'svm-results.csv', rows)


def backbone_comparison(rows: list[dict], out: Path) -> pd.DataFrame:
    indexed = {(r['draw'],r['fold'],r['shot'],r['backbone'],r['method'],r['seed']):r for r in rows}
    result = []
    for shot, method, metric in itertools.product((1,3,5), ('none','mix'), METRICS):
        delta = np.array([[100*(indexed[(0,f,shot,'dual',method,s)][metric] -
            indexed[(0,f,shot,'raw',method,s)][metric]) for s in range(3)] for f in range(4)])
        low,high = A.bootstrap(delta, f'12k/backbone/{shot}/{method}/{metric}')
        result.append(dict(shot=shot, method=method, metric=metric, contrast='dual-raw',
            mean_difference_pp=float(delta.mean()), conditional_seed_ci_low=low,conditional_seed_ci_high=high,
            positive_folds=int(np.sum(delta.mean(1)>1e-12)),negative_folds=int(np.sum(delta.mean(1)<-1e-12))))
    return csv(out,'backbone-contrasts.csv',result)


def tables_and_reports(out, manifest, summary, contrasts, classes, generation, costs, runs, svm, backbone):
    main = summary[(summary.kind=='main') & (summary.metric=='accuracy')]
    svm_summary = svm.groupby(['bundle','shot']).agg(accuracy=('accuracy','mean'),f1_macro=('f1_macro','mean'),
        fault_conditional_accuracy=('fault_conditional_accuracy','mean'),fold_sd=('accuracy','std'),
        fold_min=('accuracy','min'),fold_max=('accuracy','max'),n_folds=('fold','size')).reset_index()
    csv(out,'svm-summary.csv',svm_summary)
    gen_summary = generation.groupby(['bundle','shot']).agg(rms_ratio=('rms_ratio','mean'),
        rms_min=('rms_ratio','min'),rms_max=('rms_ratio','max'),psd_js_distance=('psd_js_distance','mean'),
        psd_correlation=('psd_correlation','mean')).reset_index()
    csv(out,'generation-summary.csv',gen_summary)
    runtime = runs.groupby(['bundle','backbone','shot','method']).agg(n_runs=('elapsed_sec','size'),
        total_median_sec=('elapsed_sec','median'),train_median_sec=('training_sec','median'),
        train_q25_sec=('training_sec',lambda x:x.quantile(.25)),train_q75_sec=('training_sec',lambda x:x.quantile(.75)),
        preprocess_median_sec=('preprocessing_sec','median'),test_evaluation_median_sec=('test_evaluation_sec','median')).reset_index()
    csv(out,'classifier-cost-summary.csv',runtime)
    gen_cost = costs.groupby(['shot','stage']).agg(n=('elapsed_sec','size'),median_sec=('elapsed_sec','median'),
        q25_sec=('elapsed_sec',lambda x:x.quantile(.25)),q75_sec=('elapsed_sec',lambda x:x.quantile(.75)),
        sum_sec=('elapsed_sec','sum')).reset_index()
    csv(out,'generator-cost-summary.csv',gen_cost)
    n_test = sorted(runs.n_test.unique().tolist())
    lines = ['# 统一12 kHz CWRU：完整主证据分析','',
        '本包仅使用明确官方记录字段、正常信号整段抗混叠降至12 kHz后生成的新划分、新生成池和新分类结果。历史1020次结果及其SVM/诊断不混入本包。', '',
        f'矩阵完整：564次分类、36个支持绑定生成池、36个确定性SVM。每折测试窗数为{n_test}，以实际数组为准；每类20个验证标签另计，不能称全流程仅K标签。S0/S1/S2分别对应新协议draw0/draw1/draw2，均非历史legacy配置。', '',
        '原始记录曾被探索，本次为统一采样率与明确通道后的复核，不是未见独立确认集。三支持配置的验证/测试集合相同，生成训练与采样随机性也随配置改变。','',
        '## 主结果','', '| 配置 | K | None | Diffusion | Jitter | Mix | 固定特征SVM |', '|---|---:|---:|---:|---:|---:|---:|']
    for bundle,k in itertools.product(SCOPES,(1,3,5)):
        part=main[(main.bundle==bundle)&(main.shot==k)].set_index('method')
        values=[part.loc[m,'mean_pct'] for m in METHODS]
        s=float(svm_summary[(svm_summary.bundle==bundle)&(svm_summary.shot==k)].accuracy.iloc[0])*100
        lines.append(f'| S{bundle[-1]} | {k} | '+' | '.join(f'{v:.3f}' for v in [*values,s])+' |')
    lines+=['','深度方法为4折×3种子等权均值；SVM每配置/折仅一次。SVM折间SD及极值另存，不制造训练种子误差。全部macro-F1与排除Normal后的故障条件准确率见method-summary.csv和svm-summary.csv。','',
        '## 预定对比效应','', '| 配置 | K | 对比 | 准确率差pp | 正/负折 | 条件seed 95%区间 | 参考p/Holm |', '|---|---:|---|---:|---|---|---|']
    for r in contrasts[(contrasts.kind=='main')&(contrasts.metric=='accuracy')].itertuples():
        hp='—' if pd.isna(r.holm_reference) else f'{r.p_reference:.3f}/{r.holm_reference:.3f}'
        lines.append(f'| S{r.draw} | {r.shot} | {r.contrast} | {r.mean_difference_pp:+.3f} | {r.positive_folds}/{r.negative_folds} | [{r.conditional_seed_ci_low:.3f}, {r.conditional_seed_ci_high:.3f}] | {hp} |')
    lines+=['','配对区间仅描述固定数据和生成池条件下的分类器随机性。四折训练负载重叠，参考符号翻转p最小0.125，不能由此宣称总体显著提升。mix-diffusion为描述对比。','',
        '## 控制范围','',
        'S0覆盖全部K的repeat、raw-none/mix；生成逐窗标准化仅覆盖S0的K=1。完整效应见paired-contrasts.csv；dual−raw配对见backbone-contrasts.csv，不扩展为未测配置结论。','',
        '## 生成诊断','', '| 配置 | K | RMS比均值 | 完整类×折范围 | PSD JS均值 | PSD相关均值 |','|---|---:|---:|---|---:|---:|']
    for r in gen_summary.itertuples():
        lines.append(f'| S{r.bundle[-1]} | {r.shot} | {r.rms_ratio:.4f} | {r.rms_min:.4f}–{r.rms_max:.4f} | {r.psd_js_distance:.4f} | {r.psd_correlation:.4f} |')
    lines+=['','全部36池×10类均保留。真实支持逐窗标准化，RMS比是模型输入尺度而非物理振动强度。PSD指标以有限支持作参照；K同时改变参照均谱窗口数、池大小和生成预算，频谱更接近不证明故障机理真实性或分类效用。不得根据诊断筛选生成池重报测试。','',
        '## 运行成本','', '| K | 阶段 | 个数 | 中位数s | IQR s |','|---|---|---:|---:|---|']
    for r in gen_cost.itertuples():
        lines.append(f'| {r.shot} | {r.stage} | {r.n} | {r.median_sec:.3f} | {r.q25_sec:.3f}–{r.q75_sec:.3f} |')
    lines+=['','classifier-cost-summary.csv单列预处理、训练、测试评价与完整运行墙钟；训练计时包含训练中的验证和选模管理，预处理含数据载入/核验和缓存访问。生成采样计时含池保存。SVM CPU计时与GPU批推理不等口径，不能计算虚构加速比。','',
        '## 结论使用边界','', '只依据本协议数据更新主表和结论。四方法、低成本SVM与所有负差值完整报告，不将局部最优或固定样本量当稳定优势/失效边界。旧字段及采样率方案结果只存档，不能以新旧分数差估计某个修复的独立因果效果。']
    (out/'analysis-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (out/'stats-appendix.md').write_text('''# 新协议统计范围

- 主要指标accuracy；次指标macro-F1、fault_conditional_accuracy。后者仅去掉真实Normal窗口，仍从10类预测中判正确，不重训九分类器，不忽略误报成Normal。
- 支持配置S0/S1/S2对应新draw0/1/2。各主比较4折×3分类器种子，先种子再折等权平均。完整矩阵前不输出主结果。
- 配对效应量为百分点；seed SD为先平均四折后的种子样本SD。条件bootstrap固定四折，对配对seed列整体重抽10000次（SHA固定随机种子），2.5/97.5百分位。仅3 seeds限制精度，不能代表支持集或负载总体不确定性。
- 4折参考双侧符号翻转枚举16向量，最小p=0.125；折训练集合重叠、符号交换性未被设计保证，不据此宣称总体显著。Holm家族为每个draw×backbone×问题组×指标，跨该组全部K；main三对比×三K共9项，repeat四对比×三K共12项，standardization两项，仅K1，raw mix-none三项。mix-diffusion描述性，无p；dual-raw仅描述配对效应与条件区间。
- 类别误差为后验描述，不增加显著性发现家族。测试每类计数由新数组核对，不能将窗口或训练种子视独立故障事件。
- SVM固定22特征与9参数格，支持拟合StandardScaler；验证只选参，不并入训练。每配置确定性一次，四折SD为描述负载差异，无伪seed CI。
- 生成诊断每配置×K完整40类折单元，其依赖结构不支持按40独立样本推断；真实支持K、池10K。PSD JS为以2为底散度平方根；窗口周期图先单位归一化再按类别平均。
- 正常原始48kHz整段抗混叠到12kHz后切窗，故障原始12kHz使用明确字段；所有输入固定1024点。新测试仍来自已探索记录，不称未见确认。旧方案所有数值与图件不进入本包。
''',encoding='utf-8')


def final_figures(out: Path) -> None:
    import paper_final_figures as F
    F.OUT=out/'final_figures';F.OUT.mkdir(parents=True,exist_ok=True)
    F.BUNDLES=SCOPES;F.SUPPORT={b:f'S{i}' for i,b in enumerate(SCOPES)}
    spec=F.configure()
    produced=[F.flow_figure(spec),F.paired_figure(pd.read_csv(out/'run-metrics-cost.csv'),spec),
        F.class_figure(pd.read_csv(out/'class-contrasts-common3.csv'),spec),
        generation_figure(pd.read_csv(out/'generation-diagnostics.csv'),F,spec)]
    captions='''# 新12 kHz协议图注与读图说明

所有图仅使用新协议结果。S0、S1、S2分别为新协议draw0、draw1、draw2，均不等于历史支持配置legacy；三个配置的固定验证/测试集合相同，全部主结果统一分类器种子0/1/2。图宽160mm，字号8.5–9pt，PNG400dpi与嵌入字体PDF；无需缩小。原始记录已被探索，此处不是未见确认实验。

图3-1（figure-00-evaluation-flow-cn）：增强方案与固定分类后端的评价流程。真实支持每类K个，条件扩散和高斯扰动池各10K；三个增强组训练集均11K，混合为真实K+扩散5K+扰动5K。共享编码器提取波形与尺度–时间特征，线性分类头输出类别。CE使用全部训练样本，原型对比损失以真实支持为锚点，两者共同训练编码器；原型不作为测试分类的额外输入。CWRU验证仅选模、测试仅最终评价。预处理细节见正文，局部/全局视图属于波形分支。

配对效应图（figure-01-paired-effects-cn）：（a）Diffusion−None；（b）Mix−Jitter。彩色/形状点为各留负载折三个共同种子的平均准确率差，黑菱形为四折等权均值，虚线为零。单位百分点，无总体置信声明。先观察正负折一致性，再跨配置比较，不能仅据正均值声称稳定有效。

全类别图（figure-02-class-recall-cn）：D−N为扩散减不增强，M−J为混合减扰动；每格四折×三种子平均召回差（百分点），蓝升红降，数值保留一位小数。B/IR/OR为滚动体/内圈/外圈，Normal正常，后三位为原数据集尺寸标记。覆盖所有18行×10类；逐类诊断不作故障机理或新的显著性结论。

生成图（figure-03-generation-cn）：左列RMS比（生成/真实支持），统一对数轴；右列类平均归一化PSD的JS距离（以2为底，0~1）。每个K下40点为全四折十类，黑线为算术均值；所有池保留。RMS真实参照已逐窗标准化，不是物理振动幅值。K同时改变支持均谱样本数、池大小与训练预算，数值更相似不证明机理真实或分类收益。图中点不作40个独立重复试验解释。
'''
    (F.OUT/'captions-and-guide.md').write_text(captions,encoding='utf-8')
    (out/'figure-catalog.md').write_text(captions,encoding='utf-8')
    P.dump(F.OUT/'export-verification.json',dict(protocol='harmonized-12k-only',figures=produced,
        source_sha256={p.name:P.file_hash(p) for p in (out/'run-metrics-cost.csv',out/'class-contrasts-common3.csv',out/'generation-diagnostics.csv')}))


def generation_figure(data: pd.DataFrame, F, spec):
    """Same compact composition, with bounds derived from every new pool."""
    import matplotlib.pyplot as plt
    if len(data)!=360 or not np.isfinite(data.rms_ratio).all() or min(data.rms_ratio)<=0:
        raise ValueError('Expected all positive finite RMS diagnostics')
    low=min(.65,float(data.rms_ratio.min())*.88)
    high=max(1.2,float(data.rms_ratio.max())*1.12)
    ticks=[x*10.**e for e in range(-8,9) for x in (1,2,5) if low<=x*10.**e<=high]
    if len(ticks)>6:ticks=ticks[::2]
    fig,axes=plt.subplots(3,2,figsize=(160/25.4,5.5))
    fig.subplots_adjust(left=.12,right=.985,top=.96,bottom=.085,wspace=.32,hspace=.28)
    for i,bundle in enumerate(SCOPES):
        for j,metric in enumerate(('rms_ratio','psd_js_distance')):
            ax=axes[i,j]
            for x,k in enumerate((1,3,5)):
                part=data[(data.bundle==bundle)&(data.shot==k)].sort_values(['fold','class_id'])
                if len(part)!=40:raise ValueError('Incomplete generation diagnostic group')
                values=part[metric].to_numpy()
                ax.scatter(x+np.tile(np.linspace(-.16,.16,10),4),values,s=8,alpha=.5,color='#0072B2',linewidths=0)
                ax.plot([x-.21,x+.21],[values.mean()]*2,color='black',linewidth=1.5)
            ax.text(.025,.94,f'S{i}',transform=ax.transAxes,va='top',fontsize=9,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.8,pad=1))
            ax.set_xticks(range(3),(1,3,5));ax.set_xlim(-.45,2.45)
            if i==2:ax.set_xlabel('每类支持样本数 K')
            if metric=='rms_ratio':
                ax.set_yscale('log');ax.set_ylim(low,high)
                ax.set_yticks(ticks,[f'{v:g}' for v in ticks]);ax.minorticks_off()
                ax.set_ylabel('RMS 比（生成/支持）');ax.axhline(1,color='#888888',ls='--',lw=.7)
            else:
                ax.set_ylim(0,1);ax.set_yticks([0,.25,.5,.75,1],['0','0.25','0.5','0.75','1'])
                ax.set_ylabel('PSD 的 JS 距离')
            ax.tick_params(length=2,pad=2);ax.grid(axis='y',lw=.5,alpha=.18)
    return F.export(fig,'figure-03-generation-cn',5.5,spec)


def analyze(out: Path=OUT, classification: Path=ROOT/'classification') -> None:
    out.mkdir(parents=True,exist_ok=True)
    manifest=require_complete(classification,out)
    rows=H.read_valid_rows(classification,manifest)
    if len(rows)!=564:
        raise ValueError('Incomplete verified classification rows')
    summary,folds,contrasts,paired=summary_statistics(rows,out)
    classes,confusions,runs=prediction_details(rows,classification,out)
    class_contrasts(classes,out)
    generation,costs=generation_details(manifest,out)
    svm=svm_details(out)
    if read(out/'svm/manifest.json')['payload']['source_audit_sha256']!=P.file_hash(
            Path(manifest['payload']['paths']['data_root'])/'source_audit.json'):
        raise ValueError('SVM and neural classifiers do not share the data preparation audit')
    backbone=backbone_comparison(rows,out)
    tables_and_reports(out,manifest,summary,contrasts,classes,generation,costs,runs,svm,backbone)
    final_figures(out)
    P.dump(out/'verification.json',dict(complete=True,created_utc=P.now(),n_classifiers=564,n_generators=36,n_svm=36,
        classification_fingerprint=manifest['fingerprint'],analysis_code_sha256=P.file_hash(Path(__file__)),
        source_audit_sha256=P.file_hash(Path(manifest['payload']['paths']['data_root'])/'source_audit.json'),
        test_counts=sorted(runs.n_test.unique().tolist()),old_protocol_results_included=False))
    P.dump(out/'status.json',dict(complete=True,completed=564,finished_utc=P.now()))
    print(f'COMPLETE: {out}',flush=True)


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',choices=('svm','analyze','all'),default='all')
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--data-root',type=Path,default=DATA)
    p.add_argument('--classification-root',type=Path,default=ROOT/'classification')
    args=p.parse_args()
    if args.phase in ('svm','all'):run_svm(args.output,args.data_root)
    if args.phase in ('analyze','all'):analyze(args.output,args.classification_root)


if __name__=='__main__':main()
