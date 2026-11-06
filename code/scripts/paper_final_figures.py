"""Chinese, 160 mm-wide paper figures from verified evidence tables only."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / 'results/paper_evidence_20261019'
OUT = EVIDENCE / 'final_figures'
BUNDLES = ('legacy', 'draw1', 'draw2')
SUPPORT = dict(zip(BUNDLES, ('S0', 'S1', 'S2')))
SHOTS = (1, 3, 5)
METHODS = ('none', 'diffusion', 'jitter', 'mix')
CONTRASTS = (('diffusion', 'none'), ('mix', 'jitter'))
CLASSES = ('Normal', 'B007', 'B014', 'B021', 'IR007', 'IR014', 'IR021', 'OR007', 'OR014', 'OR021')
FONT_PATH = Path('C:/Windows/Fonts/msyh.ttc')
WIDTH_MM = 160.0


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure() -> dict:
    if FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(FONT_PATH))
        family = font_manager.FontProperties(fname=FONT_PATH).get_name()
    else:
        available = {f.name for f in font_manager.fontManager.ttflist}
        family = next((n for n in ('Noto Sans CJK SC', 'SimHei', 'WenQuanYi Zen Hei') if n in available), None)
        if family is None:
            raise RuntimeError('Install a Chinese font such as Noto Sans CJK SC for the historical Chinese figures; use replot_benchmark_figures.py for the English manuscript figures.')
    plt.rcParams.update({'font.family': family, 'font.size': 9, 'axes.labelsize': 9,
        'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'legend.fontsize': 8.5,
        'axes.titlesize': 9, 'axes.linewidth': .65, 'lines.linewidth': 1,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'axes.unicode_minus': True})
    return dict(name='paper-160mm-chinese', font_family=family, default_raster_dpi=400)


def export(fig, name: str, height_inches: float, spec: dict) -> dict:
    fig.canvas.draw()
    visible = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
    sizes = [t.get_fontsize() for t in visible]
    if min(sizes) < 8.5 or max(sizes) > 10:
        raise ValueError(f'Unreadable fontsize in {name}: {min(sizes)}, {max(sizes)}')
    for extension in ('png', 'pdf'):
        OUT.mkdir(parents=True, exist_ok=True)
        fig.set_size_inches(WIDTH_MM / 25.4, height_inches)
        fig.savefig(OUT / f'{name}.{extension}', dpi=400)
    metadata = dict(filename=name, width_mm=WIDTH_MM, height_mm=height_inches * 25.4,
                    minimum_font_pt=min(sizes), maximum_font_pt=max(sizes),
                    dpi=400, fonts=sorted({str(t.get_fontfamily()) for t in visible}),
                    files={ext: sha(OUT / f'{name}.{ext}') for ext in ('png', 'pdf')})
    plt.close(fig)
    return metadata


def paired_figure(runs: pd.DataFrame, spec: dict) -> dict:
    part = runs[(runs.backbone == 'dual') & (runs.seed < 3) & runs.method.isin(METHODS)]
    if len(part) != 432:
        raise ValueError('Expected complete three-support, three-seed main comparison')
    group = part.groupby(['bundle', 'shot', 'method', 'fold']).accuracy_pct.mean()
    rows, means = [], []
    for bundle in BUNDLES:
        for shot in SHOTS:
            for method, baseline in CONTRASTS:
                changes = []
                for fold in range(4):
                    effect = float(group.loc[(bundle, shot, method, fold)] - group.loc[(bundle, shot, baseline, fold)])
                    rows.append(dict(bundle=bundle, support=SUPPORT[bundle], shot=shot,
                        contrast=f'{method}-{baseline}', fold=fold, accuracy_difference_pp=effect))
                    changes.append(effect)
                means.append(dict(bundle=bundle, support=SUPPORT[bundle], shot=shot,
                    contrast=f'{method}-{baseline}', accuracy_difference_pp=float(np.mean(changes))))
    folds = pd.DataFrame(rows)
    avg = pd.DataFrame(means)
    folds.to_csv(OUT / 'paired-fold-values.csv', index=False, encoding='utf-8-sig')
    avg.to_csv(OUT / 'paired-mean-values.csv', index=False, encoding='utf-8-sig')
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_MM / 25.4, 4.2), sharey=True)
    fig.subplots_adjust(left=.15, right=.98, bottom=.25, top=.92, wspace=.17)
    colors = ('#0072B2', '#D55E00', '#009E73', '#CC79A7')
    markers = ('o', 's', '^', 'v')
    keys = [(b, k) for b in BUNDLES for k in SHOTS]
    limit = np.ceil(folds.accuracy_difference_pp.abs().max() / 5) * 5
    for panel, (ax, (method, baseline)) in enumerate(zip(axes, CONTRASTS)):
        contrast = f'{method}-{baseline}'
        for y, (bundle, shot) in enumerate(keys):
            records = folds[(folds.bundle == bundle) & (folds.shot == shot) & (folds.contrast == contrast)]
            for offset, fold in zip((-.21, -.07, .07, .21), range(4)):
                x = float(records[records.fold == fold].accuracy_difference_pp.iloc[0])
                ax.scatter(x, y + offset, color=colors[fold], marker=markers[fold], s=17, linewidths=.4, zorder=3)
            ax.scatter(records.accuracy_difference_pp.mean(), y, marker='D', color='black',
                       edgecolors='white', linewidths=.4, s=25, zorder=4)
        ax.axvline(0, color='#666666', linewidth=.8, linestyle='--')
        for boundary in (2.5, 5.5):
            ax.axhline(boundary, color='#bbbbbb', linewidth=.6)
        ax.set_xlim(-limit - 2, limit + 2)
        ax.set_xticks(np.arange(-limit, limit + .1, 10))
        ax.set_ylim(8.6, -.6)
        ax.set_yticks(range(9), [f'{SUPPORT[b]}  K={k}' for b, k in keys])
        ax.set_xlabel('准确率差（百分点）')
        ax.set_title(f'（{"a" if panel == 0 else "b"}）' + ('Diffusion − None' if panel == 0 else 'Mix − Jitter'))
        ax.grid(axis='x', alpha=.18, linewidth=.5)
        ax.tick_params(length=2)
    handles = [Line2D([], [], color=c, marker=m, linestyle='none', markersize=4,
                     label=f'留出负载 {i}') for i, (c, m) in enumerate(zip(colors, markers))]
    handles.append(Line2D([], [], color='black', marker='D', linestyle='none', markersize=4, label='四折均值'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.53, .025), ncol=3,
               frameon=False, handletextpad=.4, columnspacing=1.1, labelspacing=.4)
    return export(fig, 'figure-01-paired-effects-cn', 4.2, spec)


def class_figure(contrasts: pd.DataFrame, spec: dict) -> dict:
    matrices, labels = [], []
    for bundle in BUNDLES:
        for shot in SHOTS:
            for comparison, code in (('diffusion-none', 'D−N'), ('mix-jitter', 'M−J')):
                group = contrasts[(contrasts.bundle == bundle) & (contrasts.shot == shot)
                    & (contrasts.contrast == comparison)].sort_values('class_id')
                if tuple(group.class_name) != CLASSES or set(group.n_seeds) != {3}:
                    raise ValueError('Class plot mapping or seeds changed')
                matrices.append(group.recall_difference_pp.to_numpy())
                labels.append(f'{SUPPORT[bundle]}  K={shot}  {code}')
    data = np.asarray(matrices)
    pd.DataFrame(data, columns=CLASSES, index=labels).to_csv(
        OUT / 'class-recall-heatmap.csv', encoding='utf-8-sig', index_label='configuration_contrast')
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 5.9))
    ax = fig.add_axes((.205, .21, .765, .755))
    limit = float(np.ceil(abs(data).max() / 5) * 5)
    im = ax.imshow(data, cmap='RdBu', vmin=-limit, vmax=limit, aspect='auto', interpolation='none')
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            value = data[y, x]
            ax.text(x, y, f'{value:.1f}', ha='center', va='center', fontsize=8.5,
                    color='white' if abs(value) > .58 * limit else 'black')
    ax.set_xticks(range(10), CLASSES, rotation=45, ha='right')
    ax.set_yticks(range(18), labels)
    ax.tick_params(length=0, pad=4)
    for boundary in np.arange(1.5, 17.5, 2):
        ax.axhline(boundary, color='white', lw=.7)
    for boundary in (5.5, 11.5):
        ax.axhline(boundary, color='#555555', lw=.8)
    cbax = fig.add_axes((.30, .075, .58, .02))
    cb = fig.colorbar(im, cax=cbax, orientation='horizontal', ticks=(-40, -20, 0, 20, 40))
    cb.set_label('逐类召回率差（百分点）', fontsize=9, labelpad=3)
    cb.ax.tick_params(labelsize=8.5, length=2, pad=2)
    return export(fig, 'figure-02-class-recall-cn', 5.9, spec)


def generation_figure(data: pd.DataFrame, spec: dict) -> dict:
    if len(data) != 360:
        raise ValueError('Expected all 360 class-fold pool diagnostics')
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH_MM / 25.4, 5.5))
    fig.subplots_adjust(left=.12, right=.985, top=.96, bottom=.085, wspace=.32, hspace=.28)
    for i, bundle in enumerate(BUNDLES):
        for j, metric in enumerate(('rms_ratio', 'psd_js_distance')):
            ax = axes[i, j]
            for x, shot in enumerate(SHOTS):
                part = data[(data.bundle == bundle) & (data.shot == shot)].sort_values(['fold', 'class_id'])
                if len(part) != 40:
                    raise ValueError('Missing diagnostic class/fold')
                values = part[metric].to_numpy()
                # Fixed ordering only for visual separation; not another random sample.
                offset = np.tile(np.linspace(-.16, .16, 10), 4)
                ax.scatter(x + offset, values, s=8, alpha=.5, color='#0072B2', linewidths=0, zorder=2)
                ax.plot([x-.21, x+.21], [values.mean()] * 2, color='black', linewidth=1.5, zorder=3)
            ax.text(.025, .94, SUPPORT[bundle], transform=ax.transAxes, va='top', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1))
            ax.set_xticks(range(3), SHOTS)
            if i == 2:
                ax.set_xlabel('每类支持样本数 K')
            ax.set_xlim(-.45, 2.45)
            if metric == 'rms_ratio':
                ax.set_yscale('log')
                ax.set_ylim(.65, 6.6)
                ax.set_yticks([.7, 1, 2, 4, 6], ['0.7', '1', '2', '4', '6'])
                ax.minorticks_off()
                ax.set_ylabel('RMS 比（生成/支持）')
                ax.axhline(1, color='#888888', linestyle='--', linewidth=.7)
            else:
                ax.set_ylim(0, 1)
                ax.set_yticks([0, .25, .5, .75, 1], ['0', '0.25', '0.5', '0.75', '1'])
                ax.set_ylabel('PSD 的 JS 距离')
            ax.tick_params(length=2, pad=2)
            ax.grid(axis='y', linewidth=.5, alpha=.18)
    return export(fig, 'figure-03-generation-cn', 5.5, spec)


def flow_figure(spec: dict) -> dict:
    fig = plt.figure(figsize=(WIDTH_MM / 25.4, 4.6))
    ax = fig.add_axes((.01, .015, .98, .97))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    def box(x, y, w, h, label, color='#F3F6F8'):
        patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.007,rounding_size=0.01',
                              linewidth=.8, edgecolor='#354F60', facecolor=color)
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, label, ha='center', va='center', fontsize=9, linespacing=1.45)

    def arrow(start, end, *, dashed=False, connection='arc3,rad=0'):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=9,
                     linewidth=.8, color='#354F60', linestyle='--' if dashed else '-',
                     connectionstyle=connection))

    box(.02, .66, .19, .20, '真实支持集 S\n每类 K 个')
    box(.285, .795, .28, .12, '条件扩散池\n每类 10K 个')
    box(.285, .605, .28, .12, '高斯扰动池\n每类 10K 个')
    box(.66, .60, .31, .315,
        '训练集组成（每类）\nN：K 真实\nD：K 真实 + 10K 扩散\nJ：K 真实 + 10K 扰动\nM：K 真实 + 5K 扩散\n      + 5K 扰动', '#EFF5F1')
    arrow((.217, .80), (.278, .85))
    arrow((.217, .71), (.278, .665))
    arrow((.573, .85), (.652, .85))
    arrow((.573, .665), (.652, .665))
    # Real support bypasses both synthetic pools and is retained in every arm.
    ax.plot([.115, .115, .815], [.868, .963, .963], color='#354F60', lw=.8)
    arrow((.815, .963), (.815, .923))
    ax.text(.445, .979, '真实支持直接保留', ha='center', va='bottom', fontsize=9)

    box(.02, .29, .20, .15, '输入预处理\n波形双视图\n尺度–时间图')
    box(.30, .29, .30, .15, '共享双分支编码器\n波形 / 尺度–时间特征')
    box(.68, .29, .29, .15, '线性分类头\n最终类别预测')
    ax.plot([.815, .815, .12], [.593, .515, .515], color='#354F60', lw=.8)
    arrow((.12, .515), (.12, .448))
    ax.text(.47, .54, '按 N / D / J / M 分别训练', ha='center', va='bottom', fontsize=9)
    arrow((.228, .365), (.292, .365))
    arrow((.608, .365), (.672, .365))

    box(.30, .065, .67, .13, 'CE + λ Lpcl：作用于全部训练样本\n原型仅由真实支持初始化和直接更新', '#FAF4E8')
    arrow((.45, .282), (.45, .203))
    arrow((.825, .282), (.825, .203))
    ax.plot([.293, .258, .258], [.13, .13, .365], color='#354F60', lw=.8, linestyle='--')
    arrow((.258, .365), (.292, .365), dashed=True)
    ax.text(.248, .22, '反向更新', fontsize=8.5, rotation=90, ha='right', va='center')
    ax.text(.5, .015, '验证集仅用于模型选择；测试集仅用于最终评价', ha='center', va='bottom', fontsize=9)
    return export(fig, 'figure-00-evaluation-flow-cn', 4.6, spec)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = configure()
    sources = {name: EVIDENCE / name for name in ('run-metrics-cost.csv',
               'class-contrasts-common3.csv', 'generation-diagnostics.csv')}
    figures = [flow_figure(spec), paired_figure(pd.read_csv(sources['run-metrics-cost.csv']), spec),
               class_figure(pd.read_csv(sources['class-contrasts-common3.csv']), spec),
               generation_figure(pd.read_csv(sources['generation-diagnostics.csv']), spec)]
    (OUT / 'captions-and-guide.md').write_text('''# 正文紧凑图：图注与读图说明

共同定义：S0 为既有支持配置（legacy），S1、S2 为补充支持配置（draw1、draw2）；S0 不称为独立新增抽样。所有分类差值均使用三个配置共同的分类器种子 0、1、2。原主结果的 S0 十种子统计另在主表报告，不能混用差值。三配置使用固定验证和测试集合，支持、生成训练及采样种子共同改变，不新增因果或显著性推断。

## 图 3-1：评价流程（文件 figure-00-evaluation-flow-cn）

建议图注：增强方案与固定分类后端的评价流程。每类 K 个真实支持均保留在训练集中，并分别用于训练条件扩散生成器和构造高斯扰动池。N、D、J、M 分别表示不增强、扩散增强、扰动增强与混合增强；三个增强组均为每类 11K 个训练样本，其中混合组由 K 个真实样本、5K 个扩散样本和5K个扰动样本组成。共享编码器提取波形与尺度–时间特征，线性分类头输出预测；交叉熵使用全部训练样本，原型对比损失仅以真实支持样本作为锚点，两项损失共同更新编码器。原型不作为测试分类的额外输入。CWRU 验证集仅选择模型，测试集仅作最终评价；现场无验证集时采用预定最后一步，不使用测试选模。

读图要点：真实支持的直通路径与增强路径同时存在；图中原型损失位于训练目标分支，不能理解为测试时检索原型或额外输入。预处理、编码器、分类头和优化规则在同一数据集的四个方法间保持一致。时频分支实际由原输入构造尺度–时间图，归一化双视图用于波形分支，详细操作以方法正文为准。

## 图 1：配对效应

建议图注：不同支持配置和样本数下的增强配对差值。（a）扩散增强相对不增强；（b）混合增强相对高斯扰动。S0 为既有支持配置，S1、S2 为补充配置。彩色符号表示各留负载折内三个共同分类器种子的平均准确率差，黑色菱形为四折等权均值，虚线表示零增益。正值表示前一方法更高。各点为描述性折结果，不表示独立总体重复或置信区间。

读图顺序：先看黑色均值的方向，再看四折是否一致，最后比较同 K 跨支持配置的变化。不能仅据正均值称稳定有效，也不能由几个 K 值定出确定的失效阈值。图以共同三种子为准，S0 K=1 的 mix−jitter 均值与十种子主表不同。

## 图 2：全类别差值

建议图注：增强带来的逐类召回率变化。D−N 表示扩散增强减不增强，M−J 表示混合增强减高斯扰动。每格为四个固定留负载折、三个共同分类器种子的等权平均差，数值单位为百分点；蓝色为提升，红色为下降。S0 为既有支持配置，S1、S2 为补充配置。Normal 表示正常；B、IR、OR 分别表示滚动体、内圈和外圈故障，后三位沿用数据集故障尺寸标记。该图覆盖全部类别与配置，不仅展示成功案例。

读图顺序：纵向检查同类别是否跨配置保持方向，横向辨别总准确率是否掩盖特定类别的退化。关注 B021 的改善同时保留 B007、B014、IR014 等的下降。数值属于后验描述，不是故障机理解释或新的显著性检验。

## 图 3：生成信号诊断

建议图注：完整生成池相对对应真实支持的幅值与频谱诊断。左列为生成/真实支持 RMS 比，采用三个配置共同的对数纵轴；右列为逐窗归一化功率谱类别均值间的 Jensen–Shannon 距离（以 2 为底，0–1）。每个 K 下的 40 个点覆盖四折十类，黑横线为算术均值。真实支持已逐窗标准化，虚线 RMS 比=1 仅表示输入幅值尺度一致。全部生成池保留，未依据质量指标筛选。S0 为既有支持配置，S1、S2 为补充配置。

读图顺序：先检查 RMS 极值，再比较频谱差异。完整保留 S2、K=1 的 RMS 比约 5.88。K 变化也改变支持参照均谱的窗口数、生成池大小和训练预算；JS 距离降低不能单独解释为生成保真度提高，更不保证分类增益。40 个类×折点存在依赖，不作为40个独立重复实验。

## 排版与导出

- 四图固定宽 160 mm（约6.30英寸），PNG 400 dpi，配套嵌入字体 PDF；Word 插入宽16 cm、不裁剪、不额外缩小。
- 微软雅黑，图内全部可见文字8.5–9 pt，无装饰性总标题。配对图高106.68 mm，热图高149.86 mm，生成图高139.70 mm。
- 彩色折点同时用形状区分；热图每格有精确数值，黑白打印仍可读。RMS纵轴为对数，刻度均为原始比值。
- 工具：Matplotlib 导出。绘图脚本只读已核验表格，无训练或新统计推断。
''', encoding='utf-8')
    record = dict(complete=True, plotting_library='matplotlib', source_sha256={n: sha(p) for n, p in sources.items()},
        script_sha256=sha(Path(__file__)), font=str(FONT_PATH), font_sha256=sha(FONT_PATH),
        figures=figures, supports=SUPPORT, common_classifier_seeds=[0, 1, 2],
        repeated_unit='Descriptive held-out-load fold, with shared test sets and overlapping training loads')
    (OUT / 'export-verification.json').write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
