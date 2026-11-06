# Diffusion augmentation for few-shot bearing fault diagnosis

Code and derived benchmark results for **Stability of gains from diffusion augmentation in few-shot bearing fault diagnosis across loads**.

The study compares no augmentation, diffusion augmentation, Gaussian jitter, and mixed augmentation under matched classifier budgets. The main CWRU evaluation comprises 564 classifier runs, 36 support-bound generator pools, and 36 fixed-feature SVM runs. Three training-support draws share fixed validation and test sets. The released result tables are the recorded experiment outputs; publishing this repository did not involve new model training.

## Contents

| Path | Contents |
| --- | --- |
| `code/` | Data preparation, diffusion generation, classifier training, SVM comparison, analysis, and timing scripts |
| `results/cwru/` | Main summaries, paired contrasts, per-run metrics, class recalls, confusion counts, and generation diagnostics |
| `results/pu/pu_summary.csv` | Secondary Paderborn benchmark summary |
| `replot_benchmark_figures.py` | Recreates the English manuscript's Figures 2–5 from released CSVs |
| `sha256.json` | SHA-256 hashes for released source and result files |

## Reproduce the benchmark figures

Use Python 3.10 or later. From the repository root:

```sh
python -m pip install -r requirements_plotting.txt
python replot_benchmark_figures.py
```

The script writes vector PDFs and PNG previews to `replotted_figures/`. It does not train models or access raw data. Arial is used when available; Matplotlib can substitute another sans-serif font. The numerical values are read from the released CSV files.

## CWRU training workflow

The training scripts use Python 3.10 or later, NumPy, SciPy, pandas, scikit-learn, Matplotlib, joblib, and PyTorch. Install a CUDA-enabled PyTorch build compatible with your hardware for generation and neural classifier training. The requirements file names packages; it does not claim to reconstruct the historical environment exactly.

```sh
python -m pip install -r requirements_training.txt
cd code
python scripts/paper_harmonized_prepare.py --help
python scripts/paper_harmonized_rerun.py --help
python scripts/paper_harmonized_evidence.py --help
```

Obtain the CWRU drive-end records separately. The preparation code expects `code/data/cwru/hp0/` through `hp3/`, with the following filenames and the original MATLAB channel names retained. For a load `h` in 0–3, the required channel is `X{base_id + h:03d}_DE_time`. Normal records are sampled at 48 kHz and resampled to 12 kHz before segmentation; fault records are the 12 kHz drive-end records. Outer-race faults use the 6 o'clock position.

| Class | Local filename | Base record ID |
| --- | --- | --- |
| Normal | `0_0.mat` | 97 |
| B007 | `7_2.mat` | 118 |
| B014 | `14_2.mat` | 185 |
| B021 | `21_2.mat` | 222 |
| IR007 | `7_1.mat` | 105 |
| IR014 | `14_1.mat` | 169 |
| IR021 | `21_1.mat` | 209 |
| OR007 | `7_3.mat` | 130 |
| OR014 | `14_3.mat` | 197 |
| OR021 | `21_3.mat` | 234 |

After arranging the raw data, the full workflow is:

```sh
python scripts/paper_harmonized_prepare.py
python scripts/paper_harmonized_prepare.py --generate
python scripts/paper_harmonized_rerun.py --execute
python scripts/paper_harmonized_evidence.py --phase all
```

These commands prepare data, train generators, train classifiers, and run the SVM comparison and analysis. They can be computationally expensive. Without `--execute`, the classifier entry point validates and records the run plan but does not train. The scripts preserve hashes, seeds, support-draw identities, validation-based checkpoint selection, and completeness checks. The analysis generates the historical Chinese figures; it needs a Chinese font such as Microsoft YaHei on Windows or Noto Sans CJK SC elsewhere. The standalone plotting script above generates the English submission figures.

## Reproducibility scope

The included CSV files reproduce the benchmark visualizations directly. Full training replay additionally needs the external raw data, a compatible environment, and GPU computation. Raw datasets, trained checkpoints, generated pools, and per-example prediction archives are not redistributed. Historical Paderborn generator-support provenance is incomplete, as discussed in the manuscript; the secondary summary is supplied for inspection and plotting rather than as a claim of full PU training reproducibility.

Release preparation changed only two infrastructure details relative to the experiment scripts: the machine-specific source path in `run_a_small_eval_v2.py` now resolves to the local `code/` directory, and `paper_final_figures.py` exports through Matplotlib instead of a locally installed `pubfig` helper. Chinese-font discovery includes a cross-platform fallback. Model definitions, training logic, experiment settings, and released numerical results were not changed.

## Data availability

- CWRU Bearing Data Center: https://engineering.case.edu/bearingdatacenter
- Paderborn University Bearing Data Center: https://mb.uni-paderborn.de/en/kat/research/bearing-datacenter

Obtain raw benchmark data from their original providers and follow their access and citation conditions. The drilling-pump data are not currently available for sharing because of restrictions imposed by the applicable project management regulations. No field-data files are included in this repository.

The code is distributed under the repository's MIT license. This license does not change the terms governing externally obtained datasets.
