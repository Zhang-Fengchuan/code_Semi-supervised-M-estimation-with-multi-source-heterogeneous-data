# Semi-supervised M-estimation with Multi-source Heterogeneous Data

This repository provides the reproduction code for the paper on semi-supervised
M-estimation with multi-source heterogeneous unlabeled data.  It contains the
main implementation of the proposed MST/MDSP selection method, baseline
semi-supervised estimators, simulation scripts, and the real-data analysis
pipeline used in the manuscript.

The repository is organized as a lightweight research code package.  The code
is intended for readers who want to reproduce the numerical studies, inspect
the implementation, or adapt the method to related semi-supervised problems.

## Repository Contents

```text
.
├── 核心函数/
│   ├── DataGenerator.py
│   ├── ModelSpec.py
│   ├── MstMdsp.py
│   ├── SSLogistic.py
│   ├── DRESSSSLogistic.py
│   └── SelectionViz.py
├── 数值模拟/
│   ├── MstMdsp_simulation_main.py
│   ├── _simulation_engine.py
│   └── 模拟结果/
└── 实际数据分析/
    ├── MstMdsp_real_data_main.py
    ├── FPCA.py
    ├── Fpca_compress.py
    ├── 视频姿态关键点提取.py
    └── 分析结果/
```

- `核心函数/` contains the core implementation of the data generators, working
  models, supervised estimator, DRESS, PSS, the proposed MST/MDSP selector,
  variance estimation, evaluation metrics, and plotting utilities.
- `数值模拟/MstMdsp_simulation_main.py` is the main entry point for reproducing
  the simulation tables and figures.
- `实际数据分析/MstMdsp_real_data_main.py` is the main entry point for the
  gait-video real-data analysis.
- `数值模拟/模拟结果/` stores reproducible simulation outputs that can be read by
  the plotting and table-generation routines.
- `实际数据分析/分析结果/加密表格/` stores encrypted real-data summary tables.

## Requirements

The code is written in Python and uses standard scientific-computing packages.
A typical environment needs:

```text
python >= 3.9
numpy
pandas
scipy
scikit-learn
statsmodels
matplotlib
```

For video pose extraction in the real-data pipeline, additional packages may be
needed, depending on the local video-processing setup.

## Quick Start

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Zhang-Fengchuan/Semi-supervised-M-estimation-with-multi-source-heterogeneous-data.git
cd Semi-supervised-M-estimation-with-multi-source-heterogeneous-data
python -m pip install numpy pandas scipy scikit-learn statsmodels matplotlib
```

To check that the source files can be imported and compiled:

```bash
python -m compileall .
```

## Reproducing the Simulation Results

Open `数值模拟/MstMdsp_simulation_main.py` and set the control variables at the
top of the file.  The most important variables are:

```python
TARGET = "Table2"
USE_EXISTING_RESULTS = True
RUN_SIMULATION = False
T = 500
MODEL = "linear"
```

Available targets include:

| Target | Description |
| --- | --- |
| `Table2` | Example 1, six heterogeneous unlabeled sources |
| `Figure3` | Example 2, thirty-six unlabeled sources |
| `Table3` | Example 3, homogeneous unlabeled source |
| `Table4` | Example 4, higher-order heterogeneous sources |
| `FigureS4` | MST pruning path |
| `ALL_MAIN` | Main-text simulation outputs |
| `ALL_SUPPLEMENT` | Supplementary simulation outputs |
| `ALL` | All supported simulation outputs |

If `USE_EXISTING_RESULTS = True`, the script reads saved outputs under
`数值模拟/模拟结果/` and regenerates the required tables or figures.  If
`RUN_SIMULATION = True`, the script reruns the simulations and writes new
outputs to the same result directory.

## Real-data Analysis

The real-data analysis is implemented in:

```text
实际数据分析/MstMdsp_real_data_main.py
```

The public repository does not include raw gait videos or individual-level
feature files.  To rerun the real-data analysis locally, set the following
environment variables to the corresponding local data directories:

```bash
export REALDATA_BEIJING_BASE="/path/to/beijing/data"
export REALDATA_DEYANG_BASE="/path/to/deyang/data"
```

Then run:

```bash
python 实际数据分析/MstMdsp_real_data_main.py
```

## Data Availability and Encryption

Simulation result tables in `数值模拟/模拟结果/` are included because they do not
contain individual-level information.

Real gait-video data are privacy-sensitive.  Raw videos and individual-level
feature tables are not stored in this repository.  The repository only provides
an encrypted archive of the real-data summary tables:

```text
实际数据分析/分析结果/加密表格/real_data_summary_tables.tar.gz.enc
```

The archive was encrypted with AES-256-CBC using OpenSSL.  To decrypt it, run:

```bash
openssl enc -d -aes-256-cbc -pbkdf2 -iter 100000 \
  -in 实际数据分析/分析结果/加密表格/real_data_summary_tables.tar.gz.enc \
  -out real_data_summary_tables.tar.gz
tar -xzf real_data_summary_tables.tar.gz
```

The decryption password is not included in the repository.  It should be shared
separately through a secure channel when access to the protected tables is
approved.

## Notes on the Current Manuscript Version

- The main simulation setting uses the linear regression working model.
- Examples 1--4 use the same MST/MDSP intersection-type selection rule.
- Example 4 does not use a separate z-band selector.
- The real-data analysis script keeps the final manuscript setting based on the
  selected gait-feature extraction and logistic working model.

## Citation

If you use this code, please cite the corresponding manuscript:

```bibtex
@article{semi_supervised_m_estimation_multisource,
  title   = {Semi-supervised M-estimation with Multi-source Heterogeneous Data},
  author  = {Zhang, Fengchuan and coauthors},
  year    = {2026},
  note    = {Manuscript}
}
```

