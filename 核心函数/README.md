# Core Modules

This folder contains the reusable implementation used by the simulation and
real-data scripts.  The files are kept separate from the top-level entry points
so that the estimators, data generators, and plotting utilities can be reused in
other projects.

## File Guide

| File | Purpose |
| --- | --- |
| `DataGenerator.py` | Generates labeled and unlabeled domains for the simulation examples. |
| `ModelSpec.py` | Defines the working models, losses, gradients, Hessians, and design matrices. |
| `SSLogistic.py` | Implements supervised and projection-based semi-supervised estimation tools. |
| `DRESSSSLogistic.py` | Implements the density-ratio-estimation-based semi-supervised estimator. |
| `MstMdsp.py` | Implements the proposed MST/MDSP source selection and final estimator. |
| `SelectionViz.py` | Provides helper functions for selection-frequency and MST-pruning visualizations. |
| `__init__.py` | Marks this folder as an importable Python package. |

## Main Methods

The implemented estimators are:

- `SUPERVISED`: uses only the labeled target-domain sample.
- `DRESS`: uses density-ratio weighting under a homogeneous-domain assumption.
- `PSS`: uses projection-based semi-supervised information under a homogeneous-domain assumption.
- `PROPOSED`: selects informative unlabeled sources through the MST/MDSP procedure and then constructs the semi-supervised estimator.

## Usage

The recommended way to use these modules is through the two public entry
points:

```text
数值模拟/MstMdsp_simulation_main.py
实际数据分析/MstMdsp_real_data_main.py
```

Direct imports are also possible.  For example:

```python
from 核心函数.DataGenerator import DataGenerator
from 核心函数.ModelSpec import ModelSpec
from 核心函数.MstMdsp import MstMdsp
```

When adapting the code, keep the following roles separate:

1. `DataGenerator.py` controls data-generating mechanisms.
2. `ModelSpec.py` controls the working model and estimating equations.
3. `MstMdsp.py` controls source selection and the proposed estimator.
4. `SelectionViz.py` controls visualization only.

