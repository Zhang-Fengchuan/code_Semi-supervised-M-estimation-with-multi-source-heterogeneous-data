# Code for Semi-supervised M-estimation with Multi-source Heterogeneous Data

本文件夹是论文复现和实际数据分析的正式整理版。旧代码没有删除，也没有覆盖；本目录只放当前维护需要的核心代码和两个主入口。

## 数据说明

本仓库只发布复现代码，不发布真实步态视频、由视频提取的个体级特征文件、实际数据分析结果或任何可识别个体的数据。实际数据分析脚本需要用户在本机通过环境变量指定数据目录：

- `REALDATA_BEIJING_BASE`
- `REALDATA_DEYANG_BASE`

如果需要在私下共享真实步态视频数据或个体级特征文件，请单独使用加密压缩包，并通过安全渠道发送密码。不要把这些数据提交到 GitHub。

## 目录结构

- `核心函数/`：数据生成、模型设定、SUPERVISED、DRESS、PSS、PROPOSED、MST/MDSP 选择、方差估计、指标计算和画图辅助函数。
- `数值模拟/MstMdsp_simulation_main.py`：数值模拟唯一主入口。
- `数值模拟/模拟结果/`：重新运行后保存正文和附录模拟结果。
- `实际数据分析/MstMdsp_real_data_main.py`：实际数据分析唯一主入口。
- `实际数据分析/分析结果/`：实际数据分析输出结果。

## 数值模拟怎么运行

打开 `数值模拟/MstMdsp_simulation_main.py`，修改文件开头的变量：

- `TARGET`：要复现的表或图。
- `USE_EXISTING_RESULTS`：是否读取已有结果。
- `RUN_SIMULATION`：是否重新模拟。
- `T`：模拟次数。正式结果一般设为 `500`，调试可设为 `1` 或 `2`。
- `MODEL`：正文最终模拟使用 `linear`，逻辑回归敏感性分析可改为 `logistic`。

当前支持的 `TARGET`：

- `Table2`：Example 1，六个异质无标签源，正文主表。
- `Figure3`：Example 2，三十六个无标签源，选择频次和信息集图所需数据。
- `Table3`：Example 3，同质无标签源。
- `Table4`：Example 4，高阶异质源。
- `FigureS4`：MST 剪枝路径相关数据。
- `ALL_MAIN`：正文主要结果。
- `ALL_SUPPLEMENT`：附录图表数据。
- `ALL`：全部当前支持目标。

## 实际数据分析怎么运行

打开 `实际数据分析/MstMdsp_real_data_main.py`，并在运行前设置数据路径：

- `REALDATA_BEIJING_BASE`
- `REALDATA_DEYANG_BASE`

默认使用当前正文实际数据分析口径：原始 PC1-FPCA 特征、seed=509、原始逻辑损失、阈值 0.5。输出会保存到 `实际数据分析/分析结果/`。

## 当前正式设置

- Example 1--4 均使用统一 MST/MDSP/交集式选择流程。
- Example 4 不使用 z-band 选择器。
- 线性回归正文口径使用 `LINEAR_DGP="quad1"` 和 `single_gaussian` 协变量分布。
- 结果文件名尽量采用中文，便于直接对应论文表格和图片。
