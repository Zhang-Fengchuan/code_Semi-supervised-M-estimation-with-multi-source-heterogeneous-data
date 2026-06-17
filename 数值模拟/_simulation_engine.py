"""
统一模拟调度器
==============

这个文件是 MstMdsp_simulation_main.py 背后的内部模拟实现，统一支持线性回归和逻辑回归。
它负责：

1. 按 Example 1-4 生成场景；
2. 调用 DataGenerator 生成有标签和无标签数据；
3. 使用统一 MST/MDSP/交集式选择流程筛选 source；
4. 计算 SUPERVISED、DRESS、PSS、PROPOSED；
5. 输出汇总表、逐参数结果、选择频次、运行日志和 meta 信息。

本文件不包含 z-band 选择器。普通使用时不要直接运行本文件，请运行
MstMdsp_simulation_main.py。
"""

from __future__ import annotations

import contextlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd


class config:
    """
    内部默认配置。

    MstMdsp_simulation_main.py 会在运行前覆盖这些值。这样可以避免再额外维护
    一个 config.py，保证数值模拟只有一个清楚的主入口。
    """

    SEED = 123
    LINEAR_DGP = "quad1"
    LINEAR_X_DISTRIBUTION = "single_gaussian"
    LINEAR_INTERCEPT_FROM_SUPERVISED = True
    LINEAR_BIAS_CORRECTION = False
    LOGISTIC_SS_SOLVER = "bfgs"
    LOGISTIC_SUP_SOLVER = "bfgs"
    LOGISTIC_OPTIM_TOLERANCE = 5e-3
    LOGISTIC_OPTIM_MAX_ITER = 2000
    LOGISTIC_BETA_STAR_TOLERANCE = 1e-6
    LOGISTIC_BETA_STAR_MAX_ITER = 1000


ROOT_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = ROOT_DIR / "核心函数"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from DataGenerator import DataGenerator  # noqa: E402
from DRESSSSLogistic import DRESSSSLogistic  # noqa: E402
from ModelSpec import LinearModelSpec, LogisticModelSpec  # noqa: E402
from MstMdsp import MstMdsp  # noqa: E402
from SSLogistic import SSLogistic  # noqa: E402


def reset_seed(seed: int = config.SEED) -> None:
    """固定 Python、NumPy 和 hash 随机性，保证模拟尽量可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6f") -> str:
    """将 DataFrame 写成 Markdown 表，避免依赖 tabulate。"""
    cols = list(df.columns)

    def fmt_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            return format(float(value), floatfmt)
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = [
        "| " + " | ".join(fmt_value(row[col]) for col in cols) + " |"
        for _, row in df.iterrows()
    ]
    return "\n".join([header, sep] + body) + "\n"


def latex_table(df: pd.DataFrame, model: str, caption: str, label: str) -> str:
    """生成简洁 LaTeX 表格。"""
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{llccccccccc}",
        r"\toprule",
        r"\textsc{Method} & $p$ & $n_0$ & \textsc{SE} & \textsc{SSE} & \textsc{ARE} & \textsc{BIAS} & \textsc{MSE} & \textsc{MRR} & \textsc{CP} & \textsc{TIME} \\",
        r"\midrule",
    ]
    method_order = ["SUPERVISED", "DRESS", "PSS", "PROPOSED"]
    for block_i, method in enumerate(method_order):
        block = df[df["method"] == method].sort_values("n0")
        if block.empty:
            continue
        if block_i > 0:
            lines.append(r"\addlinespace[0.5em]")
        for row_i, (_, row) in enumerate(block.iterrows()):
            values = [
                rf"\textsc{{{method}}}" if row_i == 0 else "",
                f"{int(row['p'])}" if row_i == 0 else "",
                f"{int(row['n0'])}",
                f"{row['SE']:.3f}",
                f"{row['SSE']:.3f}",
                f"{row['ARE']:.3f}",
                f"{row['BIAS']:.3f}",
                f"{row['MSE']:.3f}",
                f"{row['MRR']:.3f}",
                f"{row['CP']:.3f}",
                f"{row['TIME']:.4f}",
            ]
            lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def selection_summary(select_fields: Iterable[Iterable[str]]) -> Dict[str, int]:
    """统计 PROPOSED 在 T 次模拟中每个 source 被选中的次数。"""
    counts: Dict[str, int] = {}
    for fields in select_fields:
        for name in fields:
            counts[str(name)] = counts.get(str(name), 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: kv[0]))


def selection_counts_dataframe(
    select_fields: Iterable[Iterable[str]],
    all_fields: Iterable[str],
    scenario: Dict[str, Any],
    model: str,
    T: int,
) -> pd.DataFrame:
    """把选择频次整理成独立表格，未被选中的 source 也保留为 0。"""
    counts = selection_summary(select_fields)
    rows = []
    for source in map(str, all_fields):
        count = int(counts.get(source, 0))
        rows.append({
            "model": model,
            "example": scenario["example"],
            "source_setup": scenario["source_setup"],
            "p": int(scenario["p"]),
            "n0": int(scenario["n0"]),
            "N": int(scenario["N"]),
            "T": int(T),
            "source": source,
            "select_count": count,
            "select_rate": count / float(T) if T else np.nan,
        })
    return pd.DataFrame(rows)


def selection_by_replicate_dataframe(
    select_fields: Iterable[Iterable[str]],
    scenario: Dict[str, Any],
    model: str,
    T: int,
) -> pd.DataFrame:
    """保存每一次模拟具体选中了哪些 source，便于以后复查选择路径。"""
    rows = []
    for rep, fields in enumerate(select_fields, start=1):
        fields = list(map(str, fields))
        rows.append({
            "model": model,
            "example": scenario["example"],
            "source_setup": scenario["source_setup"],
            "p": int(scenario["p"]),
            "n0": int(scenario["n0"]),
            "N": int(scenario["N"]),
            "T": int(T),
            "replicate": rep,
            "selected_sources": ",".join(fields),
            "num_selected_sources": len(fields),
        })
    return pd.DataFrame(rows)


def metric_row(method: str, evaluate: pd.DataFrame, p: int, n0: int) -> Dict[str, float | str | int]:
    """把逐参数指标表压缩成论文表格的一行。"""
    return {
        "method": method,
        "p": p,
        "n0": n0,
        "SE": float(np.nanmean(evaluate["SE"])),
        "SSE": float(np.nanmean(evaluate["SSE"])) if "SSE" in evaluate else np.nan,
        "ARE": float(np.nanmean(evaluate["ARE"])) if "ARE" in evaluate else 1.0,
        "BIAS": float(np.nanmean(np.abs(evaluate["Bias"]))),
        "MSE": float(np.nanmean(evaluate["MSE"])),
        "MRR": float(np.nanmean(evaluate["MRR"])) if "MRR" in evaluate else 0.0,
        "CP": float(np.nanmean(evaluate["CP"])) if "CP" in evaluate else np.nan,
        "SE_SSE_ratio": float(np.nanmean(evaluate["SE_SSE_ratio"]))
        if "SE_SSE_ratio" in evaluate else np.nan,
    }


def linear_dgp_config(name: str) -> Dict[str, Any]:
    """返回线性回归真实 DGP 的误设程度配置。"""
    presets = {
        "exact": {"misspecified": False, "quadratic": 0.0, "cubic": 0.0},
        "quad0.5": {"misspecified": True, "quadratic": 0.5, "cubic": 0.0},
        "quad1": {"misspecified": True, "quadratic": 1.0, "cubic": 0.0},
        "quad2": {"misspecified": True, "quadratic": 2.0, "cubic": 0.0},
    }
    if name not in presets:
        raise ValueError(f"未知 LINEAR_DGP={name!r}，可选 {sorted(presets)}。")
    return presets[name]


def build_model_spec(model: str):
    """根据 model 名称构造模型规范对象。"""
    if model == "linear":
        dgp = linear_dgp_config(config.LINEAR_DGP)
        return LinearModelSpec(
            misspecified=bool(dgp["misspecified"]),
            noise_std=1.0,
            dgp_intercept=1.0,
            dgp_linear_coef=1.0,
            dgp_quadratic_coef=float(dgp["quadratic"]),
            dgp_cubic_coef=float(dgp["cubic"]),
            ridge_lambda=0.0,
        )

    if model == "logistic":
        return LogisticModelSpec(
            ss_solver=config.LOGISTIC_SS_SOLVER,
            sup_solver=config.LOGISTIC_SUP_SOLVER,
            dgp_intercept=-2.0,
            dgp_linear_coef=-2.0,
            dgp_quadratic_coef=1.0,
            dgp_cubic_coef=0.0,
        )

    raise ValueError("model 必须是 linear 或 logistic。")


def build_scenarios(model: str, example: str, n0_values: List[int], N: int, p: int) -> List[Dict[str, Any]]:
    """
    生成模拟场景。

    为避免再次混乱，本统一版中 Example 1-4 对线性和逻辑使用相同 source 结构：
    exm1=6 个偏移源，exm2=36 个偏移源，exm3=同质源，
    exm4=4 个高阶矩异质源 + 2 个低阶偏移有害源。
    """
    examples = ["exm1", "exm2", "exm3", "exm4"] if example == "all" else [example]
    scenarios: List[Dict[str, Any]] = []
    for exm in examples:
        for n0 in n0_values:
            if exm == "exm1":
                scenarios.append({
                    "example": exm,
                    "source_setup": "exm1_6source_shift",
                    "which_Exm": 1,
                    "data_which_Exm": 1,
                    "p": p,
                    "n0": n0,
                    "N": N,
                    "source_h_mu": None,
                    "source_h_sigma": None,
                })
            elif exm == "exm2":
                scenarios.append({
                    "example": exm,
                    "source_setup": "exm2_36source_shift",
                    "which_Exm": 2,
                    "data_which_Exm": 2,
                    "p": p,
                    "n0": n0,
                    "N": N,
                    "source_h_mu": None,
                    "source_h_sigma": None,
                })
            elif exm == "exm3":
                scenarios.append({
                    "example": exm,
                    "source_setup": "exm3_homogeneous_1source",
                    "which_Exm": 3,
                    "data_which_Exm": 1,
                    "p": p,
                    "n0": n0,
                    "N": N,
                    "source_h_mu": [0.0],
                    "source_h_sigma": [0.0],
                })
            elif exm == "exm4":
                scenarios.append({
                    "example": exm,
                    "source_setup": "exm4_highorder4_lowshift2_6source",
                    "which_Exm": 4,
                    "data_which_Exm": 4,
                    "p": p,
                    "n0": n0,
                    "N": N,
                    "source_h_mu": None,
                    "source_h_sigma": None,
                    "higher_order_sources": ["F1", "F2", "F3", "F4", "F5", "F6"],
                })
            else:
                raise ValueError(f"未知 example={exm!r}。")
    return scenarios


def method_options(model: str) -> Dict[str, Any]:
    """返回不同模型下的固定算法口径。"""
    if model == "linear":
        return {
            "tolerance": 5e-3,
            "max_iter": 2000,
            "x_distribution": config.LINEAR_X_DISTRIBUTION,
            "intercept_from_supervised": config.LINEAR_INTERCEPT_FROM_SUPERVISED,
            "bias_correction": config.LINEAR_BIAS_CORRECTION,
            "beta_star_tolerance": None,
            "beta_star_max_iter": None,
        }

    return {
        "tolerance": config.LOGISTIC_OPTIM_TOLERANCE,
        "max_iter": config.LOGISTIC_OPTIM_MAX_ITER,
        "x_distribution": "mixture_gaussian",
        "intercept_from_supervised": False,
        "bias_correction": False,
        "beta_star_tolerance": config.LOGISTIC_BETA_STAR_TOLERANCE,
        "beta_star_max_iter": config.LOGISTIC_BETA_STAR_MAX_ITER,
    }


def run_one_scenario(model: str, scenario: Dict[str, Any], T: int, out_dir: Path) -> pd.DataFrame:
    """运行一个模型和一个场景。"""
    started_wall = time.time()
    reset_seed(config.SEED)

    opts = method_options(model)
    model_spec = build_model_spec(model)
    data_generator = DataGenerator(
        random_seed=config.SEED,
        model_spec=model_spec,
        x_distribution=opts["x_distribution"],
    )

    generation_kwargs = {
        "which_Exm": int(scenario["data_which_Exm"]),
        "sample_size_n": int(scenario["n0"]),
        "sample_size_N": int(scenario["N"]),
        "p": int(scenario["p"]),
        "true_value": None,
        "simulation_times": int(T),
        "source_h_mu": scenario["source_h_mu"],
        "source_h_sigma": scenario["source_h_sigma"],
    }
    if scenario.get("higher_order_sources") is not None:
        generation_kwargs["higher_order_sources"] = scenario["higher_order_sources"]
    if opts["beta_star_tolerance"] is not None:
        generation_kwargs["beta_star_tolerance"] = opts["beta_star_tolerance"]
    if opts["beta_star_max_iter"] is not None:
        generation_kwargs["beta_star_max_iter"] = opts["beta_star_max_iter"]

    X_labeled, Y_labeled, X_unlabeled, beta_star, _, h_mu, h_sigma = data_generator.data_generation(
        **generation_kwargs
    )

    mst_mdsp = MstMdsp(random_seed=config.SEED, model_spec=model_spec)
    selection_started = time.perf_counter()
    (
        result,
        X_labeled,
        Y_labeled,
        _,
        X_unlabeled_combine,
        X_unlabeled_select,
        select_fields,
        _,
        all_fields,
        beta_star,
    ) = mst_mdsp.MstMdsp_sample_selection(
        X_labeled=X_labeled,
        X_unlabeled=X_unlabeled,
        Y_labeled=Y_labeled,
        beta_star=beta_star,
        cv_number=None,
        start_point=None,
        end_point=None,
        multiple_constant=None,
        num_lambda_mu=None,
        num_lambda_sigma=None,
        num_lambda_1=None,
        num_lambda_2=None,
        lambda_start_mu=None,
        lambda_start_sigma=None,
        c_lambda_1_start=None,
        c_lambda_2_start=None,
        k=None,
        a=None,
        residual_principle=None,
        iter_max=None,
        direct_if=None,
        lambda_range=None,
        numFolds=None,
    )
    selection_time = time.perf_counter() - selection_started

    supervised_started = time.perf_counter()
    _, evaluate_supervised, _ = mst_mdsp.solve_logistic_regression(
        X_labeled=X_labeled,
        Y_labeled=Y_labeled,
        tolerance=opts["tolerance"],
        max_iter=opts["max_iter"],
        initial_value=None,
        beta_star=beta_star,
        CP_if=1,
        lambda_range=None,
        numFolds=None,
    )
    supervised_time = time.perf_counter() - supervised_started

    dress = DRESSSSLogistic(random_seed=config.SEED, model_spec=model_spec)
    dress_started = time.perf_counter()
    _, evaluate_dress, _ = dress.dress_ss_logistic_regression(
        X_labeled=X_labeled,
        Y_labeled=Y_labeled,
        X_unlabeled=X_unlabeled_combine,
        tolerance=opts["tolerance"],
        max_iter=opts["max_iter"],
        initial_value=None,
        beta_star=beta_star,
        Evaluate_supervised=evaluate_supervised,
        result_summary=result["result_summary"],
        proposed_if=0,
        best_lambda_hat=None,
        lambda_range=None,
        numFolds=None,
        h_mu=h_mu,
        h_sigma=h_sigma,
        intercept_from_supervised=opts["intercept_from_supervised"],
    )
    dress_time = time.perf_counter() - dress_started

    pss = SSLogistic(random_seed=config.SEED, model_spec=model_spec)
    pss_started = time.perf_counter()
    _, evaluate_pss, _ = pss.ss_logistic_regression(
        X_labeled=X_labeled,
        Y_labeled=Y_labeled,
        X_unlabeled=X_unlabeled_combine,
        tolerance=opts["tolerance"],
        max_iter=opts["max_iter"],
        initial_value=None,
        beta_star=beta_star,
        Evaluate_supervised=evaluate_supervised,
        result_summary=result["result_summary"],
        proposed_if=0,
        best_lambda_hat=None,
        lambda_range=None,
        numFolds=None,
        h_mu=h_mu,
        h_sigma=h_sigma,
        intercept_from_supervised=opts["intercept_from_supervised"],
    )
    pss_time = time.perf_counter() - pss_started

    proposed = SSLogistic(random_seed=config.SEED, model_spec=model_spec)
    proposed_started = time.perf_counter()
    _, evaluate_proposed, select_times_proposed = proposed.ss_logistic_regression(
        X_labeled=X_labeled,
        Y_labeled=Y_labeled,
        X_unlabeled=X_unlabeled_select,
        tolerance=opts["tolerance"],
        max_iter=opts["max_iter"],
        initial_value=None,
        beta_star=beta_star,
        Evaluate_supervised=evaluate_supervised,
        result_summary=result["result_summary"],
        proposed_if=1,
        best_lambda_hat=None,
        lambda_range=None,
        numFolds=None,
        h_mu=h_mu,
        h_sigma=h_sigma,
        bias_correction=opts["bias_correction"],
        intercept_from_supervised=opts["intercept_from_supervised"],
    )
    proposed_time = time.perf_counter() - proposed_started

    rows = [
        metric_row("SUPERVISED", evaluate_supervised, int(scenario["p"]), int(scenario["n0"])),
        metric_row("DRESS", evaluate_dress, int(scenario["p"]), int(scenario["n0"])),
        metric_row("PSS", evaluate_pss, int(scenario["p"]), int(scenario["n0"])),
        metric_row("PROPOSED", evaluate_proposed, int(scenario["p"]), int(scenario["n0"])),
    ]
    table = pd.DataFrame(rows)
    total_time = {
        "SUPERVISED": supervised_time / T,
        "DRESS": dress_time / T,
        "PSS": pss_time / T,
        "PROPOSED": (selection_time + proposed_time) / T,
    }
    table["TIME"] = table["method"].map(total_time)
    table.insert(0, "model", model)
    table.insert(1, "example", scenario["example"])
    table.insert(2, "source_setup", scenario["source_setup"])
    table.insert(3, "N", int(scenario["N"]))
    table.insert(4, "T", int(T))

    tag = (
        f"{model}_{scenario['example']}_{scenario['source_setup']}"
        f"_p{scenario['p']}_n{scenario['n0']}_N{scenario['N']}_T{T}"
    )
    table.to_csv(out_dir / f"{tag}_summary.csv", index=False)
    evaluate_supervised.to_csv(out_dir / f"{tag}_supervised_by_param.csv", index=False)
    evaluate_dress.to_csv(out_dir / f"{tag}_dress_by_param.csv", index=False)
    evaluate_pss.to_csv(out_dir / f"{tag}_pss_by_param.csv", index=False)
    evaluate_proposed.to_csv(out_dir / f"{tag}_proposed_by_param.csv", index=False)

    selection_counts_df = selection_counts_dataframe(
        select_fields=select_fields,
        all_fields=all_fields,
        scenario=scenario,
        model=model,
        T=T,
    )
    selection_counts_df.to_csv(out_dir / f"{tag}_selection_counts.csv", index=False)
    (out_dir / f"{tag}_selection_counts.md").write_text(
        dataframe_to_markdown(selection_counts_df, floatfmt=".6f"),
        encoding="utf-8",
    )

    selection_by_rep_df = selection_by_replicate_dataframe(
        select_fields=select_fields,
        scenario=scenario,
        model=model,
        T=T,
    )
    selection_by_rep_df.to_csv(out_dir / f"{tag}_selection_by_replicate.csv", index=False)

    selection_counts = selection_summary(select_fields)
    meta = {
        "model": model,
        "scenario": scenario,
        "algorithm": {
            "selection": "unified MST/MDSP intersection rule",
            "z_band": "not used",
            "options": opts,
        },
        "beta_star": np.asarray(beta_star).ravel().tolist(),
        "all_fields": list(map(str, all_fields)),
        "h_mu": list(map(str, h_mu)),
        "h_sigma": list(map(str, h_sigma)),
        "proposed_selection_counts": selection_counts,
        "select_times_proposed": np.asarray(select_times_proposed).tolist()
        if np.asarray(select_times_proposed).size else [],
        "timing_seconds": {
            "selection_total": selection_time,
            "supervised_total": supervised_time,
            "dress_total": dress_time,
            "pss_total": pss_time,
            "proposed_estimation_total": proposed_time,
            "elapsed_total": time.time() - started_wall,
        },
    }
    (out_dir / f"{tag}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return table


def write_combined_outputs(combined: pd.DataFrame, out_dir: Path, model: str, example: str) -> None:
    """写合并结果：csv、markdown、latex。"""
    combined.to_csv(out_dir / "summary_all.csv", index=False)
    (out_dir / "summary_all.md").write_text(
        dataframe_to_markdown(combined, floatfmt=".6f"),
        encoding="utf-8",
    )
    caption = (
        f"Estimation results under the {model} working model "
        f"for {example}. TIME is average running time per replicate in seconds."
    )
    (out_dir / "summary_all.tex").write_text(
        latex_table(combined, model=model, caption=caption, label=f"tab:{model}_{example}"),
        encoding="utf-8",
    )


def write_combined_selection_outputs(out_dir: Path) -> None:
    """把当前输出目录下所有场景的选择频次表合并，便于直接查看和画图。"""
    selection_files = sorted(out_dir.glob("*_selection_counts.csv"))
    if not selection_files:
        return
    combined = pd.concat(
        [pd.read_csv(path) for path in selection_files],
        ignore_index=True,
    )
    combined = combined.sort_values(["n0", "source"]).reset_index(drop=True)
    combined.to_csv(out_dir / "selection_counts_all.csv", index=False)
    (out_dir / "selection_counts_all.md").write_text(
        dataframe_to_markdown(combined, floatfmt=".6f"),
        encoding="utf-8",
    )


def run_simulation(
    model: str,
    example: str,
    T: int,
    n0_values: List[int],
    N: int,
    p: int,
    out_root: Path,
    quiet: bool = False,
) -> Path:
    """统一模拟入口，由根目录 run.py 调用。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = out_root / f"{timestamp}_{model}_{example}_T{T}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(model=model, example=example, n0_values=n0_values, N=N, p=p)
    run_log: List[Dict[str, Any]] = []
    tables: List[pd.DataFrame] = []

    print(f"模型：{model}")
    print(f"实验：{example}")
    print(f"模拟次数 T={T}, n0={n0_values}, N={N}, p={p}")
    print("选择流程：统一 MST/MDSP/交集式选择；不使用 z-band。")
    print(f"输出目录：{out_dir}\n")

    for idx, scenario in enumerate(scenarios, start=1):
        started = time.time()
        print(f"[{idx}/{len(scenarios)}] {scenario}")
        try:
            if quiet:
                log_path = out_dir / f"scenario_{idx:02d}.log"
                with open(log_path, "w", encoding="utf-8") as log_fh:
                    with contextlib.redirect_stdout(log_fh), contextlib.redirect_stderr(log_fh):
                        table = run_one_scenario(model=model, scenario=scenario, T=T, out_dir=out_dir)
            else:
                table = run_one_scenario(model=model, scenario=scenario, T=T, out_dir=out_dir)
            elapsed = time.time() - started
            tables.append(table)
            run_log.append({"scenario": scenario, "status": "ok", "elapsed_sec": elapsed})
            print(table[["method", "SE", "SSE", "ARE", "BIAS", "MSE", "MRR", "CP", "TIME"]].to_string(index=False))
            print(f"完成，用时 {elapsed:.1f}s\n")
        except Exception as exc:
            elapsed = time.time() - started
            run_log.append({
                "scenario": scenario,
                "status": "error",
                "elapsed_sec": elapsed,
                "error": repr(exc),
            })
            print(f"错误，用时 {elapsed:.1f}s：{exc!r}\n")

        if tables:
            write_combined_outputs(pd.concat(tables, ignore_index=True), out_dir, model=model, example=example)
            write_combined_selection_outputs(out_dir)
        (out_dir / "run_log.json").write_text(
            json.dumps(run_log, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"全部任务结束：{out_dir.resolve()}")
    return out_dir
