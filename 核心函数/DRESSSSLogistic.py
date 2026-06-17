"""
DRESSSSLogistic.py
------------------
半监督 M-估计的 c1=0 加权版本。

本模块和 SSLogistic.py 共用 ModelSpec.ss_loss_and_grad。二者的实际差别只在于
传入 use_dress_c1 的取值：
    - DRESSSSLogistic: use_dress_c1=True，ModelSpec 内部令 c1=0
    - SSLogistic    : use_dress_c1=False，ModelSpec 内部令 c1=n/(n+N)

因此，本文件中的 “DRESS” 是沿用原项目/旧 MATLAB 代码中的命名。当前代码并没有显式估计
密度比模型；它使用辅助矩阵 Z 的无标签均值和有标签二阶矩阵来构造有标签样本权重。

主要类：
    DRESSSSLogistic —— c1=0 半监督 M-估计流程类，支持通过 model_spec 插入任意 M-估计模型

依赖：
    ModelSpec.py 中的 BaseModelSpec、LogisticModelSpec
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.linalg import sqrtm, det
from scipy.stats import loguniform

from ModelSpec import BaseModelSpec, LogisticModelSpec, stable_sandwich, stable_solve


class DRESSSSLogistic:
    """
    半监督 M-估计的 c1=0 权重版本。

    本类负责在每次模拟中：
      1. 选择或读取辅助特征阶数 alpha；
      2. 构造有标签/无标签辅助矩阵 Z；
      3. 调用 model_spec.ss_loss_and_grad(..., use_dress_c1=True) 优化参数；
      4. 计算 Bias、SE、MSE、ARE、RR、SSE、CP 等模拟评估量；
      5. 根据 result_summary 统计 source 选择频次。

    注意：类名保留 DRESSSSLogistic 以兼容旧代码，但当前实现不是独立的密度比估计器。
    它只是把 ModelSpec.ss_loss_and_grad 中的 c1 设为 0。

    通过 model_spec 参数，本类可以支持任意符合 BaseModelSpec 接口的 M-估计模型
    （如逻辑回归、线性回归等），实现了算法与模型的解耦。

    核心功能：
    1. c1=0 半监督参数估计（dress_ss_logistic_regression）
    2. 基于 GBIC 的多项式阶数（alpha）选择
    3. 模型评估（Bias/SE/MSE/CP/ARE/RR 等指标）
    4. 未标记数据来源选择频率统计

    属性：
        default_tolerance (float): 默认优化收敛容忍度，5e-3
        default_max_iter (int): 默认最大迭代次数，500
        default_num_folds (int): 默认交叉验证折数，5
        default_lambda_range (np.ndarray): 默认 lambda 搜索范围（对数均匀分布 100 个点）
        random_seed (int): 随机数种子，保证结果可复现
        model_spec (BaseModelSpec): M-估计模型规范对象，默认为 LogisticModelSpec()
    """

    def __init__(self, random_seed=123, model_spec=None):
        """
        初始化 DRESSSSLogistic 类，设置默认优化参数和模型规范。

        参数：
            random_seed (int): 随机数种子，默认 123，用于保证模拟结果可复现
            model_spec (BaseModelSpec, optional): M-估计模型规范对象。
                若为 None，则使用默认的逻辑回归规范（LogisticModelSpec）。
        """
        # 优化器默认参数
        self.default_tolerance = 5e-3       # 默认数值收敛判据
        self.default_max_iter = 500         # 最大迭代次数
        self.default_num_folds = 5          # 交叉验证折数
        # 对应 MATLAB 中的 logspace(-10, 2, 100)，使用对数均匀采样生成 lambda 候选值
        self.default_lambda_range = loguniform.rvs(1e-10, 1e2, size=100)

        self.random_seed = random_seed
        np.random.seed(random_seed)   # 设置全局随机种子，确保每次运行结果一致
        # 模型规范：默认使用逻辑回归；如传入自定义 model_spec，则使用传入值。
        # DRESS 与 PSS/PROPOSED 共用同一个 solve_semi_supervised 实现，
        # 唯一区别是这里固定 use_dress_c1=True，即 c1=0。
        self.model_spec = model_spec if model_spec is not None else LogisticModelSpec()

    # ========================= 核心方法：c1=0 半监督 M-估计 =========================
    def dress_ss_logistic_regression(self, X_labeled, Y_labeled, X_unlabeled,
                                     tolerance=None, max_iter=None, initial_value=None, beta_star=None,
                                     Evaluate_supervised=None, result_summary=None, proposed_if=None,
                                     best_lambda_hat=None, lambda_range=None, numFolds=None, h_mu=None, h_sigma=None,
                                     intercept_from_supervised=False):
        """
        核心方法：执行 c1=0 半监督 M-估计并汇总模拟评估指标。

        本方法沿用旧 MATLAB 函数名中的 DRESS 命名。按当前 Python 实现，它与
        SSLogistic.ss_logistic_regression 的主要区别是调用目标函数时传入
        `use_dress_c1=True`，即在 ModelSpec.ss_loss_and_grad 中令 c1=0。
        代码没有显式拟合密度比模型。

        流程：
          1. 参数默认值处理
          2. 逐次蒙特卡洛模拟，对每次模拟：
             a. 若无未标记数据 → 仅用标记数据优化（监督估计）
             b. 若有未标记数据 → GBIC 选 alpha → 构建多项式扩展特征 Z → c1=0 半监督优化
          3. 计算 Bias/SE/MSE/ARE/RR 等统计评估指标
          4. 若 proposed_if=1，调用 _calculate_sse_cp_proposed 计算精确渐近标准误和 CP
             若 proposed_if=0，调用 _calculate_sse_cp_benchmark 计算简化版渐近标准误和 CP
          5. 统计各未标记数据源的选取频次

        参数：
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p, T)，
                n=样本数，p=特征数，T=模拟次数
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1, T)
            X_unlabeled (list): 未标记数据列表，长度=T，
                每个元素为对应模拟次数的未标记特征矩阵（shape=(N_t, p)）；
                若某次模拟无未标记数据，对应元素为空数组
            tolerance (float, optional): 优化收敛容忍度，默认 5e-3
            max_iter (int, optional): 最大迭代次数，默认 500
            initial_value (np.ndarray, optional): 参数初始值，shape=(p+1, 1)，默认全零
            beta_star (np.ndarray, optional): 真实参数向量，shape=(p+1, 1)，
                默认为 [1; ones(p, 1)]
            Evaluate_supervised (pd.DataFrame, optional): 有监督估计的评估结果，
                含 'SE'、'MSE' 列，用于计算 ARE 和 RR
            result_summary (list, optional): 每次模拟的结果摘要列表，
                含 'select_alpha'、'select_index'、'Tree_lambda_mu_one_simulation' 等键
            proposed_if (int, optional): 是否使用提出的改进方法（1=是，0=否），默认 1
            best_lambda_hat (np.ndarray, optional): 各模拟的最优 L2 正则化参数，
                shape=(1, T)，默认全零
            lambda_range (np.ndarray, optional): lambda 搜索范围，默认对数均匀分布
            numFolds (int, optional): GBIC 交叉验证折数，默认 5
            h_mu (list, optional): 均值偏移参数列表（用于统计选取频次），默认空列表
            h_sigma (list, optional): 协方差偏移参数列表（用于统计选取频次），默认空列表
            intercept_from_supervised (bool, optional): True 时将截距替换为同次模拟的
                监督估计，斜率仍使用 DRESS 半监督估计。

        返回：
            beta_hat (np.ndarray): 估计的参数矩阵，shape=(p+1, T)
            Evaluate (pd.DataFrame): 模型评估指标表，含以下列：
                Bias, BIAS_MEAN, SE, SE_MEAN, SSE, SSE_MEAN, ARE, ARE_MEAN,
                MSE, MSE_MEAN, RR, MRR, SE_SSE_ratio, SE_SSE_ratio_mean, CP, CP_MEAN
            select_time (np.ndarray): 未标记数据选择次数统计矩阵，
                shape=(len(h_mu), len(h_sigma))；若 h_mu 或 h_sigma 为空则返回空数组
        """
        # ===================== 1. 参数默认值设置（对标 MATLAB isempty 判断） =====================
        tolerance = self.default_tolerance if tolerance is None else tolerance
        max_iter = self.default_max_iter if max_iter is None else max_iter
        n_features = X_labeled.shape[1]      # 特征维度 p
        n_simulations = X_labeled.shape[2]   # 模拟次数 T

        if initial_value is None:
            # 参数初始值：截距 + p 个特征系数，全部初始化为 0
            initial_value = np.zeros((n_features + 1, 1))  # +1 为截距项
        if beta_star is None:
            # 默认真实参数：截距为 1，所有特征系数为 1
            beta_star = np.vstack([1, np.ones((n_features, 1))])
        if Evaluate_supervised is None:
            # 若无监督基准，初始化为零（ARE/RR 计算时分母为零，需注意）
            Evaluate_supervised = np.zeros((n_features + 1, 1))
        proposed_if = 1 if proposed_if is None else proposed_if
        # 各模拟的最优正则化参数，默认全零（即无正则化）
        best_lambda_hat = np.zeros((1, n_simulations)) if best_lambda_hat is None else best_lambda_hat
        lambda_range = self.default_lambda_range if lambda_range is None else lambda_range
        numFolds = self.default_num_folds if numFolds is None else numFolds
        h_mu = [] if h_mu is None else h_mu
        h_sigma = [] if h_sigma is None else h_sigma

        # ===================== 2. 历史 BFGS 配置（兼容路径保留） =====================
        # 对标 MATLAB fminunc 的 quasi-newton 算法配置
        # opt_options = {
        #     'method': 'BFGS',  # 对应 quasi-newton
        #     'tol': tolerance,
        #     'maxiter': max_iter,
        #     'options': {'disp': False}  # 对应 Display='none'
        # }
        opt_options = {
            'method': 'BFGS',   # 拟牛顿法，梯度下降的高效变种
            'tol': tolerance,   # 函数值收敛容忍度
            'options': {
                'disp': False,          # 不打印优化过程
                'maxiter': max_iter     # 最大迭代次数（注意：放在 options 子字典中）
            }
        }

        # ===================== 3. 逐次模拟估计参数 =====================
        beta_hat = np.zeros((n_features + 1, n_simulations))  # 存储所有模拟的参数估计
        for t in range(n_simulations):
            X = X_labeled[:, :, t]   # 第 t 次模拟的标记特征矩阵，shape=(n, p)
            Y = Y_labeled[:, :, t]   # 第 t 次模拟的标记标签向量，shape=(n, 1)
            X2 = X_unlabeled[t]      # 第 t 次模拟的未标记特征矩阵，shape=(N_t, p)

            if X2.size == 0:
                # --- 情形 A：无未标记数据，退化为监督估计 ---
                # 通过 model_spec.solve_supervised 求解。
                # 默认实现走 BFGS（逻辑回归等）；LinearModelSpec 重写为闭式 OLS。
                beta_hat[:, t] = self.model_spec.solve_supervised(
                    X, Y,
                    lambda_reg=best_lambda_hat[0, t],
                    initial_value=initial_value,
                    tolerance=tolerance,
                    max_iter=max_iter,
                ).ravel()
            else:
                # --- 情形 B：有未标记数据，使用 c1=0 半监督估计 ---

                # Step B1：通过 GBIC 选择多项式扩展阶数 alpha
                # alpha 控制辅助函数 Z 的复杂度（Z = [1, X, X², ..., X^alpha]）
                if result_summary is None or 'select_alpha' not in result_summary[t]:
                    # result_summary 中无预先计算的 alpha，调用 GBIC 方法重新计算
                    alpha, _, _ = self.base_selection_gbic(X, Y, tolerance, max_iter,
                                                           initial_value, beta_star, 1, 5, 0, lambda_range, numFolds)
                else:
                    # 直接读取预计算的 alpha，避免重复计算
                    alpha = result_summary[t]['select_alpha']

                # Step B2：构建标记数据的多项式扩展特征矩阵 Z_labeled
                # Z_labeled 的列依次为：[1, X^1, X^2, ..., X^alpha]
                # shape = (n, 1 + p * alpha)
                Z_labeled = self.model_spec.build_z_matrix(X, alpha)

                # Step B3：构建未标记数据的多项式扩展特征矩阵 Z_unlabeled
                # 结构与 Z_labeled 相同，但基于未标记特征 X2
                Z_unlabeled = self.model_spec.build_z_matrix(X2, alpha)

                # Step B4：调用 model_spec.solve_semi_supervised 求解（c1=0，DRESS 模式）
                # 具体求解算法由 model_spec 决定：
                # LogisticModelSpec 当前复现实验默认走 BFGS；strict_wgd/newton 路径可用于诊断；
                # LinearModelSpec 重写为闭式加权 LS，避免负权重下 BFGS 沿凹方向发散。
                beta_hat[:, t] = self.model_spec.solve_semi_supervised(
                    X, Y, Z_labeled, Z_unlabeled,
                    lambda_reg=best_lambda_hat[0, t],
                    use_dress_c1=True,           # DRESS = c1=0
                    initial_value=initial_value,
                    tolerance=tolerance,
                    max_iter=max_iter,
                    intercept_from_supervised=intercept_from_supervised,
                ).ravel()

        beta_hat = self._sanitize_beta_hat(
            beta_hat, X_labeled, Y_labeled, best_lambda_hat,
            initial_value, tolerance, max_iter, label="DRESS"
        )

        # ===================== 4. 计算基础评估指标（Bias/SE/MSE/ARE/RR/MRR） =====================
        # Bias：估计量均值与真实参数之差
        Bias = (1 / n_simulations) * np.sum(beta_hat - beta_star, axis=1, keepdims=True)
        # SE：估计量的经验标准差（在所有模拟次上的标准差）
        SE = np.sqrt(np.mean(np.square(beta_hat - np.mean(beta_hat, axis=1, keepdims=True)), axis=1, keepdims=True))
        # MSE：均方误差（综合偏差和方差）
        MSE = np.mean(np.square(beta_hat - beta_star), axis=1, keepdims=True)
        # ARE：渐近相对效率（监督 SE² / 半监督 SE²），>1 表示半监督更高效
        ARE = np.float64(np.square(Evaluate_supervised['SE'])).reshape(-1,1) / np.square(SE) if isinstance(Evaluate_supervised,
                                                                                 pd.DataFrame) else np.square(Evaluate_supervised) / np.square(SE)
        # RR：相对减少率（各分量 MSE 的改进比例）
        RR = (np.float64(Evaluate_supervised['MSE']).reshape(-1,1) - MSE) / np.float64(Evaluate_supervised['MSE']).reshape(-1,1) if isinstance(
            Evaluate_supervised, pd.DataFrame) else (np.mean(Evaluate_supervised) - MSE) / np.mean(Evaluate_supervised)
        # MRR：平均相对减少率（基于所有分量 MSE 均值的整体改进）
        MRR = ((np.mean(Evaluate_supervised['MSE']) - np.mean(MSE)) / np.mean(
            Evaluate_supervised['MSE'])) * np.ones_like(MSE) if isinstance(Evaluate_supervised, pd.DataFrame) else ((np.mean(Evaluate_supervised) - np.mean(MSE)) / np.mean(
            Evaluate_supervised)) * np.ones_like(MSE)

        # ===================== 5. 计算 SSE/CP 等进阶指标（分 proposed_if=1/0 两种情况） =====================
        if proposed_if == 1:
            # 改进方法：使用精确渐近方差公式（含 V2_prime 和偏差修正 BIAS）
            SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every = self._calculate_sse_cp_proposed(
                X_labeled, Y_labeled, beta_hat, beta_star, result_summary, best_lambda_hat, SE,
                n_simulations, intercept_from_supervised=intercept_from_supervised
            )
        else:
            # 基准方法：使用简化渐近方差公式（无 V2_prime，无偏差修正）
            SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every = self._calculate_sse_cp_benchmark(
                X_labeled, Y_labeled, X_unlabeled, beta_hat, beta_star, result_summary, best_lambda_hat,
                SE, n_simulations, intercept_from_supervised=intercept_from_supervised
            )

        # ===================== 6. 处理 NaN 值并计算均值指标 =====================
        # 找出含 NaN 的列（数值不稳定的模拟次）
        nan_cols = np.unique(np.where(np.isnan(SSE_every))[1])
        if len(nan_cols) > 0:
            # 移除含 NaN 的列，保证后续均值计算的有效性
            SSE_every = np.delete(SSE_every, nan_cols, axis=1)
            CP_every = np.delete(CP_every, nan_cols, axis=1)
            SE_SSE_ratio_every = np.delete(SE_SSE_ratio_every, nan_cols, axis=1)
            SE_SSE_ratio_mean_every = np.delete(SE_SSE_ratio_mean_every, nan_cols, axis=1)

        if SSE_every.shape[1] == 0:
            SSE = np.full_like(SE, np.nan)
            CP = np.full_like(SE, np.nan)
            SE_SSE_ratio = np.full_like(SE, np.nan)
            SE_SSE_ratio_mean = np.full_like(SE, np.nan)
        else:
            # 对有效模拟次计算均值
            SSE = np.mean(SSE_every, axis=1, keepdims=True)
            # CP：置信区间覆盖率，用命中次数除以有效模拟次数
            CP = np.sum(CP_every, axis=1, keepdims=True) / SSE_every.shape[1]
            SE_SSE_ratio = np.mean(SE_SSE_ratio_every, axis=1, keepdims=True)
            SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every, axis=1, keepdims=True)

        # ===================== 7. 计算均值化统计量（方便表格展示） =====================
        MSE_MEAN = np.mean(MSE) * np.ones_like(MSE)           # 所有参数分量 MSE 的均值（标量广播）
        BIAS_MEAN = np.mean(np.abs(Bias)) * np.ones_like(MSE) # 所有参数分量绝对偏差的均值
        SE_MEAN = np.mean(SE) * np.ones_like(MSE)             # 所有参数分量 SE 的均值
        SSE_MEAN = np.mean(SSE) * np.ones_like(MSE)           # 所有参数分量 SSE 的均值
        CP_MEAN = np.mean(CP) * np.ones_like(MSE)             # 所有参数分量 CP 的均值
        ARE_MEAN = np.mean(ARE) * np.ones_like(ARE)           # ARE 的均值

        # ===================== 8. 统计各未标记数据源的选取频次 =====================
        select_time = np.array([])
        if len(h_mu) > 0 and len(h_sigma) > 0:
            # 调用内部方法，生成 (h_mu × h_sigma) 频次矩阵
            select_time = self._count_selection_times(result_summary, h_mu, h_sigma)

        # ===================== 9. 构建评估结果 DataFrame（对标 MATLAB array2table） =====================
        Evaluate = pd.DataFrame({
            'Bias': Bias.ravel(),
            'BIAS_MEAN': BIAS_MEAN.ravel(),
            'SE': SE.ravel(),
            'SE_MEAN': SE_MEAN.ravel(),
            'SSE': SSE.ravel(),
            'SSE_MEAN': SSE_MEAN.ravel(),
            'ARE': ARE.ravel() if isinstance(ARE, np.ndarray) else ARE,
            'ARE_MEAN': ARE_MEAN.ravel(),
            'MSE': MSE.ravel(),
            'MSE_MEAN': MSE_MEAN.ravel(),
            'RR': RR.ravel(),
            'MRR': MRR.ravel(),
            'SE_SSE_ratio': SE_SSE_ratio.ravel(),
            'SE_SSE_ratio_mean': SE_SSE_ratio_mean.ravel(),
            'CP': CP.ravel(),
            'CP_MEAN': CP_MEAN.ravel()
        })

        return beta_hat, Evaluate, select_time

    def _sanitize_beta_hat(self, beta_hat, X_labeled, Y_labeled, best_lambda_hat,
                           initial_value, tolerance, max_iter, label="DRESS", box=30.0):
        """
        在计算 MSE/MRR/SSE 之前拦截数值发散的参数估计。

        半监督加权目标可能含负权重；逻辑回归在有限样本下偶尔会沿近似无界方向
        给出极大的 beta。若直接进入 MSE/MRR，一个坏的模拟列就会把平均指标放大很多。
        因此把非有限或绝对值超过 box 的列回退为同一次模拟的监督估计。
        """
        box = float(getattr(self.model_spec, "beta_sanitize_box", box))
        for t in range(beta_hat.shape[1]):
            b = beta_hat[:, t]
            finite = np.isfinite(b)
            max_abs = float(np.max(np.abs(b[finite]))) if np.any(finite) else np.nan
            if (not np.all(finite)) or max_abs > box:
                print(f"[警告 {label}-estimation] t={t}: beta_hat异常(max|beta|={max_abs:.3g}); 回退监督估计")
                beta_hat[:, t] = self.model_spec.solve_supervised(
                    X_labeled[:, :, t], Y_labeled[:, :, t],
                    lambda_reg=best_lambda_hat[0, t],
                    initial_value=initial_value,
                    tolerance=tolerance,
                    max_iter=max_iter,
                ).ravel()
        return beta_hat

    # ========================= 辅助方法：Proposed 模式下计算 SSE/CP =========================
    def _calculate_sse_cp_proposed(self, X_labeled, Y_labeled, beta_hat, beta_star, result_summary, best_lambda_hat, SE,
                                   n_simulations, intercept_from_supervised=False):
        """
        Proposed 模式下计算渐近标准误（SSE）和置信区间覆盖率（CP）。

        本方法实现了改进方法的精确渐近方差公式，考虑了：
          - 残差协方差 V1（得分函数未被辅助特征 Z 解释的部分）
          - 标记数据投影方差 V2（投影方向的方差，乘以 c_star²）
          - 未标记数据投影方差 V2_prime（各数据源加权求和，乘以 (1-c_star)² * rho）
          - 偏差修正 BIAS（基于各数据源均值差的修正项）

        参数：
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p, T)
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1, T)
            beta_hat (np.ndarray): 参数估计矩阵，shape=(p+1, T)
            beta_star (np.ndarray): 真实参数，shape=(p+1, 1)
            result_summary (list): 每次模拟的结果摘要列表
            best_lambda_hat (np.ndarray): 各模拟最优正则化参数，shape=(1, T)
            SE (np.ndarray): 各参数分量的经验标准误，shape=(p+1, 1)
            n_simulations (int): 模拟次数 T

        返回：
            SSE_every (np.ndarray): 每次模拟的渐近标准误，shape=(p+1, T)
            CP_every (np.ndarray): 每次模拟的置信区间覆盖率，shape=(p+1, T)
            SE_SSE_ratio_every (np.ndarray): 每次模拟的 SE/SSE 比率，shape=(p+1, T)
            SE_SSE_ratio_mean_every (np.ndarray): 每次模拟的整体 SE/SSE 均值比，shape=(p+1, T)
        """
        n_params = beta_hat.shape[0]  # 参数维度 p+1
        SSE_every = np.zeros((n_params, n_simulations))
        CP_every = np.ones((n_params, n_simulations))
        SE_SSE_ratio_every = np.zeros((n_params, n_simulations))
        SE_SSE_ratio_mean_every = np.zeros((n_params, n_simulations))

        for t in range(n_simulations):
            beta_hat_t = beta_hat[:, t].reshape(-1, 1)   # 第 t 次模拟的参数估计
            res_sum_t = result_summary[t]

            # --- 提取结果摘要中的关键信息 ---
            # fields：未标记数据各数据源的字段名列表
            fields = list(res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_unlabeled_one_simulation'].keys())
            # h_mu：均值带宽（由 lambda_mu 和样本量共同决定）
            h_mu = res_sum_t['lambda_mu_opt_value'] * np.sqrt(
                1 / res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_labeled_one_simulation'].shape[0])

            # q：sigma 方向的分量数（用于计算协方差带宽 h_sigma）
            q = res_sum_t['Tree_lambda_sigma_one_simulation']['index_sigma'][0].shape[0] - 1
            # h_sigma：协方差带宽
            h_sigma = res_sum_t['lambda_sigma_opt_value'] * np.sqrt((q ** 2 * np.log(q)) /
                                                                    res_sum_t['Tree_lambda_sigma_one_simulation'][
                                                                        'train_Z_labeled_one_simulation'].shape[0])

            # --- 计算被选入的未标记数据总量 n_A ---
            A_number = len(res_sum_t['select_index'])  # 被选入的数据源数量
            n_A = 0
            for aa in range(A_number):
                field_idx = res_sum_t['select_index'][aa] - 1  # Python 0-based 索引
                Z_un = res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_unlabeled_one_simulation'][fields[field_idx]]
                n_A += Z_un.shape[0]

            # c_star 和 rho：用于缩放各方差项
            c_star = X_labeled.shape[0] / (X_labeled.shape[0] + n_A)
            rho = X_labeled.shape[0] / n_A if n_A > 0 else 0

            # --- 计算得分函数（损失函数对 beta 的一阶偏导数） ---
            X_t = X_labeled[:, :, t]
            Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
            Partial_l = self.model_spec.score(
                beta_hat_t, X_t, Y_t, best_lambda_hat[0, t]
            )  # shape: [n, p+1]

            # --- 投影回归：将得分函数投影到辅助特征空间 Z ---
            Z_t = res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_labeled_one_simulation']

            # 防御：线性模型 score = (Xβ - Y)·X 是无界的，BFGS 收敛到大 β 时 Xβ 可能溢出
            _pl_fin = bool(np.all(np.isfinite(Partial_l)))
            _z_fin  = bool(Z_t.size == 0 or np.all(np.isfinite(Z_t)))
            if not (_pl_fin and _z_fin):
                SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                def _mx(a):
                    a = np.asarray(a); _m = np.isfinite(a)
                    return float(np.abs(a[_m]).max()) if _m.any() else float('nan')
                print(f"[警告 DRESS-proposed] t={t}: Partial_l_finite={_pl_fin}, Z_finite={_z_fin}, "
                      f"max|beta|={_mx(beta_hat_t):.3g}, max|X|={_mx(X_t):.3g}, "
                      f"max|Z|={_mx(Z_t):.3g}, max|Partial_l|={_mx(Partial_l):.3g}; 已置 NaN")
                continue
            # ============================================================
            # 渐近方差 V 估计（论文 Theorem 3 原始三项分解）
            #   V̂ = V̂₁ + c*² V̂₂ + (1-c*)² ρ_Â V̂₂'
            # ============================================================
            gamma_hat = np.linalg.lstsq(Z_t, Partial_l, rcond=None)[0]

            # V̂₁：委托 model_spec.estimate_v1（按模型选估计方式）
            V1 = self.model_spec.estimate_v1(Z_t, Partial_l, K=5, seed=t)

            # V̂₂：labeled 中心化
            Z_centered_l = Z_t - np.mean(Z_t, axis=0)
            AZ_l_c = Z_centered_l @ gamma_hat
            V2 = (c_star ** 2) * (AZ_l_c.T @ AZ_l_c) / Z_t.shape[0]

            # V̂₂'：per-source 中心化
            V2_prime = np.zeros_like(V2)
            for aa in range(A_number):
                field_idx = res_sum_t['select_index'][aa] - 1
                Z_un = res_sum_t['Tree_lambda_mu_one_simulation'][
                    'train_Z_unlabeled_one_simulation'][fields[field_idx]]
                Z_un_c = Z_un - np.mean(Z_un, axis=0)
                AZ_un_c = Z_un_c @ gamma_hat
                n_k = Z_un.shape[0]
                V2_prime += (n_k / n_A) * (AZ_un_c.T @ AZ_un_c) / n_k
            V2_prime = ((1 - c_star) ** 2 * rho) * V2_prime

            V = V1 + V2 + V2_prime

            # --- 三明治方差估计 ---
            # M：Hessian 矩阵（损失函数对 beta 的二阶偏导数均值）
            M = self.model_spec.hessian(beta_hat_t, X_t, Y_t, best_lambda_hat[0, t])

            # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
            # 线性模型在某些 t 下 BFGS 可能不收敛 → beta_hat 溢出 → 残差/Partial_l/gamma_hat 含 inf/nan
            # 此处一旦发现 M 或 V 非有限，就把该次仿真所有 SSE/CP 置 NaN，让后续 nan_cols 过滤器丢掉
            if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                SSE_every[:, t] = np.nan
                CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan
                SE_SSE_ratio_mean_every[:, t] = np.nan
                print(f"[警告 DRESS-proposed] t={t}: 渐近方差含非有限值 "
                      f"(M_finite={bool(np.all(np.isfinite(M)))}, V_finite={bool(np.all(np.isfinite(V)))}, "
                      f"beta_finite={bool(np.all(np.isfinite(beta_hat_t)))}); 该次仿真已置 NaN")
                continue
            # SIGMA = (1/n) * M^{-1} V M^{-1}（三明治方差公式）
            SIGMA = stable_sandwich(M, V) / (Z_t @ gamma_hat).shape[0]

            # --- 计算偏差修正项 BIAS ---
            # BIAS 反映了各数据源均值漂移对估计量的影响
            M_half = sqrtm(M)   # M 的矩阵平方根
            BIAS = np.zeros_like(beta_hat_t)
            for aa in range(A_number):
                field_idx = res_sum_t['select_index'][aa] - 1
                Z_unlabeled_t = res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_unlabeled_one_simulation'][
                    fields[field_idx]]
                weight = np.sqrt(Z_unlabeled_t.shape[0] / n_A) if n_A > 0 else 0
                # 计算未标记数据源与标记数据辅助特征的均值差
                mean_diff = np.mean(Z_unlabeled_t, axis=0) - np.mean(Z_t, axis=0)
                BIAS += (weight * np.sqrt(Z_unlabeled_t.shape[0]) * mean_diff @ gamma_hat).T

            # 最终偏差修正：通过 M 的逆平方根变换到参数空间
            BIAS = (
                np.sqrt(1 / (Z_t @ gamma_hat).shape[0])
                * stable_solve(
                    M_half, np.sqrt(rho) * (1 - c_star) * BIAS,
                    symmetrize=True,
                )
            )

            # --- 计算 SSE 和 95% 置信区间覆盖率 ---
            SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()
            if intercept_from_supervised:
                beta_sup_t = self.model_spec.solve_supervised(
                    X_t, Y_t, lambda_reg=best_lambda_hat[0, t]
                )
                s_sup = self.model_spec.score(beta_sup_t, X_t, Y_t, best_lambda_hat[0, t])
                M_sup = self.model_spec.hessian(beta_sup_t, X_t, Y_t, best_lambda_hat[0, t])
                n_l = s_sup.shape[0]
                s_sup_c = s_sup - np.mean(s_sup, axis=0, keepdims=True)
                V_sup = (s_sup_c.T @ s_sup_c) / n_l
                SIGMA_sup = stable_sandwich(M_sup, V_sup) / n_l
                SSE_every[0, t] = float(np.sqrt(max(SIGMA_sup[0, 0], 0)))
                BIAS[0, 0] = 0.0
            up = (beta_hat_t - BIAS) + 1.96 * SSE_every[:, t].reshape(-1, 1)
            down = (beta_hat_t - BIAS) - 1.96 * SSE_every[:, t].reshape(-1, 1)
            # 检查真实参数 beta_star 是否落在 95% 置信区间内
            CP_every[:, t] = np.logical_and(beta_star >= down, beta_star <= up).ravel()

            # SE/SSE 比率（理想情况下接近 1）
            SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
            SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones_like(SE).ravel()

        return SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every

    # ========================= 辅助方法：基准模式下计算 SSE/CP =========================
    def _calculate_sse_cp_benchmark(self, X_labeled, Y_labeled, X_unlabeled, beta_hat, beta_star, result_summary, best_lambda_hat,
                                    SE, n_simulations, intercept_from_supervised=False):
        """
        基准模式（proposed_if=0）下计算渐近标准误（SSE）和置信区间覆盖率（CP）。

        与 _calculate_sse_cp_proposed 的主要区别：
          - 方差公式更简化（V = V1 + V2，无 V2_prime）
          - V2 的缩放系数为 (c_star² + (1 - c_star)² * rho)  # 论文 Theorem 1: V_rho = E[δδ^T] + (c1² + ρ(1-c1)²) E[A^T ZZ^T A]（合并了标记和未标记贡献）
          - 偏差修正项 BIAS 固定为零（不做偏差修正）

        参数：
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p, T)
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1, T)
            X_unlabeled (list): 未标记数据列表，长度=T
            beta_hat (np.ndarray): 参数估计矩阵，shape=(p+1, T)
            beta_star (np.ndarray): 真实参数，shape=(p+1, 1)
            result_summary (list): 每次模拟的结果摘要列表
            best_lambda_hat (np.ndarray): 各模拟最优正则化参数，shape=(1, T)
            SE (np.ndarray): 各参数分量的经验标准误，shape=(p+1, 1)
            n_simulations (int): 模拟次数 T

        返回：
            SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every
            （含义与 _calculate_sse_cp_proposed 相同）
        """
        n_params = beta_hat.shape[0]
        SSE_every = np.zeros((n_params, n_simulations))
        CP_every = np.ones((n_params, n_simulations))
        SE_SSE_ratio_every = np.zeros((n_params, n_simulations))
        SE_SSE_ratio_mean_every = np.zeros((n_params, n_simulations))

        for t in range(n_simulations):
            beta_hat_t = beta_hat[:, t].reshape(-1, 1)
            res_sum_t = result_summary[t]

            # 提取关键参数：q（sigma 分量数）、n_A（未标记数据量）、c_star、rho
            q = res_sum_t['Tree_lambda_sigma_one_simulation']['index_sigma'][0].shape[0] - 1
            A_number = len(res_sum_t['select_index'])
            n_A = X_unlabeled[t].shape[0] if X_unlabeled[t].size > 0 else 0

            c_star = X_labeled.shape[0] / (X_labeled.shape[0] + n_A) if (X_labeled.shape[0] + n_A) > 0 else 0
            rho = X_labeled.shape[0] / n_A if n_A > 0 else 0

            # 计算得分函数
            X_t = X_labeled[:, :, t]
            Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
            Partial_l = self.model_spec.score(
                beta_hat_t, X_t, Y_t, best_lambda_hat[0, t]
            )  # [n, p+1]

            # 投影回归：将得分函数投影到辅助特征空间
            Z_t = res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_labeled_one_simulation']

            # 防御：见 _calculate_sse_cp_proposed 同位置注释
            _pl_fin = bool(np.all(np.isfinite(Partial_l)))
            _z_fin  = bool(Z_t.size == 0 or np.all(np.isfinite(Z_t)))
            if not (_pl_fin and _z_fin):
                SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                def _mx(a):
                    a = np.asarray(a); _m = np.isfinite(a)
                    return float(np.abs(a[_m]).max()) if _m.any() else float('nan')
                print(f"[警告 DRESS-benchmark] t={t}: Partial_l_finite={_pl_fin}, Z_finite={_z_fin}, "
                      f"max|beta|={_mx(beta_hat_t):.3g}, max|X|={_mx(X_t):.3g}, "
                      f"max|Z|={_mx(Z_t):.3g}, max|Partial_l|={_mx(Partial_l):.3g}; 已置 NaN")
                continue
            gamma_hat = np.linalg.lstsq(Z_t, Partial_l, rcond=None)[0]
            e = Partial_l - Z_t @ gamma_hat  # 投影残差

            # 简化渐近方差公式（合并标记和未标记数据的贡献）
            V1 = (1 / e.shape[0]) * (e.T @ e)
            V2 = (1 / (Z_t @ gamma_hat).shape[0]) * ((Z_t @ gamma_hat).T @ (Z_t @ gamma_hat))
            # 合并缩放系数：c_star² 来自标记数据，(1 - c_star²) * rho 来自未标记数据
            V2 = (c_star ** 2 + (1 - c_star) ** 2 * rho) * V2
            V = V1 + V2

            # 三明治方差估计
            M = self.model_spec.hessian(beta_hat_t, X_t, Y_t, best_lambda_hat[0, t])

            # --- 防御性检查：见 _calculate_sse_cp_proposed 同位置注释 ---
            if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                SSE_every[:, t] = np.nan
                CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan
                SE_SSE_ratio_mean_every[:, t] = np.nan
                # 加强诊断：打印各关键量的极值，帮助定位根因
                def _safe_max_abs(a):
                    a = np.asarray(a)
                    finite_mask = np.isfinite(a)
                    return float(np.abs(a[finite_mask]).max()) if finite_mask.any() else float('nan')
                print(f"[警告 DRESS-benchmark] t={t}: 渐近方差含非有限值 "
                      f"M_finite={bool(np.all(np.isfinite(M)))}, V_finite={bool(np.all(np.isfinite(V)))}, "
                      f"beta_finite={bool(np.all(np.isfinite(beta_hat_t)))}, "
                      f"lambda={best_lambda_hat[0, t]}, "
                      f"max|X|={_safe_max_abs(X_t):.3g}, max|Y|={_safe_max_abs(Y_t):.3g}, "
                      f"max|beta|={_safe_max_abs(beta_hat_t):.3g}, "
                      f"max|Partial_l|={_safe_max_abs(Partial_l):.3g}, "
                      f"max|gamma_hat|={_safe_max_abs(gamma_hat):.3g}; 已置 NaN")
                continue

            SIGMA = stable_sandwich(M, V) / (Z_t @ gamma_hat).shape[0]

            # 基准模式下无偏差修正
            BIAS = np.zeros_like(beta_hat_t)

            # 计算 SSE 和置信区间覆盖率
            SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()
            if intercept_from_supervised:
                beta_sup_t = self.model_spec.solve_supervised(
                    X_t, Y_t, lambda_reg=best_lambda_hat[0, t]
                )
                s_sup = self.model_spec.score(beta_sup_t, X_t, Y_t, best_lambda_hat[0, t])
                M_sup = self.model_spec.hessian(beta_sup_t, X_t, Y_t, best_lambda_hat[0, t])
                n_l = s_sup.shape[0]
                s_sup_c = s_sup - np.mean(s_sup, axis=0, keepdims=True)
                V_sup = (s_sup_c.T @ s_sup_c) / n_l
                SIGMA_sup = stable_sandwich(M_sup, V_sup) / n_l
                SSE_every[0, t] = float(np.sqrt(max(SIGMA_sup[0, 0], 0)))
            up = (beta_hat_t - BIAS) + 1.96 * SSE_every[:, t].reshape(-1, 1)
            down = (beta_hat_t - BIAS) - 1.96 * SSE_every[:, t].reshape(-1, 1)
            CP_every[:, t] = np.logical_and(beta_star >= down, beta_star <= up).ravel()

            # SE/SSE 比率
            SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
            SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones_like(SE).ravel()

        return SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every

    # ========================= 辅助方法：统计未标记数据选择次数 =========================
    def _count_selection_times(self, result_summary, h_mu, h_sigma):
        """
        统计各未标记数据源的被选中次数，生成 (h_mu × h_sigma) 频次矩阵。

        数据源通过线性索引（1-based）标识，本方法将其转换为二维索引 (mu_idx, sigma_idx)。
        转换规则（对应 MATLAB 的 ind2sub 逻辑）：
          - 若 idx % n_sigma == 0：sigma_idx = n_sigma - 1，mu_idx = idx // n_sigma - 1
          - 否则：sigma_idx = idx % n_sigma - 1，mu_idx = idx // n_sigma

        参数：
            result_summary (list): 每次模拟的结果摘要列表，
                每个元素含 'select_index' 键（被选数据源的 1-based 线性索引列表）
            h_mu (list): 均值偏移参数列表，长度决定矩阵行数
            h_sigma (list): 协方差偏移参数列表，长度决定矩阵列数

        返回：
            select_time (np.ndarray): 频次统计矩阵，shape=(len(h_mu), len(h_sigma))，
                select_time[i, j] 表示第 (i, j) 个数据源在所有模拟中被选中的总次数
        """
        n_mu = len(h_mu)
        n_sigma = len(h_sigma)
        select_time = np.zeros((n_mu, n_sigma))

        for t in range(len(result_summary)):
            if len(result_summary[t]['select_index']) == 0:
                continue  # 当前模拟无被选数据源，跳过

            for idx in result_summary[t]['select_index']:
                idx = int(idx)  # 确保为整数
                # 将 1-based 线性索引转换为二维 (mu_idx, sigma_idx)
                if np.mod(idx, n_sigma) == 0:
                    sigma_idx = n_sigma - 1           # 最后一个 sigma 分量
                    mu_idx = int(np.floor(idx / n_sigma)) - 1
                else:
                    sigma_idx = np.mod(idx, n_sigma) - 1
                    mu_idx = int(np.floor(idx / n_sigma))

                # 边界检查，防止索引越界
                if 0 <= mu_idx < n_mu and 0 <= sigma_idx < n_sigma:
                    select_time[mu_idx, sigma_idx] += 1

        return select_time

    # ========================= 目标函数：c1=0 半监督目标函数 =========================
    def dress_ss_objective_function_logistic(self, beta, X_labeled, Y_labeled, Z_labeled=None, Z_unlabeled=None,
                                             lambda_reg=0):
        """
        c1=0 半监督目标函数包装器，供 scipy.optimize.minimize 调用。

        本方法不直接写损失公式，而是委托给 self.model_spec.ss_loss_and_grad。
        传入 use_dress_c1=True 后，ModelSpec 内部会令 c1=0，并用辅助矩阵 Z 构造样本权重。
        具体目标函数和梯度以 ModelSpec.ss_loss_and_grad 的实现为准。

        保留原方法名以兼容内部调用。

        参数：
            beta (np.ndarray): 参数向量，shape=(p+1,)，优化器传入的当前参数值
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p)
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1) 或 (n,)
            Z_labeled (np.ndarray, optional): 标记数据的多项式扩展特征矩阵，
                shape=(n, 1 + p * alpha)，默认 None（退化为纯监督模式）
            Z_unlabeled (np.ndarray, optional): 未标记数据的多项式扩展特征矩阵，
                shape=(N, 1 + p * alpha)，默认 None
            lambda_reg (float): L2 正则化参数，默认 0

        返回：
            tuple: (loss, grad)
                loss (float): 目标函数值
                grad (np.ndarray): 梯度向量，shape=(p+1,)
        """
        return self.model_spec.ss_loss_and_grad(
            beta, X_labeled, Y_labeled, Z_labeled, Z_unlabeled,
            lambda_reg, use_dress_c1=True   # 当前文件约定：c1=0
        )

    # ========================= 基础方法：基于 GBIC 的多项式阶数选择 =========================
    def base_selection_gbic(self, X_labeled, Y_labeled, tolerance=None, max_iter=None,
                            initial_value=None, beta_star=None, alpha_up=None, alpha_down=None,
                            CP_if=None, lambda_range=None, numFolds=None):
        """
        基于广义贝叶斯信息准则（GBIC）选择辅助特征矩阵 Z 的最优多项式阶数 alpha。

        alpha 控制辅助特征 Z 的复杂度：Z = [1, X, X², ..., X^alpha]。
        GBIC 越小表示模型拟合越好（平衡了拟合度和复杂度），本方法为每次模拟选择使 GBIC 最小的 alpha。

        GBIC 的计算公式（对每个分量 j）：
            GBIC = (1/n) * [SSE/sigma + n*log(sigma) + log(n)*p*alpha + trace(A) - log(det(A))]
        其中 A = (Z'Z * sigma)^{-1} * (diag(e_j) Z)' * (diag(e_j) Z)

        参数：
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p) 或 (n, p, T)
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1) 或 (n, 1, T)
            tolerance (float, optional): 优化收敛容忍度，默认 5e-3
            max_iter (int, optional): 最大迭代次数，默认 500
            initial_value (np.ndarray, optional): 参数初始值，默认全零
            beta_star (np.ndarray, optional): 真实参数，默认全 1
            alpha_up (int, optional): alpha 搜索上限，默认 5
            alpha_down (int, optional): alpha 搜索下限，默认 1
            CP_if (int, optional): 是否计算 CP（0 或 1），默认 0
            lambda_range (np.ndarray, optional): lambda 搜索范围，默认 logspace(-10, 2, 100)
            numFolds (int, optional): 交叉验证折数，默认 5

        返回：
            alpha (np.ndarray): 各模拟次的最优 alpha 值，shape=(T,)，元素为 1~alpha_up 的整数
            GBIC (np.ndarray): 各 alpha 和模拟次的 GBIC 值，shape=(alpha_up, T)
            best_lambda (np.ndarray): 各模拟次的最优 lambda 值，shape=(T,)
        """
        # 参数默认值处理
        alpha_up = 5 if alpha_up is None else alpha_up        # alpha 搜索上限
        alpha_down = 1 if alpha_down is None else alpha_down  # alpha 搜索下限
        CP_if = 0 if CP_if is None else CP_if
        lambda_range = np.logspace(-10, 2, 100) if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        # 适配输入维度：若输入为 2D（单次模拟），扩展为 3D（一次模拟）
        if len(X_labeled.shape) == 2:
            X_labeled = X_labeled.reshape(X_labeled.shape[0], X_labeled.shape[1], 1)
        if len(Y_labeled.shape) == 2:
            Y_labeled = Y_labeled.reshape(Y_labeled.shape[0], Y_labeled.shape[1], 1)

        simulation_times = X_labeled.shape[2]
        alpha = np.zeros(simulation_times)          # 存储各模拟最优 alpha
        best_lambda = np.zeros(simulation_times)    # 存储各模拟最优 lambda
        GBIC = np.zeros((alpha_up, simulation_times))  # 存储各 alpha 的 GBIC 值

        # 逐次模拟计算 GBIC
        for t in range(simulation_times):
            X = X_labeled[:, :, t]
            Y = Y_labeled[:, t].reshape(-1, 1)

            # Step 1：先用监督 model_spec 求解参数估计和最优 lambda
            beta_hat, _, best_lambda_hat = self.solve_logistic_regression(
                X.reshape(X.shape[0], X.shape[1], 1), Y.reshape(Y.shape[0], 1, 1),
                None, None, None, None, CP_if, lambda_range, numFolds
            )

            # 以下为注释掉的旧版手动计算 Partial_l 代码（已由 model_spec.score 替代）
            # # 计算Partial_l
            # X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            # exp_term = np.exp(X_aug @ beta_hat[:, t].reshape(-1, 1)) #!!!!!!!!!!!!!!!!!!改t为0
            # sigmoid = exp_term / (1 + exp_term)
            # Partial_l = (sigmoid @ np.ones((1, X_aug.shape[1]))) * X_aug - \
            #             (Y @ np.ones((1, X_aug.shape[1]))) * X_aug + \
            #             (2 * best_lambda_hat[t] * np.ones((X.shape[0], X_aug.shape[1]))) * \
            #             (np.ones((X.shape[0], 1)) @ beta_hat[:, t].reshape(1, -1))
            #
            # best_lambda[t] = best_lambda_hat[t]
            #
            # # 计算不同alpha的GBIC
            # for a in range(alpha_down, alpha_up + 1):
            #     Z = np.zeros((X_labeled.shape[0], 1 + X_labeled.shape[1] * a, X_labeled.shape[2]))
            #     for tt in range(Z.shape[2]):
            #         Z[:, 0, tt] = np.ones(X_labeled.shape[0])
            #         for i in range(1, a + 1):
            #             start_col = (i - 1) * X_labeled.shape[1] + 1
            #             end_col = i * X_labeled.shape[1] + 1
            #             Z[:, start_col:end_col, tt] = np.power(X_labeled[:, :, tt], i)
            #
            #     # 计算gamma_hat和残差
            #     gamma_hat = stable_solve(
            #         Z[:, :, t].T @ Z[:, :, t],
            #         Z[:, :, t].T @ Partial_l,
            #         symmetrize=True,
            #     )
            #     e = Partial_l - Z[:, :, t] @ gamma_hat
            #
            #     # 计算sigma和GBIC
            #     sigma = np.sum(e ** 2) / (X_labeled.shape[0] - X_labeled.shape[1] * a - 1)
            #     for j in range(X.shape[1] + 1):
            #         term1 = (1 / X_labeled.shape[0]) * (np.sum(e[:, j] ** 2) / sigma +
            #                                             X_labeled.shape[0] * np.log(sigma) +
            #                                             np.log(X_labeled.shape[0]) * X_labeled.shape[1] * a)
            #         residual_info = (np.diag(e[:, j]) @ Z[:, :, t]).T @ (np.diag(e[:, j]) @ Z[:, :, t])
            #         scaled_info = stable_solve(Z[:, :, t].T @ Z[:, :, t] * sigma, residual_info, symmetrize=True)
            #         term2 = np.trace(scaled_info)
            #         term3 = -np.log(np.linalg.det(scaled_info))
            #         GBIC[a-1, t] += term1 + term2 + term3

            # Step 2：计算得分函数（已泛化，支持任意 M-估计模型）
            Partial_l = self.model_spec.score(
                beta_hat[:, 0].reshape(-1, 1), X, Y, best_lambda_hat[t]
            )  # shape: [n, p+1]

            best_lambda[t] = best_lambda_hat[t]

            # Step 3：对各 alpha 值计算 GBIC
            for a in range(alpha_down, alpha_up + 1):
                # 构建多项式扩展特征矩阵 Z，shape=(n, 1 + p * a, T)
                Z = np.zeros((X_labeled.shape[0], 1 + X_labeled.shape[1] * a, X_labeled.shape[2]))
                for tt in range(Z.shape[2]):
                    Z[:, 0, tt] = np.ones(X_labeled.shape[0])  # 截距列
                    for i in range(1, a + 1):
                        start_col = (i - 1) * X_labeled.shape[1] + 1
                        end_col = i * X_labeled.shape[1] + 1
                        Z[:, start_col:end_col, tt] = np.power(X_labeled[:, :, tt], i)

                # 计算投影系数 gamma_hat 和残差 e
                Z_t = Z[:, :, t]
                ZtZ = Z_t.T @ Z_t
                gamma_hat = stable_solve(ZtZ, Z_t.T @ Partial_l, symmetrize=True)
                e = Partial_l - Z[:, :, t] @ gamma_hat

                # 计算各分量的残差方差 sigma（自由度修正：n - p*a - 1）
                sigma = np.sum(e ** 2, axis=0) * (1 / (X_labeled.shape[0] - X_labeled.shape[1] * a - 1))

                # 对每个参数分量 j 计算 GBIC（参数维度为截距 + p 个特征）
                for j in range(X.shape[1] + 1):
                    # term1：标准化残差 SSE + 模型复杂度惩罚（BIC 形式）
                    term1 = (np.sum(e[:, j] ** 2) / sigma[j] + X_labeled.shape[0] * np.log(sigma[j]) +
                             np.log(X_labeled.shape[0]) * X_labeled.shape[1] * a)
                    # term2：trace 项（反映信息矩阵的估计精度）。diag(e_j) @ Z 等价于逐行 e_j * Z。
                    weighted_Z = e[:, [j]] * Z_t
                    residual_info = weighted_Z.T @ weighted_Z
                    scaled_info = stable_solve(
                        ZtZ * sigma[j], residual_info, symmetrize=True
                    )
                    term2 = np.trace(scaled_info)
                    # term3：-log(det) 项（惩罚自由度）
                    term3 = -np.log(np.linalg.det(scaled_info))
                    GBIC[a - 1, t] += (1 / X_labeled.shape[0]) * (term1 + term2 + term3)

        # 选择各模拟的最优 alpha（使 GBIC 最小的 alpha 值，+1 因为从 alpha_down=1 开始）
        for t in range(simulation_times):
            alpha[t] = int(np.argmin(GBIC[:, t]) + 1)

        return alpha, GBIC, best_lambda

    # ========================= 核心方法：求解监督 M-估计 =========================
    def solve_logistic_regression(self, X_labeled, Y_labeled, tolerance=None, max_iter=None,
                                  initial_value=None, beta_star=None, CP_if=None,
                                  lambda_range=None, numFolds=None):
        """
        求解监督 M-估计（有监督基准估计），支持多次模拟的批量求解。

        本方法是半监督估计的基础，用于：
          1. 获取初始参数估计（用作半监督优化的热启动）
          2. 计算有监督基准的评估指标（作为 ARE/RR 计算的分母）

        参数：
            X_labeled (np.ndarray): 标记特征数据，shape=(n, p, T)
            Y_labeled (np.ndarray): 标记标签数据，shape=(n, 1, T)
            tolerance (float, optional): 优化收敛容忍度，默认 5e-3
            max_iter (int, optional): 最大迭代次数，默认 500
            initial_value (np.ndarray, optional): 参数初始值，shape=(p+1, 1)，默认全零
            beta_star (np.ndarray, optional): 真实参数，shape=(p+1, 1)，默认全 1
            CP_if (int, optional): 是否计算置信区间覆盖率（1=是，0=否），默认 0
            lambda_range (np.ndarray, optional): lambda 搜索范围，默认 logspace(-10, 2, 100)
            numFolds (int, optional): 交叉验证折数，默认 5

        返回：
            beta_hat (np.ndarray): 各模拟的参数估计矩阵，shape=(p+1, T)
            Evaluate (pd.DataFrame): 模型评估指标表，
                若 CP_if=1 含 Bias/SE/MSE/SSE/CP 等列，
                若 CP_if=0 仅含 Bias/SE/MSE 等基础列
            best_lambda_hat (np.ndarray): 各模拟的最优 lambda，shape=(T,)
        """
        # 参数默认值处理
        tolerance = 5e-3 if tolerance is None else tolerance
        max_iter = 500 if max_iter is None else max_iter
        # 参数初始值：p+1 维全零向量（包含截距）
        initial_value = np.zeros((X_labeled.shape[1] + 1, 1)) if initial_value is None else initial_value
        # 默认真实参数：截距为 1，所有特征系数为 1
        beta_star = np.vstack([np.ones((1, 1)), np.ones((X_labeled.shape[1], 1))]) if beta_star is None else beta_star
        CP_if = 0 if CP_if is None else CP_if
        lambda_range = np.logspace(-10, 2, 100) if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        # 初始化结果矩阵
        beta_hat = np.zeros((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        best_lambda_hat = np.zeros(X_labeled.shape[2])

        # 逐次模拟求解监督 M-估计
        for t in range(X_labeled.shape[2]):
            X = X_labeled[:, :, t]
            Y = Y_labeled[:, :, t]
            best_lambda, beta_hat_ones = self.solve_logistic_regression_single(
                X, Y, initial_value, {'maxiter': max_iter, 'gtol': tolerance}, lambda_range, numFolds
            )
            beta_hat[:, t] = beta_hat_ones.ravel()
            best_lambda_hat[t] = best_lambda

        # 计算基础评估指标
        # Bias：估计量均值偏差（对所有模拟求和后取均值）
        Bias = (1 / X_labeled.shape[2]) * np.sum(beta_hat - beta_star, axis=1).reshape(-1, 1)
        # SE：估计量的经验标准差（去均值后计算 RMS）
        SE = np.sqrt(np.mean((beta_hat - np.mean(beta_hat, axis=1, keepdims=True)) ** 2, axis=1)).reshape(-1, 1)
        # MSE：均方误差（方差 + 偏差²）
        MSE = np.mean((beta_hat - beta_star) ** 2, axis=1).reshape(-1, 1)

        # 初始化 SSE 和 CP 矩阵（若不计算 CP，默认值为 1）
        SSE_every = np.ones((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        CP_every = np.ones((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        SE_SSE_ratio_every = np.zeros_like(SSE_every)
        SE_SSE_ratio_mean_every = np.zeros_like(SSE_every)

        if CP_if == 1:
            # --- 计算渐近标准误 SSE 和置信区间覆盖率 CP ---
            for t in range(X_labeled.shape[2]):
                beta_hat_one_simulation = beta_hat[:, t].reshape(-1, 1)
                X = X_labeled[:, :, t]
                Y = Y_labeled[:, :, t]

                # 计算得分函数（一阶偏导数矩阵）
                Partial_l = self.model_spec.score(
                    beta_hat_one_simulation, X, Y, best_lambda_hat[t]
                )  # shape: [n, p+1]

                # 样本协方差 V（三明治公式的"肉"）
                V = (1 / Partial_l.shape[0]) * (Partial_l - np.mean(Partial_l, axis=0)).T @ (
                            Partial_l - np.mean(Partial_l, axis=0))
                # Hessian 矩阵 M（三明治公式的"面包"）
                M = self.model_spec.hessian(beta_hat_one_simulation, X, Y, best_lambda_hat[t])
                # 三明治方差估计：SIGMA = (1/n) * M^{-1} V M^{-1}
                SIGMA = stable_sandwich(M, V) / Partial_l.shape[0]

                # 计算 SSE 和 95% 置信区间覆盖率
                BIAS = np.zeros(SIGMA.shape[0])  # 监督估计无偏差修正
                SSE_every[:, t] = np.sqrt(np.diag(SIGMA))
                up = (beta_hat_one_simulation.ravel() - BIAS) + 1.96 * SSE_every[:, t]
                down = (beta_hat_one_simulation.ravel() - BIAS) - 1.96 * SSE_every[:, t]
                CP_every[:, t] = (beta_star.ravel() >= down) & (beta_star.ravel() <= up)
                SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
                SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones(SE.shape[0])

            # 移除含 NaN 的列（数值不稳定的模拟次）
            nan_cols = np.any(np.isnan(SSE_every), axis=0)
            SSE_every = SSE_every[:, ~nan_cols]
            CP_every = CP_every[:, ~nan_cols]
            SE_SSE_ratio_every = SE_SSE_ratio_every[:, ~nan_cols]
            SE_SSE_ratio_mean_every = SE_SSE_ratio_mean_every[:, ~nan_cols]

            # 计算有效模拟的均值指标
            SSE = np.mean(SSE_every, axis=1).reshape(-1, 1)
            CP = np.mean(CP_every, axis=1).reshape(-1, 1)
            SE_SSE_ratio = np.mean(SE_SSE_ratio_every, axis=1).reshape(-1, 1)
            SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every, axis=1).reshape(-1, 1)

            # 构建完整评估指标 DataFrame（含 SSE/CP）
            MSE_MEAN = np.mean(MSE) * np.ones_like(MSE)
            BIAS_MEAN = np.mean(np.abs(Bias)) * np.ones_like(Bias)
            SE_MEAN = np.mean(SE) * np.ones_like(SE)
            SSE_MEAN = np.mean(SSE) * np.ones_like(SSE)
            CP_MEAN = np.mean(CP) * np.ones_like(CP)

            import pandas as pd
            Evaluate = pd.DataFrame({
                'Bias': Bias.ravel(),
                'BIAS_MEAN': BIAS_MEAN.ravel(),
                'SE': SE.ravel(),
                'SE_MEAN': SE_MEAN.ravel(),
                'MSE': MSE.ravel(),
                'MSE_MEAN': MSE_MEAN.ravel(),
                'SSE': SSE.ravel(),
                'SSE_MEAN': SSE_MEAN.ravel(),
                'SE_SSE_ratio': SE_SSE_ratio.ravel(),
                'SE_SSE_ratio_mean': SE_SSE_ratio_mean.ravel(),
                'CP': CP.ravel(),
                'CP_MEAN': CP_MEAN.ravel()
            })
        else:
            # 构建基础评估指标 DataFrame（不含 SSE/CP）
            MSE_MEAN = np.mean(MSE) * np.ones_like(MSE)
            BIAS_MEAN = np.mean(np.abs(Bias)) * np.ones_like(Bias)
            SE_MEAN = np.mean(SE) * np.ones_like(SE)
            import pandas as pd
            Evaluate = pd.DataFrame({
                'Bias': Bias.ravel(),
                'BIAS_MEAN': BIAS_MEAN.ravel(),
                'SE': SE.ravel(),
                'SE_MEAN': SE_MEAN.ravel(),
                'MSE': MSE.ravel(),
                'MSE_MEAN': MSE_MEAN.ravel()
            })
        return beta_hat, Evaluate, best_lambda_hat

    def solve_logistic_regression_single(self, X, Y, initial_value, options, lambda_range, numFolds):
        """
        单次监督 M-估计优化求解（带可选 L2 正则化）。

        本方法被 solve_logistic_regression 循环调用，对单次模拟数据求解参数估计。
        函数名保留 logistic 是为了兼容旧代码；实际目标函数由 model_spec 决定。
        当前实现将 best_lambda 固定为 0（无正则化），与 MATLAB 原代码一致。
        若需启用交叉验证选 lambda，可取消相关注释。

        参数：
            X (np.ndarray): 特征矩阵，shape=(n, p)
            Y (np.ndarray): 标签向量，shape=(n, 1)
            initial_value (np.ndarray): 参数初始值，shape=(p+1, 1)，
                若为 None 则初始化为全零向量
            options (dict): SciPy 优化器参数字典，含 'maxiter'（最大迭代次数）和 'gtol'（梯度收敛容忍度）
            lambda_range (np.ndarray): lambda 搜索范围（当前版本未使用，留作扩展接口）
            numFolds (int): 交叉验证折数（当前版本未使用，留作扩展接口）

        返回：
            best_lambda (float): 最优正则化参数（当前固定为 0.0）
            beta_hat_ones (np.ndarray): 最优参数估计，shape=(p+1, 1)
        """
        # 参数默认值处理
        if initial_value is None:
            initial_value = np.zeros((X.shape[1] + 1, 1))
        if options is None:
            options = {'maxiter': 500, 'gtol': 5e-3}

        # 当前版本直接设 lambda=0（对标 MATLAB 原代码），不做交叉验证
        # 若需启用正则化，可在此处添加 cross-validation 逻辑
        best_lambda = 0.0
        # 使用 SciPy BFGS 优化器求解，同时传入梯度（jac 函数分离传入）
        res = minimize(
            fun=lambda beta: self.objective_function_logistic(beta.reshape(-1, 1), X, Y, best_lambda)[0],
            x0=initial_value.ravel(),
            jac=lambda beta: self.objective_function_logistic(beta.reshape(-1, 1), X, Y, best_lambda)[1].ravel(),
            method='BFGS',
            options=options
        )
        beta_hat_ones = res.x.reshape(-1, 1)
        return best_lambda, beta_hat_ones


    def objective_function_logistic(self, beta, X_labeled, Y_labeled, lambda_reg):
        """
        监督 M-估计目标函数（损失 + L2 正则化），含梯度。

        本方法委托给 self.model_spec.loss_and_grad 实现，
        通过 model_spec 接口实现了模型无关性（可替换为任意 M-估计模型）。
        保留原方法名以兼容内部调用。

        参数：
            beta (np.ndarray): 参数向量，shape=(p+1, 1)
            X_labeled (np.ndarray): 标记特征矩阵，shape=(n, p)
            Y_labeled (np.ndarray): 标记标签向量，shape=(n, 1)
            lambda_reg (float): L2 正则化参数（λ），>0 时对参数施加 L2 惩罚

        返回：
            tuple: (loss, grad)
                loss (float): 目标函数值（负对数似然 + λ * ||beta||²）
                grad (np.ndarray): 梯度向量，shape=(p+1, 1)
        """
        return self.model_spec.loss_and_grad(beta, X_labeled, Y_labeled, lambda_reg)
