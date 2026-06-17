"""
SSLogistic.py
-------------
投影半监督 M-估计模块（Projection Semi-Supervised M-Estimation）

本模块实现了基于投影的半监督 M-估计方法（Projection SS），
与 DRESSSSLogistic.py 中的 c1=0 加权版本相对应。
核心区别：本模块调用 model_spec.ss_loss_and_grad 时传入 use_dress_c1=False，
因此 ModelSpec 内部使用 c1=n/(n+N)。DRESSSSLogistic.py 传入 True，对应 c1=0。

主要类：
    SSLogistic —— 投影半监督 M-估计，支持通过 model_spec 插入任意 M-估计模型

依赖：
    ModelSpec.py 中的 BaseModelSpec、LogisticModelSpec
"""

import numpy as np
import pandas as pd
from numpy.linalg import lstsq, det, inv, trace
import warnings
from scipy.optimize import minimize
from scipy.linalg import sqrtm, det
from scipy.stats import loguniform
from scipy.linalg import lstsq
from sklearn.covariance import MinCovDet
from scipy.linalg import orthogonal_procrustes

from ModelSpec import BaseModelSpec, LogisticModelSpec, stable_sandwich, stable_solve

# 忽略数值计算中的 RuntimeWarning（如矩阵奇异、溢出等）
warnings.filterwarnings('ignore')

class SSLogistic:
    """
    投影半监督 M-估计类（Projection Semi-Supervised M-Estimation）

    本类实现的是当前代码中的 c1=n/(n+N) 半监督加权版本。它与 DRESSSSLogistic
    共用同一套 model_spec.ss_loss_and_grad 实现，区别只是 use_dress_c1=False。
    因此，本类不单独实现新的统计模型；它负责组织模拟循环、构造辅助矩阵 Z、调用优化器、
    计算评估指标和统计 source 选择频次。

    通过 model_spec 参数，本类可以支持任意符合 BaseModelSpec 接口的 M-估计模型
    （如逻辑回归、线性回归等），实现了算法与模型的解耦。

    核心功能：
    1. 半监督参数估计（ss_logistic_regression）
    2. 基于 GBIC 的多项式阶数（alpha）选择
    3. 模型评估（Bias/SE/MSE/CP/ARE/RR 等指标）
    4. 未标记数据来源选择频率统计

    属性：
        default_tolerance (float): 默认优化收敛容忍度，5e-3
        default_max_iter (int): 默认最大迭代次数，500
        default_num_folds (int): 默认交叉验证折数，5
        default_lambda_range (np.ndarray): 默认 lambda 搜索范围（对数均匀分布100个点）
        random_seed (int): 随机数种子，保证结果可复现
        model_spec (BaseModelSpec): M-估计模型规范对象，默认为 LogisticModelSpec()
    """

    def __init__(self, random_seed=123, model_spec=None):
        """
        初始化 SSLogistic 类，设置默认优化参数和模型规范。

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
        # 记录 model_spec 是否由调用方显式传入。若调用方自己传入了模型规范，
        # 后续不会替调用方自动改求解器步长。
        self._user_provided_model_spec = model_spec is not None
        # 模型规范：默认使用逻辑回归；如传入自定义 model_spec，则使用传入值
        self.model_spec = model_spec if model_spec is not None else LogisticModelSpec()

    # ========================= 核心方法：投影半监督 M-估计 =========================
    def ss_logistic_regression(self, X_labeled, Y_labeled, X_unlabeled,
                                     tolerance=None, max_iter=None, initial_value=None, beta_star=None,
                                     Evaluate_supervised=None, result_summary=None, proposed_if=None,
                                     best_lambda_hat=None, lambda_range=None, numFolds=None, h_mu=None, h_sigma=None,
                                     bias_correction=False, intercept_from_supervised=False):
        """
        核心方法：基于投影的半监督 M-估计参数估计与模型评估。

        本方法对应 MATLAB 中的 DRESS_ss_logistic_regression 函数（投影版本）。
        流程：
          1. 参数默认值处理
          2. 逐次蒙特卡洛模拟，对每次模拟：
             a. 若无未标记数据 → 仅用标记数据优化（监督估计）
             b. 若有未标记数据 → GBIC 选 alpha → 构建多项式扩展特征 Z → 投影半监督优化
          3. 计算 Bias/SE/MSE/ARE/RR 等统计评估指标
          4. 若 proposed_if=1，额外计算渐近标准误（SSE）和置信区间覆盖率（CP）
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
            bias_correction (bool, optional): proposed_if=1 时是否在置信区间中心加入
                论文均值漂移偏差修正项。默认 False；可打开做论文偏差项敏感性分析。
            intercept_from_supervised (bool, optional): 是否将半监督估计的截距替换为同次模拟
                的监督截距。默认 False，保持完整半监督估计器；True 仅用于有限样本敏感性分析。

        返回：
            beta_hat (np.ndarray): 估计的参数矩阵，shape=(p+1, T)
            Evaluate (pd.DataFrame): 模型评估指标表，含以下列：
                Bias, BIAS_MEAN, SE, SE_MEAN, SSE, SSE_MEAN, ARE, ARE_MEAN,
                MSE, MSE_MEAN, RR, MRR, SE_SSE_ratio, SE_SSE_ratio_mean, CP, CP_MEAN
            select_time (np.ndarray or None): 未标记数据选择次数统计矩阵，
                shape=(len(h_mu), len(h_sigma))；若 h_mu 或 h_sigma 为空则返回 None
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

        # proposed_if=0 是“所有异质无标签 source 直接合并”的 benchmark。
        # 当 N 很大时 c1=n/(n+N) 接近 0，权重尺度和 DRESS 的 c1=0 很接近；
        # 若默认严格 WGD 仍用 proposed 路径的较大步长，少数模拟会被坏权重推离监督解，
        # 进而把 MSE/MRR 放大到不合理量级。这里仅在默认逻辑回归规范下临时调小步长；
        # proposed_if=1 的筛选后 proposed 方法仍保留原默认步长。
        benchmark_wgd_step_old = None
        if (proposed_if == 0
                and not self._user_provided_model_spec
                and isinstance(self.model_spec, LogisticModelSpec)
                and getattr(self.model_spec, "ss_solver", None) == "strict_wgd"
                and getattr(self.model_spec, "wgd_step_size", 0.0) > 0.001):
            benchmark_wgd_step_old = self.model_spec.wgd_step_size
            self.model_spec.wgd_step_size = 0.001

        # 历史 BFGS 配置：保留给旧接口/兼容路径使用，当前具体求解器由 model_spec 决定。
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
                # 直接用标记数据的损失函数优化，不引入投影修正项
                # 通过 model_spec.solve_supervised 求解（无未标签数据 → 退化为监督）。
                # 默认 BFGS；LinearModelSpec 重写为闭式 OLS。
                beta_hat[:, t] = self.model_spec.solve_supervised(
                    X, Y,
                    lambda_reg=best_lambda_hat[0, t],
                    initial_value=initial_value,
                    tolerance=tolerance,
                    max_iter=max_iter,
                ).ravel()
            else:
                # --- 情形 B：有未标记数据，使用投影半监督估计 ---

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

                # Step B4：带扩展特征的投影半监督目标函数优化
                # 调用 model_spec.solve_semi_supervised 求解（c1=n/(n+N)，投影模式）。
                # 具体求解算法由 model_spec 决定：
                # LogisticModelSpec 当前复现实验默认走 BFGS；strict_wgd/newton 路径可用于诊断；
                # LinearModelSpec 重写为闭式加权 LS，避免负权重下 BFGS 沿凹方向发散。
                #
                # intercept_from_supervised=True：分块解耦（仅用于敏感性分析）
                #   - 截距方向 SS 没有方差缩减（A_0[0]=0 且 1∈span(Z) 让权重对截距贡献被抵消），
                #     反而引入随机权重的有限样本方差污染 → 截距 MSE/CP 比监督差
                #   - 斜率方向 SS 在 Z = [1, X, X²] 等基下能捕获 misspec 残差结构，
                #     方差缩减显著 → 斜率 MSE/CP 优于监督
                #   混合策略：截距走监督路径、斜率走 SS 路径，各取所长。
                beta_hat[:, t] = self.model_spec.solve_semi_supervised(
                    X, Y, Z_labeled, Z_unlabeled,
                    lambda_reg=best_lambda_hat[0, t],
                    use_dress_c1=False,          # 投影 SS 用论文推荐的 c1=n/(n+N)
                    initial_value=initial_value,
                    tolerance=tolerance,
                    max_iter=max_iter,
                    intercept_from_supervised=intercept_from_supervised,
                ).ravel()

        beta_hat = self._sanitize_beta_hat(
            beta_hat, X_labeled, Y_labeled, best_lambda_hat,
            initial_value, tolerance, max_iter, label="SS"
        )

        # ===================== 5. 计算评估指标 =====================
        # 标记哪些模拟次 t 没有未标记数据（退化为监督估计）
        position = np.zeros(X_labeled.shape[2])
        for t in range(X_labeled.shape[2]):
            X2 = X_unlabeled[t] if t < len(X_unlabeled) else np.array([])
            position[t] = 1 if X2.size == 0 else 0
        position_idx = np.where(position == 1)[0]  # 无未标记数据的模拟索引

        # 从评估中移除无未标记数据的模拟列（保持半监督评估的纯粹性）
        beta_hat_2 = np.delete(beta_hat, position_idx, axis=1)
        no_active_semisup = beta_hat_2.shape[1] == 0
        if no_active_semisup:
            # Proposed 筛选可能在强信号/大 n0 下一个 source 都不选。此时估计器退化为
            # 监督估计，评估也应按监督路径计算，而不是对空矩阵求均值。
            beta_hat_2 = beta_hat.copy()
            position_idx = np.array([], dtype=int)

        # 计算偏差（Bias）：估计量均值与真实值之差
        Bias = (1 / beta_hat_2.shape[1]) * np.sum(beta_hat_2 - beta_star, axis=1, keepdims=True)
        # 计算标准误差（SE）：估计量的经验标准差，反映估计量的变异性
        SE = np.sqrt(np.mean((beta_hat_2 - np.mean(beta_hat_2, axis=1, keepdims=True)) ** 2, axis=1, keepdims=True))
        # 计算均方误差（MSE）：偏差平方 + 方差，综合评估估计误差
        MSE = np.mean((beta_hat_2 - beta_star) ** 2, axis=1, keepdims=True)

        # 计算渐近相对效率（ARE）：监督方法 SE² / 半监督方法 SE²，>1 表示半监督更高效
        ARE = (Evaluate_supervised['SE'].values ** 2) / (SE.ravel() ** 2)
        # 计算相对减少率（RR）：（监督 MSE - 半监督 MSE）/ 监督 MSE，反映每个分量的 MSE 改进
        RR = (Evaluate_supervised['MSE'].values - MSE.ravel()) / Evaluate_supervised['MSE'].values
        # 计算平均相对减少率（MRR）：基于所有分量 MSE 均值的整体改进
        a = np.mean(Evaluate_supervised['MSE'].values)
        b = np.mean(MSE)
        MRR = ((a - b) / a) * np.ones_like(MSE)
        # 全局均值化指标（便于表格展示，所有行填同一个均值）
        MMSE = np.mean(MSE) * np.ones_like(MSE)
        MBIAS = np.mean(np.abs(Bias)) * np.ones_like(MSE)

        # ===================== 6. 计算渐近标准误 SSE 和置信区间覆盖率 CP =====================
        # 初始化存储每次模拟的 SSE 和 CP
        SSE_every = np.zeros((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        CP_every = np.ones((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        SE_SSE_ratio_every = np.zeros((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        SE_SSE_ratio_mean_every = np.zeros((X_labeled.shape[1] + 1, X_labeled.shape[2]))

        if no_active_semisup:
            for t in range(X_labeled.shape[2]):
                beta_hat_one = beta_hat[:, t].reshape(-1, 1)
                X_t = X_labeled[:, :, t]
                Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
                Partial_l = self.model_spec.score(
                    beta_hat_one, X_t, Y_t, best_lambda_hat[0, t]
                )
                Partial_l_c = Partial_l - np.mean(Partial_l, axis=0, keepdims=True)
                V = (Partial_l_c.T @ Partial_l_c) / Partial_l.shape[0]
                M = self.model_spec.hessian(beta_hat_one, X_t, Y_t, best_lambda_hat[0, t])
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                    SSE_every[:, t] = np.nan
                    CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan
                    SE_SSE_ratio_mean_every[:, t] = np.nan
                    continue
                SIGMA = stable_sandwich(M, V) / Partial_l.shape[0]
                SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()
                up = beta_hat_one + 1.96 * SSE_every[:, t].reshape(-1, 1)
                down = beta_hat_one - 1.96 * SSE_every[:, t].reshape(-1, 1)
                CP_every[:, t] = ((beta_star >= down) & (beta_star <= up)).ravel()
                SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
                SE_SSE_ratio_mean_every[:, t] = (
                    np.sum(SE) / np.sum(SSE_every[:, t])
                ) * np.ones_like(SE).ravel()

            valid_cols = ~np.any(np.isnan(SSE_every), axis=0)
            if np.any(valid_cols):
                SSE = np.mean(SSE_every[:, valid_cols], axis=1, keepdims=True)
                CP = np.sum(CP_every[:, valid_cols], axis=1, keepdims=True) / valid_cols.sum()
                SE_SSE_ratio = np.mean(SE_SSE_ratio_every[:, valid_cols], axis=1, keepdims=True)
                SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every[:, valid_cols], axis=1, keepdims=True)
            else:
                SSE = np.full_like(SE, np.nan)
                CP = np.full_like(SE, np.nan)
                SE_SSE_ratio = np.full_like(SE, np.nan)
                SE_SSE_ratio_mean = np.full_like(SE, np.nan)

        elif proposed_if == 1:
            # --- 改进方法（proposed）：使用精确的渐近方差公式 ---
            for t in range(X_labeled.shape[2]):
                if position[t] == 1:
                    continue  # 跳过无未标记数据的模拟次

                beta_hat_one = beta_hat[:, t].reshape(-1, 1)   # 第 t 次模拟的参数估计
                res_sum = result_summary[t] if t < len(result_summary) else {}

                # 从结果摘要中提取 sigma 分量数 q（用于带宽计算）
                q = len(res_sum.get('Tree_lambda_sigma_one_simulation', {}).get('index_sigma', [[]])[0]) - 1

                # 计算选入的未标记数据总量 n_A（所有被选中数据源的样本量之和）
                fields = list(
                    res_sum.get('Tree_lambda_mu_one_simulation', {}).get('train_Z_unlabeled_one_simulation', {}).keys())
                A_number = len(res_sum.get('select_index', []))
                n_A = 0
                for aa in range(A_number):
                    if aa < len(res_sum.get('select_index', [])):
                        field_idx = res_sum['select_index'][aa] - 1  # 转换为 0-based 索引
                        if field_idx < len(fields):
                            Z_un = res_sum['Tree_lambda_mu_one_simulation']['train_Z_unlabeled_one_simulation'][
                                fields[field_idx]]
                            n_A += Z_un.shape[0]  # 累加各数据源的样本数

                # c_star：标记数据占总数据（标记+未标记）的比例
                c_star = X_labeled.shape[0] / (X_labeled.shape[0] + n_A)
                # rho：标记数据量与未标记数据量的比值
                rho = X_labeled.shape[0] / n_A if n_A > 0 else 0

                # 计算得分函数 Partial_l（损失函数对 beta 的一阶偏导数）
                # shape = (n, p+1)，每行为一个样本的得分向量
                X_t = X_labeled[:, :, t]
                Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
                Partial_l = self.model_spec.score(
                    beta_hat_one, X_t, Y_t, best_lambda_hat[0, t]
                )  # [n, p+1]

                # 用最小二乘回归将得分函数投影到辅助特征空间 Z_one
                # gamma_hat = (Z'Z)^{-1} Z' Partial_l，投影系数
                Z_one = res_sum.get('Tree_lambda_mu_one_simulation', {}).get('train_Z_labeled_one_simulation',
                                                                             np.array([]))
                # 防御：线性模型 Partial_l 可能因 Xβ 溢出含 inf
                if not (np.all(np.isfinite(Partial_l)) and (Z_one.size == 0 or np.all(np.isfinite(Z_one)))):
                    SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 SS-main-proposed] t={t}: Partial_l/Z 含非有限值; 该次仿真已置 NaN")
                    continue
                # ============================================================
                # 渐近方差 V 估计（论文 Theorem 3 原始三项分解形式）
                # ------------------------------------------------------------
                #   V̂ = V̂₁ + c*² · V̂₂ + (1-c*)² ρ_Â · V̂₂'
                #
                #   V̂₁  = (1/n_l) Σ_i δ̂_i^OOF (δ̂_i^OOF)^T                [K-fold 残差]
                #   V̂₂  = (1/n_l) Σ_i (Â^T Z̃_i^l)(Â^T Z̃_i^l)^T            [labeled 中心化]
                #   V̂₂' = Σ_{src k} (n_k/N_A) · (1/n_k) Σ_{j∈k}
                #         (Â^T Z̃_j^k)(Â^T Z̃_j^k)^T                       [per-source 中心化]
                #
                # 其中 Z̃_i^l = Z_i - mean(Z^l)，Z̃_j^k = Z_j - mean(Z^k)。
                # 多源场景下 labeled 与各 unlabeled source 的 z 分布不同，
                # 因此分别估各自的二阶矩、且各 source 单独中心化以剔除均值差异。
                # V̂₁ 已优化为「K-fold OOF + 普通样本二阶矩」（去掉了旧版的
                # FMCD/OGK 鲁棒协方差混合，那会让 V₁ 在正态尾部样本下偏）。
                # ============================================================
                if Z_one.size > 0:
                    gamma_hat = np.linalg.lstsq(Z_one, Partial_l, rcond=None)[0]

                    # V̂₁：委托给 model_spec.estimate_v1（逻辑回归默认 plain plug-in）。
                    # crossfit / blend / ogk 仅作为有限样本敏感性分析，不作为默认校准。
                    V1 = self.model_spec.estimate_v1(Z_one, Partial_l, K=5, seed=t)

                    # V̂₂：labeled 整体中心化二阶矩 × c*²
                    Z_centered_l = Z_one - np.mean(Z_one, axis=0)
                    AZ_l_c = Z_centered_l @ gamma_hat
                    V2 = (c_star ** 2) * (AZ_l_c.T @ AZ_l_c) / Z_one.shape[0]

                    # V̂₂'：每个 informative source 单独中心化，按样本量加权求和
                    V2_prime = np.zeros_like(V2)
                    for aa in range(A_number):
                        if aa < len(res_sum.get('select_index', [])):
                            field_idx = res_sum['select_index'][aa] - 1
                            if field_idx < len(fields):
                                Z_un = res_sum['Tree_lambda_mu_one_simulation'][
                                    'train_Z_unlabeled_one_simulation'][fields[field_idx]]
                                Z_un_c = Z_un - np.mean(Z_un, axis=0)
                                AZ_un_c = Z_un_c @ gamma_hat
                                n_k = Z_un.shape[0]
                                V2_prime += (n_k / n_A) * (AZ_un_c.T @ AZ_un_c) / n_k
                    V2_prime = ((1 - c_star) ** 2 * rho) * V2_prime

                    V = V1 + V2 + V2_prime
                else:
                    # 无辅助特征：退化为监督
                    V = (Partial_l.T @ Partial_l) / Partial_l.shape[0]
                    gamma_hat = np.zeros((1, 1))

                # 计算 Hessian 矩阵 M（损失函数对 beta 的二阶偏导数均值）
                # M 用于将方差 V 变换到参数空间（三明治方差估计的核心）
                M = self.model_spec.hessian(beta_hat_one, X_t, Y_t, best_lambda_hat[0, t])

                # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                    SSE_every[:, t] = np.nan
                    CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan
                    SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 SS-main-proposed] t={t}: 渐近方差含非有限值; 该次仿真已置 NaN")
                    continue

                # 三明治方差公式：SIGMA = (1/n) * M^{-1} V M^{-1}
                # 这是 M-估计渐近方差的标准形式
                scale_n = (Z_one @ gamma_hat).shape[0] if Z_one.size > 0 else 1
                SIGMA = stable_sandwich(M, V) / scale_n
                # SSE：渐近标准误，取 SIGMA 对角元素的平方根
                SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()

                if intercept_from_supervised:
                    # 截距若被替换为监督估计，则其方差也必须按监督 sandwich 估计。
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

                # ============================================================
                # 偏差修正项 BIAS（论文 Eq (11) 的均值漂移项）
                # ------------------------------------------------------------
                # 论文 Eq (11) 中 β̂ 的渐近分布有非零均值项 O_p((1-c*)√ρ √(qh_μ))，
                # 反推到 β̂ 原始尺度：BIAS = √(1/n)·M̂^{-1/2}·√ρ·(1-c*)·Σ_a w_a·√n_a·Γ̂^T Δ_a
                #
                # 默认不修正 CI 中心；若 bias_correction=True，则按论文均值漂移
                # 公式做有限样本敏感性分析。
                # ============================================================
                if bias_correction and Z_one.size > 0 and n_A > 0:
                    try:
                        M_half = sqrtm(M)
                        if np.iscomplexobj(M_half):
                            M_half = np.real(M_half)
                        BIAS = np.zeros_like(beta_hat_one)
                        for aa in range(A_number):
                            if aa < len(res_sum.get('select_index', [])):
                                field_idx = res_sum['select_index'][aa] - 1
                                if field_idx < len(fields):
                                    Z_un = res_sum['Tree_lambda_mu_one_simulation'][
                                        'train_Z_unlabeled_one_simulation'][fields[field_idx]]
                                    n_k = Z_un.shape[0]
                                    weight = np.sqrt(n_k / n_A)
                                    mean_diff = np.mean(Z_un, axis=0) - np.mean(Z_one, axis=0)
                                    source_bias = (weight * np.sqrt(n_k) * mean_diff @ gamma_hat).reshape(-1, 1)
                                    BIAS += source_bias
                        BIAS = (
                            np.sqrt(1 / Z_one.shape[0])
                            * stable_solve(
                                M_half, np.sqrt(rho) * (1 - c_star) * BIAS,
                                symmetrize=True,
                            )
                        )
                        bias_shrink = getattr(self.model_spec, "bias_shrink", 1.0)
                        BIAS = bias_shrink * BIAS
                        if intercept_from_supervised:
                            # 截距若被显式替换成监督估计，则该维度不应再使用 SS 偏差修正。
                            BIAS[0, 0] = 0
                        if not np.all(np.isfinite(BIAS)):
                            BIAS = np.zeros_like(beta_hat_one)
                    except Exception as exc:
                        print(f"[警告 SS-main-proposed] t={t}: 偏差项计算失败，已置零；原因：{exc}")
                        BIAS = np.zeros_like(beta_hat_one)
                else:
                    BIAS = np.zeros_like(beta_hat_one)

                # 计算置信区间覆盖率（Coverage Probability, CP）
                up = (beta_hat_one - BIAS) + 1.96 * SSE_every[:, t].reshape(-1, 1)
                down = (beta_hat_one - BIAS) - 1.96 * SSE_every[:, t].reshape(-1, 1)
                # 检查真实参数 beta_star 是否落在置信区间内
                CP_every[:, t] = ((beta_star >= down) & (beta_star <= up)).ravel()

                # 计算 SE/SSE 比率：实际标准差与渐近标准误的比值
                # 理想情况下应接近 1，偏离 1 说明渐近方差估计有偏
                SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
                SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones_like(SE).ravel()

            # 移除无未标记数据的模拟列（position_idx）
            SSE_every2 = np.delete(SSE_every, position_idx, axis=1)
            CP_every2 = np.delete(CP_every, position_idx, axis=1)
            SE_SSE_ratio_every2 = np.delete(SE_SSE_ratio_every, position_idx, axis=1)
            SE_SSE_ratio_mean_every2 = np.delete(SE_SSE_ratio_mean_every, position_idx, axis=1)
            valid_cols = ~np.any(np.isnan(SSE_every2), axis=0)

            # 对所有有效模拟次计算均值；含 NaN 的列说明渐近方差不可用，不能当作 0 参与平均。
            if np.any(valid_cols):
                SSE = np.mean(SSE_every2[:, valid_cols], axis=1, keepdims=True)
                CP = np.sum(CP_every2[:, valid_cols], axis=1, keepdims=True) / valid_cols.sum()
                SE_SSE_ratio = np.mean(SE_SSE_ratio_every2[:, valid_cols], axis=1, keepdims=True)
                SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every2[:, valid_cols], axis=1, keepdims=True)
            else:
                SSE = np.full_like(SE, np.nan)
                CP = np.full_like(SE, np.nan)
                SE_SSE_ratio = np.full_like(SE, np.nan)
                SE_SSE_ratio_mean = np.full_like(SE, np.nan)

        else:
            # --- 基准方法（proposed_if=0）：使用简化的渐近方差公式 ---
            for t in range(X_labeled.shape[2]):
                beta_hat_one = beta_hat[:, t].reshape(-1, 1)
                res_sum = result_summary[t] if t < len(result_summary) else {}

                # 提取关键比例参数
                q = len(res_sum.get('Tree_lambda_sigma_one_simulation', {}).get('index_sigma', [[]])[0]) - 1
                n_A = X_unlabeled[t].shape[0] if t < len(X_unlabeled) and X_unlabeled[t].size > 0 else 0
                c_star = X_labeled.shape[0] / (X_labeled.shape[0] + n_A) if n_A > 0 else 1
                rho = X_labeled.shape[0] / n_A if n_A > 0 else 0

                # 计算得分函数
                X_t = X_labeled[:, :, t]
                Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
                Partial_l = self.model_spec.score(
                    beta_hat_one, X_t, Y_t, best_lambda_hat[0, t]
                )  # [n, p+1]

                # 计算投影系数和残差
                Z_one = res_sum.get('Tree_lambda_mu_one_simulation', {}).get('train_Z_labeled_one_simulation',
                                                                             np.array([]))
                # 防御：线性模型 Partial_l 可能因 Xβ 溢出含 inf
                if not (np.all(np.isfinite(Partial_l)) and (Z_one.size == 0 or np.all(np.isfinite(Z_one)))):
                    SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 SS-main-benchmark] t={t}: Partial_l/Z 含非有限值; 该次仿真已置 NaN")
                    continue
                gamma_hat = np.linalg.lstsq(Z_one, Partial_l, rcond=None)[0] if Z_one.size > 0 else np.zeros(
                    (1, 1))
                e = Partial_l - Z_one @ gamma_hat if Z_one.size > 0 else Partial_l
                # 残差协方差 V1（简化版，直接用样本协方差）
                V1 = (1 / e.shape[0]) * e.T @ e
                # 投影方差 V2（合并了标记和未标记数据的贡献）
                V2 = (1 / (Z_one @ gamma_hat).shape[0]) * (Z_one @ gamma_hat).T @ (
                            Z_one @ gamma_hat) if Z_one.size > 0 else 0
                # 简化的缩放系数：(c_star² + (1 - c_star)² * rho)  # 论文 Theorem 1: V_rho = E[δδ^T] + (c1² + ρ(1-c1)²) E[A^T ZZ^T A]
                V2 = (c_star ** 2 + (1 - c_star) ** 2 * rho) * V2 if isinstance(V2, np.ndarray) else 0
                V = V1 + V2 if isinstance(V2, np.ndarray) else V1

                # 三明治方差估计
                M = self.model_spec.hessian(beta_hat_one, X_t, Y_t, best_lambda_hat[0, t])

                # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                    SSE_every[:, t] = np.nan
                    CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan
                    SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 SS-main-benchmark] t={t}: 渐近方差含非有限值; 该次仿真已置 NaN")
                    continue

                scale_n = (Z_one @ gamma_hat).shape[0] if Z_one.size > 0 else 1
                SIGMA = stable_sandwich(M, V) / scale_n

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
                BIAS = np.zeros_like(beta_hat_one)
                up = (beta_hat_one - BIAS) + 1.96 * SSE_every[:, t].reshape(-1, 1)
                down = (beta_hat_one - BIAS) - 1.96 * SSE_every[:, t].reshape(-1, 1)
                CP_every[:, t] = ((beta_star >= down) & (beta_star <= up)).ravel()

                # SE/SSE 比率
                SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
                SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones_like(SE).ravel()

            # 含 NaN 的列说明渐近方差不可用，不能先 nan_to_num，否则失败列会以 0 参与 SSE/CP。
            valid_cols = ~np.any(np.isnan(SSE_every), axis=0)
            if np.any(valid_cols):
                SSE = np.mean(SSE_every[:, valid_cols], axis=1, keepdims=True)
                CP = np.sum(CP_every[:, valid_cols], axis=1, keepdims=True) / valid_cols.sum()
                SE_SSE_ratio = np.mean(SE_SSE_ratio_every[:, valid_cols], axis=1, keepdims=True)
                SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every[:, valid_cols], axis=1, keepdims=True)
            else:
                SSE = np.full_like(SE, np.nan)
                CP = np.full_like(SE, np.nan)
                SE_SSE_ratio = np.full_like(SE, np.nan)
                SE_SSE_ratio_mean = np.full_like(SE, np.nan)

        # ===================== 7. 计算均值化统计量（方便表格展示） =====================
        MSE_MEAN = np.mean(MSE) * np.ones_like(MSE)           # 所有参数分量 MSE 的均值（标量广播为列向量）
        BIAS_MEAN = np.mean(np.abs(Bias)) * np.ones_like(MSE) # 所有参数分量绝对偏差的均值
        SE_MEAN = np.mean(SE) * np.ones_like(MSE)             # 所有参数分量 SE 的均值
        SSE_MEAN = np.mean(SSE) * np.ones_like(MSE) if 'SSE' in locals() else np.zeros_like(MSE)
        CP_MEAN = np.mean(CP) * np.ones_like(MSE) if 'CP' in locals() else np.zeros_like(MSE)
        ARE_MEAN = np.mean(ARE) * np.ones_like(MSE)           # ARE 的均值

        # ===================== 8. 统计各未标记数据源的选取频次（h_mu 非空时） =====================
        select_time = None
        if len(h_mu) > 0 and len(h_sigma) > 0:
            # select_time[i, j] 表示均值偏移 h_mu[i]、协方差偏移 h_sigma[j] 对应的数据源被选中的总次数
            select_time = np.zeros((len(h_mu), len(h_sigma)))
            for t in range(len(result_summary)):
                select_idx = result_summary[t].get('select_index', [])
                for idx in select_idx:
                    idx = int(idx)  # 确保为整数索引
                    # 将线性索引转换为 (mu_idx, sigma_idx) 二维索引
                    sigma_mod = np.mod(idx, len(h_sigma))
                    mu_idx = idx // len(h_sigma) if sigma_mod != 0 else (idx // len(h_sigma)) - 1
                    sigma_idx = len(h_sigma) - 1 if sigma_mod == 0 else sigma_mod - 1
                    if mu_idx < len(h_mu) and sigma_idx < len(h_sigma):
                        select_time[mu_idx, sigma_idx] += 1

        # ===================== 9. 构建评估结果 DataFrame =====================
        # 将所有评估指标组合为 DataFrame，便于后续分析和输出
        Evaluate_data = np.hstack([
            Bias, BIAS_MEAN, SE, SE_MEAN, SSE, SSE_MEAN, ARE.reshape(-1, 1), ARE_MEAN,
            MSE, MSE_MEAN, RR.reshape(-1, 1), MRR, SE_SSE_ratio, SE_SSE_ratio_mean,
            CP, CP_MEAN
        ]) if 'SSE' in locals() else np.hstack([
            Bias, BIAS_MEAN, SE, SE_MEAN, MSE, MSE_MEAN,
            ARE.reshape(-1, 1), ARE_MEAN, RR.reshape(-1, 1), MRR
        ])

        Evaluate_cols = [
            'Bias', 'BIAS_MEAN', 'SE', 'SE_MEAN', 'SSE', 'SSE_MEAN', 'ARE', 'ARE_MEAN',
            'MSE', 'MSE_MEAN', 'RR', 'MRR', 'SE_SSE_ratio', 'SE_SSE_ratio_mean', 'CP', 'CP_MEAN'
        ] if 'SSE' in locals() else [
            'Bias', 'BIAS_MEAN', 'SE', 'SE_MEAN', 'MSE', 'MSE_MEAN', 'ARE', 'ARE_MEAN', 'RR', 'MRR'
        ]
        Evaluate = pd.DataFrame(Evaluate_data, columns=Evaluate_cols)

        if benchmark_wgd_step_old is not None:
            self.model_spec.wgd_step_size = benchmark_wgd_step_old

        return beta_hat, Evaluate, select_time

    def _sanitize_beta_hat(self, beta_hat, X_labeled, Y_labeled, best_lambda_hat,
                           initial_value, tolerance, max_iter, label="SS", box=30.0):
        """
        在模拟指标汇总前处理非有限或明显发散的参数估计。

        逻辑回归半监督目标含投影权重，有限样本优化偶尔会给出极大的 beta。
        若不先处理，MSE/MRR 会被单个坏列严重污染；这里用同一次模拟的监督估计回退。
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
                                   n_simulations):
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

            # --- 计算得分函数 ---
            X_t = X_labeled[:, :, t]
            Y_t = Y_labeled[:, 0, t].reshape(-1, 1)
            Partial_l = self.model_spec.score(
                beta_hat_t, X_t, Y_t, best_lambda_hat[0, t]
            )  # shape: [n, p+1]

            # --- 投影回归：将得分函数投影到辅助特征空间 ---
            Z_t = res_sum_t['Tree_lambda_mu_one_simulation']['train_Z_labeled_one_simulation']

            # 防御：线性模型 Partial_l 可能因 Xβ 溢出含 inf
            _pl_fin = bool(np.all(np.isfinite(Partial_l)))
            _z_fin  = bool(Z_t.size == 0 or np.all(np.isfinite(Z_t)))
            if not (_pl_fin and _z_fin):
                SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                def _mx(a):
                    a = np.asarray(a); _m = np.isfinite(a)
                    return float(np.abs(a[_m]).max()) if _m.any() else float('nan')
                print(f"[警告 SS-proposed] t={t}: Partial_l_finite={_pl_fin}, Z_finite={_z_fin}, "
                      f"max|beta|={_mx(beta_hat_t):.3g}, max|X|={_mx(X_t):.3g}, "
                      f"max|Z|={_mx(Z_t):.3g}, max|Partial_l|={_mx(Partial_l):.3g}; 已置 NaN")
                continue
            # ============================================================
            # 渐近方差 V 估计（论文 Theorem 3 原始三项分解）
            #   V̂ = V̂₁ + c*² V̂₂ + (1-c*)² ρ_Â V̂₂'
            # V̂₁ 的具体估计方式由 model_spec 决定。LogisticModelSpec 当前默认
            # plain plug-in；crossfit / blend / ogk 仅作为有限样本敏感性分析。
            # V̂₂ = labeled 中心化；V̂₂' = per-source 中心化。
            # ============================================================
            gamma_hat = np.linalg.lstsq(Z_t, Partial_l, rcond=None)[0]

            # V̂₁：委托 model_spec.estimate_v1（按模型选估计方式）
            V1 = self.model_spec.estimate_v1(Z_t, Partial_l, K=5, seed=t)

            # V̂₂：labeled 中心化
            Z_centered_l = Z_t - np.mean(Z_t, axis=0)
            AZ_l_c = Z_centered_l @ gamma_hat
            V2 = (c_star ** 2) * (AZ_l_c.T @ AZ_l_c) / Z_t.shape[0]

            # V̂₂'：per-source 中心化 + 按样本量加权
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
            M = self.model_spec.hessian(beta_hat_t, X_t, Y_t, best_lambda_hat[0, t])

            # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
            if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                SSE_every[:, t] = np.nan
                CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan
                SE_SSE_ratio_mean_every[:, t] = np.nan
                print(f"[警告 SS-proposed] t={t}: 渐近方差含非有限值 "
                      f"(M_finite={bool(np.all(np.isfinite(M)))}, V_finite={bool(np.all(np.isfinite(V)))}, "
                      f"beta_finite={bool(np.all(np.isfinite(beta_hat_t)))}); 该次仿真已置 NaN")
                continue

            SIGMA = stable_sandwich(M, V) / (Z_t @ gamma_hat).shape[0]

            # --- 截距 SSE 用监督公式（与解耦 β̂[0] 一致）---
            # β̂[0] 走的是监督路径（intercept_from_supervised=True），其渐近方差是
            # supervised sandwich var (M⁻¹ V_sup M⁻¹/n)，而非 SS 缩减后的方差。
            # 用 Partial_l 的中心化样本二阶矩作 V_sup，即 score covariance。
            n_l = Partial_l.shape[0]
            Partial_l_c = Partial_l - np.mean(Partial_l, axis=0, keepdims=True)
            V_sup = (Partial_l_c.T @ Partial_l_c) / n_l
            SIGMA_sup = stable_sandwich(M, V_sup) / n_l

            # --- 计算偏差修正项 BIAS ---
            # BIAS 反映了各数据源均值漂移对估计量的影响
            M_half = sqrtm(M)   # M 的矩阵平方根，用于变换到标准化空间
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
            BIAS = getattr(self.model_spec, "bias_shrink", 1.0) * BIAS

            # --- 计算 SSE 和 95% 置信区间覆盖率 ---
            SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()
            # 截距维度用 supervised SSE 替换（与 β̂[0] 的来源一致）
            SSE_every[0, t] = float(np.sqrt(max(SIGMA_sup[0, 0], 0)))
            up = (beta_hat_t - BIAS) + 1.96 * SSE_every[:, t].reshape(-1, 1)
            down = (beta_hat_t - BIAS) - 1.96 * SSE_every[:, t].reshape(-1, 1)
            CP_every[:, t] = np.logical_and(beta_star >= down, beta_star <= up).ravel()

            # SE/SSE 比率
            SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
            SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones_like(SE).ravel()

        return SSE_every, CP_every, SE_SSE_ratio_every, SE_SSE_ratio_mean_every

    # ========================= 辅助方法：基准模式下计算 SSE/CP =========================
    def _calculate_sse_cp_benchmark(self, X_labeled, Y_labeled, X_unlabeled, beta_hat, beta_star, result_summary, best_lambda_hat,
                                    SE, n_simulations):
        """
        基准模式（proposed_if=0）下计算渐近标准误（SSE）和置信区间覆盖率（CP）。

        与 _calculate_sse_cp_proposed 的主要区别：
          - 方差公式更简化（V = V1 + V2，无 V2_prime）
          - V2 的缩放系数为 (c_star² + (1-c_star)² * rho)（合并了标记和未标记贡献）
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

            # 防御：线性模型 Partial_l 可能因 Xβ 溢出含 inf
            _pl_fin = bool(np.all(np.isfinite(Partial_l)))
            _z_fin  = bool(Z_t.size == 0 or np.all(np.isfinite(Z_t)))
            if not (_pl_fin and _z_fin):
                SSE_every[:, t] = np.nan; CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan; SE_SSE_ratio_mean_every[:, t] = np.nan
                def _mx(a):
                    a = np.asarray(a); _m = np.isfinite(a)
                    return float(np.abs(a[_m]).max()) if _m.any() else float('nan')
                print(f"[警告 SS-benchmark] t={t}: Partial_l_finite={_pl_fin}, Z_finite={_z_fin}, "
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

            # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
            if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                SSE_every[:, t] = np.nan
                CP_every[:, t] = np.nan
                SE_SSE_ratio_every[:, t] = np.nan
                SE_SSE_ratio_mean_every[:, t] = np.nan
                print(f"[警告 SS-benchmark] t={t}: 渐近方差含非有限值 "
                      f"(M_finite={bool(np.all(np.isfinite(M)))}, V_finite={bool(np.all(np.isfinite(V)))}, "
                      f"beta_finite={bool(np.all(np.isfinite(beta_hat_t)))}); 该次仿真已置 NaN")
                continue

            SIGMA = stable_sandwich(M, V) / (Z_t @ gamma_hat).shape[0]

            # 基准模式下无偏差修正
            BIAS = np.zeros_like(beta_hat_t)

            # 计算 SSE 和置信区间覆盖率
            SSE_every[:, t] = np.sqrt(np.diag(SIGMA)).ravel()
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
        转换规则：
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

                # 边界检查
                if 0 <= mu_idx < n_mu and 0 <= sigma_idx < n_sigma:
                    select_time[mu_idx, sigma_idx] += 1

        return select_time

    # ========================= 目标函数：投影半监督目标函数 =========================
    def ss_objective_function_logistic(self, beta, X_labeled, Y_labeled, Z_labeled=None, Z_unlabeled=None,
                                             lambda_reg=0):
        """
        c1=n/(n+N) 半监督目标函数包装器，供 scipy.optimize.minimize 调用。

        本方法委托给 self.model_spec.ss_loss_and_grad。传入 use_dress_c1=False 后，
        ModelSpec 内部会令 c1=n/(n+N)，其中 n 是有标签样本量，N 是无标签样本量。
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
            lambda_reg, use_dress_c1=False   # 当前文件约定：c1=n/(n+N)
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

            # Step 2：计算得分函数（在当前参数估计处的一阶偏导）
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

                # 计算投影系数 gamma_hat（将得分函数投影到 Z 空间）
                Z_t = Z[:, :, t]
                ZtZ = Z_t.T @ Z_t
                gamma_hat = stable_solve(ZtZ, Z_t.T @ Partial_l, symmetrize=True)
                e = Partial_l - Z_t @ gamma_hat  # 残差

                # 计算各分量的残差方差 sigma（自由度修正：n - p*a - 1）
                sigma = np.sum(e ** 2, axis=0) * (1 / (X_labeled.shape[0] - X_labeled.shape[1] * a - 1))

                # 对每个参数分量 j 计算 GBIC（累加到 GBIC[a-1, t]）
                for j in range((X.shape[1]+1)):
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
                    # term3：-log(det) 项（反映信息矩阵行列式，惩罚自由度）
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

                # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                    SSE_every[:, t] = np.nan
                    CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan
                    SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 SS-solve] t={t}: 渐近方差含非有限值; 该次仿真已置 NaN")
                    continue

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
        # 使用 SciPy BFGS 优化器求解，同时传入梯度（jac=True 等效于 jac 函数分离传入）
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

    # def kfold_residual_covariance(self, X, Y, K=5):
    #     """
    #     K折交叉验证计算残差协方差矩阵的平均值
    #     """
    #     # 输入检查
    #     n, p = X.shape
    #     n_y, d = Y.shape
    #     if n != n_y:
    #         raise ValueError("X和Y的样本数必须相同")
    #     if n < K:
    #         K = n  # 如果样本数小于折数，调整折数
    #
    #     # 随机划分K折
    #     np.random.seed(42)  # 固定随机种子确保可复现
    #     idx = np.random.permutation(n)
    #     fold_size = n // K
    #     fold_indices = []
    #     for k in range(K):
    #         start = k * fold_size
    #         end = (k + 1) * fold_size if k < K - 1 else n
    #         fold_indices.append(idx[start:end])
    #
    #     # 初始化协方差矩阵
    #     cov_avg = np.zeros((d, d))
    #
    #     # 遍历每一折
    #     for k in range(K):
    #         # 划分训练集和测试集
    #         test_idx = fold_indices[k]
    #         train_idx = np.setdiff1d(np.arange(n), test_idx)
    #
    #         if len(train_idx) == 0:
    #             continue
    #
    #         X_train = X[train_idx, :]
    #         Y_train = Y[train_idx, :]
    #         X_test = X[test_idx, :]
    #         Y_test = Y[test_idx, :]
    #
    #         # 最小二乘拟合（添加正则项）
    #         reg_term = 1e-6 * np.eye(X_train.shape[1])
    #         gamma_k = lstsq(X_train.T @ X_train + reg_term, X_train.T @ Y_train, rcond=None)[0]
    #         Y_pred = X_test @ gamma_k
    #         e = Y_test - Y_pred  # 残差
    #
    #         # 计算残差协方差
    #         if e.shape[0] > 1:
    #             cov_e = np.cov(e.T)
    #         else:
    #             cov_e = np.eye(d) * 1e-6
    #         cov_avg += cov_e
    #
    #     # 计算平均值
    #     cov_avg /= max(K, 1)
    #
    #     return cov_avg

    def kfold_residual_covariance(self, X, Y, K=5):
        """
        K 折交叉验证计算残差二阶矩 V₁ = E[δδ^T] 的无偏估计。

        对应论文 Theorem 3 Eq (12) 中 V₁ 项的样本估计。核心思想：
          · 把残差 δ̂_i = score_i - Â_{k(i)}^T Z_i 中的 Â_{k(i)} 在
            **不含样本 i 的 K-1 个折**上拟合 → δ̂_i 与 Â_{k(i)} 独立 →
            避免「同数据 fit γ 又用同数据算残差」造成 V₁ 系统下偏。
          · 各折 OOF 残差合并，统一计算二阶矩 (1/n) Σ δ̂_i δ̂_i^T。

        本版本去掉了早期 MATLAB 移植里的 FMCD/OGK 鲁棒协方差混合
        （用户反馈鲁棒协方差对正态尾部样本下偏，且不符合论文公式）。

        参数：
            X (np.ndarray): 辅助特征 Z，shape=(n, q)
            Y (np.ndarray): 得分函数 ∂L/∂β，shape=(n, d)（d=p+1）
            K (int): 折数，默认 5

        返回：
            V1_hat (np.ndarray): 残差二阶矩估计，shape=(d, d)
        """
        n, q = X.shape
        n_y, d = Y.shape
        if n != n_y:
            raise ValueError("X 和 Y 的样本量必须相同")
        if n < K:
            raise ValueError(f"样本量 n({n}) 必须大于等于折数 K({K})")

        # K 折随机划分（由调用方种子控制可复现性）
        idx = np.random.permutation(n)
        fold_indices = np.array_split(idx, K)

        # 每个样本 i 计算其 OOF 残差 δ̂_i
        delta_oof = np.zeros((n, d))
        for k in range(K):
            test_idx = fold_indices[k]
            train_idx = np.setdiff1d(np.arange(n), test_idx)
            X_tr, Y_tr = X[train_idx], Y[train_idx]
            try:
                gamma_k = lstsq(X_tr.T @ X_tr, X_tr.T @ Y_tr, cond=None)[0]
            except Exception:
                # 病态时回退到全数据 γ̂（罕见情况）
                gamma_k = lstsq(X.T @ X, X.T @ Y, cond=None)[0]
            delta_oof[test_idx] = Y[test_idx] - X[test_idx] @ gamma_k

        # 残差二阶矩（V₁ = E[δδ^T] 的样本估计，不中心化）
        V1_hat = (delta_oof.T @ delta_oof) / n
        return V1_hat

    def ogk_covariance(self, data):
        """
        手动实现正交型 Gnanadesikan-Kettenring（OGK）鲁棒协方差估计。

        OGK 方法通过正交变换将数据投影到低相关空间，
        然后用中位数绝对偏差（MAD）估计各方向的方差，最后变换回原空间。
        相比 MCD，OGK 在高维数据上更稳定。

        算法步骤：
          1. 用中位数（而非均值）中心化数据，提高对异常值的鲁棒性
          2. 通过正交 Procrustes 变换找到最优正交矩阵 Q
          3. 在变换空间中用 MAD 估计各方向方差
          4. 逆变换回原始空间
          5. 对结果做正定化处理

        参数：
            data (np.ndarray): 输入数据矩阵，shape=(n, d)，n=样本数，d=维度

        返回：
            cov (np.ndarray): 鲁棒协方差估计矩阵，shape=(d, d)，保证正半定

        注意：
            若 n <= 1，直接返回单位矩阵乘以 1e-6（避免协方差矩阵退化）
        """
        n, d = data.shape
        if n <= 1:
            return np.eye(d) * 1e-6  # 样本数不足，返回近似零矩阵

        # 步骤 1：用中位数中心化（比均值对异常值更鲁棒）
        center = np.median(data, axis=0)
        data_centered = data - center

        # 步骤 2：正交 Procrustes 变换，找到将去均值数据最优对齐到单位矩阵的正交矩阵 Q
        Q, _ = orthogonal_procrustes(data_centered, np.eye(d))
        transformed = data_centered @ Q  # 在变换空间中，各方向近似不相关

        # 步骤 3：用 MAD 估计各正交方向的方差
        # MAD（中位数绝对偏差）是对标准差的鲁棒替代
        # 乘以 1.4826 是将 MAD 转换为正态分布下的标准差估计的一致性系数
        mad = np.median(np.abs(transformed), axis=0) * 1.4826
        mad[mad < 1e-6] = 1e-6  # 避免方差为零（奇异矩阵）
        cov_rot = np.diag(mad ** 2)  # 对角方差矩阵

        # 步骤 4：逆变换回原始空间
        cov = Q @ cov_rot @ Q.T

        # 步骤 5：确保协方差矩阵正定（数值稳定性处理）
        cov = (cov + cov.T) / 2  # 对称化（消除浮点误差导致的轻微非对称）
        min_eig = np.min(np.real(np.linalg.eigvals(cov)))
        if min_eig < 0:
            # 若最小特征值为负（非正定），通过平移使矩阵正定
            cov -= 1.1 * min_eig * np.eye(d)

        return cov
