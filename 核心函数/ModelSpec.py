"""
ModelSpec.py — M-估计模型规范抽象层
=====================================================
提供通用 M-估计框架所需的六个核心接口：

  估计接口（MstMdsp / SSLogistic / DRESSSSLogistic 使用）：
  1. loss_and_grad      — 目标函数值 + 梯度（供 scipy.optimize 使用）
  2. score              — 每样本评分函数 Partial_l（含正则项）[n, d]
  3. hessian            — Hessian 矩阵 M（含正则项）[d, d]
  4. ss_loss_and_grad   — 半监督加权目标函数 + 梯度（供 DRESS/投影 SS 使用）

  数据生成接口（DataGenerator 使用）：
  5. generate_y         — 从真实 DGP 生成响应变量 Y（可与估计模型不同，制造模型误设）
  6. default_true_value — 返回 p 维特征下默认的 DGP 真实参数

  辅助属性：
  7. dgp_feature_dim(p) — 返回 DGP 使用的特征向量维度（含截距）

使用方式：
  - 默认模型：不传 model_spec 参数，三个核心类自动使用 LogisticModelSpec
  - 切换到线性回归：
        from ModelSpec import LinearModelSpec
        dg  = DataGenerator(model_spec=LinearModelSpec())
        mst = MstMdsp(model_spec=LinearModelSpec())
  - 自定义模型：继承 BaseModelSpec，实现六个抽象方法
"""

import numpy as np
from abc import ABC, abstractmethod
from numpy.linalg import pinv
from scipy.optimize import minimize
from scipy.special import expit
from scipy.linalg import orthogonal_procrustes


def stable_solve(A: np.ndarray, B: np.ndarray, symmetrize: bool = False) -> np.ndarray:
    """
    Solve A X = B without explicitly forming A^{-1}.

    The normal path uses np.linalg.solve.  If A is singular or numerically
    problematic, fall back to least squares, and only use the Moore-Penrose
    product as the last resort.
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    if symmetrize and A.ndim == 2 and A.shape[0] == A.shape[1]:
        A = 0.5 * (A + A.T)

    try:
        X = np.linalg.solve(A, B)
        if np.all(np.isfinite(X)):
            return X
    except np.linalg.LinAlgError:
        pass

    try:
        X = np.linalg.lstsq(A, B, rcond=None)[0]
        if np.all(np.isfinite(X)):
            return X
    except np.linalg.LinAlgError:
        pass

    return pinv(A) @ B


def stable_sandwich(M: np.ndarray, V: np.ndarray, symmetrize: bool = True) -> np.ndarray:
    """
    Compute M^{-1} V M^{-1} through two linear solves.

    For symmetric M, this equals solve(M, V) followed by a right-side solve.
    The result is symmetrized to remove small numerical asymmetry.
    """
    left = stable_solve(M, V, symmetrize=symmetrize)
    out = stable_solve(np.asarray(M).T, left.T, symmetrize=symmetrize).T
    return 0.5 * (out + out.T)


def stable_left_product(A: np.ndarray, B: np.ndarray,
                        symmetrize: bool = False) -> np.ndarray:
    """Return A^{-1} B via stable_solve; kept for formula readability."""
    return stable_solve(A, B, symmetrize=symmetrize)


# =============================================================================
# 基类
# =============================================================================

class BaseModelSpec(ABC):
    """
    M-估计模型规范基类（抽象类）。

    所有子类需实现六个方法。X 均为原始特征矩阵（不含截距列）。
    截距由各子类自行决定是否添加（通常通过调用 add_intercept 辅助方法实现）。

    接口分为两组：
    - 估计接口（loss_and_grad / score / hessian / ss_loss_and_grad）：
        由 MstMdsp / SSLogistic / DRESSSSLogistic 调用，驱动参数优化过程。
    - 数据生成接口（generate_y / default_true_value）：
        由 DataGenerator 调用，可以与估计模型不同（制造有意义的模型误设），
        从而模拟真实场景中模型永远不完全正确的情况。
    """

    # ------------------------------------------------------------------
    # 辅助方法：添加截距列
    # ------------------------------------------------------------------
    def add_intercept(self, X: np.ndarray) -> np.ndarray:
        """
        在特征矩阵 X 左侧拼接全 1 截距列，构造增广设计矩阵。

        参数
        ----
        X : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。

        返回
        ----
        X_aug : np.ndarray，形状 [n, p+1]
            左侧追加全 1 列后的增广特征矩阵，第 0 列为截距项。
        """
        n = X.shape[0]
        return np.hstack([np.ones((n, 1)), X])  # 在第 0 列插入全 1，对应截距参数 β_0

    def build_z_matrix(self, X: np.ndarray, alpha) -> np.ndarray:
        """
        构造论文中的原始多项式辅助特征：
            Z = [1, X, X^2, ..., X^alpha].

        `z_feature_limit` 只用于调试高维情形：若设置为 k，则 Z 只使用
        前 k 个 X 坐标；工作模型的 beta 维度仍然是完整 p 维。
        """
        alpha = int(np.asarray(alpha).ravel()[0])
        alpha_cap = getattr(self, "max_auxiliary_alpha", None)
        if alpha_cap is not None:
            alpha = min(alpha, int(alpha_cap))
        X = np.asarray(X, dtype=float)
        z_limit = getattr(self, "z_feature_limit", None)
        if z_limit is not None:
            X = X[:, :int(z_limit)]
        n, p = X.shape
        Z = np.empty((n, 1 + p * alpha))
        Z[:, 0] = 1.0
        for a in range(1, alpha + 1):
            Z[:, 1 + (a - 1) * p: 1 + a * p] = X ** a
        return Z

    def semi_supervised_weight_term(self, Z_labeled: np.ndarray,
                                    Z_unlabeled: np.ndarray) -> np.ndarray:
        """
        通过解线性方程计算半监督权重中的投影项。

        原公式 mean(Z_unlabeled) @ pinv(Z_labeled'Z_labeled/n) @ Z_labeled'
        等价于先解 S alpha = mean(Z_unlabeled)，再返回 Z_labeled @ alpha。
        这样避免显式形成伪逆矩阵；若 S 奇异，则用最小二乘再退到 pinv 兜底。
        """
        Z_labeled = np.asarray(Z_labeled, dtype=float)
        Z_unlabeled = np.asarray(Z_unlabeled, dtype=float)
        n = Z_labeled.shape[0]
        S = (Z_labeled.T @ Z_labeled) / n
        S = 0.5 * (S + S.T)
        weight_ridge = float(getattr(self, "weight_ridge", 0.0))
        if weight_ridge > 0:
            penalty = np.eye(S.shape[0])
            penalty[0, 0] = 0.0
            S = S + weight_ridge * penalty
        rhs = np.mean(Z_unlabeled, axis=0).reshape(-1, 1)
        alpha = stable_solve(S, rhs, symmetrize=True)
        return (Z_labeled @ alpha).ravel()

    def _effective_lambda(self, lambda_reg: float) -> float:
        """
        Return the regularization used by model formulas.

        Most models use exactly the lambda supplied by the caller. LinearModelSpec
        overrides this so one fixed ridge can be applied consistently to
        beta_star, supervised, and semi-supervised estimates.
        """
        return float(lambda_reg)

    # ------------------------------------------------------------------
    # 估计接口（四个抽象方法，子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def loss_and_grad(self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                      lambda_reg: float):
        """
        计算正则化目标函数值和梯度（供 scipy.optimize.minimize 使用）。

        目标函数形式：f(β) = (1/n) Σ ρ(X_i, Y_i; β) + λ·‖β‖²
        其中 ρ 为损失函数（逻辑回归为负对数似然，线性回归为均方损失），λ 为 L2 正则系数。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量（第 0 个元素为截距）。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距列）。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量（分类为 {0,1}，回归为连续值）。
        lambda_reg : float
            L2 正则化系数 λ（λ=0 表示不正则化）。

        返回
        ----
        f : float
            目标函数值（标量）。
        g : np.ndarray，形状 [p+1,]（一维）
            目标函数关于 β 的梯度向量（一维，兼容 scipy.optimize）。
        """
        ...

    @abstractmethod
    def score(self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
              lambda_reg: float) -> np.ndarray:
        """
        计算每个样本的评分函数（即逐样本梯度，含正则项贡献）。

        评分函数 Partial_l[i, :] = ∂ρ_i/∂β + (2λ/n)·β^T，
        其中 ρ_i 为第 i 个样本的损失。此量在半监督估计的方差估计和检验中频繁使用。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        Partial_l : np.ndarray，形状 [n, p+1]
            每行对应一个样本的评分函数（p+1 维梯度贡献）。
        """
        ...

    @abstractmethod
    def hessian(self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                lambda_reg: float) -> np.ndarray:
        """
        计算平均 Hessian 矩阵（即目标函数关于 β 的二阶导数，含正则项）。

        M = (1/n) Σ ∂²ρ_i/∂β² + 2λ·I_{p+1}
        此矩阵在 M-估计渐近理论中称为"敏感度矩阵"，用于构造渐近协方差矩阵
        和计算半监督估计器的效率增益。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量（通常在 beta_hat 或 beta_star 处计算）。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        M : np.ndarray，形状 [p+1, p+1]
            平均 Hessian 矩阵（对称正定矩阵）。
        """
        ...

    @abstractmethod
    def ss_loss_and_grad(self, beta: np.ndarray, X: np.ndarray, Y: np.ndarray,
                         Z_labeled: np.ndarray, Z_unlabeled: np.ndarray,
                         lambda_reg: float, use_dress_c1: bool = False):
        """
        计算半监督辅助矩加权目标函数和梯度。

        子类需要说明自己的实际公式。当前 LogisticModelSpec 和 LinearModelSpec 的
        实现并不完全相同：线性回归实现是标准逐样本加权平方损失；逻辑回归实现中
        sigmoid 项和 Y 项的加权方式不同，详见 LogisticModelSpec.ss_loss_and_grad。

        `use_dress_c1` 在现有子类中只控制 c1：
        - True  : c1=0
        - False : c1=n/(n+N)

        参数
        ----
        beta         : np.ndarray，形状 [p+1,] 或 [p+1, 1]
            当前参数向量。
        X            : np.ndarray，形状 [n, p]
            有标签数据特征矩阵。
        Y            : np.ndarray，形状 [n,] 或 [n, 1]
            有标签数据响应变量。
        Z_labeled    : np.ndarray，形状 [n, q]
            有标签数据的辅助特征矩阵（用于构造半监督权重）。
        Z_unlabeled  : np.ndarray 或 None，形状 [N, q]
            无标签数据的辅助特征矩阵；为 None 时退化为标准监督目标函数。
        lambda_reg   : float
            L2 正则化系数。
        use_dress_c1 : bool，可选
            True 时令 c1=0；False 时令 c1=n/(n+N)。

        返回
        ----
        f : float
            目标函数值。
        g : np.ndarray，形状 [p+1,]
            目标函数关于 β 的梯度（一维）。
        """
        ...

    # ------------------------------------------------------------------
    # 数据生成接口（两个抽象方法，子类必须实现）
    # ------------------------------------------------------------------

    @abstractmethod
    def generate_y(self, X: np.ndarray, true_value: np.ndarray) -> np.ndarray:
        """
        根据真实 DGP 参数从特征 X 生成响应变量 Y（向量化，不含循环）。

        DGP 可以比估计模型更复杂（例如含高次多项式项），从而制造有意义的模型误设，
        更接近实际数据分析场景（真实 DGP 往往比我们假设的模型复杂）。

        参数
        ----
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距列）。
        true_value : np.ndarray，形状 [d_dgp, 1]
            DGP 参数向量，维度 d_dgp = dgp_feature_dim(p)（含截距）。

        返回
        ----
        Y : np.ndarray，形状 [n, 1]
            生成的响应变量（分类模型为 {0,1} 的 Bernoulli 样本，回归模型为连续值）。
        """
        ...

    @abstractmethod
    def default_true_value(self, p: int) -> np.ndarray:
        """
        返回 p 维特征下默认的 DGP 真实参数向量（用于数值模拟的参数设定）。

        此参数是 DataGenerator.data_generation 的默认 true_value，
        由各子类根据自身 DGP 结构（含或不含高次项）定义。

        参数
        ----
        p : int
            特征维度（不含截距）。

        返回
        ----
        true_value : np.ndarray，形状 [d_dgp, 1]
            DGP 参数向量，维度 = dgp_feature_dim(p)。
        """
        ...

    # ------------------------------------------------------------------
    # 求解器接口（带默认数值优化实现，子类可重写以使用闭式解等更稳定算法）
    # ------------------------------------------------------------------
    # 设计动机
    # --------
    # 不同模型的"最优数值算法"差异很大：
    #   · 线性回归 (squared loss)         : 闭式解 β = (X'WX + 2λn I)⁻¹ X'WY
    #   · 逻辑回归 (logistic loss)        : 加权梯度下降（论文 Algorithm 1）
    #   · 分位数回归 (asymmetric L1 loss) : 加权次梯度下降（论文 Section 2.4 末）
    #
    # 论文 (Song, Lin & Zhou 2023, Section 2.4) 原文明确指出：
    #   "For the linear working model, a closed-form expression of θ̂ is available;
    #    however, with many popular choices of L(·), a closed-form solution is not.
    #    Moreover, the weights w_i could be negative."
    #
    # 负权重 + 线性平方损失 → 样本目标可能非凸 → BFGS 沿凹方向发散到 β=±∞。
    # 论文的处理：线性模型直接用闭式解，绕开优化数值问题。
    #
    # 因此求解器抽离为单独接口：基类默认走 BFGS，子类可重写为更合适的算法。
    # 子类若有更稳定算法（闭式解、ADMM、坐标下降等），只需 override 这两个方法即可，
    # 调用方（DRESSSSLogistic / SSLogistic / MstMdsp）完全感知不到差异，可以无缝拓展。
    # ------------------------------------------------------------------

    def solve_supervised(self, X: np.ndarray, Y: np.ndarray,
                         lambda_reg: float = 0.0,
                         initial_value: np.ndarray = None,
                         tolerance: float = 5e-3,
                         max_iter: int = 1000) -> np.ndarray:
        """
        监督 M-估计求解器：argmin_β  (1/n) Σ L(Y_i, X_i, β) + λ‖β‖².

        默认实现：BFGS 优化 self.loss_and_grad。
        子类如有闭式解或更稳定算法可重写本方法。

        参数
        ----
        X             : np.ndarray，形状 [n, p]
        Y             : np.ndarray，形状 [n,] 或 [n, 1]
        lambda_reg    : float，L2 正则化系数，默认 0
        initial_value : np.ndarray 或 None，BFGS 初始点，None 时用零向量
        tolerance     : float，收敛容差，默认 5e-3
        max_iter      : int，最大迭代次数，默认 1000

        返回
        ----
        beta_hat : np.ndarray，形状 [p+1, 1]
        """
        p = X.shape[1]
        if initial_value is None:
            initial_value = np.zeros(p + 1)
        x0 = np.asarray(initial_value).ravel()
        res = minimize(
            fun=lambda b: self.loss_and_grad(b, X, Y, lambda_reg)[0],
            x0=x0,
            jac=lambda b: self.loss_and_grad(b, X, Y, lambda_reg)[1],
            method='BFGS',
            tol=tolerance,
            options={'maxiter': max_iter, 'gtol': tolerance, 'disp': False},
        )
        return res.x.reshape(-1, 1)

    def solve_semi_supervised(self, X: np.ndarray, Y: np.ndarray,
                              Z_labeled: np.ndarray, Z_unlabeled: np.ndarray,
                              lambda_reg: float = 0.0,
                              use_dress_c1: bool = False,
                              initial_value: np.ndarray = None,
                              tolerance: float = 5e-3,
                              max_iter: int = 1000,
                              intercept_from_supervised: bool = False) -> np.ndarray:
        """
        半监督 M-估计求解器：argmin_β  L_D^w(β) + λ‖β‖²（论文公式 (4) 加权目标）。

        默认实现：BFGS 优化 self.ss_loss_and_grad。
        线性模型重写为闭式解（避免负权重下 BFGS 发散）。

        参数
        ----
        X             : np.ndarray，形状 [n, p]
        Y             : np.ndarray，形状 [n,] 或 [n, 1]
        Z_labeled     : np.ndarray，形状 [n, q]
        Z_unlabeled   : np.ndarray 或 None，形状 [N, q]；None / 空 → 退化为监督
        lambda_reg    : float，L2 正则化系数
        use_dress_c1  : bool，True 令 c1=0；False 令 c1=n/(n+N)
        initial_value : np.ndarray 或 None
        tolerance     : float
        max_iter      : int
        intercept_from_supervised : bool，可选（默认 False）
            若为 True，β̂[0]（截距）用 self.solve_supervised 得到的监督估计替换，
            β̂[1:]（斜率）保持 SS 估计。这是「分块解耦估计」策略：
            - 理论上 SS 对截距方向 (Z 在 1 上的投影 = 1 → A_0 第 0 分量为 0) 没有
              方差缩减，反而引入随机权重 w_i 的方差污染；
            - 斜率方向 SS 通过 Z 中 X, X² 等捕获 misspec 残差结构，有显著缩减。
            混合后截距走监督路径（√n-consistent），斜率走 SS 路径（更高效率），
            实测能稳定地把截距 MSE/CP 拉回到监督水平，同时保留斜率 SS 增益。

        返回
        ----
        beta_hat : np.ndarray，形状 [p+1, 1]
        """
        p = X.shape[1]
        if initial_value is None:
            initial_value = np.zeros(p + 1)
        x0 = np.asarray(initial_value).ravel()
        try:
            res = minimize(
                fun=lambda b: self.ss_loss_and_grad(b, X, Y, Z_labeled, Z_unlabeled,
                                                    lambda_reg, use_dress_c1)[0],
                x0=x0,
                jac=lambda b: self.ss_loss_and_grad(b, X, Y, Z_labeled, Z_unlabeled,
                                                    lambda_reg, use_dress_c1)[1],
                method='BFGS',
                tol=tolerance,
                options={'maxiter': max_iter, 'gtol': tolerance, 'disp': False},
            )
            beta_ss = res.x.reshape(-1, 1)
        except Exception:
            beta_ss = np.full((p + 1, 1), np.nan)

        # ============================================================
        # 健康检查：防止 BFGS 偶发性发散污染下游指标
        # ------------------------------------------------------------
        # DRESS 路径 (c1=0) 和 SS 路径在 w_i 取极端负值时 BFGS 可能发散，
        # β 被推到 ±1e4 量级 → exp(Xβ) 溢出 → Partial_l = NaN
        # → MSE 单次仿真巨大 → MC 平均后 MRR 出现 -1e7 级灾难。
        #
        # 触发条件（任一）：β 含 NaN/Inf；max|β| 超过 BOX=30。
        # 处理：回退到 supervised β̂（必收敛、必合理）。
        # BOX=30 远超 sigmoid 饱和阈值（|Xβ|≈5 就饱和），不会误伤正常估计。
        # ============================================================
        BOX = 30.0
        if (not np.all(np.isfinite(beta_ss))
                or np.max(np.abs(beta_ss)) > BOX):
            beta_sup_safe = self.solve_supervised(
                X, Y, lambda_reg=lambda_reg,
                initial_value=initial_value, tolerance=tolerance, max_iter=max_iter,
            )
            beta_ss = beta_sup_safe

        # 分块解耦：截距用监督估计替换
        if intercept_from_supervised:
            beta_sup = self.solve_supervised(
                X, Y, lambda_reg=lambda_reg,
                initial_value=initial_value, tolerance=tolerance, max_iter=max_iter,
            )
            beta_ss[0, 0] = beta_sup[0, 0]
        return beta_ss

    # ------------------------------------------------------------------
    # SSE/CP 估计辅助接口：残差二阶矩 V1 = E[δδ^T]
    # ------------------------------------------------------------------
    # 设计动机
    # --------
    # 不同模型的 β̂ 性质差异很大，对应的 V1 估计策略也不同：
    #   · 严格 M-估计器（如 LinearModelSpec 的闭式加权 LS）→ β̂ 是严格论文估计量,
    #     真实 Var 与论文渐近 V/n 一致 → V1 用 K-fold OOF 准确估
    #   · 部分数值稳定化估计器（如 LogisticModelSpec 的 strict WGD）在有限样本下
    #     与理想严格根方程仍有差异；当前默认先试 K-fold OOF 以缓解 CP 偏低，
    #     如出现过保守，可通过 LogisticModelSpec(v1_method="plain") 回退到全数据残差。
    #
    # 调用方（SSLogistic、DRESSSSLogistic）调 model_spec.estimate_v1(...) 即可,
    # 不需要关心具体模型用哪种估计方式。
    # ------------------------------------------------------------------

    def estimate_v1(self, Z, score, K=5, seed=None):
        """
        默认 V1 = E[δδ^T] 估计：K-fold OOF 残差二阶矩（适用于严格 M-估计器）。

        参数
        ----
        Z     : np.ndarray, shape (n, q)
            辅助特征矩阵。
        score : np.ndarray, shape (n, d)
            得分函数 ∂L_i/∂β。
        K     : int, K-fold 折数
        seed  : int 或 None，K-fold permutation 随机种子，None 不固定

        返回
        ----
        V1_hat : np.ndarray, shape (d, d)
        """
        n, q = Z.shape
        _, d = score.shape
        if n < K:
            # 样本太少：退化为全数据 plain
            try:
                gamma_full = np.linalg.lstsq(Z, score, rcond=None)[0]
            except np.linalg.LinAlgError:
                gamma_full = stable_solve(Z, score)
            e = score - Z @ gamma_full
            return (e.T @ e) / n

        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = np.random
        perm = rng.permutation(n)
        fold_indices = np.array_split(perm, K)

        delta_oof = np.zeros((n, d))
        try:
            gamma_full = np.linalg.lstsq(Z, score, rcond=None)[0]
        except np.linalg.LinAlgError:
            gamma_full = stable_solve(Z, score)

        for k in range(K):
            test_idx = fold_indices[k]
            train_mask = np.ones(n, dtype=bool)
            train_mask[test_idx] = False
            train_idx = np.flatnonzero(train_mask)
            Z_tr, S_tr = Z[train_idx], score[train_idx]
            try:
                gamma_k = np.linalg.lstsq(Z_tr, S_tr, rcond=None)[0]
            except np.linalg.LinAlgError:
                gamma_k = gamma_full
            delta_oof[test_idx] = score[test_idx] - Z[test_idx] @ gamma_k
        return (delta_oof.T @ delta_oof) / n

    def dgp_feature_dim(self, p: int) -> int:
        """
        返回 DGP 特征向量的维度（含截距）。

        默认实现：DGP 与估计模型使用相同的特征维度 p+1（截距 + p 个线性项）。
        若子类的 DGP 包含高次项（如二次项、三次项），需覆盖此方法返回正确维度，
        否则 DataGenerator 在构造 X_poly 时会出现维度不匹配错误。

        参数
        ----
        p : int
            特征维度（不含截距）。

        返回
        ----
        d_dgp : int
            DGP 特征向量维度（含截距），默认返回 p+1。
        """
        return p + 1


# =============================================================================
# 逻辑回归模型规范
# =============================================================================

class LogisticModelSpec(BaseModelSpec):
    """
    逻辑回归 M-估计规范（二分类，负对数似然 + L2 正则）。

    DGP 与估计模型存在有意设计的误设（常见于半监督 M-估计文献中）：
    - DGP：P(Y=1|X) = expit(β_0 + β_1ᵀX + β_2ᵀX² + β_3ᵀX³)，含三次多项式项
      （特征向量为 [1; X; X²; X³]，维度 = 1+3p）
    - 估计模型：logit 线性 [1|X]，只用截距和一次项
      （参数维度 = p+1，比 DGP 少 2p 个参数）
    这种误设使 beta_star（估计模型的总体最优参数）不等于 DGP 真值，
    模拟了实际中模型总是近似而非精确正确的情形。
    """

    def __init__(self, ss_solver: str = "bfgs",
                 wgd_step_size: float = 1.0,
                 wgd_max_iter: int = 50,
                 wgd_tolerance: float = 1e-5,
                 wgd_rel_tolerance: float = 0.0,
                 wgd_start: str = "zero",
                 wgd_ridge: float = 0.0,
                 wgd_max_abs_beta: float = 30.0,
                 newton_max_iter: int = 100,
                 newton_min_curvature: float = 1e-6,
                 newton_max_step_norm: float = 2.0,
                 v1_method: str = "plain",
                 v1_blend_alpha: float = 0,
                 bias_shrink: float = 1.0,
                 dgp_intercept: float = -2.0,
                 dgp_linear_coef: float = -2.0,
                 dgp_quadratic_coef: float = 1.0,
                 dgp_cubic_coef: float = 0.0,
                 sup_solver: str = "bfgs",
                 class_weight: str = "none"):
        """
        初始化逻辑回归模型规范。

        参数
        ----
        ss_solver : {"bfgs", "safe_newton", "weighted_newton", "strict_wgd", "pseudo_bfgs"}
            逻辑回归半监督估计的求解方式。
            - "bfgs"：用 scipy BFGS 优化严格论文加权目标/梯度。
              这不同于 "pseudo_bfgs"，后者保留旧版伪梯度。
            - "strict_wgd"：按多源论文计算部分的 ADMM + GD 口径，
              使用严格加权梯度下降，即梯度为 n^{-1}Σ_i w_i(σ_i-Y_i)X_i。
            - "safe_newton"：稳定的一步/多步局部 Newton 求根。以监督估计为局部锚点，
              用正定 Fisher 曲率近似缩放加权 score，并用 trust radius 与 score-norm
              阻尼线搜索控制负权重带来的非凸漂移。
            - "weighted_newton"：逻辑回归损失二阶可导时的完全求解诊断选项，
              按 weighted Newton 迭代求解加权 score 方程；为处理负权重和近奇异
              Hessian，实现中使用特征值地板、trust step 和 score-norm 阻尼线搜索。
            - "pseudo_bfgs"：历史兼容模式。使用旧版伪梯度 + BFGS，可复现实验旧结果。
        sup_solver : {"bfgs", "wgd"}
            逻辑回归【监督基线】的求解方式。
            - "bfgs"：默认，保持原行为。基类 scipy BFGS 完全收敛到监督 M-估计。
            - "wgd"：与半监督完全一致的固定步长梯度法（权重恒为 1 的 strict WGD）。
              监督即 w_i≡1 的加权 M-估计，故复用 ss_loss_and_grad，步长/迭代上限/
              收敛阈值/数值盒约束均与半监督相同。用于把监督基线与半监督放到同一
              优化器与同一迭代预算下比较，消除 BFGS（充分收敛）与 WGD（固定迭代）
              之间的求解器不对称。
        class_weight : {"none", "balanced"}
            实际数据分类不平衡时的可选诊断参数。默认 "none" 保持原逻辑损失；
            "balanced" 使用训练样本中 0/1 两类的反频率权重。
        wgd_step_size : float
            Algorithm 1 中的固定步长 γ。
        wgd_max_iter : int
            WGD 最大迭代次数。
        wgd_tolerance : float
            WGD 绝对终止阈值 δ_abs，即严格加权梯度范数小于该值时停止。
        wgd_rel_tolerance : float
            WGD 相对终止阈值 δ_rel。strict_wgd 中使用
            max(δ_abs, δ_rel * ||g_0||) 作为半监督早停阈值。默认 0 保持旧行为。
        wgd_start : {"zero", "initial", "supervised"}
            WGD 初始点。论文只要求给定 θ^(0)，默认使用零向量；"initial" 使用外部传入
            initial_value；"supervised" 使用监督估计作为热启动，主要用于敏感性分析。
        wgd_ridge : float
            可选的极小 ridge 稳定项。默认 0，保持论文严格加权梯度；若设为正数，
            梯度中额外加入 2*wgd_ridge*beta。
        wgd_max_abs_beta : float
            数值安全阈值。若迭代中 max|beta| 超过该值，回退监督估计。
        newton_max_iter : int
            weighted Newton 最大迭代次数。
        newton_min_curvature : float
            weighted Hessian 特征值绝对值下界，避免负权重或高维 Z 导致近奇异反演。
        newton_max_step_norm : float
            单次 Newton 更新的最大欧氏范数，用于 trust-step 稳定化。
        v1_method : {"plain", "crossfit", "blend", "ogk"}
            逻辑回归 SSE 中 V1 = E(delta_Z delta_Z^T) 的估计方式。
            - "plain"：默认。全数据拟合 Gamma 后直接计算投影残差二阶矩，对应论文
              中 V1 的 plug-in 估计。
            - "crossfit"：K-fold out-of-fold 投影残差二阶矩，作为有限样本敏感性分析。
            - "blend"：V1 = α V1_crossfit + (1-α) V1_plain，作为有限样本敏感性分析。
            - "ogk"：使用全数据残差的 OGK 鲁棒二阶矩。该选项用于有限样本校准试验。
        v1_blend_alpha : float
            v1_method="blend" 时 crossfit V1 的混合权重 α，取值范围 [0, 1]。
        bias_shrink : float
            proposed 置信区间中心的偏差修正收缩系数。0 表示不修正，1 表示全量修正。
        dgp_intercept, dgp_linear_coef, dgp_quadratic_coef, dgp_cubic_coef : float
            逻辑回归数据生成模型的系数：
                logit P(Y=1|X) = b0 + b1·sum_j X_j
                                + b2·sum_j X_j^2 + b3·sum_j X_j^3.
            估计模型仍只用线性 logit [1|X]，因此 b2/b3 控制模型误设强度。
            默认值完全复现原始 DGP：[-2; -2·1_p; 1_p; 0_p]。
        """
        if ss_solver not in {"bfgs", "safe_newton", "weighted_newton", "strict_wgd", "pseudo_bfgs"}:
            raise ValueError("ss_solver must be 'bfgs', 'safe_newton', 'weighted_newton', 'strict_wgd', or 'pseudo_bfgs'")
        if wgd_start not in {"zero", "initial", "supervised"}:
            raise ValueError("wgd_start must be 'zero', 'initial', or 'supervised'")
        if v1_method not in {"crossfit", "plain", "blend", "ogk"}:
            raise ValueError("v1_method must be 'crossfit', 'plain', 'blend', or 'ogk'")
        if not (0.0 <= float(v1_blend_alpha) <= 1.0):
            raise ValueError("v1_blend_alpha must be in [0, 1]")
        if not (0.0 <= float(bias_shrink) <= 1.0):
            raise ValueError("bias_shrink must be in [0, 1]")
        if sup_solver not in {"bfgs", "wgd"}:
            raise ValueError("sup_solver must be 'bfgs' or 'wgd'")
        if class_weight not in {"none", "balanced"}:
            raise ValueError("class_weight must be 'none' or 'balanced'")
        if float(wgd_tolerance) < 0:
            raise ValueError("wgd_tolerance must be non-negative")
        if float(wgd_rel_tolerance) < 0:
            raise ValueError("wgd_rel_tolerance must be non-negative")
        self.ss_solver = ss_solver
        self.sup_solver = sup_solver
        self.wgd_step_size = float(wgd_step_size)
        self.wgd_max_iter = int(wgd_max_iter)
        self.wgd_tolerance = float(wgd_tolerance)
        self.wgd_rel_tolerance = float(wgd_rel_tolerance)
        self.wgd_start = wgd_start
        self.wgd_ridge = float(wgd_ridge)
        self.wgd_max_abs_beta = float(wgd_max_abs_beta)
        self.newton_max_iter = int(newton_max_iter)
        self.newton_min_curvature = float(newton_min_curvature)
        self.newton_max_step_norm = float(newton_max_step_norm)
        self.v1_method = v1_method
        self.v1_blend_alpha = float(v1_blend_alpha)
        self.bias_shrink = float(bias_shrink)
        self.dgp_intercept = float(dgp_intercept)
        self.dgp_linear_coef = float(dgp_linear_coef)
        self.dgp_quadratic_coef = float(dgp_quadratic_coef)
        self.dgp_cubic_coef = float(dgp_cubic_coef)
        self.class_weight = class_weight
        self.wgd_diagnostics = []

    def _class_weight_vector(self, Y):
        """返回每个样本的类别权重；默认全 1，balanced 时使用反频率权重。"""
        Y_col = np.asarray(Y).reshape(-1, 1)
        if self.class_weight == "none":
            return np.ones_like(Y_col, dtype=float)

        y_flat = Y_col.ravel()
        n = len(y_flat)
        n0 = int(np.sum(y_flat == 0))
        n1 = int(np.sum(y_flat == 1))
        if n0 == 0 or n1 == 0:
            return np.ones_like(Y_col, dtype=float)
        w0 = n / (2.0 * n0)
        w1 = n / (2.0 * n1)
        return np.where(Y_col == 1, w1, w0).astype(float)

    # -------- 数据生成接口 --------

    def dgp_feature_dim(self, p: int) -> int:
        """
        返回逻辑回归 DGP 的特征向量维度。

        DGP 使用三次多项式特征 [1; X; X²; X³]，维度 = 1 + 3p。

        参数
        ----
        p : int
            原始特征维度。

        返回
        ----
        d_dgp : int
            DGP 特征维度，等于 1 + 3p。
        """
        return 1 + 3 * p

    def default_true_value(self, p: int) -> np.ndarray:
        """
        返回逻辑回归 DGP 的默认参数设定，参数由初始化时的
        dgp_intercept / dgp_linear_coef / dgp_quadratic_coef / dgp_cubic_coef 控制。

        参数含义（对应特征 [1; X; X²; X³]）：
        - 截距 β_0 = -2
        - 一次项系数 β_1 = -2·1_p
        - 二次项系数 β_2 = 1·1_p（估计模型漏掉，制造模型误设）
        - 三次项系数 β_3 = 0·1_p

        参数
        ----
        p : int
            特征维度。

        返回
        ----
        true_value : np.ndarray，形状 [1+3p, 1]
            DGP 参数向量。
        """
        return np.vstack([
            np.array([[self.dgp_intercept]]),                 # 截距项
            self.dgp_linear_coef * np.ones((p, 1)),           # X 的一次项系数
            self.dgp_quadratic_coef * np.ones((p, 1)),        # X² 的二次项系数
            self.dgp_cubic_coef * np.ones((p, 1))             # X³ 的三次项系数
        ])

    def generate_y(self, X: np.ndarray, true_value: np.ndarray) -> np.ndarray:
        """
        向量化逻辑回归 DGP：基于三次多项式特征生成二值响应变量。

        计算流程：
        1. 构造多项式特征：X_poly = [1 | X | X² | X³]，形状 [n, 1+3p]
           （X² 和 X³ 为逐元素幂次，不是矩阵幂）
        2. 计算线性预测值（logit）：logit = X_poly @ true_value，形状 [n,]
        3. 数值稳定的 sigmoid：P = sigmoid(logit)，避免大数值下的溢出
        4. 抽样：Y ~ Bernoulli(P)

        参数
        ----
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        true_value : np.ndarray，形状 [1+3p, 1]
            DGP 参数向量（含截距和三次项系数）。

        返回
        ----
        Y : np.ndarray，形状 [n, 1]
            生成的二值响应变量（0 或 1，float 类型）。
        """
        import scipy.stats as stats
        n, p = X.shape
        true_value = true_value.reshape(-1, 1)
        # 构造多项式特征矩阵 [n, 1+3p]：截距 + 一次项 + 二次项 + 三次项
        X_poly = np.hstack([np.ones((n, 1)), X, X ** 2, X ** 3])
        logits = (X_poly @ true_value).ravel()         # 线性预测值 [n,]

        # 数值稳定的 sigmoid 函数，避免极端 logit 下 exp 上溢/下溢。
        P = expit(logits)
        # 按 Bernoulli(P) 独立抽样，生成二值响应变量
        Y = stats.binom.rvs(n=1, p=P, size=n).astype(float).reshape(-1, 1)
        return Y

    # -------- 估计接口 --------

    def loss_and_grad(self, beta, X, Y, lambda_reg):
        """
        计算逻辑回归的正则化负对数似然目标函数值和梯度。

        目标函数：f(β) = (1/n) Σ [log(1 + exp(X_aug_i · β)) - Y_i · (X_aug_i · β)] + λ·‖β‖²
        梯度：∇f(β) = (1/n) X_aug^T · (σ(X_aug·β) - Y) + 2λ·β
        其中 σ(z) = exp(z)/(1+exp(z)) 为 sigmoid 函数。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量（β_0 为截距，β_1...β_p 为特征系数）。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            二值响应变量（0 或 1）。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        f : float
            目标函数值。
        g : np.ndarray，形状 [p+1,]
            目标函数梯度（一维，兼容 scipy.optimize.minimize）。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        beta = beta.reshape(-1, 1)
        X_aug = self.add_intercept(X)   # [n, p+1]，添加截距列

        eta = X_aug @ beta                                                  # [n, 1]
        Y_col = Y.reshape(-1, 1)
        class_weight_vec = self._class_weight_vector(Y_col)
        # 负对数似然：log(1+exp(Xβ)) - Y·Xβ（逐样本损失的均值）
        loss_per = np.logaddexp(0.0, eta) - Y_col * eta
        log_likelihood = np.mean(class_weight_vec * loss_per)
        # L2 正则化项：λ·‖β‖²（注意不对截距正则化的情形需另行处理，此处对全部 β 正则化）
        reg_term = lambda_reg * float(beta.T @ beta)
        f = float(log_likelihood) + reg_term

        sigmoid = expit(eta)                                                # [n, 1]
        # 梯度：(1/n)·X_aug^T·(σ-Y) + 2λ·β
        grad = np.mean(class_weight_vec * (sigmoid - Y_col) * X_aug, axis=0).reshape(-1, 1) + \
               2 * lambda_reg * beta
        g = grad.ravel()                                                   # 返回一维梯度
        return f, g

    def score(self, beta, X, Y, lambda_reg):
        """
        计算逻辑回归的逐样本评分函数（含正则项贡献）。

        评分函数（第 i 个样本）：
        Partial_l[i, :] = (σ(X_aug_i·β) - Y_i) · X_aug_i + 2λ·β^T
        其中第一项为负对数似然对第 i 个样本的梯度贡献，第二项为正则项均摊到每个样本。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        Partial_l : np.ndarray，形状 [n, p+1]
            每行为对应样本的评分函数向量（p+1 维）。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        beta = beta.reshape(-1, 1)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]

        eta = X_aug @ beta
        sigmoid = expit(eta)                                       # [n, 1]，sigmoid 预测概率

        Y_col = Y.reshape(-1, 1)
        class_weight_vec = self._class_weight_vector(Y_col)
        # 逐样本梯度：(σ_i - Y_i)·X_aug_i 加上正则项（正则项对所有样本相同，广播展开）
        Partial_l = class_weight_vec * (sigmoid - Y_col) * X_aug + \
                    2 * lambda_reg * (np.ones((n, 1)) @ beta.T)  # [n, p+1]
        return Partial_l

    def hessian(self, beta, X, Y, lambda_reg):
        """
        计算逻辑回归的平均 Hessian 矩阵（含正则项）。

        Hessian 公式：
        M = (1/n) X_aug^T · diag(σ(1-σ)) · X_aug + 2λ·I_{p+1}
        其中 σ·(1-σ) 为 sigmoid 函数的导数（即逻辑回归的权重矩阵 W 的对角元素），
        W = diag(σ_1·(1-σ_1), ..., σ_n·(1-σ_n))。

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量（Hessian 计算中实际不依赖 Y，但保留接口一致性）。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        M : np.ndarray，形状 [p+1, p+1]
            平均 Hessian 矩阵（对称正定，用于方差估计和 Newton 步）。
        """
        beta = beta.reshape(-1, 1)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]

        eta = X_aug @ beta
        sigmoid = expit(eta)                                       # [n, 1]
        # 只需要对 X_aug 的每一行乘以 σ_i(1-σ_i)，无需显式构造 n×n 对角矩阵。
        # 该写法与 X_aug.T @ diag(w_i) @ X_aug 完全等价，但时间和内存复杂度更低。
        class_weight_vec = self._class_weight_vector(Y).ravel()
        fisher_weight = (class_weight_vec * sigmoid.ravel() * (1 - sigmoid.ravel()))          # [n,]
        M = (1 / n) * X_aug.T @ (fisher_weight[:, None] * X_aug) + \
            2 * lambda_reg * np.eye(X_aug.shape[1])
        return M                                                   # [p+1, p+1]

    def _semi_supervised_weight_vector(self, n, Z_labeled=None, Z_unlabeled=None,
                                       use_dress_c1=False):
        """
        计算半监督加权估计中的样本权重向量 w。

        权重只由 Z_labeled、Z_unlabeled、c1 决定，与 beta 无关。因此在 WGD
        迭代前预先计算一次即可，避免每一步重复求伪逆。
        """
        if Z_unlabeled is None or (hasattr(Z_unlabeled, 'size') and Z_unlabeled.size == 0):
            return np.ones(n)

        N = Z_unlabeled.shape[0]
        c1 = 0 if use_dress_c1 else (n / (n + N) if (n + N) > 0 else 0)
        weight_term = self.semi_supervised_weight_term(Z_labeled, Z_unlabeled)
        return np.asarray(c1 + (1 - c1) * weight_term).ravel()

    def ss_loss_and_grad(self, beta, X, Y, Z_labeled=None, Z_unlabeled=None,
                         lambda_reg=0, use_dress_c1=False, weight_vec=None):
        """
        计算当前代码实现的半监督辅助矩加权逻辑回归目标和梯度。

        代码先用有标签辅助矩阵 Z_labeled 和无标签辅助矩阵 Z_unlabeled 构造
        一个长度为 n 的样本权重向量。权重公式为：

            solve((Z_labeled.T @ Z_labeled) / n, mean(Z_unlabeled)) -> alpha
            weight_term = Z_labeled @ alpha
            diag(w) = c1 + (1 - c1) * weight_term

        这里的 use_dress_c1 只控制 c1 的取值：
        - True  : c1 = 0
        - False : c1 = n / (n + N)

        ss_solver="bfgs"、"strict_wgd"、"safe_newton" 和 "weighted_newton" 都使用
        论文 Eq.(4) 的严格加权梯度形式：

                (1/n) * sum_i w_i * (sigmoid_i - Y_i) * X_aug_i + 2*lambda*beta

        历史兼容 ss_solver="pseudo_bfgs" 时，梯度改用旧版伪梯度：

                (1/n) * sum_i [w_i * sigmoid_i - Y_i] * X_aug_i + 2*lambda*beta

        两者只有"Y 项是否被 w_i 加权"的差别。旧版伪梯度是工程稳定化，不是论文
        公式本身；保留它只是为了复现实验旧结果。

        — 论文 Eq (4) 的 w_i 可以是 NEGATIVE（论文 Section 2.4 line 162 明确指出）。
          当 w_i < 0 时，严格加权 log-likelihood
                w_i * log(1+exp(z_i)) - w_i * Y_i * z_i
          中 log 项变成 concave，整个目标在 z_i → +∞ 方向 unbounded below。
        — 当前复现实验默认使用 BFGS，因为它与之前输出的表格口径一致；
          strict_wgd / safe_newton / weighted_newton 保留为诊断选项。

        参数
        ----
        beta : np.ndarray, shape (p+1,) 或 (p+1, 1)
            当前优化参数，第一项为截距。
        X : np.ndarray, shape (n, p)
            有标签样本的原始协变量，不含截距列。
        Y : np.ndarray, shape (n,) 或 (n, 1)
            有标签样本响应变量，逻辑回归中应为 0/1。
        Z_labeled : np.ndarray, shape (n, q) 或 None
            有标签样本的辅助矩阵。Z_unlabeled 非空时必须提供。
        Z_unlabeled : np.ndarray, shape (N, q) 或 None
            无标签样本的辅助矩阵；为空时函数退化为监督逻辑回归。
        lambda_reg : float, 默认 0
            L2 正则化系数。当前实现会正则化包括截距在内的全部参数。
        use_dress_c1 : bool, 默认 False
            是否把 c1 设为 0。False 时使用 c1=n/(n+N)。

        返回
        ----
        f : float
            当前实现的加权目标函数值。
        g : np.ndarray, shape (p+1,)
            当前实现的梯度，一维数组，供 scipy.optimize.minimize 使用。
        """
        beta = beta.reshape(-1, 1)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]

        eta = X_aug @ beta
        sigmoid = expit(eta)   # [n, 1]

        # 构造样本权重向量。数学上等价于 diag(w_i)，但不显式生成 n×n 对角矩阵。
        if weight_vec is None:
            weight_vec = self._semi_supervised_weight_vector(
                n, Z_labeled, Z_unlabeled, use_dress_c1
            )
        else:
            weight_vec = np.asarray(weight_vec).ravel()

        Y_col = Y.reshape(-1, 1)
        class_weight_vec = self._class_weight_vector(Y_col)
        # 逐样本逻辑回归负对数似然。
        loss_per = np.logaddexp(0.0, eta) - Y_col * eta
        lambda_eff = max(lambda_reg, self.wgd_ridge) if self.ss_solver != "pseudo_bfgs" else lambda_reg
        combined_weight = weight_vec[:, None] * class_weight_vec
        f = float(np.mean(combined_weight * loss_per) + lambda_eff * float(beta.T @ beta))

        if self.ss_solver == "pseudo_bfgs":
            # 历史兼容：旧版伪梯度，只对 sigmoid 项加权，Y 项不加权。
            grad_ll = np.sum(combined_weight * sigmoid * X_aug - class_weight_vec * Y_col * X_aug,
                             axis=0).reshape(-1, 1)
        else:
            # 论文 Algorithm 1 / Eq.(4) 的严格加权梯度。
            grad_ll = np.sum(combined_weight * (sigmoid - Y_col) * X_aug,
                             axis=0).reshape(-1, 1)
        penalty = 2 * lambda_eff * beta
        g = ((1 / n) * grad_ll + penalty).ravel()
        return f, g

    def _weighted_logistic_hessian(self, beta, X, weight_vec, lambda_reg=0.0):
        """
        Weighted Newton 使用的 Hessian / score Jacobian:
        n^{-1} X_aug^T diag(w_i * pi_i * (1-pi_i)) X_aug + 2 lambda I.
        负权重允许该矩阵不定，因此后续只把它作为 score 方程的 Jacobian。
        """
        beta = np.asarray(beta).reshape(-1, 1)
        X_aug = self.add_intercept(X)
        eta = X_aug @ beta
        sigmoid = expit(eta).ravel()
        fisher = sigmoid * (1.0 - sigmoid)
        weight_vec = np.asarray(weight_vec).ravel()
        lambda_eff = max(lambda_reg, self.wgd_ridge)
        H = (X_aug.T @ ((weight_vec * fisher)[:, None] * X_aug)) / X.shape[0]
        H = H + 2.0 * lambda_eff * np.eye(X_aug.shape[1])
        return 0.5 * (H + H.T)

    def _regularized_newton_delta(self, H, grad):
        """
        反演可能不定或近奇异的 weighted Hessian。保留特征值符号，但把绝对值
        小于阈值的方向抬高到阈值，避免高维 Z 或负权重造成数值爆炸。
        """
        grad = np.asarray(grad).reshape(-1, 1)
        eigvals, eigvecs = np.linalg.eigh(H)
        scale = max(float(np.max(np.abs(eigvals))), 1.0)
        floor = max(self.newton_min_curvature, 1e-8 * scale)
        signs = np.where(eigvals >= 0.0, 1.0, -1.0)
        eigvals_safe = signs * np.maximum(np.abs(eigvals), floor)
        return eigvecs @ ((eigvecs.T @ grad) / eigvals_safe[:, None])

    def _positive_logistic_curvature(self, beta, X, lambda_reg=0.0):
        """
        safe_newton 使用的正定曲率近似。

        加权 logistic Hessian 在 w_i 为负时可能不定，直接用于 Newton 会把迭代
        推向凹方向。这里改用未加权 Fisher 曲率作为局部尺度矩阵，相当于求解
        加权 score 方程时采用稳定的 Fisher-scoring/Levenberg 预条件器。
        """
        beta = np.asarray(beta).reshape(-1, 1)
        X_aug = self.add_intercept(X)
        n, d = X_aug.shape
        eta = X_aug @ beta
        sigmoid = expit(eta).ravel()
        fisher = sigmoid * (1.0 - sigmoid)
        lambda_eff = max(lambda_reg, self.wgd_ridge)
        H = (X_aug.T @ (fisher[:, None] * X_aug)) / n
        H = H + 2.0 * lambda_eff * np.eye(d)
        H = 0.5 * (H + H.T)
        eig_max = float(np.linalg.eigvalsh(H).max()) if d > 0 else 1.0
        floor = max(self.newton_min_curvature, 1e-8 * max(eig_max, 1.0))
        return H + floor * np.eye(d)

    @staticmethod
    def _project_to_ball(beta, center, radius):
        """把 beta 投影到以 center 为中心的欧氏球内。"""
        if radius is None or radius <= 0:
            return beta
        diff = beta - center
        norm = float(np.linalg.norm(diff))
        if np.isfinite(norm) and norm > radius:
            beta = center + (radius / norm) * diff
        return beta

    def _solve_safe_weighted_newton(self, X, Y, Z_labeled, Z_unlabeled,
                                    lambda_reg, use_dress_c1, beta, beta_sup,
                                    tolerance, max_iter):
        """
        稳定的加权 score 求根器。

        这个求解器不把可能非凸/无下界的加权 logistic objective 当作普通凸目标
        完全最小化，而是在监督解附近求解加权 score:

            n^{-1} sum_i w_i (sigma_i - Y_i) X_i + 2 lambda beta = 0.

        每一步用正定未加权 Fisher 曲率缩放 score，并通过 score-norm 线搜索、
        单步长度限制和围绕监督解的 trust radius 控制有限样本负权重造成的漂移。
        """
        weight_vec = self._semi_supervised_weight_vector(
            X.shape[0], Z_labeled, Z_unlabeled, use_dress_c1
        )
        n_iter = min(int(max_iter), self.wgd_max_iter, self.newton_max_iter)
        trust_radius = max(float(self.newton_max_step_norm), 1e-8)
        stop_reason = "max_iter"
        valid = True
        steps_done = 0

        _, grad0 = self.ss_loss_and_grad(
            beta, X, Y, Z_labeled, Z_unlabeled, lambda_reg, use_dress_c1,
            weight_vec=weight_vec
        )
        grad = np.asarray(grad0).reshape(-1, 1)
        grad_norm0 = float(np.linalg.norm(grad))
        grad_norm = grad_norm0
        stop_threshold = max(
            float(tolerance),
            self.wgd_tolerance,
            self.wgd_rel_tolerance * grad_norm0 if np.isfinite(grad_norm0) else 0.0,
        )
        best_beta = beta.copy()
        best_grad_norm = grad_norm0
        max_abs_beta_seen = float(np.max(np.abs(beta)))

        if not np.isfinite(grad_norm0):
            valid = False
            stop_reason = "invalid_grad"
        else:
            for k in range(n_iter):
                if grad_norm <= stop_threshold:
                    stop_reason = "tol"
                    break

                try:
                    H_base = self._positive_logistic_curvature(beta, X, lambda_reg)
                except np.linalg.LinAlgError:
                    valid = False
                    stop_reason = "invalid_curvature"
                    break

                accepted = False
                damping = max(self.newton_min_curvature, 1e-8)
                for _ in range(10):
                    H = H_base + damping * np.eye(H_base.shape[0])
                    try:
                        delta = stable_solve(H, grad, symmetrize=True)
                    except Exception:
                        damping *= 10.0
                        continue
                    delta_norm = float(np.linalg.norm(delta))
                    if not np.isfinite(delta_norm):
                        damping *= 10.0
                        continue
                    if delta_norm > self.newton_max_step_norm:
                        delta *= self.newton_max_step_norm / delta_norm

                    step = 1.0
                    for _ in range(25):
                        beta_next = beta - step * delta
                        beta_next = self._project_to_ball(beta_next, beta_sup, trust_radius)
                        beta_next = np.clip(
                            beta_next, -self.wgd_max_abs_beta, self.wgd_max_abs_beta
                        )
                        if not np.all(np.isfinite(beta_next)):
                            step *= 0.5
                            continue
                        _, grad_next = self.ss_loss_and_grad(
                            beta_next, X, Y, Z_labeled, Z_unlabeled,
                            lambda_reg, use_dress_c1, weight_vec=weight_vec
                        )
                        grad_next = np.asarray(grad_next).reshape(-1, 1)
                        grad_next_norm = float(np.linalg.norm(grad_next))
                        enough_decrease = grad_next_norm <= grad_norm * (1.0 - 1e-4 * step)
                        best_decrease = grad_next_norm < best_grad_norm
                        if np.isfinite(grad_next_norm) and (enough_decrease or best_decrease):
                            beta = beta_next
                            grad = grad_next
                            grad_norm = grad_next_norm
                            steps_done = k + 1
                            max_abs_beta_seen = max(
                                max_abs_beta_seen, float(np.max(np.abs(beta)))
                            )
                            if grad_next_norm < best_grad_norm:
                                best_grad_norm = grad_next_norm
                                best_beta = beta_next.copy()
                            accepted = True
                            break
                        step *= 0.5

                    if accepted:
                        break
                    damping *= 10.0

                if not accepted:
                    stop_reason = "line_search_failed"
                    break

            if stop_reason == "max_iter" and steps_done < n_iter:
                stop_reason = "stalled"

        if not valid:
            best_beta = beta_sup.copy()
            grad_norm = np.nan

        self.wgd_diagnostics.append({
            "solver": self.ss_solver,
            "use_dress_c1": bool(use_dress_c1),
            "n_labeled": int(X.shape[0]),
            "n_unlabeled": 0 if Z_unlabeled is None else int(Z_unlabeled.shape[0]),
            "n_iter_cap": int(n_iter),
            "steps_done": int(steps_done),
            "stop_reason": stop_reason,
            "abs_tolerance": float(max(float(tolerance), self.wgd_tolerance)),
            "rel_tolerance": float(self.wgd_rel_tolerance),
            "stop_threshold": float(stop_threshold),
            "grad_norm0": float(grad_norm0),
            "grad_norm_end": float(grad_norm),
            "grad_ratio": float(grad_norm / grad_norm0)
            if np.isfinite(grad_norm) and np.isfinite(grad_norm0) and grad_norm0 > 0
            else np.nan,
            "best_grad_norm": float(best_grad_norm),
            "weight_min": float(np.min(weight_vec)),
            "weight_max": float(np.max(weight_vec)),
            "weight_mean": float(np.mean(weight_vec)),
            "max_abs_beta": float(max_abs_beta_seen),
            "trust_radius": float(trust_radius),
            "fallback_to_supervised": bool(not valid),
        })
        return best_beta

    def _solve_weighted_newton(self, X, Y, Z_labeled, Z_unlabeled,
                               lambda_reg, use_dress_c1, beta, beta_sup,
                               tolerance, max_iter):
        weight_vec = self._semi_supervised_weight_vector(
            X.shape[0], Z_labeled, Z_unlabeled, use_dress_c1
        )
        n_iter = min(int(max_iter), self.wgd_max_iter, self.newton_max_iter)
        tol = max(float(tolerance), self.wgd_tolerance)
        valid = True

        for _ in range(n_iter):
            _, grad = self.ss_loss_and_grad(
                beta, X, Y, Z_labeled, Z_unlabeled, lambda_reg, use_dress_c1,
                weight_vec=weight_vec
            )
            grad = grad.reshape(-1, 1)
            grad_norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad_norm):
                valid = False
                break
            if grad_norm <= tol:
                break

            H = self._weighted_logistic_hessian(beta, X, weight_vec, lambda_reg)
            try:
                delta = self._regularized_newton_delta(H, grad)
            except np.linalg.LinAlgError:
                valid = False
                break
            delta_norm = float(np.linalg.norm(delta))
            if not np.isfinite(delta_norm):
                valid = False
                break
            if delta_norm > self.newton_max_step_norm:
                delta *= self.newton_max_step_norm / delta_norm

            accepted = False
            step = 1.0
            for _ in range(20):
                beta_next = beta - step * delta
                beta_next = np.clip(
                    beta_next, -self.wgd_max_abs_beta, self.wgd_max_abs_beta
                )
                if not np.all(np.isfinite(beta_next)):
                    step *= 0.5
                    continue
                _, grad_next = self.ss_loss_and_grad(
                    beta_next, X, Y, Z_labeled, Z_unlabeled, lambda_reg, use_dress_c1,
                    weight_vec=weight_vec
                )
                grad_next_norm = float(np.linalg.norm(grad_next))
                if np.isfinite(grad_next_norm) and grad_next_norm <= grad_norm * (1.0 - 1e-4 * step):
                    beta = beta_next
                    accepted = True
                    break
                step *= 0.5

            if not accepted:
                break

        if not valid:
            beta = beta_sup.copy()
        return beta

    def solve_supervised(self, X, Y, lambda_reg=0.0, initial_value=None,
                         tolerance=5e-3, max_iter=1000):
        """
        逻辑回归监督求解器。

        - sup_solver="bfgs"（默认）：委托基类 scipy BFGS，保持原行为。
        - sup_solver="wgd"：与半监督完全相同的固定步长梯度法，区别仅在于权重恒为 1。
          监督 M-估计 = w_i≡1 的加权 M-估计，因此直接复用 ss_loss_and_grad 计算
          严格（不加权）梯度 (1/n)Σ_i(σ_i-Y_i)X_aug_i + 2λβ。
          逻辑回归监督目标为凸且有下界（w_i≥0），这里用监督目标 Hessian 的全局
          Lipschitz 上界自动选择固定步长 γ=1/L，并给足迭代预算，使 WGD 收敛到
          与 BFGS 相同的监督 M-估计，而不是形成早停正则化版本；万一出现数值异常
          仍回退 BFGS，保证下游指标不被污染。
        """
        if self.sup_solver == "bfgs":
            return super().solve_supervised(
                X, Y, lambda_reg=lambda_reg, initial_value=initial_value,
                tolerance=tolerance, max_iter=max_iter,
            )

        # sup_solver == "wgd"
        p = X.shape[1]
        n = X.shape[0]
        if self.wgd_start == "initial" and initial_value is not None:
            beta = np.asarray(initial_value).reshape(-1, 1).copy()
        else:
            # "zero" 与 "supervised"（监督自身无可借用初值）均用零向量起步
            beta = np.zeros((p + 1, 1))

        weight_vec = np.ones(n)   # 监督 = 权重恒为 1 的加权梯度下降
        X_aug = self.add_intercept(X)
        lambda_eff = max(lambda_reg, self.wgd_ridge)
        lipschitz = 0.25 * float(np.linalg.eigvalsh((X_aug.T @ X_aug) / n).max()) + 2 * lambda_eff
        step_size = 1.0 / max(lipschitz, 1e-12)
        n_iter = max(int(max_iter), self.wgd_max_iter, 10000)
        tol = min(float(tolerance), self.wgd_tolerance)
        valid = True
        for _ in range(n_iter):
            _, grad = self.ss_loss_and_grad(
                beta, X, Y, None, None, lambda_reg, False, weight_vec=weight_vec
            )
            grad = grad.reshape(-1, 1)
            grad_norm = float(np.linalg.norm(grad))
            if not np.isfinite(grad_norm):
                valid = False
                break
            if grad_norm <= tol:
                break
            beta_next = beta - step_size * grad
            if (not np.all(np.isfinite(beta_next))
                    or np.max(np.abs(beta_next)) > self.wgd_max_abs_beta):
                valid = False
                break
            beta = beta_next

        if not valid:
            # 凸问题理论上不会发散；数值异常时回退 BFGS（必收敛、必合理）。
            return super().solve_supervised(
                X, Y, lambda_reg=lambda_reg, initial_value=initial_value,
                tolerance=tolerance, max_iter=max_iter,
            )
        return beta

    def solve_semi_supervised(self, X, Y, Z_labeled, Z_unlabeled,
                              lambda_reg=0.0,
                              use_dress_c1=False,
                              initial_value=None,
                              tolerance=5e-3,
                              max_iter=1000,
                              intercept_from_supervised=False):
        """
        逻辑回归半监督求解器。

        默认 bfgs 委托基类用 scipy BFGS 优化严格论文加权目标/梯度。
        strict_wgd 按多源论文计算部分的 ADMM + GD 口径做固定步长 weighted
        gradient descent。safe_newton 是以监督估计为局部锚点的稳定 score 求根路径；
        weighted_newton 为二阶可导 logistic loss 的完全求解诊断选项。
        pseudo_bfgs 为旧版兼容模式，使用旧版伪梯度。
        """
        if self.ss_solver in {"bfgs", "pseudo_bfgs"}:
            return super().solve_semi_supervised(
                X, Y, Z_labeled, Z_unlabeled,
                lambda_reg=lambda_reg,
                use_dress_c1=use_dress_c1,
                initial_value=initial_value,
                tolerance=tolerance,
                max_iter=max_iter,
                intercept_from_supervised=intercept_from_supervised,
            )

        p = X.shape[1]
        beta_sup = self.solve_supervised(
            X, Y, lambda_reg=lambda_reg,
            initial_value=initial_value,
            tolerance=tolerance,
            max_iter=max_iter,
        )

        if self.wgd_start == "supervised":
            beta = beta_sup.copy()
        elif self.wgd_start == "initial" and initial_value is not None:
            beta = np.asarray(initial_value).reshape(-1, 1).copy()
        else:
            beta = np.zeros((p + 1, 1))

        if self.ss_solver == "safe_newton":
            # safe_newton 是局部 one-step / Fisher-scoring 修正，理论上应围绕
            # 监督根求解；若调用方显式传 initial_value，则尊重该初值。
            if self.wgd_start == "initial" and initial_value is not None:
                beta = np.asarray(initial_value).reshape(-1, 1).copy()
            else:
                beta = beta_sup.copy()
            beta = self._solve_safe_weighted_newton(
                X, Y, Z_labeled, Z_unlabeled,
                lambda_reg, use_dress_c1, beta, beta_sup, tolerance, max_iter
            )
            if intercept_from_supervised:
                beta[0, 0] = beta_sup[0, 0]
            return beta

        if self.ss_solver == "weighted_newton":
            beta = self._solve_weighted_newton(
                X, Y, Z_labeled, Z_unlabeled,
                lambda_reg, use_dress_c1, beta, beta_sup, tolerance, max_iter
            )
            if intercept_from_supervised:
                beta[0, 0] = beta_sup[0, 0]
            return beta

        n_iter = min(int(max_iter), self.wgd_max_iter)
        valid = True
        stop_reason = "max_iter"
        steps_done = 0
        grad_norm0 = np.nan
        grad_norm = np.nan
        stop_threshold = max(float(tolerance), self.wgd_tolerance)
        max_abs_beta_seen = float(np.max(np.abs(beta)))
        weight_vec = self._semi_supervised_weight_vector(
            X.shape[0], Z_labeled, Z_unlabeled, use_dress_c1
        )
        for k in range(n_iter):
            _, grad = self.ss_loss_and_grad(
                beta, X, Y, Z_labeled, Z_unlabeled, lambda_reg, use_dress_c1,
                weight_vec=weight_vec
            )
            grad = grad.reshape(-1, 1)
            grad_norm = float(np.linalg.norm(grad))
            if k == 0:
                grad_norm0 = grad_norm
                stop_threshold = max(
                    float(tolerance),
                    self.wgd_tolerance,
                    self.wgd_rel_tolerance * grad_norm0,
                )
            if not np.isfinite(grad_norm):
                stop_reason = "invalid_grad"
                valid = False
                break
            if grad_norm <= stop_threshold:
                stop_reason = "tol"
                break

            beta_next = beta - self.wgd_step_size * grad
            steps_done = k + 1
            if (not np.all(np.isfinite(beta_next))
                    or np.max(np.abs(beta_next)) > self.wgd_max_abs_beta):
                stop_reason = "invalid_beta"
                valid = False
                break
            beta = beta_next
            max_abs_beta_seen = max(max_abs_beta_seen, float(np.max(np.abs(beta))))

        self.wgd_diagnostics.append({
            "solver": self.ss_solver,
            "use_dress_c1": bool(use_dress_c1),
            "n_labeled": int(X.shape[0]),
            "n_unlabeled": 0 if Z_unlabeled is None else int(Z_unlabeled.shape[0]),
            "n_iter_cap": int(n_iter),
            "steps_done": int(steps_done),
            "stop_reason": stop_reason,
            "abs_tolerance": float(max(float(tolerance), self.wgd_tolerance)),
            "rel_tolerance": float(self.wgd_rel_tolerance),
            "stop_threshold": float(stop_threshold),
            "grad_norm0": float(grad_norm0),
            "grad_norm_end": float(grad_norm),
            "grad_ratio": float(grad_norm / grad_norm0)
            if np.isfinite(grad_norm) and np.isfinite(grad_norm0) and grad_norm0 > 0
            else np.nan,
            "weight_min": float(np.min(weight_vec)),
            "weight_max": float(np.max(weight_vec)),
            "weight_mean": float(np.mean(weight_vec)),
            "max_abs_beta": float(max_abs_beta_seen),
            "fallback_to_supervised": bool(not valid),
        })

        # WGD 可能因负权重沿无界方向漂移；一旦数值异常，回退监督估计，避免污染 MC 指标。
        if not valid:
            beta = beta_sup.copy()

        if intercept_from_supervised:
            beta[0, 0] = beta_sup[0, 0]
        return beta

    # ------------------------------------------------------------------
    # V₁ 估计：可切换 K-fold OOF 交叉拟合或全数据 plain 二阶矩。
    # ------------------------------------------------------------------
    def estimate_v1(self, Z, score, K=5, seed=None):
        if self.v1_method == "crossfit":
            return super().estimate_v1(Z, score, K=K, seed=seed)

        try:
            gamma_full = np.linalg.lstsq(Z, score, rcond=None)[0]
        except np.linalg.LinAlgError:
            gamma_full = stable_solve(Z, score)
        e = score - Z @ gamma_full
        v_plain = (e.T @ e) / Z.shape[0]
        if self.v1_method == "plain":
            return v_plain
        if self.v1_method == "ogk":
            return self._ogk_second_moment(e)

        v_cross = super().estimate_v1(Z, score, K=K, seed=seed)
        return self.v1_blend_alpha * v_cross + (1.0 - self.v1_blend_alpha) * v_plain

    @staticmethod
    def _ogk_second_moment(residual: np.ndarray) -> np.ndarray:
        """Return a positive semidefinite OGK-style robust second-moment matrix."""
        residual = np.asarray(residual, dtype=float)
        n, d = residual.shape
        if n <= 1:
            return np.eye(d) * 1e-8
        center = np.median(residual, axis=0, keepdims=True)
        centered = residual - center
        try:
            Q, _ = orthogonal_procrustes(centered, np.eye(d))
            rotated = centered @ Q
        except Exception:
            Q = np.eye(d)
            rotated = centered
        mad = np.median(np.abs(rotated), axis=0) * 1.4826
        mad = np.maximum(mad, 1e-8)
        cov = Q @ np.diag(mad ** 2) @ Q.T
        cov = (cov + cov.T) / 2.0
        min_eig = np.min(np.linalg.eigvalsh(cov))
        if min_eig < 0:
            cov -= 1.05 * min_eig * np.eye(d)
        return cov


# =============================================================================
# 线性回归模型规范
# =============================================================================

class LinearModelSpec(BaseModelSpec):
    """
    线性回归 M-估计规范（OLS，均方误差损失 + L2 正则）。

    DGP 与估计模型存在轻度误设（可通过 misspecified 参数控制）：
    - misspecified=True（默认）：
        DGP：Y = β_0 + β_1ᵀX + β_2ᵀX² + ε，ε ~ N(0, noise_std²)，含二次项
        估计模型：线性 [1|X]，只用截距和一次项（参数维度 = p+1）
    - misspecified=False：
        DGP 与估计模型完全一致，均为线性，无模型误设

    如需无误设实验，初始化时传入 misspecified=False：
        LinearModelSpec(misspecified=False)
    """

    def __init__(self, misspecified: bool = True, noise_std: float = 1.0,
                 dgp_intercept: float = 0.0,
                 dgp_linear_coef: float = 1.0,
                 dgp_quadratic_coef: float = 0.5,
                 dgp_cubic_coef: float = 0.0,
                 dgp_nonlinear_feature_limit: int = None,
                 ridge_lambda: float = 0.0,
                 weight_ridge: float = 0.0,
                 weight_clip_min: float = None,
                 weight_clip_max: float = None,
                 z_feature_limit: int = None):
        """
        初始化线性回归模型规范。

        参数
        ----
        misspecified : bool，可选
            True（默认）= DGP 含二次项（制造轻度误设，DGP 特征维度 = 1+2p）；
            False = DGP 与估计模型完全一致（均为线性，DGP 特征维度 = 1+p）。
        noise_std    : float，可选
            DGP 残差的标准差 σ_ε，控制信噪比，默认 1.0。
        dgp_intercept, dgp_linear_coef, dgp_quadratic_coef, dgp_cubic_coef : float
            默认 DGP 系数。misspecified=True 时二次项/三次项制造线性工作模型误设；
            dgp_cubic_coef=0 时不显式加入三次项，保持旧的 1+2p 参数维度。
        dgp_nonlinear_feature_limit : int, optional
            若给定，仅前 k 个原始特征带有二次/三次误设项；其余非线性系数设为 0。
            p<=k 时等价于所有特征都有同强度误设。
        """
        self.misspecified = misspecified
        self.noise_std = noise_std
        self.dgp_intercept = dgp_intercept
        self.dgp_linear_coef = dgp_linear_coef
        self.dgp_quadratic_coef = dgp_quadratic_coef
        self.dgp_cubic_coef = dgp_cubic_coef
        self.dgp_nonlinear_feature_limit = dgp_nonlinear_feature_limit
        self.ridge_lambda = ridge_lambda
        self.weight_ridge = weight_ridge
        self.weight_clip_min = weight_clip_min
        self.weight_clip_max = weight_clip_max
        self.z_feature_limit = z_feature_limit
        # 当前线性正文实验只使用 exact/quad DGP。quad 响应里虽然最高是 X^2，
        # 但斜率 score 为 residual * X，会产生 X^3；因此最多保留到三次辅助项。
        if not self.misspecified:
            self.max_auxiliary_alpha = 1
        else:
            self.max_auxiliary_alpha = 3 if self.dgp_cubic_coef == 0.0 else 4
        # 线性闭式解已有 _stable_solve 防御，汇总阶段不需要像逻辑回归那样
        # 用较小 box 截断参数。
        self.beta_sanitize_box = 1e6

    def _effective_lambda(self, lambda_reg: float) -> float:
        return max(float(lambda_reg), float(self.ridge_lambda))

    def _stabilize_weights(self, w_vec: np.ndarray) -> np.ndarray:
        lo = self.weight_clip_min
        hi = self.weight_clip_max
        if lo is None and hi is None:
            return w_vec
        lo_val = -np.inf if lo is None else float(lo)
        hi_val = np.inf if hi is None else float(hi)
        return np.clip(w_vec, lo_val, hi_val)

    # -------- 数据生成接口 --------

    def dgp_feature_dim(self, p: int) -> int:
        """
        返回线性回归 DGP 的特征向量维度。

        - misspecified=True ：DGP 含二次项，特征为 [1; X; X²]，维度 = 1+2p
        - misspecified=False：DGP 为纯线性，特征为 [1; X]，维度 = 1+p

        参数
        ----
        p : int
            原始特征维度。

        返回
        ----
        d_dgp : int
            DGP 特征向量维度（含截距）。
        """
        if not self.misspecified:
            return 1 + p
        return 1 + 2 * p + (p if self.dgp_cubic_coef != 0.0 else 0)

    def default_true_value(self, p: int) -> np.ndarray:
        """
        返回线性回归 DGP 的默认参数设定。

        - misspecified=True ：[b0; b1·1_p; b2·1_p]，默认 b0=0, b1=1, b2=0.5
          （长度 = 1+2p，含二次项使 DGP 与估计模型不一致）
        - misspecified=False：[b0; b1·1_p]，默认 b0=0, b1=1
          （长度 = 1+p，DGP 与估计模型完全一致）

        参数
        ----
        p : int
            特征维度。

        返回
        ----
        true_value : np.ndarray，形状 [d_dgp, 1]
            DGP 参数向量。
        """
        if not self.misspecified:
            return np.vstack([
                np.array([[self.dgp_intercept]]),
                self.dgp_linear_coef * np.ones((p, 1))
            ])
        nonlinear_active = p
        if self.dgp_nonlinear_feature_limit is not None:
            nonlinear_active = max(0, min(int(self.dgp_nonlinear_feature_limit), p))
        quadratic_coef = np.zeros((p, 1))
        quadratic_coef[:nonlinear_active, :] = self.dgp_quadratic_coef
        parts = [
            np.array([[self.dgp_intercept]]),
            self.dgp_linear_coef * np.ones((p, 1)),
            quadratic_coef,
        ]
        if self.dgp_cubic_coef != 0.0:
            cubic_coef = np.zeros((p, 1))
            cubic_coef[:nonlinear_active, :] = self.dgp_cubic_coef
            parts.append(cubic_coef)
        return np.vstack(parts)

    def generate_y(self, X: np.ndarray, true_value: np.ndarray) -> np.ndarray:
        """
        向量化线性 DGP：基于（含或不含二次项的）特征生成连续响应变量。

        计算流程：
        1. 构造特征矩阵：X_feat = [1|X|X²]（misspecified=True）或 [1|X]（False）
        2. 计算条件均值：μ_Y = X_feat @ true_value
        3. 添加高斯噪声：Y = μ_Y + ε，ε ~ N(0, noise_std²)

        参数
        ----
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        true_value : np.ndarray，形状 [d_dgp, 1]
            DGP 参数向量（含截距）。

        返回
        ----
        Y : np.ndarray，形状 [n, 1]
            生成的连续响应变量。
        """
        n, p = X.shape
        true_value = true_value.reshape(-1, 1)
        d = true_value.shape[0]
        if d == 1 + p:
            X_feat = np.hstack([np.ones((n, 1)), X])
        elif d == 1 + 2 * p:
            X_feat = np.hstack([np.ones((n, 1)), X, X ** 2])
        elif d == 1 + 3 * p:
            X_feat = np.hstack([np.ones((n, 1)), X, X ** 2, X ** 3])
        else:
            raise ValueError(
                f"LinearModelSpec true_value length {d} is incompatible with p={p}; "
                f"expected {1 + p}, {1 + 2 * p}, or {1 + 3 * p}."
            )
        mu_y = X_feat @ true_value                       # 条件均值 [n, 1]
        eps = np.random.randn(n, 1) * self.noise_std     # 独立同分布高斯噪声
        return mu_y + eps                                # [n, 1]

    # -------- 估计接口 --------

    def loss_and_grad(self, beta, X, Y, lambda_reg):
        """
        计算线性回归的正则化均方误差目标函数值和梯度。

        目标函数：f(β) = (1/n) Σ (1/2)·(X_aug_i·β - Y_i)² + λ·‖β‖²
        梯度：∇f(β) = (1/n) X_aug^T·(X_aug·β - Y) + 2λ·β

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵（不含截距）。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            连续响应变量。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        f : float
            目标函数值。
        g : np.ndarray，形状 [p+1,]
            目标函数梯度（一维）。
        """
        beta = beta.reshape(-1, 1)
        X_aug = self.add_intercept(X)   # [n, p+1]

        residual = X_aug @ beta - Y.reshape(-1, 1)      # 残差向量 [n, 1]
        # 均方损失（系数 1/2 使梯度不含 2 的倍数）+ L2 正则化项
        f = float(np.mean(0.5 * residual ** 2) + lambda_reg * float(beta.T @ beta))

        # 梯度：(1/n)·X^T·residual + 2λ·β
        grad = np.mean(residual * X_aug, axis=0).reshape(-1, 1) + \
               2 * lambda_reg * beta
        g = grad.ravel()
        return f, g

    def score(self, beta, X, Y, lambda_reg):
        """
        计算线性回归的逐样本评分函数（含正则项贡献）。

        评分函数（第 i 个样本）：
        Partial_l[i, :] = (X_aug_i·β - Y_i) · X_aug_i + 2λ·β^T

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        Partial_l : np.ndarray，形状 [n, p+1]
            每行为对应样本的评分函数向量。
        """
        beta = beta.reshape(-1, 1)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]

        residual = X_aug @ beta - Y.reshape(-1, 1)      # 残差 [n, 1]
        # 逐样本梯度：残差·X_aug（损失函数的梯度贡献）+ 正则项（广播到每个样本）
        Partial_l = residual * X_aug + \
                    2 * lambda_reg * (np.ones((n, 1)) @ beta.T)  # [n, p+1]
        return Partial_l

    def hessian(self, beta, X, Y, lambda_reg):
        """
        计算线性回归的平均 Hessian 矩阵（含正则项）。

        Hessian 公式（线性回归的 Hessian 与 β 无关，为常数矩阵）：
        M = (1/n) X_aug^T · X_aug + 2λ·I_{p+1}
        （线性回归的二阶导数不依赖 β 或 Y，故参数仅保留接口一致性）

        参数
        ----
        beta       : np.ndarray，形状 [p+1, 1] 或 [p+1,]
            当前参数向量（线性回归 Hessian 与 β 无关，但保留接口统一性）。
        X          : np.ndarray，形状 [n, p]
            原始特征矩阵。
        Y          : np.ndarray，形状 [n, 1] 或 [n,]
            响应变量（线性回归 Hessian 与 Y 无关，但保留接口统一性）。
        lambda_reg : float
            L2 正则化系数。

        返回
        ----
        M : np.ndarray，形状 [p+1, p+1]
            平均 Hessian 矩阵（常数矩阵，不随 β 变化）。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]
        # 格拉姆矩阵 + 正则项：(1/n)·X^T·X + 2λ·I
        M = (1 / n) * X_aug.T @ X_aug + \
            2 * lambda_reg * np.eye(X_aug.shape[1])
        return M                        # [p+1, p+1]

    def ss_loss_and_grad(self, beta, X, Y, Z_labeled=None, Z_unlabeled=None,
                         lambda_reg=0, use_dress_c1=False):
        """
        计算线性回归的半监督加权目标函数值和梯度。

        权重构造方式与 LogisticModelSpec.ss_loss_and_grad 相同。区别是这里的梯度
        是标准逐样本加权平方误差梯度，w_i 同时作用在 residual_i 和 X_i 上。

        目标函数：f_SS = (1/n) Σ_i w_i · (1/2)·(X_aug_i·β - Y_i)² + λ‖β‖²
        梯度：∇f_SS = (1/n)·[w·residual·X_aug].sum(axis=0) + 2λ·β

        参数
        ----
        beta         : np.ndarray，形状 [p+1,] 或 [p+1, 1]
            当前参数向量。
        X            : np.ndarray，形状 [n, p]
            有标签数据特征矩阵。
        Y            : np.ndarray，形状 [n,] 或 [n, 1]
            有标签数据响应变量（连续值）。
        Z_labeled    : np.ndarray，形状 [n, q]
            有标签数据辅助特征矩阵。
        Z_unlabeled  : np.ndarray 或 None，形状 [N, q]
            无标签数据辅助特征矩阵；None 时退化为标准监督学习。
        lambda_reg   : float，可选
            L2 正则化系数，默认 0。
        use_dress_c1 : bool，可选
            True 时令 c1=0；False 时令 c1=n/(n+N)。

        返回
        ----
        f : float
            半监督加权目标函数值。
        g : np.ndarray，形状 [p+1,]
            目标函数关于 β 的梯度（一维）。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        beta = beta.reshape(-1, 1)
        n = X.shape[0]
        X_aug = self.add_intercept(X)   # [n, p+1]

        residual = X_aug @ beta - Y.reshape(-1, 1)   # 残差 [n, 1]

        # 构造半监督权重向量。c1 的含义与 LogisticModelSpec 中一致。
        if Z_unlabeled is None or (hasattr(Z_unlabeled, 'size') and Z_unlabeled.size == 0):
            # 无无标签数据：退化为标准监督学习（单位权重）
            weight_vec = np.ones(n)
        else:
            N = Z_unlabeled.shape[0]
            if use_dress_c1:
                c1 = 0
            else:
                c1 = n / (n + N) if (n + N) > 0 else 0
            # 半监督权重向量（基于矩条件）
            weight_term = self.semi_supervised_weight_term(Z_labeled, Z_unlabeled)
            weight_vec = np.asarray(c1 + (1 - c1) * weight_term).ravel()
            weight_vec = self._stabilize_weights(weight_vec)

        # 每个样本的均方损失（系数 1/2）[n, 1]
        loss_per = 0.5 * residual ** 2
        # 加权目标函数值
        f = float(np.mean(weight_vec[:, None] * loss_per) + lambda_reg * float(beta.T @ beta))

        # 加权梯度：d/dβ [(1/n)·Σ_i w_i·(1/2)·(X_i·β - Y_i)²]
        # = (1/n)·Σ_i w_i·(X_i·β - Y_i)·X_i
        # 注意：weight_vec[:, None] * (...) 等价于 diag(w_i) @ (...)
        grad_ll = (weight_vec[:, None] * residual * X_aug).sum(axis=0).reshape(-1, 1)
        g = ((1 / n) * grad_ll + 2 * lambda_reg * beta).ravel()
        return f, g

    # ------------------------------------------------------------------
    # 求解器接口（重写为闭式解，对应论文 Section 2.4 推荐做法）
    # ------------------------------------------------------------------
    # 论文 (Song, Lin & Zhou 2023) Section 2.4 第二段开头明确指出：
    #   "For the linear working model in Example 1, a closed-form expression of
    #    θ̂ in (5) is available; ... Moreover, the weights w_i ... could be negative."
    #
    # 因此本类不走 BFGS（负权重会导致样本目标非凸 → BFGS 沿凹方向发散），
    # 直接由 score equation 解出唯一 critical point：
    #
    #   监督：  β̂ = (X'X / n + 2λ I)⁻¹ · (X'Y / n)
    #              = (X'X + 2λn I)⁻¹ · X'Y
    #
    #   半监督：β̂ = (X'WX + 2λn I)⁻¹ · X'WY
    #          其中 W = diag(w_i), w_i = c1 + (1-c1) · weight_term_i
    #          weight_term = Z_labeled @ alpha,
    #          solve((Z_labeled'Z_labeled)/n, mean(Z_unlabeled)) -> alpha
    #
    # 即便 W 含负元素，闭式公式作为 score equation 的唯一解依然是论文的正确估计；
    # 总体层面 E{L_D^w(β)} 是严格凸的，所以这个 critical point 就是渐近上的目标。
    # ------------------------------------------------------------------

    def solve_supervised(self, X, Y, lambda_reg=0.0, initial_value=None,
                         tolerance=5e-3, max_iter=1000):
        """
        监督线性回归闭式解（带 L2 正则化）：
            β̂ = (X_aug' X_aug + 2λn I)⁻¹ · X_aug' Y

        参数与基类 solve_supervised 一致；tolerance / max_iter / initial_value
        在闭式解中不使用（保留接口签名以便统一调用）。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        X_aug = self.add_intercept(X)               # [n, p+1]
        Y_col = np.asarray(Y).reshape(-1, 1)         # [n, 1]
        n, d = X_aug.shape

        A = X_aug.T @ X_aug + 2.0 * lambda_reg * n * np.eye(d)   # [d, d]
        b = X_aug.T @ Y_col                                       # [d, 1]
        beta_hat = self._stable_solve(A, b, X_aug, Y_col, d)
        return beta_hat                                           # [d, 1]

    def solve_semi_supervised(self, X, Y, Z_labeled, Z_unlabeled,
                              lambda_reg=0.0, use_dress_c1=False,
                              initial_value=None, tolerance=5e-3, max_iter=1000,
                              intercept_from_supervised=False):
        """
        半监督线性回归闭式解（论文公式 (4)+(5) 在 squared loss 下的解析形式）：
            β̂ = (X_aug' W X_aug + 2λn I)⁻¹ · X_aug' W Y

        权重构造与 ss_loss_and_grad 严格一致（保证闭式解与目标函数理论一致）。

        intercept_from_supervised : bool (默认 False)
            参考 BaseModelSpec.solve_semi_supervised 同名参数。
            True 时 β̂[0] 用监督 OLS 替换，β̂[1:] 保持 SS。
        """
        lambda_reg = self._effective_lambda(lambda_reg)
        X_aug = self.add_intercept(X)                # [n, p+1]
        Y_col = np.asarray(Y).reshape(-1, 1)          # [n, 1]
        n, d = X_aug.shape

        # ── 构造对角权重向量 w_i，与 ss_loss_and_grad 中一致 ──────────────────
        if (Z_unlabeled is None
                or (hasattr(Z_unlabeled, 'size') and Z_unlabeled.size == 0)):
            # 无无标签数据 → 退化为监督最小二乘
            w_vec = np.ones(n)
        else:
            N = Z_unlabeled.shape[0]
            c1 = 0.0 if use_dress_c1 else (n / (n + N) if (n + N) > 0 else 0.0)
            # 通过解 S alpha = mean(Z_unlabeled) 得到投影项，避免显式形成 S^{-1}。
            weight_term = self.semi_supervised_weight_term(Z_labeled, Z_unlabeled)
            w_vec = c1 + (1.0 - c1) * weight_term     # [n,]
            w_vec = self._stabilize_weights(w_vec)

        # ── 解加权正规方程 (X' W X + 2λn I) β = X' W Y ───────────────────────
        WX = w_vec.reshape(-1, 1) * X_aug             # [n, d]
        A = X_aug.T @ WX + 2.0 * lambda_reg * n * np.eye(d)
        b = X_aug.T @ (w_vec.reshape(-1, 1) * Y_col)
        beta_hat = self._stable_solve(A, b, X_aug, Y_col, d)

        # 分块解耦：截距用监督 OLS 替换
        if intercept_from_supervised:
            beta_sup = self.solve_supervised(X, Y, lambda_reg=lambda_reg)
            beta_hat[0, 0] = beta_sup[0, 0]
        return beta_hat                                # [d, 1]

    # ------------------------------------------------------------------
    # 内部工具：数值稳定线性求解器（防止 W 含负权重导致 β 发散到 Inf）
    # ------------------------------------------------------------------
    @staticmethod
    def _stable_solve(A, b, X_aug, Y_col, d):
        """
        三级防御策略求解 A·β = b：

        Level 1 — 直接解原始正规方程。这是线性工作模型下论文闭式解对应的方程。

        Level 2 — 若原方程奇异、非有限或 |β|_max > 1e6，则加入很小的 Tikhonov
                  ridge 后再解，避免单次病态样本把仿真拖到 Inf。

        Level 3 — 监督 OLS 兜底：完全忽略权重，用纯 X'X 解。这必然 finite,
                  虽损失了 SS 的效率增益但保证 t 这次仿真有可用估计。
        """
        # ---- Level 1: 原始正规方程 ----
        beta_hat = stable_solve(A, b, symmetrize=True)
        if np.all(np.isfinite(beta_hat)) and np.max(np.abs(beta_hat)) <= 1e6:
            return beta_hat

        # ---- Level 2: Tikhonov 兜底 ----
        ridge = max(1e-6, 1e-4 * abs(np.trace(A) / d))
        A_reg = A + ridge * np.eye(d)
        try:
            beta_hat = np.linalg.solve(A_reg, b)
        except np.linalg.LinAlgError:
            beta_hat = stable_solve(A_reg, b, symmetrize=True)

        # ---- Level 3: 健康检查失败时退回监督 OLS ----
        if (not np.all(np.isfinite(beta_hat))
                or np.max(np.abs(beta_hat)) > 1e6):
            A_sup = X_aug.T @ X_aug + 1e-6 * np.eye(d)
            beta_hat = np.linalg.solve(A_sup, X_aug.T @ Y_col)
            print(f"[_stable_solve] level3 fired, fallback max|β|={float(np.max(np.abs(beta_hat))):.3g}")

        return beta_hat
