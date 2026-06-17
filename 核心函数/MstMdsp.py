import numpy as np
import networkx as nx
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from scipy.linalg import eig, sqrtm, norm
import warnings

warnings.filterwarnings('ignore')

from ModelSpec import BaseModelSpec, LogisticModelSpec, stable_sandwich, stable_solve
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


class MstMdsp:
    """
    多源无标签数据筛选与监督基线估计工具类。

    当前实现的主入口是 MstMdsp_sample_selection。代码先把每个 source 的原始 X
    映射为辅助矩阵 Z=[1, X, X^2, ..., X^alpha]，再分别比较：
      - Z 的样本均值；
      - Z 的样本协方差矩阵。

    均值和协方差两条路径分别用 source 之间的距离构造最小生成树（MST）。给定
    lambda_mu 或 lambda_sigma 后，代码删除 MST 中权重大于 lambda 的边，连通分量即为
    当前 lambda 下的 source 聚类。后续 ADMM 相关函数在这些聚类结果上继续选择候选 source。

    重要说明
    --------
    本文件大量保留了 MATLAB 迁移代码的变量名和 1-based 索引习惯：
      - 对外返回的 select_index 多数仍是 1-based；
      - 内部访问 Python list/array 前会再减 1；
      - result_summary 中保存的中间对象字段较多，主要供 SSLogistic/DRESSSSLogistic 读取。

    属性
    ----
    random_seed : int
        写入 NumPy 全局随机状态，用于复现实验划分和随机置换。
    model_spec : BaseModelSpec 子类实例
        监督估计、GBIC 中 score/Hessian 计算所用的模型规范。默认是 LogisticModelSpec。
    """
    def __init__(self, random_seed=123, model_spec=None):
        """
        初始化 MST-MDSP 选择器。
        
        参数
        ----
        random_seed : int, 默认 123
            NumPy 全局随机种子；影响交叉验证划分、置换和模拟流程中的随机步骤。
        model_spec : BaseModelSpec 或 None
            M-估计模型规范。None 时使用 LogisticModelSpec；传入 LinearModelSpec 或自定义
            BaseModelSpec 子类时，损失、梯度、Hessian 和 score 均由该对象提供。
        """
        # self.random_seed = random_seed
        # np.random.seed(random_seed)
        # self.beta_star = None  # 存储估计的真实参数

        self.random_seed = random_seed
        # 初始化全局随机状态（np.random.permutation 等均依赖此）
        np.random.seed(random_seed)
        # 模型规范：默认逻辑回归，可替换为任意 BaseModelSpec 子类
        self.model_spec = model_spec if model_spec is not None else LogisticModelSpec()

    def _build_Z_matrix(self, X, alpha):
        """
        构造半监督矩条件使用的多项式辅助特征矩阵。
        
        Z 的列顺序固定为 [1, X, X^2, ..., X^alpha]，其中幂次是逐元素幂，不是矩阵幂。
        该矩阵用于比较有标签域与每个无标签 source 的均值/协方差，并进一步生成 MST。
        
        参数
        ----
        X : np.ndarray, shape (n, p)
            原始协变量矩阵，不含截距列。
        alpha : int 或 array-like
            多项式最高阶数；若来自 GBIC 通常是形如 [alpha] 的数组，函数内部转为 int。
        
        返回
        ----
        Z : np.ndarray, shape (n, 1 + p * alpha)
            带截距列的多项式辅助特征矩阵。
        """
        return self.model_spec.build_z_matrix(X, alpha)


    def MstMdsp_sample_selection(self, X_labeled, X_unlabeled, Y_labeled, beta_star,
                                 cv_number=None, start_point=None, end_point=None,
                                 multiple_constant=None, num_lambda_mu=None, num_lambda_sigma=None,
                                 num_lambda_1=None, num_lambda_2=None, lambda_start_mu=None,
                                 lambda_start_sigma=None, c_lambda_1_start=None, c_lambda_2_start=None,
                                 k=None, a=None, residual_principle=None, iter_max=None, direct_if=None,
                                 lambda_range=None, numFolds=None):
        """
        执行当前代码实现的完整多源无标签数据筛选流程。
        
        实际执行顺序如下：
        1. cv_mst：选择每个模拟轮次的辅助阶数 alpha，并生成均值/协方差 MST 的 lambda 路径；
        2. cv_penalty_parameter_mu/sigma：估计 ADMM 惩罚参数 lambda_1/lambda_2 的全局搜索范围；
        3. cv_penalty_mu/sigma：在交叉验证结果上选择每轮模拟的最优 lambda 索引；
        4. Mst：在完整数据上重新生成 MST 路径；
        5. admm3_mu_one_simulation 和 admm3_sigma_one_simulation：并行处理每个模拟轮次；
        6. mu_sigma_combine：合并均值路径和协方差路径选择结果。

        函数不会修改输入数组本身；返回值中重复带回 X_labeled、Y_labeled、X_unlabeled 是为了兼容
        旧 MATLAB 风格主程序。
        
        参数
        ----
        X_labeled : np.ndarray, shape (n, p, T)
            有标签协变量；T 为模拟次数。
        X_unlabeled : dict[str, list[np.ndarray]]
            多源无标签数据。每个键对应一个 source，值为长度 T 的列表，列表元素形状为 (N_k, p)。
        Y_labeled : np.ndarray, shape (n, 1, T)
            有标签响应变量。
        beta_star : np.ndarray, shape (p+1, 1)
            评估时使用的目标参数真值或伪真值。
        cv_number, start_point, end_point, multiple_constant : optional
            交叉验证次数、测试集切片比例和 lambda 搜索的几何放大倍数。None 时使用内部默认值。
        num_lambda_mu, num_lambda_sigma : optional
            MST 均值路径/协方差路径的 lambda 网格数量。
        num_lambda_1, num_lambda_2 : optional
            ADMM 二阶段惩罚参数网格数量。
        lambda_start_mu, lambda_start_sigma : optional
            自动扩展 MST lambda 区间时的起始值。
        c_lambda_1_start, c_lambda_2_start : optional
            ADMM 惩罚参数搜索起点。
        k, a, residual_principle, iter_max : optional
            ADMM 更新中的步长/阈值/收敛残差/最大迭代次数。
        direct_if : optional
            传递给 ADMM/lambda 搜索函数的模式标记；当前默认值为 1。
        lambda_range, numFolds : optional
            监督模型正则化交叉验证使用的候选 lambda 和折数。
        
        返回
        ----
        tuple
            (Result, X_labeled, Y_labeled, X_unlabeled, X_unlabeled_combine,
            X_unlabeled_select, select_fields, select_index, all_fields, beta_star)。Result 中保留
            MST、ADMM 和最终选择摘要，后续 SS/DRESS 估计会读取 result_summary。
        """
        # 参数默认值初始化（对应MATLAB的isempty判断）
        cv_number = 10 if cv_number is None else cv_number
        start_point = 0.0 if start_point is None else start_point
        end_point = 0.9 if end_point is None else end_point
        multiple_constant = 1.1 if multiple_constant is None else multiple_constant
        num_lambda_mu = 100 if num_lambda_mu is None else num_lambda_mu
        num_lambda_sigma = 100 if num_lambda_sigma is None else num_lambda_sigma
        lambda_start_mu = 0.001 if lambda_start_mu is None else lambda_start_mu
        lambda_start_sigma = 0.001 if lambda_start_sigma is None else lambda_start_sigma
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        num_lambda_1 = 10 if num_lambda_1 is None else num_lambda_1
        num_lambda_2 = 10 if num_lambda_2 is None else num_lambda_2
        lambda_range = 0.0 if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds
        direct_if = 1 if direct_if is None else direct_if

        # 初始化结果字典（对应MATLAB的struct）
        Result = {}

        # 1. 交叉验证得到MST总体参数范围并生成所有MST结果
        result_mst = self.cv_mst(X_labeled, X_unlabeled, Y_labeled, cv_number, multiple_constant,
                                 num_lambda_mu, num_lambda_sigma, lambda_start_mu, lambda_start_sigma,
                                 start_point, end_point, lambda_range, numFolds)

        # 2. 交叉验证得到惩罚参数(mu/sigma)的总体参数范围
        Information_mu, Lambda_1_small, Lambda_1_big = self.cv_penalty_parameter_mu(
            result_mst, c_lambda_1_start, k, a, residual_principle, iter_max,
            multiple_constant, num_lambda_1
        )
        Information_sigma, Lambda_2_small, Lambda_2_big = self.cv_penalty_parameter_sigma(
            result_mst, c_lambda_2_start, k, a, residual_principle, iter_max,
            multiple_constant, num_lambda_2
        )

        # 3. 交叉验证得到MST和惩罚参数的最优值
        data_mu = self.cv_penalty_mu(
            result_mst, Lambda_1_small, Lambda_1_big, Information_mu, c_lambda_1_start,
            k, a, residual_principle, iter_max, multiple_constant, num_lambda_1, direct_if
        )
        data_sigma = self.cv_penalty_sigma(
            result_mst, Lambda_2_small, Lambda_2_big, Information_sigma, c_lambda_2_start,
            k, a, residual_principle, iter_max, multiple_constant, num_lambda_2, direct_if
        )

        # 4. 生成整体数据的最小生成树
        Tree_mu_output, Tree_sigma_output, Lambda_mu, Lambda_sigma, Lambda_mu_small, Lambda_mu_big, \
            Lambda_sigma_small, Lambda_sigma_big, _, _ = self.Mst(
            X_labeled, Y_labeled, X_unlabeled, X_labeled, Y_labeled, X_unlabeled,
            lambda_start_mu, lambda_start_sigma, multiple_constant, num_lambda_mu, num_lambda_sigma,
            data_mu['Lambda_mu_small'], data_mu['Lambda_mu_big'],
            data_sigma['Lambda_sigma_small'], data_sigma['Lambda_sigma_big'],
            data_mu['result_mst']['Tree_lambda_mu'][0], result_mst['alpha_prepare']
        )

        # 5. 逐次模拟处理
        simulation_times = X_labeled.shape[2]
        X_unlabeled_select = [None] * simulation_times
        result_summary = [None] * simulation_times
        X_unlabeled_combine = [None] * simulation_times

        # ---- 并行化处理每次模拟 ----
        def _process_one_t(t):
            """
            处理单个模拟轮次的最终均值/协方差路径选择。
            
            该内部函数由 ThreadPoolExecutor 并行调用，先分别运行均值路径和协方差路径的
            ADMM 选择，再把轮次编号与两条路径结果一起返回给主线程。
            """
            r_mu = self.admm3_mu_one_simulation(
                t + 1, Tree_mu_output, Lambda_mu, len(Lambda_mu[0]), c_lambda_1_start, k, a,
                residual_principle, iter_max, multiple_constant, num_lambda_1, X_labeled,
                Y_labeled, X_unlabeled, X_labeled, Y_labeled, X_unlabeled, direct_if,
                data_mu['Lambda_1_small'], data_mu['Lambda_1_big'],
                data_mu['which_lambda_1_opt'][t], data_mu['which_lambda_mu_opt'][t]
            )
            r_sigma = self.admm3_sigma_one_simulation(
                t + 1, Tree_sigma_output, Lambda_sigma, len(Lambda_sigma[0]), c_lambda_2_start, k, a,
                residual_principle, iter_max, multiple_constant, num_lambda_2, X_labeled,
                Y_labeled, X_unlabeled, X_labeled, Y_labeled, X_unlabeled, direct_if,
                data_sigma['Lambda_2_small'], data_sigma['Lambda_2_big'],
                data_sigma['which_lambda_2_opt'][t], data_sigma['which_lambda_sigma_opt'][t]
            )
            return t, r_mu, r_sigma

        _sim_bar = tqdm(total=simulation_times, desc='Step 7/7 | 最终数据源选择', unit='sim',
                        ncols=90, colour='green', leave=True)
        max_workers = min(simulation_times, 4)  # 最多 4 线程，避免内存压力
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_process_one_t, t): t for t in range(simulation_times)}
            for future in as_completed(futures):
                t, result_mu_output, result_sigma_output = future.result()
                X_unlabeled_select_one_simulation, result_summary_one_simulation = self.mu_sigma_combine(
                    t + 1, result_mu_output, result_sigma_output, X_unlabeled
                )
                X_unlabeled_select[t] = X_unlabeled_select_one_simulation
                result_summary[t] = result_summary_one_simulation
                fields = list(X_unlabeled.keys())
                if 'combine' in fields:
                    fields.remove('combine')
                combine_store = [X_unlabeled[f][t] for f in fields]
                X_unlabeled_combine[t] = np.vstack(combine_store) if combine_store else np.array([])
                _sim_bar.update(1)
                _sim_bar.set_postfix({'已完成': f'{t+1}次'})
        _sim_bar.close()

        # 封装最终结果
        Result['X_unlabeled_select'] = X_unlabeled_select
        Result['result_summary'] = result_summary
        Result['X_unlabeled_combine'] = X_unlabeled_combine
        Result['X_unlabeled'] = X_unlabeled
        Result['Y_labeled'] = Y_labeled
        Result['X_labeled'] = X_labeled
        Result['beta_star'] = beta_star

        # ========================= 4. 提取结果（对标MATLAB变量赋值） =========================
        # MATLAB: X_labeled = Result.X_labeled; Y_labeled = Result.Y_labeled; X_unlabeled = Result.X_unlabeled;
        X_unlabeled_select = Result['X_unlabeled_select']  # 最终选择的未标记数据【列表】, X_unlabeled_select[0]为选择出的样本数组
        X_unlabeled_combine = Result['X_unlabeled_combine']  # 合并后的未标记数据【列表】, X_unlabeled_combine[0]为合并后的样本数组
        result_summary = Result['result_summary']  # 选择结果汇总（指标/参数等）【列表】,result_summary[0]为选择出的信息表
        all_fields = list(X_unlabeled.keys())
        select_fields = []
        select_index = []
        for i in range(len(result_summary)):
            select_fields.append(result_summary[i]['select_fields'])
            select_index.append(result_summary[i]['select_index'])
        beta_star = Result['beta_star']  # 真实参数（冗余，仅对齐MATLAB）
        X_labeled = Result['X_labeled']  # 更新后的标记特征数据【数组】
        Y_labeled = Result['Y_labeled']  # 更新后的标记标签数据【数组】
        X_unlabeled = Result['X_unlabeled']  # 原始未标记数据【字典】[无标签数据集个数个键和值, X_unlabeled['m1s1'][0]为数组
        # 键为m1s1,...mksk,值为列表，每个列表中包含着模拟次数(通常为1)个数组]
        beta_star = Result['beta_star'] 
        return (Result, X_labeled, Y_labeled, X_unlabeled,
                X_unlabeled_combine, X_unlabeled_select, select_fields, select_index, all_fields, beta_star)

    def cv_mst(self, X_labeled, X_unlabeled, Y_labeled, cv_number, multiple_constant,
               num_lambda_mu, num_lambda_sigma, lambda_start_mu, lambda_start_sigma,
               start_point, end_point, lambda_range, numFolds):
        """
        通过交叉验证确定 MST 聚合阈值的全局搜索范围。
        
        先调用 cv_mst_parameter 得到每个模拟轮次的辅助阶数 alpha、训练/测试划分和
        lambda 上下界；随后在选定划分上构建均值路径与协方差路径的 MST 树。
        
        参数
        ----
        X_labeled, Y_labeled, X_unlabeled : array/dict
            有标签和无标签数据，形状约定同 MstMdsp_sample_selection。
        cv_number : int
            候选交叉验证划分数量。
        multiple_constant : float
            lambda 自动扩展时的几何倍数。
        num_lambda_mu, num_lambda_sigma : int
            均值路径/协方差路径上的 lambda 网格数量。
        lambda_start_mu, lambda_start_sigma : float
            lambda 搜索初值。
        start_point, end_point : float
            测试集在随机置换索引中的切片比例。
        lambda_range, numFolds : array-like, int
            GBIC/监督模型正则化交叉验证参数。
        
        返回
        ----
        result_mst : dict
            包含 Tree_lambda_mu、Tree_lambda_sigma、Lambda_mu、Lambda_sigma、训练/测试数据、
            alpha_prepare 和 best_lambda 等中间结果。
        """
        # 1. 先获取MST参数初始范围
        Tree_before_alpha_store, Lambda_mu_small, Lambda_mu_big, \
            Lambda_sigma_small, Lambda_sigma_big, cv_which_store, best_lambda, alpha_prepare = self.cv_mst_parameter(
            X_labeled, X_unlabeled, Y_labeled, cv_number, multiple_constant,
            num_lambda_mu, num_lambda_sigma, lambda_start_mu, lambda_start_sigma,
            start_point, end_point, lambda_range, numFolds
        )

        # MATLAB中硬编码的cv_number_new=1
        cv_number_new = 1
        Tree_lambda_mu = [None] * cv_number_new
        Tree_lambda_sigma = [None] * cv_number_new
        Lambda_mu = [None] * cv_number_new
        Lambda_sigma = [None] * cv_number_new
        train_X_labeled = [None] * cv_number_new
        train_Y_labeled = [None] * cv_number_new
        train_X_unlabeled = [None] * cv_number_new
        test_X_labeled = [None] * cv_number_new
        test_Y_labeled = [None] * cv_number_new
        test_X_unlabeled = [None] * cv_number_new

        # 逐次交叉验证
        for cv in range(cv_number_new):
            # 数据划分
            train_X_labeled_single_cv, train_Y_labeled_single_cv, train_X_unlabeled_single_cv, \
                test_X_labeled_single_cv, test_Y_labeled_single_cv, test_X_unlabeled_single_cv = self.split(
                X_labeled, X_unlabeled, Y_labeled, start_point, end_point, cv_which_store
            )

            # 生成MST
            Tree_lambda_mu_single_cv, Tree_lambda_sigma_single_cv, Lambda_mu_single_cv, \
                Lambda_sigma_single_cv, Lambda_mu_start_overall, Lambda_mu_end_overall, \
                Lambda_sigma_start_overall, Lambda_sigma_end_overall, num_lambda_mu, num_lambda_sigma = self.Mst(
                train_X_labeled_single_cv, train_Y_labeled_single_cv, train_X_unlabeled_single_cv,
                test_X_labeled_single_cv, test_Y_labeled_single_cv, test_X_unlabeled_single_cv,
                lambda_start_mu, lambda_start_sigma, multiple_constant, num_lambda_mu, num_lambda_sigma,
                Lambda_mu_small, Lambda_mu_big, Lambda_sigma_small, Lambda_sigma_big,
                Tree_before_alpha_store[cv], alpha_prepare
            )

            # 存储结果
            Tree_lambda_mu[cv] = Tree_lambda_mu_single_cv
            Tree_lambda_sigma[cv] = Tree_lambda_sigma_single_cv
            Lambda_mu[cv] = Lambda_mu_single_cv
            Lambda_sigma[cv] = Lambda_sigma_single_cv
            train_X_labeled[cv] = train_X_labeled_single_cv
            train_Y_labeled[cv] = train_Y_labeled_single_cv
            train_X_unlabeled[cv] = train_X_unlabeled_single_cv
            test_X_labeled[cv] = test_X_labeled_single_cv
            test_Y_labeled[cv] = test_Y_labeled_single_cv
            test_X_unlabeled[cv] = test_X_unlabeled_single_cv

        # 封装结果
        result_mst = {
            'Tree_lambda_mu': Tree_lambda_mu,
            'Tree_lambda_sigma': Tree_lambda_sigma,
            'train_X_labeled': train_X_labeled,
            'train_Y_labeled': train_Y_labeled,
            'train_X_unlabeled': train_X_unlabeled,
            'test_X_labeled': test_X_labeled,
            'test_Y_labeled': test_Y_labeled,
            'test_X_unlabeled': test_X_unlabeled,
            'Lambda_mu': Lambda_mu,
            'Lambda_sigma': Lambda_sigma,
            'Lambda_mu_small': Lambda_mu_small,
            'Lambda_mu_big': Lambda_mu_big,
            'Lambda_sigma_small': Lambda_sigma_small,
            'Lambda_sigma_big': Lambda_sigma_big,
            'cv_which_store': cv_which_store,
            'best_lambda': best_lambda,
            'alpha_prepare': alpha_prepare
        }

        return result_mst

    def cv_mst_parameter(self, X_labeled, X_unlabeled, Y_labeled, cv_number, multiple_constant,
                         num_lambda_mu, num_lambda_sigma, lambda_start_mu, lambda_start_sigma,
                         start_point, end_point, lambda_range, numFolds):
        """
        为 MST 阈值搜索准备辅助阶数、交叉验证划分和 lambda 区间。
        
        代码行为分三段：
        1. 对每个模拟轮次调用 base_selection_gbic，得到 alpha_prepare[t]；
        2. 对 cv=1..cv_number 构造随机划分，比较“测试有标签协方差”到
           “训练有标签协方差”和“训练中最接近 source 协方差”的距离差；
        3. 每个模拟轮次选取上述距离差最大的 cv 编号，再用该划分重新生成 MST，
           汇总 lambda_mu/lambda_sigma 的全局上下界。

        注意：这里的 cv_which_store 主要作为随机种子/划分编号使用，并不是 sklearn 意义下
        互斥且覆盖全样本的 K-fold 交叉验证。
        
        参数
        ----
        X_labeled, X_unlabeled, Y_labeled : array/dict
            原始模拟数据。
        cv_number : int
            随机划分次数；每个划分由 cv_which_store 控制随机种子。
        multiple_constant, num_lambda_mu, num_lambda_sigma : numeric
            lambda 区间扩展和网格生成参数。
        lambda_start_mu, lambda_start_sigma : float
            均值/协方差路径搜索初始 lambda。
        start_point, end_point : float
            测试集切片比例。
        lambda_range, numFolds : optional
            base_selection_gbic 使用的正则化候选网格和折数。
        
        返回
        ----
        tuple
            Tree_before_alpha_store、四个 lambda 上下界、cv_which_store、best_lambda、alpha_prepare。
        """
        # 参数默认值
        cv_number = 3 if cv_number is None else cv_number
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_mu = 100 if num_lambda_mu is None else num_lambda_mu
        num_lambda_sigma = 100 if num_lambda_sigma is None else num_lambda_sigma
        lambda_start_mu = 0.001 if lambda_start_mu is None else lambda_start_mu
        lambda_start_sigma = 0.001 if lambda_start_sigma is None else lambda_start_sigma
        start_point = 0.0 if start_point is None else start_point
        end_point = 0.5 if end_point is None else end_point
        lambda_range = np.logspace(-10, 2, 100) if (lambda_range is None or isinstance(lambda_range, (int,
                                                                                                      float)) and lambda_range == 0) else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        simulation_times = X_labeled.shape[2]
        cv_which_panduan_before = np.ones((simulation_times, cv_number))
        alpha_prepare = [None] * simulation_times

        # 1. 计算alpha准备值
        for t in tqdm(range(simulation_times), desc='Step 1/7 | 选择辅助特征阶数 (GBIC)', unit='sim', ncols=90, leave=True):
            alpha_prepare[t], _, best_lambda = self.base_selection_gbic(
                X_labeled[:, :, t], Y_labeled[:, :, t], None, None, None, None,
                None, None, 0, lambda_range, numFolds
            )

        # 2. 交叉验证计算距离判断矩阵
        _cv_bar = tqdm(range(cv_number), desc='Step 2/7 | 构建 MST 参数区间', unit='fold', ncols=90, leave=True)
        for cv in _cv_bar:
            cv_which_store_before_single = (cv + 1) * np.ones((simulation_times, 1))  # MATLAB 1-based
            # 数据划分(有放回抽样cv_number次，划分训练集测试集)
            train_X_labeled, train_Y_labeled, train_X_unlabeled, \
                test_X_labeled, test_Y_labeled, test_X_unlabeled = self.split(
                X_labeled, X_unlabeled, Y_labeled, start_point, end_point, cv_which_store_before_single
            )

            _cv_bar.set_postfix(fold=f'{cv+1}/{cv_number}')
            for t in range(simulation_times):
                # 单次模拟数据提取
                train_X_labeled_one_simulation, train_Y_labeled_one_simulation, train_X_unlabeled_one_simulation, fields = self.one_simulation(
                    train_X_labeled, train_Y_labeled, train_X_unlabeled, t + 1
                )
                test_X_labeled_one_simulation, test_Y_labeled_one_simulation, test_X_unlabeled_one_simulation, fields = self.one_simulation(
                    test_X_labeled, test_Y_labeled, test_X_unlabeled, t + 1
                )

                # 合并标记数据
                X_labeled_one_simulation = np.vstack([train_X_labeled_one_simulation, test_X_labeled_one_simulation])
                Y_labeled_one_simulation = np.vstack([train_Y_labeled_one_simulation, test_Y_labeled_one_simulation])

                # 构建测试集Z矩阵
                test_Z_unlabeled_one_simulation = {}
                for f in range(len(fields)):
                    field_name = fields[f]
                    test_Z_unlabeled_one_simulation[field_name] = self._build_Z_matrix(
                        test_X_unlabeled_one_simulation[field_name], alpha_prepare[t])

                # 测试集标记数据Z矩阵
                test_Z_labeled_one_simulation = self._build_Z_matrix(
                    test_X_labeled_one_simulation, alpha_prepare[t])

                # 测试集统计量计算
                w_test = {}
                element_store = []
                for f in range(len(fields)):
                    element_store.append(np.mean(test_Z_unlabeled_one_simulation[fields[f]], axis=0))
                w_test['labeled_mean'] = np.mean(test_Z_labeled_one_simulation, axis=0)
                w_test['labeled_sigma'] = (1 / test_Z_labeled_one_simulation.shape[0]) * \
                                          (test_Z_labeled_one_simulation.T - w_test['labeled_mean'].reshape(-1, 1) ) @ (
                                                      test_Z_labeled_one_simulation.T - w_test['labeled_mean'].reshape(-1, 1)).T
                # 训练集Z矩阵构建
                train_Z_unlabeled_one_simulation = {}
                for f in range(len(fields)):
                    field_name = fields[f]
                    train_Z_unlabeled_one_simulation[field_name] = self._build_Z_matrix(
                        train_X_unlabeled_one_simulation[field_name], alpha_prepare[t])

                # 训练集标记数据Z矩阵
                train_Z_labeled_one_simulation = self._build_Z_matrix(
                    train_X_labeled_one_simulation, alpha_prepare[t])

                # 训练集统计量计算
                w_train = {}
                element_store = []
                for f in range(len(fields)):
                    element_store.append(np.mean(train_Z_unlabeled_one_simulation[fields[f]], axis=0))
                w_train['labeled_mean'] = np.mean(train_Z_labeled_one_simulation, axis=0)
                w_train['unlabeled_mean'] = np.vstack(element_store)

                # 均值距离计算
                distances = np.linalg.norm((w_train['unlabeled_mean'].T - w_train['labeled_mean'].reshape(-1, 1)).T, ord=2,
                                           axis=1)
                minIndex = np.argmin(distances)
                closestColumn_mean = w_train['unlabeled_mean'][minIndex, :]

                # 协方差计算
                w_train['labeled_sigma'] = (1 / train_Z_labeled_one_simulation.shape[0]) * \
                                           (train_Z_labeled_one_simulation.T - w_train['labeled_mean'].reshape(-1, 1)) @ (
                                                       train_Z_labeled_one_simulation.T - w_train['labeled_mean'].reshape(-1, 1)).T
                w_train['unlabeled_sigma'] = [None] * len(fields)
                for f in range(len(fields)):
                    field_name = fields[f]
                    w_train['unlabeled_sigma'][f] = (1 / train_Z_unlabeled_one_simulation[field_name].shape[0]) * \
                                                    (train_Z_unlabeled_one_simulation[field_name].T - \
                                                     w_train['unlabeled_mean'][f, :].reshape(-1, 1) ) @ \
                                                    (train_Z_unlabeled_one_simulation[field_name].T - \
                                                     w_train['unlabeled_mean'][f, :].reshape(-1, 1)).T
                # 协方差距离计算
                distances = np.zeros(len(fields))
                for f in range(len(fields)):
                    distances[f] = norm(w_train['unlabeled_sigma'][f] - w_train['labeled_sigma'], 'fro')
                min_index = np.argmin(distances)
                closestColumn_sigma = w_train['unlabeled_sigma'][min_index]
                # 距离判断值
                cv_which_panduan_before[t, cv] = norm(w_test['labeled_sigma'] - w_train['labeled_sigma'], 'fro') - \
                                                 norm(w_test['labeled_sigma'] - closestColumn_sigma, 'fro')
        # 3. 确定最优交叉验证索引
        cv_which_store = np.argmax(cv_which_panduan_before, axis=1) + 1  # 转回MATLAB 1-based

        # 4. 重新计算MST参数范围
        cv_number_new = 1
        Lambda_mu_start_store = np.zeros((cv_number_new, 1))
        Lambda_mu_end_store = np.zeros((cv_number_new, 1))
        Lambda_sigma_start_store = np.zeros((cv_number_new, 1))
        Lambda_sigma_end_store = np.zeros((cv_number_new, 1))
        Tree_before_alpha_store = [None] * cv_number_new

        for cv in range(cv_number_new):
            # 数据划分
            train_X_labeled, train_Y_labeled, train_X_unlabeled, \
                test_X_labeled, test_Y_labeled, test_X_unlabeled = self.split(
                X_labeled, X_unlabeled, Y_labeled, start_point, end_point, cv_which_store
            )

            # 生成MST
            Tree_before_alpha, _, _, _, Lambda_mu_small, Lambda_mu_big, \
                Lambda_sigma_small, Lambda_sigma_big, _, _ = self.Mst(
                train_X_labeled, train_Y_labeled, train_X_unlabeled,
                test_X_labeled, test_Y_labeled, test_X_unlabeled,
                lambda_start_mu, lambda_start_sigma, multiple_constant,
                num_lambda_mu, num_lambda_sigma, [], [], [], [], [], alpha_prepare
            )

            Tree_before_alpha_store[cv] = Tree_before_alpha
            Lambda_mu_start_store[cv, 0] = Lambda_mu_small
            Lambda_mu_end_store[cv, 0] = Lambda_mu_big
            Lambda_sigma_start_store[cv, 0] = Lambda_sigma_small
            Lambda_sigma_end_store[cv, 0] = Lambda_sigma_big

        # 计算全局lambda范围
        Lambda_mu_small = np.min(Lambda_mu_start_store)
        Lambda_mu_big = np.max(Lambda_mu_end_store)
        Lambda_sigma_small = np.min(Lambda_sigma_start_store)
        Lambda_sigma_big = np.max(Lambda_sigma_end_store)

        return Tree_before_alpha_store, Lambda_mu_small, Lambda_mu_big, Lambda_sigma_small, Lambda_sigma_big, cv_which_store, best_lambda, alpha_prepare

    def split(self, X_labeled, X_unlabeled, Y_labeled, start_point, end_point, cv_which_store):
        """
        按模拟轮次随机划分有标签和无标签数据为训练集/测试集。

        该函数不是分层抽样，也不是 K-fold 划分。对每个模拟轮次 k，它用
        cv_which_store[k] 设置 NumPy 随机种子，分别打乱有标签样本和每个无标签 source 的样本。
        有标签测试集索引为：

            permute[floor(start_point*n) : n - floor((1-end_point)*n)]

        无标签测试集索引为：

            permute[floor(start_point*N) : floor(end_point*N)]

        两者公式并不完全相同，这是当前代码的真实行为。
        
        参数
        ----
        X_labeled : np.ndarray, shape (n, p, T)
            有标签协变量。
        X_unlabeled : dict[str, list[np.ndarray]]
            无标签 source 字典。若包含 combine 键会在拆分单源时忽略，最后重新生成 combine。
        Y_labeled : np.ndarray, shape (n, 1, T)
            有标签响应。
        start_point, end_point : float
            在随机置换后的索引中截取测试集的起止比例。
        cv_which_store : np.ndarray
            每个模拟轮次使用的随机种子/划分编号，沿用 MATLAB 的 1-based 语义。
        
        返回
        ----
        tuple
            train_X_labeled, train_Y_labeled, train_X_unlabeled, test_X_labeled,
            test_Y_labeled, test_X_unlabeled。
        """
        simulation_times = X_labeled.shape[2]
        n_labeled = X_labeled.shape[0]

        # 初始化训练/测试标记数据
        train_X_labeled = np.zeros((int((1 - end_point) * n_labeled), X_labeled.shape[1], simulation_times))
        train_Y_labeled = np.zeros((int((1 - end_point) * n_labeled), 1, simulation_times))
        test_X_labeled = np.zeros((int( n_labeled - int((1 - end_point) * n_labeled)), X_labeled.shape[1], simulation_times))
        test_Y_labeled = np.zeros((int( n_labeled - int((1 - end_point) * n_labeled)), 1, simulation_times))

        # 标记数据划分
        for k in range(simulation_times):
            # 设置随机种子（对应MATLAB rng）
            np.random.seed(int(cv_which_store[k, 0]) if len(cv_which_store.shape) > 1 else int(cv_which_store[k]))
            permute = np.random.permutation(n_labeled)

            # 划分索引（Python 0-based）
            idx_test_start = int(np.floor(start_point * n_labeled))
            idx_test_end = int(n_labeled - int((1 - end_point) * n_labeled))

            index_labeled_test = permute[idx_test_start:idx_test_end]
            index_labeled_train = np.setdiff1d(permute, index_labeled_test)

            # 存储划分结果
            train_X_labeled[:, :, k] = X_labeled[index_labeled_train, :, k]
            train_Y_labeled[:, :, k] = Y_labeled[index_labeled_train, :, k]
            test_X_labeled[:, :, k] = X_labeled[index_labeled_test, :, k]
            test_Y_labeled[:, :, k] = Y_labeled[index_labeled_test, :, k]

        # 未标记数据划分
        fields = list(X_unlabeled.keys())
        if 'combine' in fields:
            fields.remove('combine')

        train_X_unlabeled = {}
        test_X_unlabeled = {}
        index_unlabeled_test = {}
        index_unlabeled_train = {}

        for f in range(len(fields)):
            field_name = fields[f]
            train_X_unlabeled[field_name] = [None] * simulation_times
            test_X_unlabeled[field_name] = [None] * simulation_times
            n_unlabeled = X_unlabeled[field_name][0].shape[0]

            for k in range(simulation_times):
                # 设置随机种子
                np.random.seed(int(cv_which_store[k, 0]) if len(cv_which_store.shape) > 1 else int(cv_which_store[k]))
                permute = np.random.permutation(n_unlabeled)

                # 划分索引
                idx_test_start = int(np.floor(start_point * n_unlabeled))
                idx_test_end = int(np.floor(end_point * n_unlabeled))
                index_unlabeled_test[field_name] = permute[idx_test_start:idx_test_end]
                index_unlabeled_train[field_name] = np.setdiff1d(permute, index_unlabeled_test[field_name])

                # 存储划分结果
                train_X_unlabeled[field_name][k] = X_unlabeled[field_name][k][index_unlabeled_train[field_name], :]
                test_X_unlabeled[field_name][k] = X_unlabeled[field_name][k][index_unlabeled_test[field_name], :]

        # 组合未标记数据
        train_X_unlabeled['combine'] = [None] * simulation_times
        test_X_unlabeled['combine'] = [None] * simulation_times

        for k in range(simulation_times):
            combined_elements = []
            for f in range(len(fields)):
                combined_elements.append(train_X_unlabeled[fields[f]][k])
            train_X_unlabeled['combine'][k] = np.vstack(combined_elements) if combined_elements else np.array([])

            combined_elements = []
            for f in range(len(fields)):
                combined_elements.append(test_X_unlabeled[fields[f]][k])
            test_X_unlabeled['combine'][k] = np.vstack(combined_elements) if combined_elements else np.array([])

        return train_X_labeled, train_Y_labeled, train_X_unlabeled, test_X_labeled, test_Y_labeled, test_X_unlabeled

    def Mst(self, train_X_labeled, train_Y_labeled, train_X_unlabeled,
            test_X_labeled, test_Y_labeled, test_X_unlabeled, lambda_start_mu,
            lambda_start_sigma, multiple_constant, num_lambda_mu, num_lambda_sigma,
            lambda_mu_input1=None, lambda_mu_input2=None, lambda_sigma_input1=None,
            lambda_sigma_input2=None, Tree_before_alpha=None, alpha_prepare=None):
        """
        批量生成所有模拟轮次的均值路径和协方差路径 MST。
        
        参数
        ----
        train_X_labeled, train_Y_labeled, train_X_unlabeled : array/dict
            训练集有标签和无标签数据。
        test_X_labeled, test_Y_labeled, test_X_unlabeled : array/dict
            测试集数据；主要用于保持接口与交叉验证流程一致。
        lambda_start_mu, lambda_start_sigma : float
            自动搜索 lambda 范围时的起点。
        multiple_constant : float
            lambda 几何扩展倍数。
        num_lambda_mu, num_lambda_sigma : int
            网格点数量。
        lambda_mu_input1, lambda_mu_input2, lambda_sigma_input1, lambda_sigma_input2 : optional
            若给定，则直接在指定上下界内生成 lambda 网格。
        Tree_before_alpha : list/dict, optional
            已生成的 MST 结果，用于复用其中的 alpha。
        alpha_prepare : list, optional
            每个模拟轮次预先选择的多项式阶数。
        
        返回
        ----
        tuple
            Tree_lambda_mu、Tree_lambda_sigma、Lambda_mu、Lambda_sigma 及全局上下界。
        """
        # 空值处理
        lambda_mu_input1 = [] if lambda_mu_input1 is None else lambda_mu_input1
        lambda_mu_input2 = [] if lambda_mu_input2 is None else lambda_mu_input2
        lambda_sigma_input1 = [] if lambda_sigma_input1 is None else lambda_sigma_input1
        lambda_sigma_input2 = [] if lambda_sigma_input2 is None else lambda_sigma_input2
        Tree_before_alpha = [] if Tree_before_alpha is None else Tree_before_alpha

        simulation_times = train_X_labeled.shape[2]
        Tree_lambda_mu = [None] * simulation_times
        Tree_lambda_sigma = [None] * simulation_times
        Lambda_mu = [None] * simulation_times
        Lambda_sigma = [None] * simulation_times
        Lambda_mu_small = [None] * simulation_times
        Lambda_mu_big = [None] * simulation_times
        Lambda_sigma_small = [None] * simulation_times
        Lambda_sigma_big = [None] * simulation_times

        # 逐次模拟生成MST
        for t in range(simulation_times):
            Tree_lambda_mu_single, Tree_lambda_sigma_single, lambda_mu, lambda_sigma, \
                num_lambda_mu, num_lambda_sigma = self.Mst_aggregation(
                t + 1, train_X_labeled, train_Y_labeled, train_X_unlabeled,
                test_X_labeled, test_Y_labeled, test_X_unlabeled, lambda_start_mu,
                lambda_start_sigma, multiple_constant, num_lambda_mu, num_lambda_sigma,
                lambda_mu_input1, lambda_mu_input2, lambda_sigma_input1, lambda_sigma_input2,
                Tree_before_alpha, alpha_prepare
            )

            Tree_lambda_mu[t] = Tree_lambda_mu_single
            Tree_lambda_sigma[t] = Tree_lambda_sigma_single
            Lambda_mu[t] = lambda_mu
            Lambda_sigma[t] = lambda_sigma
            Lambda_mu_small[t] = np.min(lambda_mu)
            Lambda_mu_big[t] = np.max(lambda_mu)
            Lambda_sigma_small[t] = np.min(lambda_sigma)
            Lambda_sigma_big[t] = np.max(lambda_sigma)

        # 计算全局lambda范围
        Lambda_mu_small_global = np.min(Lambda_mu_small)
        Lambda_mu_big_global = np.max(Lambda_mu_big)
        Lambda_sigma_small_global = np.min(Lambda_sigma_small)
        Lambda_sigma_big_global = np.max(Lambda_sigma_big)

        return Tree_lambda_mu, Tree_lambda_sigma, Lambda_mu, Lambda_sigma, \
            Lambda_mu_small_global, Lambda_mu_big_global, Lambda_sigma_small_global, Lambda_sigma_big_global, \
            num_lambda_mu, num_lambda_sigma

    def Mst_aggregation(self, t, train_X_labeled, train_Y_labeled, train_X_unlabeled,
                        test_X_labeled, test_Y_labeled, test_X_unlabeled, lambda_start_mu,
                        lambda_start_sigma, multiple_constant, num_lambda_mu, num_lambda_sigma,
                        lambda_mu_input1, lambda_mu_input2, lambda_sigma_input1, lambda_sigma_input2,
                        Tree_before_alpha, alpha_prepare):
        """
        针对单个模拟轮次生成 MST 聚合路径。
        
        函数先抽取第 t 次模拟的数据，构造辅助矩阵 Z，再调用 mst_generation_single 得到
        均值距离 MST 和协方差距离 MST。若未传入 lambda 上下界，代码会从 lambda_start 开始
        按 multiple_constant 放大，直到聚类数变为 1；随后再反向缩小，直到聚类数回到 source
        个数。最后在该区间上生成对数等距 lambda 网格。
        
        参数
        ----
        t : int
            模拟轮次编号，沿用 MATLAB 习惯，传入值从 1 开始。
        train_X_labeled, train_Y_labeled, train_X_unlabeled : array/dict
            训练集数据。
        test_X_labeled, test_Y_labeled, test_X_unlabeled : array/dict
            测试集数据。
        lambda_start_mu, lambda_start_sigma, multiple_constant : float
            lambda 自动搜索控制参数。
        num_lambda_mu, num_lambda_sigma : int
            lambda 网格点数。
        lambda_mu_input1/2, lambda_sigma_input1/2 : optional
            指定搜索区间。
        Tree_before_alpha, alpha_prepare : optional
            复用的 alpha 或先验树。
        
        返回
        ----
        tuple
            单轮次的均值路径树、协方差路径树、lambda_mu、lambda_sigma 和网格数量。
        """
        # 参数默认值
        t = 1 if t is None else t
        lambda_start_mu = 0.001 if lambda_start_mu is None else lambda_start_mu
        lambda_start_sigma = 0.001 if lambda_start_sigma is None else lambda_start_sigma
        multiple_constant = 1.1 if multiple_constant is None else multiple_constant
        num_lambda_mu = 100 if num_lambda_mu is None else num_lambda_mu
        num_lambda_sigma = 100 if num_lambda_sigma is None else num_lambda_sigma

        # 空值处理
        lambda_mu_input1 = [] if lambda_mu_input1 is None else lambda_mu_input1
        lambda_mu_input2 = [] if lambda_mu_input2 is None else lambda_mu_input2
        lambda_sigma_input1 = [] if lambda_sigma_input1 is None else lambda_sigma_input1
        lambda_sigma_input2 = [] if lambda_sigma_input2 is None else lambda_sigma_input2
        Tree_before_alpha = [] if Tree_before_alpha is None else Tree_before_alpha


        if isinstance(lambda_mu_input1, np.float64):
            lambda_mu_input1 = [lambda_mu_input1]
        if isinstance(lambda_mu_input2, np.float64):
            lambda_mu_input2 = [lambda_mu_input2]
        if isinstance(lambda_sigma_input1, np.float64):
            lambda_sigma_input1 = [lambda_sigma_input1]
        if isinstance(lambda_sigma_input2, np.float64):
            lambda_sigma_input2 = [lambda_sigma_input2]

        # 分支1：无lambda输入，自动计算lambda范围
        if len(lambda_mu_input1) == 0:
            # 提取单次模拟数据
            train_X_labeled_one_simulation, train_Y_labeled_one_simulation, train_X_unlabeled_one_simulation, fields = self.one_simulation(
                train_X_labeled, train_Y_labeled, train_X_unlabeled, t
            )
            test_X_labeled_one_simulation, test_Y_labeled_one_simulation, test_X_unlabeled_one_simulation, fields = self.one_simulation(
                test_X_labeled, test_Y_labeled, test_X_unlabeled, t
            )

            # 合并标记数据
            X_labeled_one_simulation = np.vstack([train_X_labeled_one_simulation, test_X_labeled_one_simulation])
            Y_labeled_one_simulation = np.vstack([train_Y_labeled_one_simulation, test_Y_labeled_one_simulation])

            # 生成MST
            T_mu, T_sigma, w, fields, train_Z_labeled_one_simulation, \
                train_Z_unlabeled_one_simulation, alpha = self.mst_generation_single(
                train_X_labeled_one_simulation, train_Y_labeled_one_simulation,
                train_X_unlabeled_one_simulation, fields, X_labeled_one_simulation,
                Y_labeled_one_simulation, alpha_prepare[t - 1]
            )
            # 自动搜索lambda_mu范围
            lambda_mu_step = lambda_start_mu
            Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu_step,
                                                 train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                 alpha)
            while Cluster_mu['cluster_number'] > 1:
                lambda_mu_step *= multiple_constant
                Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu_step,
                                                     train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                     alpha)
            lambda_end_mu = lambda_mu_step

            lambda_mu_step = lambda_end_mu
            Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu_step,
                                                 train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                 alpha)
            while Cluster_mu['cluster_number'] < len(Cluster_mu['fields']):
                lambda_mu_step /= multiple_constant
                Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu_step,
                                                     train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                     alpha)
            lambda_start_mu = lambda_mu_step
            # 自动搜索lambda_sigma范围
            lambda_sigma_step = lambda_start_sigma
            Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma_step,
                                                       train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                       alpha)
            while Cluster_sigma['cluster_number'] > 1:
                lambda_sigma_step *= multiple_constant
                Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma_step,
                                                           train_Z_labeled_one_simulation,
                                                           train_Z_unlabeled_one_simulation, alpha)
            lambda_end_sigma = lambda_sigma_step

            lambda_sigma_step = lambda_end_sigma
            Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma_step,
                                                       train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                       alpha)
            while Cluster_sigma['cluster_number'] < len(Cluster_sigma['fields']):
                lambda_sigma_step /= multiple_constant
                Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma_step,
                                                           train_Z_labeled_one_simulation,
                                                           train_Z_unlabeled_one_simulation, alpha)
            lambda_start_sigma = lambda_sigma_step

            # 生成lambda数组（对数间距）
            lambda_mu = np.exp(np.linspace(np.log(lambda_start_mu), np.log(lambda_end_mu), num_lambda_mu))
            lambda_sigma = np.exp(np.linspace(np.log(lambda_start_sigma), np.log(lambda_end_sigma), num_lambda_sigma))

            # 生成不同lambda下的MST树
            Tree_lambda_mu_single = {}
            for i in range(len(lambda_mu)):
                Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu[i],
                                                     train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                     alpha)
                Tree_lambda_mu_single[f'lambda_mu{i + 1}'] = Cluster_mu  # MATLAB 1-based命名

            Tree_lambda_sigma_single = {}
            for i in range(len(lambda_sigma)):
                Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma[i],
                                                           train_Z_labeled_one_simulation,
                                                           train_Z_unlabeled_one_simulation, alpha)
                Tree_lambda_sigma_single[f'lambda_sigma{i + 1}'] = Cluster_sigma
        # 分支2：有lambda输入，使用指定范围
        else:
            # 提取单次模拟数据
            train_X_labeled_one_simulation, train_Y_labeled_one_simulation, train_X_unlabeled_one_simulation, fields = self.one_simulation(
                train_X_labeled, train_Y_labeled, train_X_unlabeled, t
            )
            test_X_labeled_one_simulation, test_Y_labeled_one_simulation, test_X_unlabeled_one_simulation, fields = self.one_simulation(
                test_X_labeled, test_Y_labeled, test_X_unlabeled, t
            )

            # 合并标记数据
            X_labeled_one_simulation = np.vstack([train_X_labeled_one_simulation, test_X_labeled_one_simulation])
            Y_labeled_one_simulation = np.vstack([train_Y_labeled_one_simulation, test_Y_labeled_one_simulation])

            # 生成MST（使用先验alpha）
            T_mu, T_sigma, w, fields, train_Z_labeled_one_simulation, \
                train_Z_unlabeled_one_simulation, alpha = self.mst_generation_single(
                train_X_labeled_one_simulation, train_Y_labeled_one_simulation,
                train_X_unlabeled_one_simulation, fields, X_labeled_one_simulation,
                Y_labeled_one_simulation, Tree_before_alpha[t - 1][f'lambda_mu1']['alpha']
            )

            # 生成指定范围的lambda数组
            lambda_mu = np.exp(np.linspace(np.log(lambda_mu_input1), np.log(lambda_mu_input2), num_lambda_mu))
            lambda_sigma = np.exp(
                np.linspace(np.log(lambda_sigma_input1), np.log(lambda_sigma_input2), num_lambda_sigma))

            # 生成不同lambda下的MST树
            Tree_lambda_mu_single = {}
            for i in range(len(lambda_mu)):
                Cluster_mu = self.mst_mu_aggregation(T_mu, w, fields, lambda_mu[i],
                                                     train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation,
                                                     alpha)
                Tree_lambda_mu_single[f'lambda_mu{i + 1}'] = Cluster_mu

            Tree_lambda_sigma_single = {}
            for i in range(len(lambda_sigma)):
                Cluster_sigma = self.mst_sigma_aggregation(T_sigma, w, fields, lambda_sigma[i],
                                                           train_Z_labeled_one_simulation,
                                                           train_Z_unlabeled_one_simulation, alpha)
                Tree_lambda_sigma_single[f'lambda_sigma{i + 1}'] = Cluster_sigma

        return Tree_lambda_mu_single, Tree_lambda_sigma_single, lambda_mu, lambda_sigma, num_lambda_mu, num_lambda_sigma

    def one_simulation(self, train_X_labeled, train_Y_labeled, train_X_unlabeled, t):
        """
        抽取第 t 个模拟轮次的有标签和无标签数据。
        
        参数
        ----
        train_X_labeled : np.ndarray, shape (n, p, T)
            多轮次有标签协变量。
        train_Y_labeled : np.ndarray, shape (n, 1, T)
            多轮次有标签响应。
        train_X_unlabeled : dict[str, list[np.ndarray]]
            多源无标签数据；combine 键会被排除。
        t : int
            MATLAB 风格的 1-based 模拟编号。
        
        返回
        ----
        tuple
            X_t、Y_t、X_unlabeled_t 字典和 source 名称列表 fields。
        """
        # 转换为Python 0-based
        t_idx = t - 1

        # 提取标记数据
        train_X_labeled_one_simulation = train_X_labeled[:, :, t_idx]
        train_Y_labeled_one_simulation = train_Y_labeled[:, :, t_idx]

        # 提取未标记数据
        train_X_unlabeled_one_simulation = {}
        fields = []
        if train_X_unlabeled is not None and len(train_X_unlabeled) > 0:
            fields = list(train_X_unlabeled.keys())
            if 'combine' in fields:
                fields.remove('combine')
            for f in fields:
                train_X_unlabeled_one_simulation[f] = train_X_unlabeled[f][t_idx]
        else:
            train_X_unlabeled_one_simulation = None

        return train_X_labeled_one_simulation, train_Y_labeled_one_simulation, train_X_unlabeled_one_simulation, fields

    def mst_generation_single(self, train_X_labeled_one_simulation, train_Y_labeled_one_simulation,
                              train_X_unlabeled_one_simulation, fields, X_labeled_one_simulation,
                              Y_labeled_one_simulation, alpha):
        """
        为单个模拟轮次计算 MST 所需的统计量和最小生成树。
        
        参数
        ----
        train_X_labeled_one_simulation, train_Y_labeled_one_simulation : np.ndarray
            单轮训练集有标签数据。
        train_X_unlabeled_one_simulation : dict[str, np.ndarray]
            单轮训练集多源无标签数据。
        fields : list[str]
            参与计算的 source 名称。
        X_labeled_one_simulation, Y_labeled_one_simulation : np.ndarray
            训练集和测试集合并后的有标签数据，用于必要时重新选择 alpha。
        alpha : int/list/None
            多项式辅助特征阶数；为空时调用 base_selection_gbic 自动选择。
        
        返回
        ----
        tuple
            T_mu、T_sigma、统计量字典 w、fields、Z_labeled、Z_unlabeled 和 alpha。
        """
        # 自动计算alpha（如果未指定）
        if alpha is None or len(alpha) == 0:
            alpha, _, best_lambda = self.base_selection_gbic(
                X_labeled_one_simulation, Y_labeled_one_simulation, None, None, None, None,
                None, None, 0, None, None
            )

        # 构建未标记数据Z矩阵
        train_Z_unlabeled_one_simulation = {}
        for f in range(len(fields)):
            field_name = fields[f]
            train_Z_unlabeled_one_simulation[field_name] = self._build_Z_matrix(
                train_X_unlabeled_one_simulation[field_name], alpha)

        # 构建标记数据Z矩阵
        train_Z_labeled_one_simulation = self._build_Z_matrix(
            train_X_labeled_one_simulation, alpha)

        # 计算统计量w
        w = {}
        element_store = []
        for f in range(len(fields)):
            element_store.append(np.mean(train_Z_unlabeled_one_simulation[fields[f]], axis=0))
        w['labeled_mean'] = np.mean(train_Z_labeled_one_simulation, axis=0)
        w['unlabeled_mean'] = np.vstack(element_store)

        # ---- 构建均值（mu）路径的距离矩阵和 MST ----
        # 以各未标记数据源的特征均值向量为节点，欧氏距离为边权，构建全连通图
        # squareform(pdist(...)) 一次性计算所有节点对的成对距离矩阵
        D_mu = squareform(pdist(w['unlabeled_mean']))
        # 将精确为0的距离（除对角线外不应出现）设为极小值，避免 MST 边权为0
        D_mu[D_mu == 0] = 1e-10
        np.fill_diagonal(D_mu, 0)  # 对角线（自身距离）保持为0
        G_mu = nx.from_numpy_array(D_mu)  # 构建加权无向图
        T_mu = nx.minimum_spanning_tree(G_mu)  # Kruskal/Prim 算法求最小生成树

        # ---- 计算标记数据和各未标记数据源的协方差矩阵 ----
        # 标记数据协方差：Σ_L = (1/n_L) * (Z_L - μ_L)^T (Z_L - μ_L)
        w['labeled_sigma'] = (1 / train_Z_labeled_one_simulation.shape[0]) * \
                             (train_Z_labeled_one_simulation.T - w['labeled_mean'].reshape(-1,1)) @ (
                                         train_Z_labeled_one_simulation.T - w['labeled_mean'].reshape(-1,1)).T
        w['unlabeled_sigma'] = [None] * len(fields)

        for f in range(len(fields)):
            field_name = fields[f]
            # 第 f 个未标记数据源的协方差：Σ_f = (1/n_f) * (Z_f - μ_f)^T (Z_f - μ_f)
            w['unlabeled_sigma'][f] = (1 / train_Z_unlabeled_one_simulation[field_name].shape[0]) * \
                                      (train_Z_unlabeled_one_simulation[field_name].T - w['unlabeled_mean'][f, :].reshape(-1,1)) @ \
                                      (train_Z_unlabeled_one_simulation[field_name].T - w['unlabeled_mean'][f, :].reshape(-1,1)).T

        # ---- 构建协方差（sigma）路径的距离矩阵和 MST ----
        # 以各未标记数据源的协方差矩阵为节点，Frobenius 范数距离为边权
        D_sigma = np.zeros((len(fields), len(fields)))
        for f1 in range(len(fields)):
            for f2 in range(len(fields)):
                # Frobenius 范数衡量两个协方差矩阵的差异
                D_sigma[f1, f2] = norm(w['unlabeled_sigma'][f1] - w['unlabeled_sigma'][f2], 'fro')

        D_sigma[D_sigma == 0] = 1e-10  # 同样避免精确零边权
        np.fill_diagonal(D_sigma, 0)
        G_sigma = nx.from_numpy_array(D_sigma)
        T_sigma = nx.minimum_spanning_tree(G_sigma)  # 协方差路径的 MST

        return T_mu, T_sigma, w, fields, train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation, alpha

    def mst_mu_aggregation(self, T_mu, w, fields, lambda_mu,
                           train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation, alpha):
        """
        在给定 lambda_mu 下对均值 MST 进行边收缩聚类。
        
        参数
        ----
        T_mu : networkx.Graph
            以 source 均值距离为边权的最小生成树。
        w : dict
            含 labeled_mean、unlabeled_mean 等统计量。
        fields : list[str]
            source 名称。
        lambda_mu : float
            均值路径聚合阈值；边权小于该阈值的节点倾向于被合并。
        train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation : array/dict
            有标签/无标签辅助矩阵。
        alpha : int
            辅助矩阵多项式阶数。
        
        返回
        ----
        Cluster_mu : dict
            聚类个数、每个聚类包含的 source、与有标签均值最接近的候选等信息。
        """
        # 裁剪MST（移除权重>lambda_mu的边）
        edges_to_remove = [(u, v) for u, v, d in T_mu.edges(data=True) if d['weight'] > lambda_mu]
        T_mu_cut = T_mu.copy()
        T_mu_cut.remove_edges_from(edges_to_remove)

        # 计算连通分量（聚类）
        clusters = list(nx.connected_components(T_mu_cut))
        num_clusters = len(clusters)

        # 构建聚类标签（映射节点到聚类ID）
        cluster_labels = np.zeros(len(fields), dtype=int)
        for i, cluster in enumerate(clusters):
            for node in cluster:
                cluster_labels[node] = i

        # 初始化聚类结果
        Cluster_mu = {
            'cluster_number': num_clusters,
            'index': [None] * num_clusters,
            'index_mean': [None] * num_clusters,
            'fields': fields,
            'train_Z_unlabeled_one_simulation': train_Z_unlabeled_one_simulation,
            'train_Z_labeled_one_simulation': train_Z_labeled_one_simulation,
            'labeled_mean': w['labeled_mean'],
            'unlabeled_mean': w['unlabeled_mean'],
            'alpha': alpha
        }

        # 计算每个聚类的索引和均值
        for i in range(num_clusters):
            Cluster_mu['index'][i] = np.where(cluster_labels == i)[0] + 1  # 转回MATLAB 1-based
            new_z_mean = np.zeros_like(w['labeled_mean']).reshape(-1, 1)
            n_sum = 0

            for j in range(len(Cluster_mu['index'][i])):
                which = Cluster_mu['index'][i][j] - 1  # Python 0-based
                field_name = fields[which]
                n = train_Z_unlabeled_one_simulation[field_name].shape[0]
                new_z_mean += w['unlabeled_mean'][which, :].reshape(-1, 1) * n
                n_sum += n

            if n_sum > 0:
                Cluster_mu['index_mean'][i] = new_z_mean / n_sum
            else:
                Cluster_mu['index_mean'][i] = new_z_mean

        return Cluster_mu

    def mst_sigma_aggregation(self, T_sigma, w, fields, lambda_sigma,
                              train_Z_labeled_one_simulation, train_Z_unlabeled_one_simulation, alpha):
        """
        在给定 lambda_sigma 下对协方差 MST 进行边收缩聚类。
        
        参数与 mst_mu_aggregation 类似，但距离度量由均值向量欧氏距离替换为协方差矩阵
        Frobenius 距离。
        
        返回
        ----
        Cluster_sigma : dict
            协方差路径下的聚类结果、候选 source 和诊断统计量。
        """
        # 裁剪MST（移除权重>lambda_sigma的边）
        edges_to_remove = [(u, v) for u, v, d in T_sigma.edges(data=True) if d['weight'] > lambda_sigma]
        T_sigma_cut = T_sigma.copy()
        T_sigma_cut.remove_edges_from(edges_to_remove)

        # 计算连通分量（聚类）
        clusters = list(nx.connected_components(T_sigma_cut))
        num_clusters = len(clusters)

        # 构建聚类标签
        cluster_labels = np.zeros(len(fields), dtype=int)
        for i, cluster in enumerate(clusters):
            for node in cluster:
                cluster_labels[node] = i

        # 初始化聚类结果
        Cluster_sigma = {
            'cluster_number': num_clusters,
            'index': [None] * num_clusters,
            'index_sigma': [None] * num_clusters,
            'fields': fields,
            'train_Z_unlabeled_one_simulation': train_Z_unlabeled_one_simulation,
            'train_Z_labeled_one_simulation': train_Z_labeled_one_simulation,
            'labeled_sigma': w['labeled_sigma'],
            'unlabeled_sigma': w['unlabeled_sigma'],
            'alpha': alpha
        }

        # 计算每个聚类的索引和协方差
        for i in range(num_clusters):
            Cluster_sigma['index'][i] = np.where(cluster_labels == i)[0] + 1  # 转回MATLAB 1-based
            new_z_sigma = np.zeros_like(w['labeled_sigma'])
            n_sum = 0

            for j in range(len(Cluster_sigma['index'][i])):
                which = Cluster_sigma['index'][i][j] - 1  # Python 0-based
                field_name = fields[which]
                n = train_Z_unlabeled_one_simulation[field_name].shape[0]
                new_z_sigma += w['unlabeled_sigma'][which] * n
                n_sum += n

            if n_sum > 0:
                Cluster_sigma['index_sigma'][i] = new_z_sigma / n_sum
            else:
                Cluster_sigma['index_sigma'][i] = new_z_sigma

        return Cluster_sigma

    def base_selection_gbic(self, X_labeled, Y_labeled, tolerance=None, max_iter=None,
                            initial_value=None, beta_star=None, alpha_up=None, alpha_down=None,
                            CP_if=None, lambda_range=None, numFolds=None):
        """
        使用 GBIC 在候选多项式阶数 alpha 中选择辅助特征复杂度。
        
        对 alpha_up 到 alpha_down 的候选阶数，函数先用当前 model_spec 拟合监督模型，
        再基于 score、Hessian 和辅助矩阵 Z 计算广义 BIC，选择准则最小的 alpha。
        
        参数
        ----
        X_labeled, Y_labeled : np.ndarray
            单轮次有标签数据。
        tolerance, max_iter, initial_value : optional
            监督模型优化器参数。
        beta_star : np.ndarray or None
            评估参考参数；GBIC 本身不依赖该值。
        alpha_up, alpha_down : int
            alpha 搜索下界和上界。
        CP_if, lambda_range, numFolds : optional
            是否交叉验证正则化参数及其候选网格/折数。
        
        返回
        ----
        alpha : np.ndarray
            选择的多项式阶数，保持 MATLAB 兼容的数组形式。
        GBIC_store : np.ndarray
            各候选 alpha 的 GBIC 值。
        best_lambda : float
            监督模型交叉验证选出的正则化参数。
        """
        # 参数默认值
        alpha_up = 5 if alpha_up is None else alpha_up
        alpha_down = 1 if alpha_down is None else alpha_down
        CP_if = 0 if CP_if is None else CP_if
        lambda_range = np.logspace(-10, 2, 100) if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        # 适配输入维度（处理单次模拟数据）
        if len(X_labeled.shape) == 2:
            X_labeled = X_labeled.reshape(X_labeled.shape[0], X_labeled.shape[1], 1)
        if len(Y_labeled.shape) == 2:
            Y_labeled = Y_labeled.reshape(Y_labeled.shape[0], Y_labeled.shape[1], 1)

        simulation_times = X_labeled.shape[2]
        alpha = np.zeros(simulation_times)
        best_lambda = np.zeros(simulation_times)
        GBIC = np.zeros((alpha_up, simulation_times))

        # 逐次模拟计算GBIC
        for t in range(simulation_times):
            X = X_labeled[:, :, t]
            Y = Y_labeled[:, t].reshape(-1, 1)

            # 求解监督 model_spec，得到 score 计算所需的 beta_hat 和 lambda
            beta_hat, _, best_lambda_hat = self.solve_logistic_regression(
                X.reshape(X.shape[0], X.shape[1], 1), Y.reshape(Y.shape[0], 1, 1),
                None, None, None, None, CP_if, lambda_range, numFolds
            )

            # # 计算Partial_l
            # X_aug = np.hstack([np.ones((X.shape[0], 1)), X])
            # exp_term = np.exp(X_aug @ beta_hat[:, t].reshape(-1, 1))  #!!!!!!!!!!!!!!!!!!改t为0
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

            # 计算得分函数矩阵 Partial_l（已泛化：委托给 self.model_spec）
            # 含义：score[i, j] = ∂ℓ_i / ∂β_j，即第 i 个样本对第 j 个参数的得分（梯度）
            # 注：原代码用 beta_hat[:,0]（固定用第0次模拟的beta），此处保持一致
            Partial_l = self.model_spec.score(
                beta_hat[:, 0].reshape(-1, 1), X, Y, best_lambda_hat[t]
            )  # shape: [n, p+1]

            best_lambda[t] = best_lambda_hat[t]

            # 对多项式阶数 alpha 从 alpha_down 到 alpha_up 逐一计算 GBIC
            # GBIC (Generalized Bayes Information Criterion) 用于选择最优的辅助特征阶数
            for a in range(alpha_down, alpha_up + 1):
                # 构建论文原始辅助矩阵 Z = [1, X, X^2, ..., X^a]。
                # 统一走 model_spec.build_z_matrix，避免各处手写 Z 构造不一致。
                Z_first = self._build_Z_matrix(X_labeled[:, :, 0], a)
                Z = np.zeros((X_labeled.shape[0], Z_first.shape[1], X_labeled.shape[2]))
                Z[:, :, 0] = Z_first
                for tt in range(1, Z.shape[2]):
                    Z[:, :, tt] = self._build_Z_matrix(X_labeled[:, :, tt], a)
                z_df = Z.shape[1] - 1

                # 用 Z 对 Partial_l 做线性回归，估计辅助模型参数 gamma_hat
                # gamma_hat = (Z'Z)^{-1} Z' * Partial_l，即最小二乘估计
                Z_t = Z[:, :, t]
                ZtZ = Z_t.T @ Z_t
                gamma_hat = stable_solve(ZtZ, Z_t.T @ Partial_l, symmetrize=True)
                # 残差矩阵：e[i, j] 为第 i 个样本在第 j 个坐标方向的残差
                e = Partial_l - Z_t @ gamma_hat

                # 计算残差方差（每列独立）：σ^2 = Σe^2 / (n - dim(Z))。
                denom = max(X_labeled.shape[0] - z_df - 1, 1)
                sigma = np.sum(e ** 2, axis=0) * (1 / denom)
                # 按参数维度逐一累加 GBIC
                for j in range((X.shape[1]+1)):
                    # term1：标准 BIC 项 = Σe^2/σ + n*ln(σ) + ln(n)*dim(Z非截距)
                    term1 = (np.sum(e[:, j] ** 2) / sigma[j] + X_labeled.shape[0] * np.log(sigma[j]) +
                             np.log(X_labeled.shape[0]) * z_df)
                    # term2：修正项。diag(e_j) @ Z 等价于逐行 e_j * Z，避免构造 n×n 对角矩阵。
                    weighted_Z = e[:, [j]] * Z_t
                    residual_info = weighted_Z.T @ weighted_Z
                    scaled_info = stable_solve(
                        ZtZ * sigma[j], residual_info, symmetrize=True
                    )
                    term2 = np.trace(scaled_info)
                    # term3：-log det 项，对行列式取对数后取负，与 term2 共同构成广义 BIC 修正
                    term3 = -np.log(np.linalg.det(scaled_info))
                    # 将三项之和（除以 n 归一化）累加到 GBIC[a-1, t]
                    GBIC[a - 1, t] += (1 / X_labeled.shape[0]) * (term1 + term2 + term3)

        # 选择最优alpha
        for t in range(simulation_times):
            alpha[t] = int(np.argmin(GBIC[:, t]) + 1)

        return alpha, GBIC, best_lambda

    def solve_logistic_regression(self, X_labeled, Y_labeled, tolerance=None, max_iter=None,
                                  initial_value=None, beta_star=None, CP_if=None,
                                  lambda_range=None, numFolds=None):
        """
        求解监督 M-估计，并计算多种评估指标（Bias、SE、MSE，可选 CP/SSE）。

        函数名保留 logistic 是为了兼容旧主程序；实际优化目标由 self.model_spec 实现。

        参数
        ----
        X_labeled : np.ndarray, shape [n, p, T]
            标记样本特征矩阵，T 为模拟次数（第三维）。
        Y_labeled : np.ndarray, shape [n, 1, T]
            标记样本标签。
        tolerance : float, 可选
            优化器收敛梯度阈值，默认 5e-3。
        max_iter : int, 可选
            优化器最大迭代次数，默认 500。
        initial_value : np.ndarray, shape [p+1, 1], 可选
            参数初始值，默认全零（含截距项）。
        beta_star : np.ndarray, shape [p+1, 1], 可选
            真实参数，用于计算 Bias/MSE，默认全1。
        CP_if : int, 可选
            是否计算置信区间覆盖概率（CP）。0=否，1=是，默认0。
        lambda_range : np.ndarray, 可选
            正则化参数搜索范围，默认 logspace(-10, 2, 100)。
        numFolds : int, 可选
            交叉验证折数，默认5。

        返回
        ----
        beta_hat : np.ndarray, shape [p+1, T]
            每次模拟的参数估计值（每列对应一次模拟）。
        Evaluate : pd.DataFrame
            评估结果表（含 Bias/SE/MSE，CP_if=1 时额外含 SSE/CP）。
        best_lambda_hat : np.ndarray, shape [T,]
            每次模拟选出的最优正则化参数。
        """
        # 参数默认值
        tolerance = 5e-3 if tolerance is None else tolerance
        max_iter = 500 if max_iter is None else max_iter
        initial_value = np.zeros((X_labeled.shape[1] + 1, 1)) if initial_value is None else initial_value
        beta_star = np.vstack([np.ones((1, 1)), np.ones((X_labeled.shape[1], 1))]) if beta_star is None else beta_star
        CP_if = 0 if CP_if is None else CP_if
        lambda_range = np.logspace(-10, 2, 100) if lambda_range is None else lambda_range
        numFolds = 5 if numFolds is None else numFolds

        # 初始化结果
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

        # 计算评估指标
        Bias = (1 / X_labeled.shape[2]) * np.sum(beta_hat - beta_star, axis=1).reshape(-1, 1)
        SE = np.sqrt(np.mean((beta_hat - np.mean(beta_hat, axis=1).reshape(-1, 1)) ** 2, axis=1)).reshape(-1, 1)
        MSE = np.mean((beta_hat - beta_star) ** 2, axis=1).reshape(-1, 1)

        # CP相关计算（如果开启）
        SSE_every = np.ones((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        CP_every = np.ones((X_labeled.shape[1] + 1, X_labeled.shape[2]))
        SE_SSE_ratio_every = np.zeros_like(SSE_every)
        SE_SSE_ratio_mean_every = np.zeros_like(SSE_every)

        if CP_if == 1:
            for t in range(X_labeled.shape[2]):
                beta_hat_one_simulation = beta_hat[:, t].reshape(-1, 1)
                X = X_labeled[:, :, t]
                Y = Y_labeled[:, :, t]

                # 计算Partial_l（已泛化：委托给 self.model_spec）
                Partial_l = self.model_spec.score(
                    beta_hat_one_simulation, X, Y, best_lambda_hat[t]
                )  # [n, p+1]

                # 计算协方差矩阵V和M（已泛化：M 委托给 self.model_spec）
                V = (1 / Partial_l.shape[0]) * (Partial_l - np.mean(Partial_l, axis=0)).T @ (
                            Partial_l - np.mean(Partial_l, axis=0))
                M = self.model_spec.hessian(beta_hat_one_simulation, X, Y, best_lambda_hat[t])

                # --- 防御性检查：M / V 非有限值时跳过本次仿真 ---
                # 监督估计若 BFGS 未收敛或线性模型残差爆炸，M/V 可能含 inf/nan
                if not (np.all(np.isfinite(M)) and np.all(np.isfinite(V))):
                    SSE_every[:, t] = np.nan
                    CP_every[:, t] = np.nan
                    SE_SSE_ratio_every[:, t] = np.nan
                    SE_SSE_ratio_mean_every[:, t] = np.nan
                    print(f"[警告 MstMdsp-supervised] t={t}: 渐近方差含非有限值 "
                          f"(M_finite={bool(np.all(np.isfinite(M)))}, V_finite={bool(np.all(np.isfinite(V)))}, "
                          f"beta_finite={bool(np.all(np.isfinite(beta_hat_one_simulation)))}); 该次仿真已置 NaN")
                    continue

                SIGMA = stable_sandwich(M, V) / Partial_l.shape[0]

                # 计算置信区间和覆盖概率
                BIAS = np.zeros(SIGMA.shape[0])
                SSE_every[:, t] = np.sqrt(np.diag(SIGMA))
                up = (beta_hat_one_simulation.ravel() - BIAS) + 1.96 * SSE_every[:, t]
                down = (beta_hat_one_simulation.ravel() - BIAS) - 1.96 * SSE_every[:, t]
                CP_every[:, t] = (beta_star.ravel() >= down) & (beta_star.ravel() <= up)
                SE_SSE_ratio_every[:, t] = SE.ravel() / SSE_every[:, t]
                SE_SSE_ratio_mean_every[:, t] = (np.sum(SE) / np.sum(SSE_every[:, t])) * np.ones(SE.shape[0])

            # 处理NaN值
            nan_cols = np.any(np.isnan(SSE_every), axis=0)
            SSE_every = SSE_every[:, ~nan_cols]
            CP_every = CP_every[:, ~nan_cols]
            SE_SSE_ratio_every = SE_SSE_ratio_every[:, ~nan_cols]
            SE_SSE_ratio_mean_every = SE_SSE_ratio_mean_every[:, ~nan_cols]

            # 计算评估指标均值
            SSE = np.mean(SSE_every, axis=1).reshape(-1, 1)
            CP = np.mean(CP_every, axis=1).reshape(-1, 1)
            SE_SSE_ratio = np.mean(SE_SSE_ratio_every, axis=1).reshape(-1, 1)
            SE_SSE_ratio_mean = np.mean(SE_SSE_ratio_mean_every, axis=1).reshape(-1, 1)

            # 封装评估结果为DataFrame（替代MATLAB的table）
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
            # 封装评估结果为DataFrame（替代MATLAB的table）
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
        对单次模拟数据执行监督 M-估计（带 L2 正则化）优化，返回最优参数估计。

        函数名保留 logistic 是历史兼容命名；实际目标函数由 self.model_spec 决定。

        参数
        ----
        X : np.ndarray, shape [n, p]
            特征矩阵（不含截距列，函数内部会将截距合并进 beta）。
        Y : np.ndarray, shape [n, 1]
            二分类标签（0/1）。
        initial_value : np.ndarray, shape [p+1, 1]
            优化初始点（含截距项，共 p+1 维）。
        options : dict
            SciPy minimize 的 options 参数，如 {'maxiter': 500, 'gtol': 5e-3}。
        lambda_range : np.ndarray
            正则化参数候选范围。若只传入一个数，则本函数把它作为固定
            L2 正则化系数使用；若为空，则使用无正则化的原始设定。
        numFolds : int
            交叉验证折数（当前实现未使用，保留接口以兼容上层调用）。

        返回
        ----
        best_lambda : float
            本次使用的正则化参数。
        beta_hat_ones : np.ndarray, shape [p+1, 1]
            BFGS 优化收敛后的参数估计值（含截距项）。
        """
        # 参数默认值
        if initial_value is None:
            initial_value = np.zeros((X.shape[1] + 1, 1))
        if options is None:
            options = {'maxiter': 500, 'gtol': 5e-3}

        # 原 MATLAB 代码中直接设为 0。实际数据分析中可通过传入单个
        # lambda 值启用同一套固定 L2 正则化，避免小样本逻辑回归过拟合。
        if lambda_range is None:
            best_lambda = 0.0
        else:
            lambda_values = np.asarray(lambda_range, dtype=float).ravel()
            best_lambda = float(lambda_values[0]) if lambda_values.size == 1 else 0.0
        # 委托给 self.model_spec.solve_supervised：
        # - LogisticModelSpec 走默认 BFGS（基类实现）
        # - LinearModelSpec   走闭式 OLS（避免线性损失负权重情形下 BFGS 发散）
        # 这样新增模型时无需改动本函数，只需在对应 ModelSpec 中按需重写求解器。
        beta_hat_ones = self.model_spec.solve_supervised(
            X, Y,
            lambda_reg=best_lambda,
            initial_value=initial_value,
            tolerance=options.get('gtol', 5e-3),
            max_iter=options.get('maxiter', 1000),
        )
        return best_lambda, beta_hat_ones


    def objective_function_logistic(self, beta, X_labeled, Y_labeled, lambda_reg):
        """
        目标函数（负对数似然/损失+L2正则）和梯度。
        已泛化：实际计算委托给 self.model_spec.loss_and_grad。
        保留原名称以兼容外部调用。

        参数：
        - beta: 参数向量 [p+1, 1] 或 [p+1,]
        - X_labeled: 特征矩阵 [n, p]（不含截距）
        - Y_labeled: 标签向量 [n, 1]
        - lambda_reg: L2正则化参数
        返回：
        - f: 目标函数值 (float)
        - g: 梯度向量 [p+1,]（一维）
        """
        return self.model_spec.loss_and_grad(beta, X_labeled, Y_labeled, lambda_reg)

    def admm3_mu_one_simulation(self, t, Tree_lambda_mu, Lambda_mu, num_lambda_mu, c_lambda_1_start, k, a,
                                residual_principle, iter_max, multiple_constant, num_lambda_1, train_X_labeled,
                                train_Y_labeled, train_X_unlabeled, test_X_labeled, test_Y_labeled,
                                test_X_unlabeled,
                                direct_if, lambda_1_start_overall, lambda_1_end_overall, which_lambda_1_opt_input,
                                which_lambda_mu_opt_input):
        """
        对单个模拟轮次执行均值路径的第三层 ADMM/lambda 搜索。
        
        该函数在给定 MST lambda_mu 路径和 ADMM 惩罚参数范围内，计算每个组合对应的
        候选选择结果，并根据验证准则选择最优 lambda_mu 与 lambda_1。
        
        参数
        ----
        t : int
            1-based 模拟轮次。
        Tree_lambda_mu, Lambda_mu, num_lambda_mu : list/dict/array
            均值路径 MST 结果及 lambda 网格。
        c_lambda_1_start, k, a, residual_principle, iter_max : numeric
            ADMM 惩罚与收敛参数。
        multiple_constant, num_lambda_1 : numeric
            lambda_1 区间扩展和网格大小。
        train_*/test_* : array/dict
            训练集和测试集数据。
        direct_if : int
            是否直接使用传入的最优索引。
        lambda_1_start_overall, lambda_1_end_overall : float
            全局 lambda_1 搜索区间。
        which_lambda_1_opt_input, which_lambda_mu_opt_input : int
            外层交叉验证给出的最优索引。
        
        返回
        ----
        result_mu_output : dict
            均值路径的最优聚类、选择 source、lambda 索引和诊断量。
        """
        # 参数默认值
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_1 = 100 if num_lambda_1 is None else num_lambda_1
        direct_if = None if direct_if is None else direct_if
        lambda_1_start_overall = None if lambda_1_start_overall is None else lambda_1_start_overall
        lambda_1_end_overall = None if lambda_1_end_overall is None else lambda_1_end_overall
        which_lambda_1_opt_input = None if which_lambda_1_opt_input is None else which_lambda_1_opt_input
        which_lambda_mu_opt_input = None if which_lambda_mu_opt_input is None else which_lambda_mu_opt_input

        result = {}
        fields_mu = list(Tree_lambda_mu[t - 1].keys())  # Python 0-based
        Error_mu_one_simulation = np.zeros((num_lambda_1, num_lambda_mu))
        Lambda_1_one_simulation = np.zeros((num_lambda_1, num_lambda_mu))
        Lambda_mu_one_simulation = Lambda_mu[t - 1]

        # ── 提取公共数据（不依赖 lambda，只需算一次）──────────────────
        train_X_one, train_Y_one, train_X_un_one, _ = self.one_simulation(
            train_X_labeled, train_Y_labeled, train_X_unlabeled, t
        )
        test_X_one, test_Y_one, test_X_un_one, fields = self.one_simulation(
            test_X_labeled, test_Y_labeled, test_X_unlabeled, t
        )
        X_labeled_one = np.vstack([train_X_one, test_X_one])
        Y_labeled_one = np.vstack([train_Y_one, test_Y_one])
        _alpha_mu = Tree_lambda_mu[t - 1][fields_mu[0]]['alpha']
        _, _, _w_mu, _, _, _, _ = self.mst_generation_single(
            test_X_one, test_Y_one, test_X_un_one, fields,
            X_labeled_one, Y_labeled_one, _alpha_mu
        )
        A1 = _w_mu['labeled_mean']

        # ── 顺序遍历所有 lambda_mu（A1 已在循环外算好，避免冗余调用）──
        for lambda_mu_i in range(num_lambda_mu):
            tree = Tree_lambda_mu[t - 1][fields_mu[lambda_mu_i]]
            _err, _lam = self.admm2_mu_one_simulation_one_lambda_mu_i(
                tree, c_lambda_1_start, k, a, residual_principle,
                iter_max, multiple_constant, num_lambda_1, A1, direct_if,
                lambda_1_start_overall, lambda_1_end_overall, t
            )
            Error_mu_one_simulation[:, lambda_mu_i] = _err.ravel()
            Lambda_1_one_simulation[:, lambda_mu_i] = _lam.ravel()

        # 分支1: 无输入最优索引，自动选择
        if direct_if is None:
            # 寻找最小误差对应的索引
            min_err_idx = np.unravel_index(np.argmin(Error_mu_one_simulation), Error_mu_one_simulation.shape)
            which_lambda_1_opt = min_err_idx[0] + 1  # 转回1-based
            which_lambda_mu_opt = min_err_idx[1] + 1

            # 最优参数值
            lambda_mu_opt_value = Lambda_mu_one_simulation[which_lambda_mu_opt - 1]
            lambda_1_opt_value = Lambda_1_one_simulation[which_lambda_1_opt - 1, which_lambda_mu_opt - 1]

            # 基于最优参数求解ADMM
            Tree_opt = Tree_lambda_mu[t - 1][fields_mu[which_lambda_mu_opt - 1]]
            mu_hat, select_if, which_aux, select_if_pro = self.admm_mu_one_simulation_one_lambda_mu_i(
                Tree_opt, lambda_1_opt_value, k, a, residual_principle, iter_max
            )

            # 封装结果
            result['Tree_lambda_mu_one_simulation'] = Tree_opt
            result['Error_mu_one_simulation'] = Error_mu_one_simulation
            result['Lambda_1_one_simulation'] = Lambda_1_one_simulation
            result['Lambda_mu_one_simulation'] = Lambda_mu_one_simulation
            result['which_lambda_mu_opt'] = which_lambda_mu_opt
            result['which_lambda_1_opt'] = which_lambda_1_opt
            result['lambda_mu_opt_value'] = lambda_mu_opt_value
            result['lambda_1_opt_value'] = lambda_1_opt_value
            result['mu_hat'] = mu_hat
            result['select_if'] = select_if
            result['select_if_pro'] = select_if_pro
            result['which_aux'] = which_aux
            result['select_mean'] = Tree_opt['index_mean'][which_aux - 1]  # 1-based转0-based
            result['select_index'] = Tree_opt['index'][which_aux - 1]
            result['select_alpha'] = Tree_opt['alpha']
            result['select_fields'] = [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]]
        # 分支2: 有输入最优索引或直接模式
        else:
            if which_lambda_1_opt_input is None:
                result['Error_mu_one_simulation'] = Error_mu_one_simulation
                result['fields_mu'] = fields_mu
                result['Lambda_1_one_simulation'] = Lambda_1_one_simulation
                result['Lambda_mu_one_simulation'] = Lambda_mu_one_simulation
            else:
                which_lambda_1_opt = which_lambda_1_opt_input
                which_lambda_mu_opt = which_lambda_mu_opt_input
                lambda_mu_opt_value = Lambda_mu_one_simulation[which_lambda_mu_opt - 1]
                lambda_1_opt_value = Lambda_1_one_simulation[which_lambda_1_opt - 1, which_lambda_mu_opt - 1]
                Tree_opt = Tree_lambda_mu[t - 1][fields_mu[which_lambda_mu_opt - 1]]

                # 求解ADMM
                mu_hat, select_if, which_aux, select_if_pro = self.admm_mu_one_simulation_one_lambda_mu_i(
                    Tree_opt, lambda_1_opt_value, k, a, residual_principle, iter_max
                )

                # 封装结果
                result.update({
                    'Tree_lambda_mu_one_simulation': Tree_opt,
                    'Error_mu_one_simulation': Error_mu_one_simulation,
                    'Lambda_1_one_simulation': Lambda_1_one_simulation,
                    'Lambda_mu_one_simulation': Lambda_mu_one_simulation,
                    'which_lambda_mu_opt': which_lambda_mu_opt,
                    'which_lambda_1_opt': which_lambda_1_opt,
                    'lambda_mu_opt_value': lambda_mu_opt_value,
                    'lambda_1_opt_value': lambda_1_opt_value,
                    'mu_hat': mu_hat,
                    'select_if': select_if,
                    'select_if_pro': select_if_pro,
                    'which_aux': which_aux,
                    'select_mean': Tree_opt['index_mean'][which_aux - 1],
                    'select_index': Tree_opt['index'][which_aux - 1],
                    'select_alpha': Tree_opt['alpha'],
                    'select_fields': [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]],
                    'fields_mu': fields_mu
                })
        return result


    def admm2_mu_one_simulation_one_lambda_mu_i(self, Tree_lambda_mu_one_simulation, c_lambda_1_start, k, a,
                                                residual_principle, iter_max, multiple_constant, num_lambda_1, A2,
                                                direct_if, lambda_1_start_overall, lambda_1_end_overall, t):
        """
        在固定 lambda_mu 下扫描均值路径 ADMM 的 lambda_1 候选值。
        
        参数
        ----
        Tree_lambda_mu_one_simulation : dict
            某个 lambda_mu 下的 MST 聚类结构。
        c_lambda_1_start, lambda_1_start_overall, lambda_1_end_overall : float
            lambda_1 的初值或全局搜索区间。
        k, a, residual_principle, iter_max : numeric
            ADMM 更新和收敛控制参数。
        multiple_constant, num_lambda_1 : numeric
            搜索区间扩展倍数和候选网格数量。
        A2 : np.ndarray
            验证准则中使用的目标统计量。
        direct_if : int
            是否直接使用指定候选区间。
        t : int
            当前模拟轮次。
        
        返回
        ----
        dict
            每个 lambda_1 候选对应的 ADMM 结果、准则值和最优索引。
        """
        # 参数默认值
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_1 = 10 if num_lambda_1 is None else num_lambda_1
        direct_if = None if direct_if is None else direct_if
        lambda_1_start_overall = None if lambda_1_start_overall is None else lambda_1_start_overall
        lambda_1_end_overall = None if lambda_1_end_overall is None else lambda_1_end_overall
        t = None if t is None else t

        # 分支1: 无直接模式，自动搜索lambda_1区间
        if direct_if is None:
            # 搜索lambda_1的上下界
            c_lambda_1_start, c_lambda_1_end = self.interval_admm_mu_one_simulation_one_lambda_mu_i(
                Tree_lambda_mu_one_simulation, c_lambda_1_start, k, a, residual_principle, iter_max,
                multiple_constant
            )
            Lambda_1 = np.linspace(c_lambda_1_start, c_lambda_1_end, num_lambda_1)
        # 分支2: 直接模式，使用输入区间
        else:
            Lambda_1 = np.linspace(lambda_1_start_overall[t - 1], lambda_1_end_overall[t - 1], num_lambda_1)

        # 计算每个lambda_1对应的误差
        Error = np.zeros((num_lambda_1, 1))
        for i in range(num_lambda_1):
            mu_hat, _, _, _ = self.admm_mu_one_simulation_one_lambda_mu_i(
                Tree_lambda_mu_one_simulation, Lambda_1[i], k, a, residual_principle, iter_max
            )
            Error[i] = np.linalg.norm(A2.reshape(-1, 1) - mu_hat)

        return Error, Lambda_1.reshape(-1, 1)

    def admm2_sigma_one_simulation_one_lambda_sigma_i(self, Tree_lambda_sigma_one_simulation, c_lambda_2_start, k,
                                                      a,
                                                      residual_principle, iter_max, multiple_constant, num_lambda_2,
                                                      A2,
                                                      direct_if, lambda_2_start_overall, lambda_2_end_overall, t):
        """
        在固定 lambda_sigma 下扫描协方差路径 ADMM 的 lambda_2 候选值。
        
        参数与 admm2_mu_one_simulation_one_lambda_mu_i 对称，只是输入树和惩罚参数来自
        协方差路径；函数用于确定当前 lambda_sigma 下最合适的协方差融合强度。
        
        返回
        ----
        dict
            每个 lambda_2 候选对应的 ADMM 结果、验证准则和最优索引。
        """
        # 参数默认值
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_2 = 10 if num_lambda_2 is None else num_lambda_2
        direct_if = None if direct_if is None else direct_if
        lambda_2_start_overall = None if lambda_2_start_overall is None else lambda_2_start_overall
        lambda_2_end_overall = None if lambda_2_end_overall is None else lambda_2_end_overall
        t = None if t is None else t

        # 分支1: 自动搜索lambda_2区间
        if direct_if is None:
            c_lambda_2_start, c_lambda_2_end = self.interval_admm_sigma_one_simulation_one_lambda_sigma_i(
                Tree_lambda_sigma_one_simulation, c_lambda_2_start, k, a, residual_principle, iter_max,
                multiple_constant
            )
            Lambda_2 = np.linspace(c_lambda_2_start, c_lambda_2_end, num_lambda_2)
        # 分支2: 使用输入区间
        else:
            Lambda_2 = np.linspace(lambda_2_start_overall[t - 1], lambda_2_end_overall[t - 1], num_lambda_2)

        # 计算每个lambda_2对应的误差
        Error = np.zeros((num_lambda_2, 1))
        for i in range(num_lambda_2):
            sigma_hat, _, _, _, _ = self.admm_sigma_one_simulation_one_lambda_sigma_i(
                Tree_lambda_sigma_one_simulation, Lambda_2[i], k, a, residual_principle, iter_max
            )
            Error[i] = norm(A2 - sigma_hat, 'fro')

        return Error, Lambda_2.reshape(-1, 1)

    def admm_mu_one_simulation_one_lambda_mu_i(self, Tree_lambda_mu_one_simulation, c_lambda_1, k, a,
                                               residual_principle, iter_max):
        """
        执行固定 lambda_mu 与 lambda_1 下的均值路径 ADMM 更新。
        
        ADMM 在 MST 聚类结构上估计每个 source 的调整向量，利用软阈值/融合惩罚将
        分布接近的 source 收缩到同一组，最终输出与有标签均值最接近的候选集合。
        """
        # 对惩罚参数做样本数缩放：lambda_1 = c_lambda_1 * sqrt(1/n)
        # 这样做使惩罚强度随样本增大而减弱，保证参数估计一致性
        n_labeled = Tree_lambda_mu_one_simulation['train_Z_labeled_one_simulation'].shape[0]
        lambda_1 = c_lambda_1 * np.sqrt(1 / n_labeled)

        # 初始化 ADMM 三个变量：
        # mu_hat：主变量（待估均值），初始为标记数据的样本均值
        # delta：辅助变量（与 mu_hat 耦合约束），初始与 mu_hat 相同
        # nu：对偶变量（乘子），初始为0
        mu_hat = Tree_lambda_mu_one_simulation['labeled_mean'].reshape(-1, 1)
        delta = mu_hat.copy()
        nu = np.zeros_like(mu_hat)

        # 将各聚类的加权均值横向拼接为矩阵，每列对应一个聚类
        # shape: [d, num_clusters]，用于后续近端算子中寻找最近聚类
        xi_store = np.hstack(Tree_lambda_mu_one_simulation['index_mean'])

        residual = 1.0
        select_if = 0      # 是否精确收敛到某聚类均值（coeff=0时为1）
        select_if_pro = 0  # 是否落入某聚类的邻近区域（进入 else 分支时为1）
        iter = 0

        # ADMM 迭代（交替方向乘子法）：
        # 目标：min_{mu} (1/2)||mu - labeled_mean||^2 + λ * φ(mu)
        # 其中 φ(mu) = min_i ||mu - ξ_i||（与最近聚类均值的距离惩罚）
        while residual > residual_principle and iter < iter_max:
            # 步骤1：更新主变量 mu_hat
            # 闭合形式解：mu_hat = 1/(k+1) * (labeled_mean + k*delta - nu)
            # 等价于在数据项与耦合约束之间做加权平均
            mu_hat = (1 / (k + 1)) * (Tree_lambda_mu_one_simulation['labeled_mean'].reshape(-1, 1) + k * delta - nu)

            # 步骤2：更新辅助变量 delta（近端映射步）
            # Bar_delta 是 mu_hat 经对偶变量校正后的中间量
            Bar_delta = mu_hat + nu / k
            # 计算 Bar_delta 到每个聚类均值 xi_i 的欧氏距离
            dists = np.linalg.norm(xi_store - Bar_delta, axis=0)
            which_aux = np.argmin(dists) + 1  # 最近聚类的 1-based 索引
            diff_norm = dists[which_aux - 1]   # 到最近聚类的距离

            if diff_norm >= a * lambda_1:
                # 距离超过阈值 a*λ：不施加惩罚，delta 直接等于 Bar_delta
                delta = Bar_delta
                select_if = 0
                select_if_pro = 0
            else:
                # 距离在阈值内：施加近端收缩，将 delta 拉向最近聚类均值
                # coeff = max(1 - λ/(k*dist), 0)，当 coeff=0 时精确吸附到聚类中心
                coeff = max(1 - lambda_1 / (k * diff_norm), 0)
                delta = xi_store[:, which_aux - 1].reshape(-1, 1) + coeff * (1 / (1 - 1 / (a * k))) * (
                        Bar_delta - xi_store[:, which_aux - 1].reshape(-1, 1))
                select_if = 1 if coeff == 0 else 0  # coeff=0 表示精确选择该聚类
                select_if_pro = 1  # 进入近端区域，概率性选择标记置1

            # 步骤3：更新对偶变量 nu（乘子更新，惩罚原始残差）
            nu = nu + k * (mu_hat - delta)

            # 原始残差：mu_hat 与 delta 的差范数，用于判断收敛
            residual = np.linalg.norm(mu_hat - delta)
            iter += 1

        # 迭代结束后，根据最终 Bar_delta 再次确认最近聚类
        Bar_delta = mu_hat + nu / k
        dists = np.linalg.norm(xi_store - Bar_delta, axis=0)
        which_aux = np.argmin(dists) + 1  # 返回最终选定的聚类索引（1-based）

        return mu_hat, select_if, which_aux, select_if_pro

    def admm_sigma_one_simulation_one_lambda_sigma_i(self, Tree_lambda_sigma_one_simulation, c_lambda_2, k, a,
                                                     residual_principle, iter_max):
        """
        执行固定 lambda_sigma 与 lambda_2 下的协方差路径 ADMM 更新。
        
        该过程与均值路径 ADMM 对称，只是节点统计量从辅助特征均值替换为协方差矩阵，
        距离使用 Frobenius 范数。
        """
        # 对惩罚参数做样本数和矩阵维度缩放：
        # lambda_2 = c_lambda_2 * sqrt(d^2 * ln(d) / n)
        # 协方差矩阵参数量为 d^2，故引入 d^2*ln(d) 因子，保证估计一致性
        n_labeled = Tree_lambda_sigma_one_simulation['train_Z_labeled_one_simulation'].shape[0]
        d = Tree_lambda_sigma_one_simulation['labeled_sigma'].shape[0]  # 协方差矩阵维度
        lambda_2 = c_lambda_2 * np.sqrt((d ** 2 * np.log(d)) / n_labeled)

        # 将协方差矩阵向量化（reshape 为列向量）后进行 ADMM 迭代
        # 这样可以复用与均值（向量）完全相同的 ADMM 框架
        sigma_hat = Tree_lambda_sigma_one_simulation['labeled_sigma'].copy()
        delta = sigma_hat.copy()
        vec_delta = delta.reshape(-1, 1)    # 辅助变量的向量化形式，shape: [d^2, 1]
        nu = np.zeros_like(sigma_hat)
        vec_nu = nu.reshape(-1, 1)          # 对偶变量的向量化形式，shape: [d^2, 1]

        # 将各聚类协方差矩阵向量化后横向拼接，shape: [d^2, num_clusters]
        xi_store = np.hstack([mat.reshape(-1, 1) for mat in Tree_lambda_sigma_one_simulation['index_sigma']])

        residual = 1.0
        select_if = 0
        select_if_pro = 0
        iter = 0

        # ADMM 迭代（与均值路径完全对称，操作对象换为向量化协方差矩阵）
        while residual > residual_principle and iter < iter_max:
            # 步骤1：更新主变量 sigma_hat（向量化形式）
            # 闭合形式同均值路径：vec_sigma_hat = 1/(k+1) * (sigma + k*delta - nu)
            vec_sigma_hat = (1 / (k + 1)) * (sigma_hat.reshape(-1, 1) + k * vec_delta - vec_nu)

            # 步骤2：更新辅助变量 delta（近端映射，寻找最近聚类协方差）
            vec_Bar_delta = vec_sigma_hat + vec_nu / k  # 对偶校正后的中间量
            # 计算向量化的 Bar_delta 与各聚类协方差的欧氏距离
            dists = np.linalg.norm(xi_store - vec_Bar_delta, axis=0)
            which_aux = np.argmin(dists) + 1  # 最近聚类的 1-based 索引
            diff_norm = dists[which_aux - 1]

            if diff_norm >= a * lambda_2:
                # 距离超过阈值：不收缩，delta 保持为 Bar_delta
                vec_delta = vec_Bar_delta
                select_if = 0
                select_if_pro = 0
            else:
                # 距离在阈值内：近端收缩，将 delta 拉向最近聚类协方差
                coeff = max(1 - lambda_2 / (k * diff_norm), 0)
                vec_delta = xi_store[:, which_aux - 1].reshape(-1, 1) + coeff * (1 / (1 - 1 / (a * k))) * (
                        vec_Bar_delta - xi_store[:, which_aux - 1].reshape(-1, 1))
                select_if = 1 if coeff == 0 else 0
                select_if_pro = 1

            # 步骤3：更新对偶变量（乘子更新）
            vec_nu = vec_nu + k * (vec_sigma_hat - vec_delta)

            # 残差（判断收敛）：主辅变量差的向量范数
            residual = np.linalg.norm(vec_sigma_hat - vec_delta)
            iter += 1

        # 迭代结束，确认最终最近聚类并将结果重构为 d×d 矩阵
        vec_Bar_delta = vec_sigma_hat + vec_nu / k
        dists = np.linalg.norm(xi_store - vec_Bar_delta, axis=0)
        which_aux = np.argmin(dists) + 1
        sigma_hat = vec_sigma_hat.reshape(d, d)  # 向量化结果还原为协方差矩阵形式

        return sigma_hat, select_if, which_aux, iter, select_if_pro

    def interval_admm_mu_one_simulation_one_lambda_mu_i(self, Tree_lambda_mu_one_simulation, c_lambda_1_start, k, a,
                                                        residual_principle, iter_max, multiple_constant):
        """
        为均值路径 ADMM 自动寻找 lambda_1 的有效搜索区间。
        
        从 c_lambda_1_start 开始不断按 multiple_constant 放大惩罚参数，记录 ADMM 解从
        不充分融合到充分融合的变化范围。该区间随后会被离散成 num_lambda_1 个候选点。
        
        返回
        ----
        tuple[float, float]
            lambda_1 搜索下界和上界。
        """
        # 参数默认值
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant

        A1 = Tree_lambda_mu_one_simulation['labeled_mean'].reshape(-1, 1)
        c_lambda_1_mid = c_lambda_1_start

        # 搜索上界：直到select_if=1
        _, select_if, _, _ = self.admm_mu_one_simulation_one_lambda_mu_i(
            Tree_lambda_mu_one_simulation, c_lambda_1_mid, k, a, residual_principle, iter_max
        )
        while select_if != 1:
            c_lambda_1_mid *= multiple_constant
            _, select_if, _, _ = self.admm_mu_one_simulation_one_lambda_mu_i(
                Tree_lambda_mu_one_simulation, c_lambda_1_mid, k, a, residual_principle, iter_max
            )
        c_lambda_1_end = c_lambda_1_mid

        # 搜索下界：直到误差为0
        mu_hat, _, _, _ = self.admm_mu_one_simulation_one_lambda_mu_i(
            Tree_lambda_mu_one_simulation, c_lambda_1_mid, k, a, residual_principle, iter_max
        )
        diff = np.linalg.norm(A1 - mu_hat)
        while diff > 1e-6:  # 数值精度替代diff~=0
            c_lambda_1_mid /= multiple_constant
            mu_hat, _, _, _ = self.admm_mu_one_simulation_one_lambda_mu_i(
                Tree_lambda_mu_one_simulation, c_lambda_1_mid, k, a, residual_principle, iter_max
            )
            diff = np.linalg.norm(A1 - mu_hat)
        c_lambda_1_start = c_lambda_1_mid

        return c_lambda_1_start, c_lambda_1_end

    def interval_admm_sigma_one_simulation_one_lambda_sigma_i(self, Tree_lambda_sigma_one_simulation,
                                                              c_lambda_2_start, k, a,
                                                              residual_principle, iter_max, multiple_constant):
        """
        为协方差路径 ADMM 自动寻找 lambda_2 的有效搜索区间。
        
        逻辑与 interval_admm_mu_one_simulation_one_lambda_mu_i 对称，但 ADMM 变量表示
        协方差矩阵差异，收敛判断使用 Frobenius 范数相关残差。
        
        返回
        ----
        tuple[float, float]
            lambda_2 搜索下界和上界。
        """
        # 参数默认值
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant

        A1 = Tree_lambda_sigma_one_simulation['labeled_sigma']
        c_lambda_2_mid = c_lambda_2_start

        # 搜索上界：直到select_if=1
        _, select_if, _, _, _ = self.admm_sigma_one_simulation_one_lambda_sigma_i(
            Tree_lambda_sigma_one_simulation, c_lambda_2_mid, k, a, residual_principle, iter_max
        )
        while select_if != 1:
            c_lambda_2_mid *= multiple_constant
            _, select_if, _, _, _ = self.admm_sigma_one_simulation_one_lambda_sigma_i(
                Tree_lambda_sigma_one_simulation, c_lambda_2_mid, k, a, residual_principle, iter_max
            )
        c_lambda_2_end = c_lambda_2_mid

        # 搜索下界：直到误差为0
        sigma_hat, _, _, _, _ = self.admm_sigma_one_simulation_one_lambda_sigma_i(
            Tree_lambda_sigma_one_simulation, c_lambda_2_mid, k, a, residual_principle, iter_max
        )
        diff = norm(A1 - sigma_hat, 'fro')
        while diff > 1e-6:
            c_lambda_2_mid /= multiple_constant
            sigma_hat, _, _, _, _ = self.admm_sigma_one_simulation_one_lambda_sigma_i(
                Tree_lambda_sigma_one_simulation, c_lambda_2_mid, k, a, residual_principle, iter_max
            )
            diff = norm(A1 - sigma_hat, 'fro')
        c_lambda_2_start = c_lambda_2_mid

        return c_lambda_2_start, c_lambda_2_end

    def mu_sigma_combine(self, t, result_mu_output, result_sigma_output, X_unlabeled):
        """
        合并均值路径和协方差路径的筛选结果。
        
        只有同时通过均值相似性和协方差相似性诊断的 source 会进入最终选择；若交集为空，
        函数会按诊断准则选择保守候选，并生成 result_summary 供后续估计和可视化使用。
        
        参数
        ----
        t : int
            1-based 模拟轮次。
        result_mu_output, result_sigma_output : dict
            两条路径的 ADMM 选择结果。
        X_unlabeled : dict[str, list[np.ndarray]]
            原始多源无标签数据。
        
        返回
        ----
        X_unlabeled_select_one_simulation : np.ndarray
            第 t 轮最终选中的无标签样本合并矩阵。
        result_summary_one_simulation : dict
            选择字段、索引、alpha、lambda 和诊断信息。
        """
        result_summary = {}
        # 合并mu结果
        result_summary.update({
            'Tree_lambda_mu_one_simulation': result_mu_output['Tree_lambda_mu_one_simulation'],
            'Error_mu_one_simulation': result_mu_output['Error_mu_one_simulation'],
            'Lambda_1_one_simulation': result_mu_output['Lambda_1_one_simulation'],
            'Lambda_mu_one_simulation': result_mu_output['Lambda_mu_one_simulation'],
            'which_lambda_mu_opt': result_mu_output['which_lambda_mu_opt'],
            'which_lambda_1_opt': result_mu_output['which_lambda_1_opt'],
            'lambda_mu_opt_value': result_mu_output['lambda_mu_opt_value'],
            'lambda_1_opt_value': result_mu_output['lambda_1_opt_value'],
            'mu_hat': result_mu_output['mu_hat'],
            'select_if_mu': result_mu_output['select_if'],
            'select_if_mu_pro': result_mu_output['select_if_pro'],
            'which_aux': result_mu_output['which_aux'],
            'select_mean': result_mu_output['select_mean'],
            'select_index_mu': result_mu_output['select_index'],
            'select_fields_mu_lambda_name': result_mu_output.get('fields_mu', [])
        })
        # 合并sigma结果
        result_summary.update({
            'Tree_lambda_sigma_one_simulation': result_sigma_output['Tree_lambda_sigma_one_simulation'],
            'Error_sigma_one_simulation': result_sigma_output['Error_sigma_one_simulation'],
            'Lambda_2_one_simulation': result_sigma_output['Lambda_2_one_simulation'],
            'Lambda_sigma_one_simulation': result_sigma_output['Lambda_sigma_one_simulation'],
            'which_lambda_sigma_opt': result_sigma_output['which_lambda_sigma_opt'],
            'which_lambda_2_opt': result_sigma_output['which_lambda_2_opt'],
            'lambda_sigma_opt_value': result_sigma_output['lambda_sigma_opt_value'],
            'lambda_2_opt_value': result_sigma_output['lambda_2_opt_value'],
            'sigma_hat': result_sigma_output['sigma_hat'],
            'select_if_sigma': result_sigma_output['select_if'],
            'select_if_sigma_pro': result_sigma_output['select_if_pro'],
            'select_sigma': result_sigma_output['select_sigma'],
            'select_index_sigma': result_sigma_output['select_index'],
            'select_fields_sigma_lambda_name': result_sigma_output.get('fields_sigma', [])
        })
        # 综合判断：只有 μ 和 Σ 两条路径同时精确选择（select_if==1）才认为整体精确选择
        result_summary['select_alpha'] = result_mu_output['select_alpha']
        result_summary['select_if'] = 1 if (
                result_summary['select_if_sigma'] == 1 and result_summary['select_if_mu'] == 1) else 0
        # 概率性选择：μ 或 Σ 任一路径进入聚类邻近区域即标记
        result_summary['select_if_pro'] = 1 if (
                result_summary['select_if_mu_pro'] == 1 and result_summary['select_if_sigma_pro'] == 1) else 0

        # 取 μ 路径和 Σ 路径选出的数据源索引的交集，作为最终选择
        # 交集策略：同时满足均值相似和协方差相似的数据源才被纳入
        select_index_mu = set(result_summary['select_index_mu'])
        select_index_sigma = set(result_summary['select_index_sigma'])
        result_summary['select_index'] = sorted(select_index_mu.intersection(select_index_sigma))
        result_summary['select_fields_mu'] = result_mu_output['select_fields']
        result_summary['select_fields_sigma'] = result_sigma_output['select_fields']

        if result_summary['select_if_pro'] == 1:
            select_fields_mu = set(result_summary['select_fields_mu'])
            select_fields_sigma = set(result_summary['select_fields_sigma'])
            # 主策略：取两条路径选出字段的交集
            result_summary['select_fields'] = sorted(select_fields_mu.intersection(select_fields_sigma))
            # 回退策略：交集为空且精确选择时，退化为取并集（保证至少选出一个数据源）
            if not result_summary['select_fields'] and result_summary['select_if'] == 1:
                result_summary['select_fields'] = sorted(select_fields_mu.union(select_fields_sigma))
                result_summary['select_index'] = sorted(select_index_mu.union(select_index_sigma))
        else:
            result_summary['select_fields'] = []

        # 选择未标记数据
        combined_elements = []
        for field in result_summary['select_fields']:
            if field in X_unlabeled:
                combined_elements.append(X_unlabeled[field][t - 1])  # Python 0-based
        X_unlabeled_select_one_simulation = np.vstack(combined_elements) if combined_elements else np.array([])

        return X_unlabeled_select_one_simulation, result_summary

    def cv_penalty_parameter_mu(self, result_mst, c_lambda_1_start, k, a, residual_principle, iter_max,
                                multiple_constant, num_lambda_1):
        """
        通过交叉验证确定均值路径 ADMM 惩罚参数 lambda_1 的全局范围。
        
        输入 cv_mst 生成的 MST 路径，先为每个模拟和 lambda_mu 自动扩展 lambda_1 区间，
        再汇总得到所有模拟共享的搜索上下界，避免后续网格过窄。
        """
        # 参数默认值
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_1 = 100 if num_lambda_1 is None else num_lambda_1

        cv_number = len(result_mst['Lambda_mu'])
        simulation_times = len(result_mst['Lambda_mu'][0])
        Information_mu = [{} for _ in range(simulation_times)]

        # ── 顺序执行（保证结果可复现）─────────────────────────────
        for t in tqdm(range(simulation_times), desc='Step 3/7 | μ-ADMM 惩罚范围 (CV)', unit='sim', ncols=90, leave=True):
            info = {
                'Lambda_1_min': np.zeros((cv_number, 1)), 'Lambda_1_max': np.zeros((cv_number, 1)),
                'Lambda_mu': [None]*cv_number, 'Error_mu_one_simulation': [None]*cv_number,
                'Lambda_1_one_simulation': [None]*cv_number, 'lambda_1_opt_value': np.zeros((cv_number,1)),
                'lambda_mu_opt_value': np.zeros((cv_number,1)), 'which_lambda_1_opt': np.zeros((cv_number,1)),
                'which_lambda_mu_opt': np.zeros((cv_number,1)), 'mu_hat': [None]*cv_number,
                'select_alpha': np.zeros((cv_number,1)), 'select_fields': [None]*cv_number,
                'select_if': np.zeros((cv_number,1)), 'select_if_pro': np.zeros((cv_number,1)),
                'select_index': [None]*cv_number, 'select_mean': [None]*cv_number,
                'Tree_lambda_mu_one_simulation': [None]*cv_number, 'which_aux': np.zeros((cv_number,1))
            }
            for cv in range(cv_number):
                result = self.admm3_mu_one_simulation(
                    t + 1, result_mst['Tree_lambda_mu'][cv], result_mst['Lambda_mu'][cv],
                    len(result_mst['Lambda_mu'][0][0]), c_lambda_1_start, k, a, residual_principle,
                    iter_max, multiple_constant, num_lambda_1, result_mst['train_X_labeled'][cv],
                    result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
                    result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
                    result_mst['test_X_unlabeled'][cv], None, None, None, None, None
                )
                info['Lambda_1_min'][cv] = np.min(result['Lambda_1_one_simulation'])
                info['Lambda_1_max'][cv] = np.max(result['Lambda_1_one_simulation'])
                info['Lambda_mu'][cv] = result['Lambda_mu_one_simulation']
                info['Error_mu_one_simulation'][cv] = result['Error_mu_one_simulation']
                info['Lambda_1_one_simulation'][cv] = result['Lambda_1_one_simulation']
                info['lambda_1_opt_value'][cv] = result['lambda_1_opt_value']
                info['lambda_mu_opt_value'][cv] = result['lambda_mu_opt_value']
                info['which_lambda_1_opt'][cv] = result['which_lambda_1_opt']
                info['which_lambda_mu_opt'][cv] = result['which_lambda_mu_opt']
                info['mu_hat'][cv] = result['mu_hat']
                info['select_alpha'][cv] = result['select_alpha']
                info['select_fields'][cv] = result['select_fields']
                info['select_if'][cv] = result['select_if']
                info['select_if_pro'][cv] = result['select_if_pro']
                info['select_index'][cv] = result['select_index']
                info['select_mean'][cv] = result['select_mean']
                info['Tree_lambda_mu_one_simulation'][cv] = result['Tree_lambda_mu_one_simulation']
                info['which_aux'][cv] = result['which_aux']
            Information_mu[t] = info

        # 计算全局lambda_1范围
        Lambda_1_small = [np.min(info['Lambda_1_min']) for info in Information_mu]
        Lambda_1_big = [np.max(info['Lambda_1_max']) for info in Information_mu]

        return Information_mu, Lambda_1_small, Lambda_1_big






    # def admm3_sigma_one_simulation(self, t, Tree_lambda_sigma, Lambda_sigma, num_lambda_sigma, c_lambda_2_start, k,
    #                                a,
    #                                residual_principle, iter_max, multiple_constant, num_lambda_2, train_X_labeled,
    #                                train_Y_labeled, train_X_unlabeled, test_X_labeled, test_Y_labeled,
    #                                test_X_unlabeled,
    #                                direct_if, lambda_2_start_overall, lambda_2_end_overall,
    #                                which_lambda_2_opt_input,
    #                                which_lambda_sigma_opt_input):
    #     """
    #     ADMM优化针对sigma的单次模拟主流程（逻辑与mu完全对称）
    #     参数：与admm3_mu_one_simulation一致，替换mu为sigma
    #     返回：result: dict, 包含sigma优化结果和参数
    #     t = t + 1,
    #     Tree_lambda_sigma = Tree_sigma_output
    #     Lambda_sigma = Lambda_sigma
    #     num_lambda_sigma = len(Lambda_sigma[0])
    #     c_lambda_2_start = c_lambda_2_start
    #     train_X_labeled = X_labeled
    #     train_Y_labeled = Y_labeled
    #     train_X_unlabeled = X_unlabeled
    #     test_X_labeled = X_labeled
    #     test_Y_labeled = Y_labeled
    #     test_X_unlabeled = X_unlabeled
    #     lambda_2_start_overall = data_sigma['Lambda_2_small']
    #     lambda_2_end_overall = data_sigma['Lambda_2_big']
    #     which_lambda_2_opt_input = data_sigma['which_lambda_2_opt'][t]
    #     which_lambda_sigma_opt_input = data_sigma['which_lambda_sigma_opt'][t]
    #     """
    #     # 参数默认值
    #     c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
    #     k = 1 if k is None else k
    #     a = 3 if a is None else a
    #     residual_principle = 1e-3 if residual_principle is None else residual_principle
    #     iter_max = 50 if iter_max is None else iter_max
    #     multiple_constant = 2 if multiple_constant is None else multiple_constant
    #     num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2
    #     direct_if = None if direct_if is None else direct_if
    #     lambda_2_start_overall = None if lambda_2_start_overall is None else lambda_2_start_overall
    #     lambda_2_end_overall = None if lambda_2_end_overall is None else lambda_2_end_overall
    #     which_lambda_2_opt_input = None if which_lambda_2_opt_input is None else which_lambda_2_opt_input
    #     which_lambda_sigma_opt_input = None if which_lambda_sigma_opt_input is None else which_lambda_sigma_opt_input
    #
    #     result = {}
    #     fields_sigma = list(Tree_lambda_sigma[t - 1].keys())
    #     Error_sigma_one_simulation = np.zeros((num_lambda_2, num_lambda_sigma))
    #     Lambda_2_one_simulation = np.zeros((num_lambda_2, num_lambda_sigma))
    #     Lambda_sigma_one_simulation = Lambda_sigma[t - 1]
    #
    #     # 遍历所有lambda_sigma
    #     for lambda_sigma_i in range(num_lambda_sigma):
    #         Tree_lambda_sigma_one_simulation = Tree_lambda_sigma[t - 1][fields_sigma[lambda_sigma_i]]
    #         # 提取单次模拟数据
    #         train_X_one, train_Y_one, train_X_un_one, _ = self.one_simulation(
    #             train_X_labeled, train_Y_labeled, train_X_unlabeled, t
    #         )
    #         test_X_one, test_Y_one, test_X_un_one, fields = self.one_simulation(
    #             test_X_labeled, test_Y_labeled, test_X_unlabeled, t
    #         )
    #         X_labeled_one = np.vstack([train_X_one, test_X_one])
    #         Y_labeled_one = np.vstack([train_Y_one, test_Y_one])
    #
    #         # 生成MST统计量
    #         _, _, w, _, _, _, _ = self.mst_generation_single(
    #             test_X_one, test_Y_one, test_X_un_one, fields, X_labeled_one, Y_labeled_one,
    #             Tree_lambda_sigma_one_simulation['alpha']
    #         )
    #         A1 = w['labeled_sigma']
    #
    #         # 求解ADMM得到误差和lambda_2
    #         Error, Lambda_2 = self.admm2_sigma_one_simulation_one_lambda_sigma_i(
    #             Tree_lambda_sigma_one_simulation, c_lambda_2_start, k, a, residual_principle,
    #             iter_max, multiple_constant, num_lambda_2, A1, direct_if,
    #             lambda_2_start_overall, lambda_2_end_overall, t
    #         )
    #         Error_sigma_one_simulation[:, lambda_sigma_i] = Error.ravel()
    #         Lambda_2_one_simulation[:, lambda_sigma_i] = Lambda_2.ravel()
    #
    #     # 分支1: 无输入最优索引，自动选择
    #     if direct_if is None:
    #         min_err_idx = np.unravel_index(np.argmin(Error_sigma_one_simulation), Error_sigma_one_simulation.shape)
    #         which_lambda_2_opt = min_err_idx[0] + 1
    #         which_lambda_sigma_opt = min_err_idx[1] + 1
    #
    #         lambda_sigma_opt_value = Lambda_sigma_one_simulation[which_lambda_sigma_opt - 1]
    #         lambda_2_opt_value = Lambda_2_one_simulation[which_lambda_2_opt - 1, which_lambda_sigma_opt - 1]
    #         Tree_opt = Tree_lambda_sigma[t - 1][fields_sigma[which_lambda_sigma_opt - 1]]
    #
    #         # 求解ADMM
    #         sigma_hat, select_if, which_aux, _, select_if_pro = self.admm_sigma_one_simulation_one_lambda_sigma_i(
    #             Tree_opt, lambda_2_opt_value, k, a, residual_principle, iter_max
    #         )
    #
    #         # 封装结果
    #         result['Tree_lambda_sigma_one_simulation'] = Tree_opt
    #         result['Error_sigma_one_simulation'] = Error_sigma_one_simulation
    #         result['Lambda_2_one_simulation'] = Lambda_2_one_simulation
    #         result['Lambda_sigma_one_simulation'] = Lambda_sigma_one_simulation
    #         result['which_lambda_sigma_opt'] = which_lambda_sigma_opt
    #         result['which_lambda_2_opt'] = which_lambda_2_opt
    #         result['lambda_sigma_opt_value'] = lambda_sigma_opt_value
    #         result['lambda_2_opt_value'] = lambda_2_opt_value
    #         result['sigma_hat'] = sigma_hat
    #         result['select_if'] = select_if
    #         result['select_if_pro'] = select_if_pro
    #         result['which_aux'] = which_aux
    #         result['select_sigma'] = Tree_opt['index_sigma'][which_aux - 1]
    #         result['select_index'] = Tree_opt['index'][which_aux - 1]
    #         result['select_alpha'] = Tree_opt['alpha']
    #         result['select_fields'] = [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]]
    #     # 分支2: 有输入最优索引或直接模式
    #     else:
    #         if which_lambda_2_opt_input is None:
    #             result['Error_sigma_one_simulation'] = Error_sigma_one_simulation
    #             result['fields_sigma'] = fields_sigma
    #             result['Lambda_2_one_simulation'] = Lambda_2_one_simulation
    #             result['Lambda_sigma_one_simulation'] = Lambda_sigma_one_simulation
    #         else:
    #             which_lambda_2_opt = which_lambda_2_opt_input[0]
    #             which_lambda_sigma_opt = which_lambda_sigma_opt_input[0]
    #             lambda_sigma_opt_value = Lambda_sigma_one_simulation[which_lambda_sigma_opt - 1]
    #             lambda_2_opt_value = Lambda_2_one_simulation[which_lambda_2_opt - 1, which_lambda_sigma_opt - 1]
    #             Tree_opt = Tree_lambda_sigma[t - 1][fields_sigma[which_lambda_sigma_opt - 1]]
    #
    #             # 求解ADMM
    #             sigma_hat, select_if, which_aux, _, select_if_pro = self.admm_sigma_one_simulation_one_lambda_sigma_i(
    #                 Tree_opt, lambda_2_opt_value, k, a, residual_principle, iter_max
    #             )
    #
    #             # 封装结果
    #             result.update({
    #                 'Tree_lambda_sigma_one_simulation': Tree_opt,
    #                 'Error_sigma_one_simulation': Error_sigma_one_simulation,
    #                 'Lambda_2_one_simulation': Lambda_2_one_simulation,
    #                 'Lambda_sigma_one_simulation': Lambda_sigma_one_simulation,
    #                 'which_lambda_sigma_opt': which_lambda_sigma_opt,
    #                 'which_lambda_2_opt': which_lambda_2_opt,
    #                 'lambda_sigma_opt_value': lambda_sigma_opt_value,
    #                 'lambda_2_opt_value': lambda_2_opt_value,
    #                 'sigma_hat': sigma_hat,
    #                 'select_if': select_if,
    #                 'select_if_pro': select_if_pro,
    #                 'which_aux': which_aux,
    #                 'select_sigma': Tree_opt['index_sigma'][which_aux - 1],
    #                 'select_index': Tree_opt['index'][which_aux - 1],
    #                 'select_alpha': Tree_opt['alpha'],
    #                 'select_fields': [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]],
    #                 'fields_sigma': fields_sigma
    #             })
    #     return result
    #
    # def cv_penalty_parameter_sigma(self, result_mst, c_lambda_2_start, k, a, residual_principle, iter_max,
    #                                multiple_constant, num_lambda_2):
    #     """
    #     交叉验证确定sigma的惩罚参数范围（与mu对称）
    #     参数：与cv_penalty_parameter_mu一致，替换mu为sigma
    #     返回：Information_sigma: 交叉验证信息, Lambda_2_small: lambda_2下界, Lambda_2_big: lambda_2上界
    #     """
    #     # 参数默认值
    #     c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
    #     k = 1 if k is None else k
    #     a = 3 if a is None else a
    #     residual_principle = 1e-3 if residual_principle is None else residual_principle
    #     iter_max = 50 if iter_max is None else iter_max
    #     multiple_constant = 2 if multiple_constant is None else multiple_constant
    #     num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2
    #
    #     cv_number = len(result_mst['Lambda_sigma'])
    #     simulation_times = len(result_mst['Lambda_sigma'][0])
    #     Information_sigma = [{} for _ in range(simulation_times)]
    #
    #     # 逐次模拟+交叉验证
    #     for t in range(simulation_times):
    #         info = {
    #             'Lambda_2_min': np.zeros((cv_number, 1)),
    #             'Lambda_2_max': np.zeros((cv_number, 1)),
    #             'Lambda_sigma': [None] * cv_number,
    #             'Error_sigma_one_simulation': [None] * cv_number,
    #             'Lambda_2_one_simulation': [None] * cv_number,
    #             'lambda_2_opt_value': np.zeros((cv_number, 1)),
    #             'lambda_sigma_opt_value': np.zeros((cv_number, 1)),
    #             'which_lambda_2_opt': np.zeros((cv_number, 1)),
    #             'which_lambda_sigma_opt': np.zeros((cv_number, 1)),
    #             'sigma_hat': [None] * cv_number,
    #             'select_alpha': np.zeros((cv_number, 1)),
    #             'select_fields': [None] * cv_number,
    #             'select_if': np.zeros((cv_number, 1)),
    #             'select_if_pro': np.zeros((cv_number, 1)),
    #             'select_index': [None] * cv_number,
    #             'select_sigma': [None] * cv_number,
    #             'Tree_lambda_sigma_one_simulation': [None] * cv_number,
    #             'which_aux': np.zeros((cv_number, 1))
    #         }
    #         # 逐折交叉验证
    #         for cv in range(cv_number):
    #             # 单次交叉验证的ADMM优化
    #             result = self.admm3_sigma_one_simulation(
    #                 t + 1, result_mst['Tree_lambda_sigma'][cv], result_mst['Lambda_sigma'][cv],
    #                 len(result_mst['Lambda_sigma'][0][0]), c_lambda_2_start, k, a, residual_principle,
    #                 iter_max, multiple_constant, num_lambda_2, result_mst['train_X_labeled'][cv],
    #                 result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
    #                 result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
    #                 result_mst['test_X_unlabeled'][cv], None, None, None, None, None
    #             )
    #             # 存储交叉验证结果
    #             info['Lambda_2_min'][cv] = np.min(result['Lambda_2_one_simulation'])
    #             info['Lambda_2_max'][cv] = np.max(result['Lambda_2_one_simulation'])
    #             info['Lambda_sigma'][cv] = result['Lambda_sigma_one_simulation']
    #             info['Error_sigma_one_simulation'][cv] = result['Error_sigma_one_simulation']
    #             info['Lambda_2_one_simulation'][cv] = result['Lambda_2_one_simulation']
    #             info['lambda_2_opt_value'][cv] = result['lambda_2_opt_value']
    #             info['lambda_sigma_opt_value'][cv] = result['lambda_sigma_opt_value']
    #             info['which_lambda_2_opt'][cv] = result['which_lambda_2_opt']
    #             info['which_lambda_sigma_opt'][cv] = result['which_lambda_sigma_opt']
    #             info['sigma_hat'][cv] = result['sigma_hat']
    #             info['select_alpha'][cv] = result['select_alpha']
    #             info['select_fields'][cv] = result['select_fields']
    #             info['select_if'][cv] = result['select_if']
    #             info['select_if_pro'][cv] = result['select_if_pro']
    #             info['select_index'][cv] = result['select_index']
    #             info['select_sigma'][cv] = result['select_sigma']
    #             info['Tree_lambda_sigma_one_simulation'][cv] = result['Tree_lambda_sigma_one_simulation']
    #             info['which_aux'][cv] = result['which_aux']
    #             print(f'已完成第{t + 1}次模拟的协方差惩罚参数区间估计')
    #         Information_sigma[t] = info
    #
    #     # 计算全局lambda_2范围
    #     Lambda_2_small = [np.min(info['Lambda_2_min']) for info in Information_sigma]
    #     Lambda_2_big = [np.max(info['Lambda_2_max']) for info in Information_sigma]
    #
    #     return Information_sigma, Lambda_2_small, Lambda_2_big
    #
    # def cv_penalty_mu(self, result_mst, Lambda_1_small, Lambda_1_big, Information_mu, c_lambda_1_start, k, a,
    #                   residual_principle, iter_max, multiple_constant, num_lambda_1, direct_if):
    #     """
    #     基于固定区间的mu惩罚参数交叉验证主流程
    #     参数：
    #     - result_mst: MST交叉验证结果
    #     - Lambda_1_small/Lambda_1_big: lambda_1上下界
    #     - Information_mu: 先验交叉验证信息
    #     - 其他参数：ADMM超参数和模式
    #     返回：data: 交叉验证结果 dict
    #     """
    #     # 参数默认值
    #     c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
    #     k = 1 if k is None else k
    #     a = 3 if a is None else a
    #     residual_principle = 1e-3 if residual_principle is None else residual_principle
    #     iter_max = 50 if iter_max is None else iter_max
    #     multiple_constant = 1.1 if multiple_constant is None else multiple_constant
    #     num_lambda_1 = 100 if num_lambda_1 is None else num_lambda_1
    #     direct_if = 1 if direct_if is None else direct_if
    #
    #     simulation_times = len(result_mst['Lambda_mu'][0])
    #     cv_number = len(result_mst['Lambda_mu'])
    #     data = {}
    #
    #     # 初始化结果存储
    #     data['result_mst'] = result_mst
    #     data['Lambda_mu'] = result_mst['Lambda_mu']
    #     data['Information_mu'] = Information_mu
    #     data['Error_mu_one_simulation'] = [[None for _ in range(cv_number)] for __ in range(simulation_times)]
    #     data['Lambda_1'] = [[None for _ in range(cv_number)] for __ in range(simulation_times)]
    #     data['Lambda_1_small'] = Lambda_1_small
    #     data['Lambda_1_big'] = Lambda_1_big
    #     data['Lambda_mu_small'] = result_mst['Lambda_mu_small']
    #     data['Lambda_mu_big'] = result_mst['Lambda_mu_big']
    #
    #     # 逐次模拟+交叉验证
    #     for t in range(simulation_times):
    #         for cv in range(cv_number):
    #             result = self.admm3_mu_one_simulation(
    #                 t + 1, result_mst['Tree_lambda_mu'][cv], result_mst['Lambda_mu'][cv],
    #                 len(result_mst['Lambda_mu'][0][0]), c_lambda_1_start, k, a, residual_principle,
    #                 iter_max, multiple_constant, num_lambda_1, result_mst['train_X_labeled'][cv],
    #                 result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
    #                 result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
    #                 result_mst['test_X_unlabeled'][cv], direct_if, Lambda_1_small, Lambda_1_big, None, None
    #             )
    #             data['Error_mu_one_simulation'][t][cv] = result['Error_mu_one_simulation']
    #             data['Lambda_1'][t][cv] = result['Lambda_1_one_simulation'][:, 0]
    #             print(f'已完成第{t + 1}次模拟的均值固定区间估计')
    #
    #     # 确定最优lambda索引
    #     data['Lambda_mu_fields'] = result_mst.get('fields_mu', [])
    #     data['Sum_Error_mu_one_simulation'] = [None] * simulation_times
    #     data['which_lambda_1_opt'] = [None] * simulation_times
    #     data['which_lambda_mu_opt'] = [None] * simulation_times
    #
    #     for t in range(simulation_times):
    #         # 选择误差变化最大的折
    #         panduan_which_store = []
    #         for cv in range(cv_number):
    #             err_mat = data['Error_mu_one_simulation'][t][cv]
    #             panduan_which_store.append(err_mat[0, 0] - err_mat[-1, 0])
    #         panduan_which = np.argmax(panduan_which_store)
    #         data['Sum_Error_mu_one_simulation'][t] = data['Error_mu_one_simulation'][t][panduan_which]
    #
    #         # 寻找最优lambda索引
    #         min_err_idx = np.unravel_index(np.argmin(data['Sum_Error_mu_one_simulation'][t]),
    #                                        data['Sum_Error_mu_one_simulation'][t].shape)
    #         which_lambda_1 = min_err_idx[0] + 1
    #         which_lambda_mu = min_err_idx[1] + 1
    #         # 选择最大lambda_1对应的索引
    #         if isinstance(min_err_idx[0], np.int64) and isinstance(min_err_idx[1], np.int64):
    #             min_err_idx = [[min_err_idx[0]], [min_err_idx[1]]]
    #         which_lambda_1_max = [idx for idx in min_err_idx[0] if idx == np.max(min_err_idx[0])]
    #         which_lambda_mu_max = [idx for idx in min_err_idx[1] if idx == np.min(min_err_idx[1])]
    #         data['which_lambda_1_opt'][t] = which_lambda_1_max[0] if which_lambda_1_max else which_lambda_1
    #         data['which_lambda_mu_opt'][t] = which_lambda_mu_max[0] if which_lambda_mu_max else which_lambda_mu
    #     return data
    #
    # def cv_penalty_sigma(self, result_mst, Lambda_2_small, Lambda_2_big, Information_sigma, c_lambda_2_start, k, a,
    #                      residual_principle, iter_max, multiple_constant, num_lambda_2, direct_if):
    #     """
    #     基于固定区间的sigma惩罚参数交叉验证主流程（与mu对称）
    #     参数：与cv_penalty_mu一致，替换mu为sigma
    #     返回：data: 交叉验证结果 dict
    #     """
    #     # 参数默认值
    #     c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
    #     k = 1 if k is None else k
    #     a = 3 if a is None else a
    #     residual_principle = 1e-3 if residual_principle is None else residual_principle
    #     iter_max = 50 if iter_max is None else iter_max
    #     multiple_constant = 1.1 if multiple_constant is None else multiple_constant
    #     num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2
    #     direct_if = 1 if direct_if is None else direct_if
    #
    #     simulation_times = len(result_mst['Lambda_sigma'][0])
    #     cv_number = len(result_mst['Lambda_sigma'])
    #     data = {}
    #
    #     # 初始化结果存储
    #     data['result_mst'] = result_mst
    #     data['Lambda_sigma'] = result_mst['Lambda_sigma']
    #     data['Information_sigma'] = Information_sigma
    #     data['Error_sigma_one_simulation'] = [[None for _ in range(cv_number)] for __ in range(simulation_times)]
    #     data['Lambda_2'] = [[None for _ in range(cv_number)] for __ in range(simulation_times)]
    #     data['Lambda_2_small'] = Lambda_2_small
    #     data['Lambda_2_big'] = Lambda_2_big
    #     data['Lambda_sigma_small'] = result_mst['Lambda_sigma_small']
    #     data['Lambda_sigma_big'] = result_mst['Lambda_sigma_big']
    #
    #     # 逐次模拟+交叉验证
    #     for t in range(simulation_times):
    #         for cv in range(cv_number):
    #             result = self.admm3_sigma_one_simulation(
    #                 t + 1, result_mst['Tree_lambda_sigma'][cv], result_mst['Lambda_sigma'][cv],
    #                 len(result_mst['Lambda_sigma'][0][0]), c_lambda_2_start, k, a, residual_principle,
    #                 iter_max, multiple_constant, num_lambda_2, result_mst['train_X_labeled'][cv],
    #                 result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
    #                 result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
    #                 result_mst['test_X_unlabeled'][cv], direct_if, Lambda_2_small, Lambda_2_big, None, None
    #             )
    #             data['Error_sigma_one_simulation'][t][cv] = result['Error_sigma_one_simulation']
    #             data['Lambda_2'][t][cv] = result['Lambda_2_one_simulation'][:, 0]
    #             print(f'已完成第{t + 1}次模拟第{cv + 1}次交叉验证的协方差固定区间估计')
    #
    #     # 确定最优lambda索引
    #     data['Lambda_sigma_fields'] = result_mst.get('fields_sigma', [])
    #     data['Sum_Error_sigma_one_simulation'] = [None] * simulation_times
    #     data['which_lambda_2_opt'] = [None] * simulation_times
    #     data['which_lambda_sigma_opt'] = [None] * simulation_times
    #
    #     for t in range(simulation_times):
    #         # 选择误差变化最大的折
    #         panduan_which_store = []
    #         for cv in range(cv_number):
    #             err_mat = data['Error_sigma_one_simulation'][t][cv]
    #             panduan_which_store.append(err_mat[0, 0] - err_mat[-1, 0])
    #         panduan_which = np.argmax(panduan_which_store)
    #         data['Sum_Error_sigma_one_simulation'][t] = data['Error_sigma_one_simulation'][t][panduan_which]
    #
    #         # 寻找最优lambda索引
    #         min_err_idx = np.unravel_index(np.argmin(data['Sum_Error_sigma_one_simulation'][t]),
    #                                        data['Sum_Error_sigma_one_simulation'][t].shape)
    #         which_lambda_2 = min_err_idx[0] + 1
    #         which_lambda_sigma = min_err_idx[1] + 1
    #         # 选择最大lambda_2对应的索引
    #         if isinstance(min_err_idx[0], np.int64) and isinstance(min_err_idx[1], np.int64):
    #             min_err_idx = [[min_err_idx[0]], [min_err_idx[1]]]
    #         which_lambda_2_max = [idx for idx in [min_err_idx[0]] if idx == np.max([min_err_idx[0]])]
    #         which_lambda_sigma_max = [idx for idx in [min_err_idx[1]] if idx == np.min([min_err_idx[1]])]
    #         data['which_lambda_2_opt'][t] = which_lambda_2_max[0] if which_lambda_2_max else which_lambda_2
    #         data['which_lambda_sigma_opt'][t] = which_lambda_sigma_max[
    #             0] if which_lambda_sigma_max else which_lambda_sigma
    #
    #     return data










    def admm3_sigma_one_simulation(self, t, Tree_lambda_sigma, Lambda_sigma, num_lambda_sigma, c_lambda_2_start, k,
                                   a, residual_principle, iter_max, multiple_constant, num_lambda_2, train_X_labeled,
                                   train_Y_labeled, train_X_unlabeled, test_X_labeled, test_Y_labeled,
                                   test_X_unlabeled, direct_if, lambda_2_start_overall, lambda_2_end_overall,
                                   which_lambda_2_opt_input, which_lambda_sigma_opt_input):
        """
        对单个模拟轮次执行协方差路径的第三层 ADMM/lambda 搜索。
        
        参数含义与 admm3_mu_one_simulation 对称；区别在于使用 Tree_lambda_sigma、
        Lambda_sigma 和 lambda_2，并以协方差矩阵距离作为选择准则。
        
        返回
        ----
        result_sigma_output : dict
            协方差路径的最优聚类、选择 source、lambda 索引和诊断量。
        """
        # 参数默认值
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2
        direct_if = None if direct_if is None else direct_if
        lambda_2_start_overall = None if lambda_2_start_overall is None else lambda_2_start_overall
        lambda_2_end_overall = None if lambda_2_end_overall is None else lambda_2_end_overall
        which_lambda_2_opt_input = None if which_lambda_2_opt_input is None else which_lambda_2_opt_input
        which_lambda_sigma_opt_input = None if which_lambda_sigma_opt_input is None else which_lambda_sigma_opt_input

        result = {}
        fields_sigma = list(Tree_lambda_sigma[t - 1].keys())
        Error_sigma_one_simulation = np.zeros((num_lambda_2, num_lambda_sigma))
        Lambda_2_one_simulation = np.zeros((num_lambda_2, num_lambda_sigma))
        Lambda_sigma_one_simulation = Lambda_sigma[t - 1]

        # ── 提取公共数据（不依赖 lambda，只需算一次）──────────────────
        train_X_one_s, train_Y_one_s, train_X_un_one_s, _ = self.one_simulation(
            train_X_labeled, train_Y_labeled, train_X_unlabeled, t
        )
        test_X_one_s, test_Y_one_s, test_X_un_one_s, fields = self.one_simulation(
            test_X_labeled, test_Y_labeled, test_X_unlabeled, t
        )
        X_labeled_one_s = np.vstack([train_X_one_s, test_X_one_s])
        Y_labeled_one_s = np.vstack([train_Y_one_s, test_Y_one_s])
        _alpha_sig = Tree_lambda_sigma[t - 1][fields_sigma[0]]['alpha']
        _, _, _w_sig, _, _, _, _ = self.mst_generation_single(
            test_X_one_s, test_Y_one_s, test_X_un_one_s, fields,
            X_labeled_one_s, Y_labeled_one_s, _alpha_sig
        )
        A1 = _w_sig['labeled_sigma']

        # ── 顺序遍历所有 lambda_sigma（A1 已在循环外算好，避免冗余调用）──
        for lambda_sigma_i in range(num_lambda_sigma):
            tree = Tree_lambda_sigma[t - 1][fields_sigma[lambda_sigma_i]]
            _err, _lam = self.admm2_sigma_one_simulation_one_lambda_sigma_i(
                tree, c_lambda_2_start, k, a, residual_principle,
                iter_max, multiple_constant, num_lambda_2, A1, direct_if,
                lambda_2_start_overall, lambda_2_end_overall, t
            )
            Error_sigma_one_simulation[:, lambda_sigma_i] = _err.ravel()
            Lambda_2_one_simulation[:, lambda_sigma_i] = _lam.ravel()

        # 分支1: 无输入最优索引，自动选择
        if direct_if is None:
            min_err_idx = np.unravel_index(np.argmin(Error_sigma_one_simulation), Error_sigma_one_simulation.shape)
            which_lambda_2_opt = min_err_idx[0] + 1
            which_lambda_sigma_opt = min_err_idx[1] + 1

            lambda_sigma_opt_value = Lambda_sigma_one_simulation[which_lambda_sigma_opt - 1]
            lambda_2_opt_value = Lambda_2_one_simulation[which_lambda_2_opt - 1, which_lambda_sigma_opt - 1]
            Tree_opt = Tree_lambda_sigma[t - 1][fields_sigma[which_lambda_sigma_opt - 1]]

            # 求解ADMM
            sigma_hat, select_if, which_aux, _, select_if_pro = self.admm_sigma_one_simulation_one_lambda_sigma_i(
                Tree_opt, lambda_2_opt_value, k, a, residual_principle, iter_max
            )

            # 封装结果
            result['Tree_lambda_sigma_one_simulation'] = Tree_opt
            result['Error_sigma_one_simulation'] = Error_sigma_one_simulation
            result['Lambda_2_one_simulation'] = Lambda_2_one_simulation
            result['Lambda_sigma_one_simulation'] = Lambda_sigma_one_simulation
            result['which_lambda_sigma_opt'] = which_lambda_sigma_opt
            result['which_lambda_2_opt'] = which_lambda_2_opt
            result['lambda_sigma_opt_value'] = lambda_sigma_opt_value
            result['lambda_2_opt_value'] = lambda_2_opt_value
            result['sigma_hat'] = sigma_hat
            result['select_if'] = select_if
            result['select_if_pro'] = select_if_pro
            result['which_aux'] = which_aux
            result['select_sigma'] = Tree_opt['index_sigma'][which_aux - 1]
            result['select_index'] = Tree_opt['index'][which_aux - 1]
            result['select_alpha'] = Tree_opt['alpha']
            result['select_fields'] = [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]]
        # 分支2: 有输入最优索引或直接模式
        else:
            if which_lambda_2_opt_input is None:
                result['Error_sigma_one_simulation'] = Error_sigma_one_simulation
                result['fields_sigma'] = fields_sigma
                result['Lambda_2_one_simulation'] = Lambda_2_one_simulation
                result['Lambda_sigma_one_simulation'] = Lambda_sigma_one_simulation
            else:
                # ========== 核心修复：安全处理标量/数组索引 ==========
                # 判断输入是否为标量，标量直接使用，数组才取[0]
                if np.isscalar(which_lambda_2_opt_input):
                    which_lambda_2_opt = which_lambda_2_opt_input
                else:
                    which_lambda_2_opt = which_lambda_2_opt_input[0] if len(which_lambda_2_opt_input) > 0 else 1

                if np.isscalar(which_lambda_sigma_opt_input):
                    which_lambda_sigma_opt = which_lambda_sigma_opt_input
                else:
                    which_lambda_sigma_opt = which_lambda_sigma_opt_input[0] if len(
                        which_lambda_sigma_opt_input) > 0 else 1

                lambda_sigma_opt_value = Lambda_sigma_one_simulation[which_lambda_sigma_opt - 1]
                lambda_2_opt_value = Lambda_2_one_simulation[which_lambda_2_opt - 1, which_lambda_sigma_opt - 1]
                Tree_opt = Tree_lambda_sigma[t - 1][fields_sigma[which_lambda_sigma_opt - 1]]

                # 求解ADMM
                sigma_hat, select_if, which_aux, _, select_if_pro = self.admm_sigma_one_simulation_one_lambda_sigma_i(
                    Tree_opt, lambda_2_opt_value, k, a, residual_principle, iter_max
                )

                # 封装结果
                result.update({
                    'Tree_lambda_sigma_one_simulation': Tree_opt,
                    'Error_sigma_one_simulation': Error_sigma_one_simulation,
                    'Lambda_2_one_simulation': Lambda_2_one_simulation,
                    'Lambda_sigma_one_simulation': Lambda_sigma_one_simulation,
                    'which_lambda_sigma_opt': which_lambda_sigma_opt,
                    'which_lambda_2_opt': which_lambda_2_opt,
                    'lambda_sigma_opt_value': lambda_sigma_opt_value,
                    'lambda_2_opt_value': lambda_2_opt_value,
                    'sigma_hat': sigma_hat,
                    'select_if': select_if,
                    'select_if_pro': select_if_pro,
                    'which_aux': which_aux,
                    'select_sigma': Tree_opt['index_sigma'][which_aux - 1],
                    'select_index': Tree_opt['index'][which_aux - 1],
                    'select_alpha': Tree_opt['alpha'],
                    'select_fields': [Tree_opt['fields'][idx - 1] for idx in Tree_opt['index'][which_aux - 1]],
                    'fields_sigma': fields_sigma
                })
        return result


    def cv_penalty_parameter_sigma(self, result_mst, c_lambda_2_start, k, a, residual_principle, iter_max,
                                   multiple_constant, num_lambda_2):
        """
        通过交叉验证确定协方差路径 ADMM 惩罚参数 lambda_2 的全局搜索范围。
        
        函数遍历 result_mst 中所有模拟轮次和 lambda_sigma 候选，调用区间搜索函数得到每个
        局部问题的 lambda_2 上下界，再取全局最小下界和最大上界作为后续网格搜索区间。
        
        返回
        ----
        tuple
            Information_sigma 以及 Lambda_2_small、Lambda_2_big。
        """
        # 参数默认值
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 2 if multiple_constant is None else multiple_constant
        num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2

        cv_number = len(result_mst['Lambda_sigma'])
        simulation_times = len(result_mst['Lambda_sigma'][0])
        Information_sigma = [{} for _ in range(simulation_times)]

        # ── 顺序执行（保证结果可复现）─────────────────────────────
        for t in tqdm(range(simulation_times), desc='Step 4/7 | Σ-ADMM 惩罚范围 (CV)', unit='sim', ncols=90, leave=True):
            info = {
                'Lambda_2_min': np.zeros((cv_number,1)), 'Lambda_2_max': np.zeros((cv_number,1)),
                'Lambda_sigma': [None]*cv_number, 'Error_sigma_one_simulation': [None]*cv_number,
                'Lambda_2_one_simulation': [None]*cv_number, 'lambda_2_opt_value': np.zeros((cv_number,1)),
                'lambda_sigma_opt_value': np.zeros((cv_number,1)), 'which_lambda_2_opt': np.zeros((cv_number,1)),
                'which_lambda_sigma_opt': np.zeros((cv_number,1)), 'sigma_hat': [None]*cv_number,
                'select_alpha': np.zeros((cv_number,1)), 'select_fields': [None]*cv_number,
                'select_if': np.zeros((cv_number,1)), 'select_if_pro': np.zeros((cv_number,1)),
                'select_index': [None]*cv_number, 'select_sigma': [None]*cv_number,
                'Tree_lambda_sigma_one_simulation': [None]*cv_number, 'which_aux': np.zeros((cv_number,1))
            }
            for cv in range(cv_number):
                result = self.admm3_sigma_one_simulation(
                    t + 1, result_mst['Tree_lambda_sigma'][cv], result_mst['Lambda_sigma'][cv],
                    len(result_mst['Lambda_sigma'][0][0]), c_lambda_2_start, k, a, residual_principle,
                    iter_max, multiple_constant, num_lambda_2, result_mst['train_X_labeled'][cv],
                    result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
                    result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
                    result_mst['test_X_unlabeled'][cv], None, None, None, None, None
                )
                info['Lambda_2_min'][cv] = np.min(result['Lambda_2_one_simulation'])
                info['Lambda_2_max'][cv] = np.max(result['Lambda_2_one_simulation'])
                info['Lambda_sigma'][cv] = result['Lambda_sigma_one_simulation']
                info['Error_sigma_one_simulation'][cv] = result['Error_sigma_one_simulation']
                info['Lambda_2_one_simulation'][cv] = result['Lambda_2_one_simulation']
                info['lambda_2_opt_value'][cv] = result['lambda_2_opt_value']
                info['lambda_sigma_opt_value'][cv] = result['lambda_sigma_opt_value']
                info['which_lambda_2_opt'][cv] = result['which_lambda_2_opt']
                info['which_lambda_sigma_opt'][cv] = result['which_lambda_sigma_opt']
                info['sigma_hat'][cv] = result['sigma_hat']
                info['select_alpha'][cv] = result['select_alpha']
                info['select_fields'][cv] = result['select_fields']
                info['select_if'][cv] = result['select_if']
                info['select_if_pro'][cv] = result['select_if_pro']
                info['select_index'][cv] = result['select_index']
                info['select_sigma'][cv] = result['select_sigma']
                info['Tree_lambda_sigma_one_simulation'][cv] = result['Tree_lambda_sigma_one_simulation']
                info['which_aux'][cv] = result['which_aux']
            Information_sigma[t] = info

        # 计算全局lambda_2范围
        Lambda_2_small = [np.min(info['Lambda_2_min']) for info in Information_sigma]
        Lambda_2_big = [np.max(info['Lambda_2_max']) for info in Information_sigma]

        return Information_sigma, Lambda_2_small, Lambda_2_big

    def cv_penalty_mu(self, result_mst, Lambda_1_small, Lambda_1_big, Information_mu, c_lambda_1_start,
                      k, a, residual_principle, iter_max, multiple_constant, num_lambda_1, direct_if):
        """
        在均值路径上执行最终交叉验证网格搜索。
        
        输入由 cv_penalty_parameter_mu 得到的 lambda_1 全局区间和 MST 结果，函数逐模拟扫描
        lambda_mu 与 lambda_1 的组合，保存验证准则最优的索引和对应中间结果。
        
        返回
        ----
        data_mu : dict
            包含最优 lambda_mu/lambda_1 索引、全局区间和 result_mst 引用。
        """
        # 参数默认值
        c_lambda_1_start = 0.001 if c_lambda_1_start is None else c_lambda_1_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 1.1 if multiple_constant is None else multiple_constant
        num_lambda_1 = 100 if num_lambda_1 is None else num_lambda_1
        direct_if = 1 if direct_if is None else direct_if

        simulation_times = len(result_mst['Lambda_mu'][0])
        cv_number = len(result_mst['Lambda_mu'])
        data = {}

        # 基础信息存储
        data['result_mst'] = result_mst
        data['Lambda_mu'] = result_mst['Lambda_mu']
        data['Information_mu'] = Information_mu
        data['Error_mu_one_simulation'] = [[None for _ in range(cv_number)] for _ in range(simulation_times)]
        data['Lambda_1'] = [[None for _ in range(cv_number)] for _ in range(simulation_times)]
        data['Lambda_1_small'] = Lambda_1_small
        data['Lambda_1_big'] = Lambda_1_big
        data['Lambda_mu_small'] = result_mst['Lambda_mu_small']
        data['Lambda_mu_big'] = result_mst['Lambda_mu_big']

        # ── 顺序执行（保证结果可复现）─────────────────────────────
        for t in tqdm(range(simulation_times), desc='Step 5/7 | μ-ADMM 最优参数 (CV)', unit='sim', ncols=90, leave=True):
            for cv in range(cv_number):
                result = self.admm3_mu_one_simulation(
                    t + 1, result_mst['Tree_lambda_mu'][cv], result_mst['Lambda_mu'][cv],
                    len(result_mst['Lambda_mu'][0][0]), c_lambda_1_start, k, a, residual_principle,
                    iter_max, multiple_constant, num_lambda_1, result_mst['train_X_labeled'][cv],
                    result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
                    result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
                    result_mst['test_X_unlabeled'][cv], direct_if, Lambda_1_small, Lambda_1_big, None, None
                )
                data['Error_mu_one_simulation'][t][cv] = result['Error_mu_one_simulation']
                data['Lambda_1'][t][cv] = result['Lambda_1_one_simulation'][:, 0]

        # 字段和误差汇总
        data['Lambda_mu_fields'] = result_mst.get('fields_mu', [])
        data['Sum_Error_mu_one_simulation'] = [None] * simulation_times
        data['which_lambda_1_opt'] = [None] * simulation_times
        data['which_lambda_mu_opt'] = [None] * simulation_times

        for t in range(simulation_times):
            # 选择误差变化最大的折
            panduan_which_store = np.zeros(cv_number)
            for cv in range(cv_number):
                err_mat = data['Error_mu_one_simulation'][t][cv]
                panduan_which_store[cv] = err_mat[0, 0] - err_mat[-1, 0]
            panduan_which = np.argmax(panduan_which_store)
            data['Sum_Error_mu_one_simulation'][t] = data['Error_mu_one_simulation'][t][panduan_which]

            # 寻找最优lambda索引
            min_err_idx = np.unravel_index(np.argmin(data['Sum_Error_mu_one_simulation'][t]),
                                           data['Sum_Error_mu_one_simulation'][t].shape)
            which_lambda_1 = min_err_idx[0] + 1
            which_lambda_mu = min_err_idx[1] + 1

            # 安全处理标量/数组
            which_lambda_1_max = which_lambda_1[np.argmax(which_lambda_1)] if isinstance(which_lambda_1,
                                                                                         np.ndarray) else which_lambda_1
            which_lambda_mu_max = which_lambda_mu[0] if isinstance(which_lambda_mu, np.ndarray) else which_lambda_mu

            data['which_lambda_1_opt'][t] = which_lambda_1_max
            data['which_lambda_mu_opt'][t] = which_lambda_mu_max

        return data

    def cv_penalty_sigma(self, result_mst, Lambda_2_small, Lambda_2_big, Information_sigma, c_lambda_2_start,
                         k, a, residual_principle, iter_max, multiple_constant, num_lambda_2, direct_if):
        """
        在协方差路径上执行最终交叉验证网格搜索。
        
        输入由 cv_penalty_parameter_sigma 得到的 lambda_2 全局区间和 MST 结果，函数逐模拟扫描
        lambda_sigma 与 lambda_2 的组合，保存验证准则最优的索引和对应中间结果。
        
        返回
        ----
        data_sigma : dict
            包含最优 lambda_sigma/lambda_2 索引、全局区间和 result_mst 引用。
        """
        # 参数默认值
        c_lambda_2_start = 0.001 if c_lambda_2_start is None else c_lambda_2_start
        k = 1 if k is None else k
        a = 3 if a is None else a
        residual_principle = 1e-3 if residual_principle is None else residual_principle
        iter_max = 50 if iter_max is None else iter_max
        multiple_constant = 1.1 if multiple_constant is None else multiple_constant
        num_lambda_2 = 100 if num_lambda_2 is None else num_lambda_2
        direct_if = 1 if direct_if is None else direct_if

        simulation_times = len(result_mst['Lambda_sigma'][0])
        cv_number = len(result_mst['Lambda_sigma'])
        data = {}

        # 基础信息存储
        data['result_mst'] = result_mst
        data['Lambda_sigma'] = result_mst['Lambda_sigma']
        data['Information_sigma'] = Information_sigma
        data['Error_sigma_one_simulation'] = [[None for _ in range(cv_number)] for _ in range(simulation_times)]
        data['Lambda_2'] = [[None for _ in range(cv_number)] for _ in range(simulation_times)]
        data['Lambda_2_small'] = Lambda_2_small
        data['Lambda_2_big'] = Lambda_2_big
        data['Lambda_sigma_small'] = result_mst['Lambda_sigma_small']
        data['Lambda_sigma_big'] = result_mst['Lambda_sigma_big']

        # ── 顺序执行（保证结果可复现）─────────────────────────────
        for t in tqdm(range(simulation_times), desc='Step 6/7 | Σ-ADMM 最优参数 (CV)', unit='sim', ncols=90, leave=True):
            for cv in range(cv_number):
                result = self.admm3_sigma_one_simulation(
                    t + 1, result_mst['Tree_lambda_sigma'][cv], result_mst['Lambda_sigma'][cv],
                    len(result_mst['Lambda_sigma'][0][0]), c_lambda_2_start, k, a, residual_principle,
                    iter_max, multiple_constant, num_lambda_2, result_mst['train_X_labeled'][cv],
                    result_mst['train_Y_labeled'][cv], result_mst['train_X_unlabeled'][cv],
                    result_mst['test_X_labeled'][cv], result_mst['test_Y_labeled'][cv],
                    result_mst['test_X_unlabeled'][cv], direct_if, Lambda_2_small, Lambda_2_big,
                    None, None
                )
                data['Error_sigma_one_simulation'][t][cv] = result['Error_sigma_one_simulation']
                data['Lambda_2'][t][cv] = result['Lambda_2_one_simulation'][:, 0]

        # 字段和误差汇总
        data['Lambda_sigma_fields'] = result_mst.get('fields_sigma', [])
        data['Sum_Error_sigma_one_simulation'] = [None] * simulation_times
        data['which_lambda_2_opt'] = [None] * simulation_times
        data['which_lambda_sigma_opt'] = [None] * simulation_times

        for t in range(simulation_times):
            # 选择误差变化最大的折
            panduan_which_store = np.zeros(cv_number)
            for cv in range(cv_number):
                err_mat = data['Error_sigma_one_simulation'][t][cv]
                panduan_which_store[cv] = err_mat[0, 0] - err_mat[-1, 0]
            panduan_which = np.argmax(panduan_which_store)
            data['Sum_Error_sigma_one_simulation'][t] = data['Error_sigma_one_simulation'][t][panduan_which]

            # 寻找最优lambda索引
            min_err_idx = np.unravel_index(np.argmin(data['Sum_Error_sigma_one_simulation'][t]),
                                           data['Sum_Error_sigma_one_simulation'][t].shape)
            which_lambda_2 = min_err_idx[0] + 1
            which_lambda_sigma = min_err_idx[1] + 1

            # ========== 关键修改2：安全处理标量/数组 ==========
            which_lambda_2_max = which_lambda_2[np.argmax(which_lambda_2)] if isinstance(which_lambda_2,
                                                                                         np.ndarray) else which_lambda_2
            which_lambda_sigma_max = which_lambda_sigma[0] if isinstance(which_lambda_sigma,
                                                                         np.ndarray) else which_lambda_sigma

            data['which_lambda_2_opt'][t] = which_lambda_2_max
            data['which_lambda_sigma_opt'][t] = which_lambda_sigma_max

        return data

    def final_selection(self, data_mu, data_sigma, X_unlabeled, simulation_times):
        """
        最终未标记数据选择：整合均值(mu)与协方差(sigma)两条优化路径的
        交叉验证结果，对每次模拟输出最优数据源集合及对应未标记样本。

        参数
        ----
        data_mu : dict
            mu 路径的交叉验证汇总结果（由 cv_penalty_mu 返回），包含
            'which_lambda_1_opt'、'which_lambda_mu_opt'、'Lambda_mu'、
            'Lambda_1'、'Information_mu' 等键。
        data_sigma : dict
            sigma 路径的交叉验证汇总结果（由 cv_penalty_sigma 返回），
            结构与 data_mu 对称，字段名中 mu 替换为 sigma。
        X_unlabeled : dict
            未标记数据字典，键为数据源名称（如 'm1s1'），值为长度等于
            simulation_times 的列表，每个元素为 [n_unlabeled, p] 数组。
        simulation_times : int
            模拟次数（即 X_labeled.shape[2]）。

        返回
        ----
        final_result : dict
            包含以下键：
            - 'selected_fields_all_simulations': list[list[str]]
              每次模拟选出的数据源名称列表。
            - 'selected_data_all_simulations': list[np.ndarray]
              每次模拟对应的合并未标记样本矩阵 [n_sel, p]。
            - 'optimal_lambda_all_simulations': list[dict]
              每次模拟使用的最优 lambda 值（mu 和 sigma 各一套）。
            - 'selection_metrics': list[dict]
              每次模拟的选择率、选择样本数、总未标记样本数。
            - 'global_metrics': dict
              跨模拟的汇总统计（平均选择率、标准差、最高频字段等）。
        """
        final_result = {
            'selected_fields_all_simulations': [],
            'selected_data_all_simulations': [],
            'optimal_lambda_all_simulations': [],
            'selection_metrics': []
        }

        for t in range(simulation_times):
            # 获取单次模拟的最优lambda索引
            which_lambda_1_opt = data_mu['which_lambda_1_opt'][t]
            which_lambda_mu_opt = data_mu['which_lambda_mu_opt'][t]
            which_lambda_2_opt = data_sigma['which_lambda_2_opt'][t]
            which_lambda_sigma_opt = data_sigma['which_lambda_sigma_opt'][t]

            # 获取mu和sigma的最优字段集合
            fields_mu = data_mu['Lambda_mu_fields']
            fields_sigma = data_sigma['Lambda_sigma_fields']
            selected_fields_mu = data_mu['Information_mu'][t]['select_fields'][which_lambda_mu_opt - 1]
            selected_fields_sigma = data_sigma['Information_sigma'][t]['select_fields'][which_lambda_sigma_opt - 1]

            # 字段交集作为最终选择
            selected_fields = sorted(set(selected_fields_mu) & set(selected_fields_sigma))
            # 交集为空时取并集，保证选择有效性
            if not selected_fields:
                selected_fields = sorted(set(selected_fields_mu) | set(selected_fields_sigma))
                tqdm.write(f'  [注意] 第{t+1}次模拟 μ/Σ 选择字段无交集，已取并集')

            # 提取未标记数据
            selected_data = []
            for field in selected_fields:
                if field in X_unlabeled:
                    selected_data.append(X_unlabeled[field][t])
            selected_data = np.vstack(selected_data) if selected_data else np.array([])

            # 存储单次模拟结果
            final_result['selected_fields_all_simulations'].append(selected_fields)
            final_result['selected_data_all_simulations'].append(selected_data)
            final_result['optimal_lambda_all_simulations'].append({
                'mu': {
                    'lambda_1': data_mu['Lambda_1'][t][which_lambda_mu_opt - 1][which_lambda_1_opt - 1],
                    'lambda_mu': data_mu['Lambda_mu'][0][t][which_lambda_mu_opt - 1]
                },
                'sigma': {
                    'lambda_2': data_sigma['Lambda_2'][t][which_lambda_sigma_opt - 1][which_lambda_2_opt - 1],
                    'lambda_sigma': data_sigma['Lambda_sigma'][0][t][which_lambda_sigma_opt - 1]
                }
            })

            # 计算选择指标：选择率 = 选择样本数 / 总未标记样本数
            total_unlabeled = sum([arr.shape[0] for arr in X_unlabeled.values()])
            selection_rate = selected_data.shape[0] / total_unlabeled if total_unlabeled > 0 else 0
            final_result['selection_metrics'].append({
                'selection_rate': selection_rate,
                'n_selected': selected_data.shape[0],
                'n_total_unlabeled': total_unlabeled
            })

            tqdm.write(f'  [完成] 第{t+1}次模拟  选源={len(selected_fields)}个  样本数={selected_data.shape[0]}')

        # 计算全局统计指标
        selection_rates = [m['selection_rate'] for m in final_result['selection_metrics']]
        final_result['global_metrics'] = {
            'mean_selection_rate': np.mean(selection_rates),
            'std_selection_rate': np.std(selection_rates),
            'mean_n_selected': np.mean([m['n_selected'] for m in final_result['selection_metrics']]),
            'most_frequent_fields': max(
                [f for fields in final_result['selected_fields_all_simulations'] for f in fields],
                key=lambda x: [f for fields in final_result['selected_fields_all_simulations'] for f in
                               fields].count(x)
            ) if any(final_result['selected_fields_all_simulations']) else None
        }

        return final_result
