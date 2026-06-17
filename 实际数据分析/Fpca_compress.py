#%%
import os
from operator import concat
from xml.sax import default_parser_list

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import glob
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import autocast, GradScaler
import argparse
from tqdm import tqdm
import gc
from FPCA import FPCA1D
# FPCA1D使用方法:
# numpy.ndarray: [n*t]
# fp = FPCA1D(n_components = k).fit(theta, t=t) #pve=0.95
import argparse
from sklearn.preprocessing import StandardScaler


def process_csv_file(file_path):
    """
    读取单个角度 CSV 文件，提取数值型角度序列。

    文件格式约定：第一行为标题行，从第二行起为角度数值数据（每行一个浮点数）。
    若文件不可读、行数不足或数值转换失败，返回 None。

    参数
    ----
    file_path : str
        CSV 文件的完整路径。

    返回
    ----
    vector : np.ndarray 或 None
        从文件中读取的角度数值数组（去掉标题行后的所有行）。
        若读取或解析失败，则返回 None。
    """
    try:
        # 以文本方式逐行读取，兼容编码问题
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        # 确保文件至少有1行（含标题）
        if len(lines) < 1:
            print(f"文件 {file_path} 行数不足")
            return None
        else:
            try:
                # 跳过第一行标题，将剩余行转为浮点数数组
                vector = np.array([float(x) for x in lines[1:]])
            except ValueError as e:
                print(f"转换数据时出错: {str(e)}")
                return None
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {str(e)}")
        return None
    return vector


def collect_all_data(dataset_dirs, pattern='*摆动角度.csv'):
    """
    从多个数据集目录中批量收集角度序列数据。

    遍历所有匹配目录，按文件名模式搜索 CSV 文件，
    读取角度序列后对异常帧（跳帧）进行均值替换，
    仅保留长度恰好为 599 的有效序列。

    参数
    ----
    dataset_dirs : list of str
        数据集目录路径列表（支持 glob 通配符）。
    pattern : str, 默认 '*摆动角度.csv'
        用于匹配目标 CSV 文件名的 glob 模式。

    返回
    ----
    all_vectors : np.ndarray, shape (N, 599) 或 None
        所有有效角度序列堆叠成的二维数组，N 为有效样本数。
    file_info : list of dict 或 None
        每个有效样本的文件信息列表，每项包含：
        - 'file'   : str，原始 CSV 文件路径
        - 'vector' : np.ndarray，原始角度序列
        - 'index'  : int，在 all_vectors 中的行索引
    """
    all_vectors = []
    file_info = []  # 存储文件信息，用于后续处理
    for dataset_dir in dataset_dirs:
        print(f"从数据集收集数据: {dataset_dir}")
        # 处理front和side文件夹
        # for view in ['front', 'side']:
        #     csv_dir = os.path.join(dataset_dir, 'Analysis_CSV/2025-05-18', view)
        csv_dirs = glob.glob(dataset_dir)  # csv_dir
        for csv_dir_path in csv_dirs:
            csv_files = glob.glob(os.path.join(csv_dir_path, pattern))
            for csv_file in csv_files:
                print(f"处理文件: {csv_file}")
                # 读取单个 CSV 文件的角度序列
                vector = process_csv_file(csv_file)
                if vector is None:
                    # 文件读取失败或格式不合规时跳过，避免后续均值/方差计算报错
                    continue
                # 跳帧异常值处理：若某帧与均值的偏差超过 5 倍标准差，则替换为均值
                # 这是一种简单的离群点修复策略，防止姿态估计的跳帧误差影响后续分析
                vector_mean = np.mean(vector)
                outlier_mask = (vector - vector_mean) ** 2 > 5 * np.std(vector) ** 2
                vector[outlier_mask] = vector_mean
                # 绘制提取的某关节角度变化曲线
                # plt.figure()
                # plt.plot(all_vectors[315], color = 'g') # all_vectors[0], vector
                # plt.show()
                all_vectors.append(vector)
                # all_vectors.append(vector2)
                file_info.append({
                    'file': csv_file,
                    'vector': vector,
                    'index': len(all_vectors) - 1
                })
    if not all_vectors:
        print("没有找到有效数据")
        return None, None

    # 只保留长度为 599 的序列，确保所有样本时间长度一致（FPCA 要求等长输入）
    all_vectors_delete = []
    file_info_delete = []
    for i in range(len(all_vectors)):  # 读取的向量的个数
        if len(all_vectors[i]) == 599:
            all_vectors_delete.append(all_vectors[i])
            file_info_delete.append(file_info[i])
    all_vectors = all_vectors_delete
    file_info = file_info_delete
    # 将所有向量堆叠成一个二维数组，行为样本，列为时间点
    all_vectors = np.array(all_vectors)
    print(f"总共收集到 {len(all_vectors)} 个向量，维度为 {all_vectors.shape[1]}")
    return all_vectors, file_info


def compress_vectors(model, vectors, device, file_info, joint_names):
    """
    对每个关节的角度序列分别进行函数型 PCA 压缩，提取低维主成分得分。

    首先按关节名称对所有样本进行分组（通过解析文件名中的关节标识符），
    然后对每组关节数据独立拟合 FPCA1D 模型，
    最后将各关节的主成分得分合并返回。

    参数
    ----
    model : FPCA1D
        未拟合的 FPCA1D 模型实例（主要用于获取 n_components 参数，
        实际拟合在函数内部通过 args.n_components 重新创建）。
    vectors : np.ndarray, shape (N, T)
        所有样本的角度序列（各关节混合，按 file_info 中的顺序排列）。
    device : torch.device
        计算设备（本函数实际未使用，保留接口兼容性）。
    file_info : list of dict
        样本文件信息列表，每项需包含 'file'、'vector'、'index' 字段。
    joint_names : list of str
        所有关节名称列表，用于初始化分组字典及确定关节标识。

    返回
    ----
    compressed_vectors : np.ndarray, shape (N_out, K)
        所有样本压缩后的主成分得分矩阵，K 为 args.n_components。
    compressed_file_info : list of dict
        与压缩向量对应的文件信息列表，每项包含：
        - 'file'              : str，原始文件路径
        - 'vector'            : np.ndarray，原始角度序列
        - 'compressed_vector' : np.ndarray，压缩后的主成分得分
        - 'index'             : int，在原始 file_info 中的索引
    """
    compressed_vectors = []
    compressed_file_info = []

    # 初始化按关节分组的存储结构
    vectors_every_joint = {}      # {关节名: [角度序列列表]}
    file_info_every_joint = {}    # {关节名: {'file':[], 'vector':[], 'index':[]}}
    for item in joint_names:
        vectors_every_joint[item] = []
        file_info_every_joint[item] = {}
        file_info_every_joint[item]['file'] = []
        file_info_every_joint[item]['vector'] = []
        file_info_every_joint[item]['index'] = []

    # 解析文件名以提取关节标识，并按关节分组
    # 文件名格式示例："前缀-XX-后缀.csv"，提取两个连字符之间的关节缩写（2字符）
    for i, file_info_single in enumerate(file_info):
        name_file_single = list(os.path.basename(file_info_single['file']))
        # 找出文件名中所有 '-' 的位置
        position = [i for i, _ in enumerate(name_file_single) if name_file_single[i] == '-']
        # 取第一个 '-' 之后的2个字符作为关节标识
        index = [position[0] + 1, position[0] + 3]
        name_this = ''.join(name_file_single[index[0]:index[1]])
        # 将当前样本归入对应关节的分组
        vectors_every_joint[name_this].append(file_info_single['vector'])
        file_info_every_joint[name_this]['file'].append(file_info_single['file'])
        file_info_every_joint[name_this]['vector'].append(file_info_single['vector'])
        file_info_every_joint[name_this]['index'].append(file_info_single['index'])

    """使用FPCA模型压缩向量得到低维特征并记录"""
    # 对每个关节独立进行 FPCA，确保各关节的主成分互不干扰
    for joint in joint_names:
        theta = vectors_every_joint[joint]
        # 用该关节的所有样本拟合 FPCA 模型，提取主成分得分
        fp = FPCA1D(n_components=args.n_components).fit(theta)  # pve=0.95
        print("scores:", fp.scores_.shape, "first explained ratio:", np.sum(fp.explained_ratio_), f"{joint}")
        # plt.figure()
        # t = np.linspace(0, 1, len(fp.scores_.T))
        # plt.plot(t, fp.scores_.T)  # 把 (n, T) 转成 (T, n)
        # plt.xlabel("time")
        # plt.ylabel("theta")
        # plt.title("theta(t) — all samples")
        # plt.show()

        # 将该关节每个样本的主成分得分及对应文件信息存入结果列表
        for i in range(len(fp.scores_)):
            compressed_vectors.append(fp.scores_[i])
            compressed_file_info_single = {
                'file' : file_info_every_joint[joint]['file'][i],
                'vector' : file_info_every_joint[joint]['vector'][i],
                'compressed_vector' : fp.scores_[i],  # 当前文件对应的低维得分
                'index' :  file_info_every_joint[joint]['index'][i]
            }
            compressed_file_info.append(compressed_file_info_single)
    return np.array(compressed_vectors), compressed_file_info


def save_compressed_results(file_info, compressed_vectors, output_dir='compressed_results'):
    """
    将压缩后的向量按原始文件路径分组，依次保存为 CSV 文件。

    遍历 file_info 列表，检测文件路径变化，将属于同一原始文件的
    压缩向量收集后调用 save_file_result() 写出。

    参数
    ----
    file_info : list of dict
        压缩后的样本文件信息列表，每项须包含 'file' 字段。
    compressed_vectors : np.ndarray, shape (N, K)
        所有样本压缩后的主成分得分矩阵，行顺序与 file_info 一致。
    output_dir : str, 默认 'compressed_results'
        压缩结果的根输出目录，不存在时自动创建。

    返回
    ----
    None
    """
    # 创建输出目录（如已存在则忽略）
    os.makedirs(output_dir, exist_ok=True)

    # 用于追踪当前正在处理的文件及其压缩向量列表
    current_file = None
    current_vectors = []

    for i, info in enumerate(file_info):
        if current_file != info['file']:
            # 检测到新的文件路径，先保存前一个文件的所有压缩向量
            if current_file is not None and current_vectors:
                save_file_result(current_file, current_vectors, output_dir)

            # 切换到新文件，重置向量缓冲区
            current_file = info['file']
            current_vectors = []

        # 将当前样本的压缩向量加入缓冲区
        current_vectors.append(compressed_vectors[i])

    # 循环结束后保存最后一个文件的压缩结果
    if current_file is not None and current_vectors:
        save_file_result(current_file, current_vectors, output_dir)


def save_file_result(file_path, vectors, output_dir):
    """
    保存单个原始文件对应的压缩向量到 CSV 文件。

    在输出目录下重建与原始文件相同的子目录结构（仅保留最后一级目录名），
    然后以原文件名写出压缩向量（取 vectors[0]，即该文件的第一个压缩向量）。

    参数
    ----
    file_path : str
        原始 CSV 文件的完整路径，用于确定输出文件名和目录。
    vectors : list of np.ndarray
        该文件对应的压缩向量列表（每项为一个主成分得分数组）。
    output_dir : str
        压缩结果的根输出目录。

    返回
    ----
    None
    """
    # 取原文件所在目录的最后一级名称，保持目录结构一致
    last_dir = os.path.basename(os.path.dirname(file_path))
    result_dir = os.path.join(output_dir, last_dir)
    # result_dir = os.path.dirname(file_path).replace(f"{file_path[:2]}*2025-05-20", output_dir)

    os.makedirs(result_dir, exist_ok=True)
    # 输出文件与原始文件同名，存放在对应的子目录下
    result_file = os.path.join(result_dir, os.path.basename(file_path))

    # 将压缩后的主成分得分写出为单列 CSV 文件
    result_df = pd.DataFrame({
        'vector_compressed': vectors[0]
    })
    result_df.to_csv(result_file, index=False)

    print(f"已保存压缩结果到: {result_file}")


def extraction(file_info, output_dir, dimension):
    """
    从已保存的压缩 CSV 文件中抽取数据，整合成按受试者和任务类型分类的特征矩阵。

    流程：
    1. 遍历 file_info，从文件名前缀推断关节点类别，构建列名列表 dataframe_name。
    2. 按受试者目录（index_sigle）分组，每组对应一个受试者的一次测量。
    3. 对每个受试者，读取各关节对应的压缩 CSV，拼接为一行宽表特征向量。
    4. 按目录后缀（'_1' 或 '_2'）区分单任务和双任务，分别存储。
    5. 最终返回单/双任务的 DataFrame 及对应的受试者 ID 列表。

    参数
    ----
    file_info : list of dict
        压缩后的样本文件信息列表，每项包含 'file' 字段（完整路径）。
        注意：本函数会**原地修改**传入的 file_info 列表（逐步弹出已处理项）。
    output_dir : str
        压缩结果的根输出目录，与 save_compressed_results() 保存位置一致。
    dimension : int
        每个关节压缩后的特征维数（即 FPCA 主成分数 n_components），
        用于确定拼接后特征向量的列索引区间。

    返回
    ----
    dataframe_name : list of str
        关节点类别名称列表（去重后的文件名前缀部分），作为 DataFrame 的列标签基础。
    vector_single_store : pd.DataFrame, shape (N_single, len(dataframe_name)*dimension)
        单任务受试者的宽表特征矩阵，每行对应一个受试者，列为各关节各主成分得分。
    vector_double_store : pd.DataFrame, shape (N_double, len(dataframe_name)*dimension)
        双任务受试者的宽表特征矩阵，格式同上。
    index_sinle_store : list of str
        单任务受试者的目录名列表（后缀为 '_1'），用于后续匹配结局变量。
    index_double_store : list of str
        双任务受试者的目录名列表（后缀为 '_2'），用于后续匹配结局变量。
    """
    # output_dir = 'compressed_vectors'
    # 步骤 1：生成列名列表 dataframe_name
    # 从文件名前缀（矢状面5字符或冠状面4字符）提取关节点类别标识，去重保序
    dataframe_name = []
    for i in range(len(file_info)):
        if os.path.basename(file_info[i]['file'])[0] == '矢':
            # 矢状面文件名以"矢"开头，取前5个字符作为关节类别标识
            if os.path.basename(file_info[i]['file']).replace(os.path.basename(file_info[i]['file'])[5:],
                                                              '') not in dataframe_name:
                dataframe_name.append(
                    os.path.basename(file_info[i]['file']).replace(os.path.basename(file_info[i]['file'])[5:], '')
                )
        else:
            # 冠状面文件名，取前4个字符作为关节类别标识
            if os.path.basename(file_info[i]['file']).replace(os.path.basename(file_info[i]['file'])[4:],
                                                              '') not in dataframe_name:
                dataframe_name.append(
                    os.path.basename(file_info[i]['file']).replace(os.path.basename(file_info[i]['file'])[4:], '')
                )

    # 用于分别存储单任务（_1）和双任务（_2）的特征行向量及受试者 ID
    vector_single_store = []
    vector_double_store = []
    index_sinle_store = []
    index_double_store = []

    # 步骤 2-4：按受试者目录分组处理
    while file_info:
        if not file_info:
            break
        file_info_single = []
        # 取当前第一条记录的受试者目录名作为分组键
        index_sigle = os.path.basename(os.path.dirname(file_info[0]['file']))
        # 根据目录后缀区分单任务（_1）和双任务（_2）
        if index_sigle[-2:] == '_1':
            index_sinle_store.append(index_sigle)
        elif index_sigle[-2:] == '_2':
            index_double_store.append(index_sigle)
        # 收集同一受试者目录下的所有文件记录
        for i in range(len(file_info)):
            if os.path.basename(os.path.dirname(file_info[i]['file'])) == index_sigle:
                file_info_single.append({'file': file_info[i]['file']})

        # 创建该受试者的特征行向量（所有关节拼接），初始化为空数组
        # 总列数 = 关节数 * 每关节主成分维数
        vector_after = np.empty((1, (int(len(dataframe_name))*dimension)))

        # 步骤 3：读取各关节的压缩 CSV，填入对应列区间
        for i in range(len(file_info_single)):
            # 解析文件名前缀，确定该关节在 dataframe_name 中的列位置
            if os.path.basename(file_info_single[i]['file'])[0] == '矢':
                a = os.path.basename(file_info_single[i]['file']).replace(
                    os.path.basename(file_info_single[i]['file'])[5:],
                    '')
            else:
                a = os.path.basename(file_info_single[i]['file']).replace(
                    os.path.basename(file_info_single[i]['file'])[4:],
                    '')
            # 找到该关节在列名列表中的位置，计算起止列索引
            col = dataframe_name.index(a)
            # 读取该关节对应的压缩 CSV 文件（跳过标题行，取数值部分）
            with open(os.path.join(
                    os.path.join(output_dir, os.path.basename(os.path.dirname(file_info_single[i]['file']))),
                    os.path.basename(file_info_single[i]['file'])), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            # 将该关节的 dimension 维主成分得分写入特征行向量的对应列区间
            vector_after[0, col*dimension:(col*dimension+dimension)] = np.float32(lines[1:])

        # 从 file_info 中移除已处理的受试者记录，继续处理下一个受试者
        file_info_single_copy = [item['file'] for item in file_info_single]
        file_info = [item for _, item in enumerate(file_info) if item['file'] not in file_info_single_copy]
        print(f"压缩向量{index_sigle[2:5]}抽取中")
        # 按任务类型分别存储
        if index_sigle[-2:] == '_1':
            vector_single_store.append(vector_after)
        elif index_sigle[-2:] == '_2':
            vector_double_store.append(vector_after)

    # 将列表转为 DataFrame，并设置多级列名（每个关节重复 dimension 次）
    vector_single_store = pd.DataFrame(np.array(vector_single_store).squeeze())
    vector_double_store = pd.DataFrame(np.array(vector_double_store).squeeze())
    vector_single_store.columns = np.repeat(dataframe_name, dimension).tolist()
    vector_double_store.columns = np.repeat(dataframe_name, dimension).tolist()
    return dataframe_name, vector_single_store, vector_double_store, index_sinle_store, index_double_store


def read_outcome(outcome_dir, outcome_name, index_single_store, index_double_store):
    """
    读取结局变量 CSV 文件，按受试者 ID 匹配并提取多个协变量。

    提取的结局/协变量包括：衰弱标签（Y）、年龄、性别、身高、体重、
    六米步速、最大握力、起坐时间，分别按单任务和双任务受试者分组返回。
    对标签 Y 进行二值重编码（1->0，其他->1），并对缺失值用列均值填补。

    参数
    ----
    outcome_dir : str
        结局变量 CSV 文件所在目录。
    outcome_name : str
        结局变量 CSV 文件名（含扩展名）。
    index_single_store : list of str
        单任务受试者的目录名列表（后缀为 '_1'），前几位为受试者 ID。
    index_double_store : list of str
        双任务受试者的目录名列表（后缀为 '_2'），前几位为受试者 ID。

    返回
    ----
    tuple，共 16 个元素，按以下顺序返回（每项均为 pd.DataFrame）：
    Y_single, Y_double             : 衰弱标签（0=非衰弱，1=衰弱前期/衰弱）
    Year_single, Year_double       : 年龄（岁）
    Gender_single, Gender_double   : 性别
    Height_single, Height_double   : 身高（cm）
    Weight_single, Weight_double   : 体重（kg）
    Speed_single, Speed_double     : 六米步速（m/s）
    Power_single, Power_double     : 最大握力（kg）
    Seat_single, Seat_double       : 起坐实验时间（s）
    """
    file_path = os.path.join(outcome_dir, outcome_name)
    outcome = pd.read_csv(file_path)
    ID = np.array(outcome['ID'])
    # 读取衰弱分组标签：1=非衰弱，2=衰弱前期+衰弱
    # Y = np.array(outcome['F_认知受损_G2  【0=否，1=是】'])
    Y = np.array(outcome['F_FRAIL_G2【1=非衰弱（0）,2=衰弱前期+衰弱（1-2）】'])  # 当前使用的结局变量
    # Y = np.array(outcome['F_ICOPE_G3【1=受损数0-1,2=受损数2-3，3=受损数4-5】'])
    # 二值重编码：原始值 1 映射为 0（非衰弱），其余映射为 1（衰弱）
    for i in range(len(Y)):
        if Y[i] == 1:
            Y[i] = 0  # 非衰弱 -> 0
        else:
            Y[i] = 1  # 衰弱前期/衰弱 -> 1
    Year = np.array(outcome['年龄（岁）'])
    # 按受试者 ID 匹配，提取单任务和双任务对应的标签
    # idx[:-2] 去掉目录名末尾的 '_1' 或 '_2' 后缀，得到受试者 ID
    Y_single = [Y[ID == idx[:-2]][0] for idx in index_single_store]
    Y_double = [Y[ID == idx[:-2]][0] for idx in index_double_store]
    Y_single = pd.DataFrame({'if': np.array(Y_single)})
    Y_double = pd.DataFrame({'if': np.array(Y_double)})

    Year_single = [Year[ID == idx[:-2]][0] for idx in index_single_store]
    Year_double = [Year[ID == idx[:-2]][0] for idx in index_double_store]
    Year_single = pd.DataFrame({'年龄': np.array(Year_single)})
    Year_double = pd.DataFrame({'年龄': np.array(Year_double)})

    Gender = np.array(outcome['性别'])
    Gender_single = [Gender[ID == idx[:-2]][0] for idx in index_single_store]
    Gender_double = [Gender[ID == idx[:-2]][0] for idx in index_double_store]
    Gender_single = pd.DataFrame({'性别': np.array(Gender_single)})
    Gender_double = pd.DataFrame({'性别': np.array(Gender_double)})

    Height = np.array(outcome['身高（cm）'])
    Height_single = [Height[ID == idx[:-2]][0] for idx in index_single_store]
    Height_double = [Height[ID == idx[:-2]][0] for idx in index_double_store]
    Height_single = pd.DataFrame({'身高': np.array(Height_single)})
    Height_double = pd.DataFrame({'身高': np.array(Height_double)})

    Weight = np.array(outcome['体重（kg）'])
    Weight_single = [Weight[ID == idx[:-2]][0] for idx in index_single_store]
    Weight_double = [Weight[ID == idx[:-2]][0] for idx in index_double_store]
    Weight_single = pd.DataFrame({'体重': np.array(Weight_single)})
    Weight_double = pd.DataFrame({'体重': np.array(Weight_double)})

    Speed = np.array(outcome['6米步速（m/s）'])
    Speed_single = [Speed[ID == idx[:-2]][0] for idx in index_single_store]
    Speed_double = [Speed[ID == idx[:-2]][0] for idx in index_double_store]
    Speed_single = pd.DataFrame({'六米步速': np.array(Speed_single)})
    Speed_double = pd.DataFrame({'六米步速': np.array(Speed_double)})

    Power = np.array(outcome['最大握力（kg）'])
    Power_single = [Power[ID == idx[:-2]][0] for idx in index_single_store]
    Power_double = [Power[ID == idx[:-2]][0] for idx in index_double_store]
    Power_single = pd.DataFrame({'最大握力': np.array(Power_single)})
    Power_double = pd.DataFrame({'最大握力': np.array(Power_double)})

    Seat = np.array(outcome['起坐实验时间（s）'])
    Seat_single = [Seat[ID == idx[:-2]][0] for idx in index_single_store]
    Seat_double = [Seat[ID == idx[:-2]][0] for idx in index_double_store]
    Seat_single = pd.DataFrame({'起坐时间': np.array(Seat_single)})
    Seat_double = pd.DataFrame({'起坐时间': np.array(Seat_double)})

    # 对所有变量进行缺失值填补：用列均值替换 NaN，避免后续计算报错
    Y_single.fillna(Y_single.mean(numeric_only=True), inplace=True)
    Y_double.fillna(Y_double.mean(numeric_only=True), inplace=True)
    Year_single.fillna(Year_single.mean(numeric_only=True), inplace=True)
    Year_double.fillna(Year_double.mean(numeric_only=True), inplace=True)
    Gender_single.fillna(Gender_single.mean(numeric_only=True), inplace=True)
    Gender_double.fillna(Gender_double.mean(numeric_only=True), inplace=True)
    Height_single.fillna(Height_single.mean(numeric_only=True), inplace=True)
    Height_double.fillna(Height_double.mean(numeric_only=True), inplace=True)
    Weight_single.fillna(Weight_single.mean(numeric_only=True), inplace=True)
    Weight_double.fillna(Weight_double.mean(numeric_only=True), inplace=True)
    Speed_single.fillna(Speed_single.mean(numeric_only=True), inplace=True)
    Speed_double.fillna(Speed_double.mean(numeric_only=True), inplace=True)
    Power_single.fillna(Power_single.mean(numeric_only=True), inplace=True)
    Power_double.fillna(Power_double.mean(numeric_only=True), inplace=True)
    Seat_single.fillna(Seat_single.mean(numeric_only=True), inplace=True)
    Seat_double.fillna(Seat_double.mean(numeric_only=True), inplace=True)

    return (Y_single, Y_double, Year_single, Year_double, Gender_single, Gender_double, Height_single, Height_double,
            Weight_single, Weight_double, Speed_single, Speed_double, Power_single, Power_double, Seat_single, Seat_double)


def save_Outcome_result(Outcome_single, Outcome_double, outcome_output_dir):
    """
    将单任务和双任务的整合特征矩阵分别保存为 CSV 文件。

    参数
    ----
    Outcome_single : pd.DataFrame
        单任务整合数据（结局变量 + 协变量 + 关节主成分特征）。
    Outcome_double : pd.DataFrame
        双任务整合数据，格式同上。
    outcome_output_dir : str
        输出目录路径，不存在时自动创建。

    返回
    ----
    None
    """
    # 创建结果文件夹（如已存在则忽略）
    os.makedirs(outcome_output_dir, exist_ok=True)
    # 固定输出文件名，便于后续 split() 函数用 glob 搜索
    file_name_1 = '单任务整合文件.csv'
    file_name_2 = '双任务整合文件.csv'
    result_file_1 = os.path.join(outcome_output_dir, os.path.basename(file_name_1))
    result_file_2 = os.path.join(outcome_output_dir, os.path.basename(file_name_2))
    # 以 GBK 编码保存，兼容中文列名在 Windows Excel 中的显示
    Outcome_single.to_csv(result_file_1, index=False, encoding='gbk')
    Outcome_double.to_csv(result_file_2, index=False, encoding='gbk')
    print(f"已保存合并结果到{result_file_1} 和 {result_file_2}")


def split(outcome_output_dir, test_ratio=0.3, unlabeled_ratio=0.5, random_state_each=42):
    """
    读取整合特征 CSV 文件，划分有标签/无标签数据集及训练/测试集。

    对 outcome_output_dir 中的每个 CSV 文件依次执行：
    1. 按 unlabeled_ratio 将全部样本划分为有标签集和无标签集；
    2. 对有标签集按 test_ratio 进一步划分训练集和测试集；
    3. 对测试集额外截去前2行（去除可能的异常样本）。

    参数
    ----
    outcome_output_dir : str
        整合特征 CSV 文件所在目录（由 save_Outcome_result() 写出）。
    test_ratio : float, 默认 0.3
        有标签数据中测试集的比例。
    unlabeled_ratio : float, 默认 0.5
        全部数据中无标签数据的比例。
    random_state_each : int, 默认 42
        随机划分的随机种子，保证结果可复现。

    返回
    ----
    dict : dict
        以文件路径为键，每个键对应一个子字典，包含以下字段：
        - 'X_labeled'       : pd.DataFrame，有标签特征
        - 'X_unlabeled'     : pd.DataFrame，无标签特征
        - 'Y_labeled'       : pd.Series，有标签标签
        - 'Y_unlabeled'     : pd.Series，无标签标签
        - 'X_labeled_train' : pd.DataFrame，有标签训练集特征
        - 'X_labeled_test'  : pd.DataFrame，有标签测试集特征（已去掉前2行）
        - 'Y_labeled_train' : pd.Series，有标签训练集标签
        - 'Y_labeled_test'  : pd.DataFrame，有标签测试集标签（已去掉前2行）
    """
    # 搜索目录下所有 CSV 文件
    combine_file = glob.glob(os.path.join(outcome_output_dir, r'*.csv'))
    dict = {}
    for i, file in enumerate(combine_file):
        dict[file] = {}
        data = pd.read_csv(file, encoding='gbk')
        X = data.iloc[:, 1:]   # 特征列（去掉第一列结局变量）
        Y = data.iloc[:, 0]    # 结局变量列
        # 第一步：按 unlabeled_ratio 划分有标签集和无标签集
        X_labeled, X_unlabeled, Y_labeled, Y_unlabeled = (
            train_test_split(X, Y, test_size=unlabeled_ratio, random_state=random_state_each))

        dict[file]['X_labeled'], dict[file]['X_unlabeled'], dict[file]['Y_labeled'], dict[file]['Y_unlabeled'] = \
            X_labeled, X_unlabeled, Y_labeled, Y_unlabeled

        # 第二步：对有标签集按 test_ratio 划分训练集和测试集
        X_labeled_train, X_labeled_test, Y_labeled_train, Y_labeled_test = \
            train_test_split(X_labeled, Y_labeled, test_size=test_ratio, random_state=random_state_each)

        # 去掉测试集前2行（经验性做法，可能用于去除索引对齐问题或异常样本）
        X_labeled_test = pd.DataFrame(np.array(X_labeled_test)[2:, :])
        Y_labeled_test = pd.DataFrame(np.array(Y_labeled_test)[2:])

        dict[file]['X_labeled_train'], dict[file]['X_labeled_test'], dict[file]['Y_labeled_train'], dict[file][
            'Y_labeled_test'] = \
            X_labeled_train, X_labeled_test, Y_labeled_train, Y_labeled_test
        # 查看输出情况
        print(f"{os.path.basename(file)[0:3]}有标签域数据集大小:{X_labeled.shape}")
        print(f"{os.path.basename(file)[0:3]}无标签域数据集大小:{X_unlabeled.shape}")
        print(f"{os.path.basename(file)[0:3]}有标签域训练集大小:{X_labeled_train.shape}")
        print(f"{os.path.basename(file)[0:3]}有标签域测试集大小:{X_labeled_test.shape}")
    return dict


def save_combine(dict):
    """
    将一次随机划分的训练集、测试集和无标签集保存为 CSV 文件。

    在 args.outcome_output_dir_date 子目录下，以文件前3字符为前缀
    分别保存 6 个子集文件（X_labeled_train/test、Y_labeled_train/test、
    X_unlabeled、Y_unlabeled）。

    参数
    ----
    dict : dict
        由 split() 函数返回的数据划分字典，键为原始文件路径，
        值为包含各子集 DataFrame 的子字典。

    返回
    ----
    None（通过 print 输出完成提示）
    """
    keys = list(dict.keys())
    # 在与整合 CSV 相同的目录下创建以日期命名的子目录
    os.makedirs(os.path.join(os.path.dirname(keys[0]), args.outcome_output_dir_date), exist_ok=True)
    for key in keys:
        # 依次保存 6 个子集，文件名格式：前缀（3字符） + 子集类型 + '.csv'
        dict[key]['X_labeled_train'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_X_labeled_train.csv')), index=False, encoding='gbk')
        dict[key]['X_labeled_test'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_X_labeled_test.csv')), index=False, encoding='gbk')
        dict[key]['Y_labeled_train'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_Y_labeled_train.csv')), index=False, encoding='gbk')
        dict[key]['Y_labeled_test'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_Y_labeled_test.csv')), index=False, encoding='gbk')
        dict[key]['X_unlabeled'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_X_unlabeled.csv')), index=False, encoding='gbk')
        dict[key]['Y_unlabeled'].to_csv(os.path.join(
            os.path.join(os.path.dirname(key), args.outcome_output_dir_date, os.path.basename(key)[0:3]
                         + '_Y_unlabeled.csv')), index=False, encoding='gbk')
    return print(f"训练测试划分结果保存完成")


def save_multi_datasets(outcome_output_dir):
    """
    对整合特征数据进行多次随机训练/测试集划分并保存全部结果。

    对 outcome_output_dir 中每个 CSV 文件，先按 args.unlabeled_ratio
    一次性划分有标签/无标签集并保存，然后对有标签集重复 args.num_split 次
    随机训练/测试划分（每次使用不同随机种子 i+1），将每次划分结果独立保存
    为带编号的 CSV 文件，便于后续多次重复实验的统计分析。

    参数
    ----
    outcome_output_dir : str
        整合特征 CSV 文件所在目录（由 save_Outcome_result() 写出）。

    返回
    ----
    None
    """
    times = args.num_split  # 重复划分次数
    combine_file = glob.glob(os.path.join(outcome_output_dir, r'*.csv'))
    for file_index, file in enumerate(combine_file):
        data = pd.read_csv(file, encoding='gbk')
        X = data.iloc[:, 1:]   # 特征列
        Y = data.iloc[:, 0]    # 结局变量列
        # 一次性划分有标签/无标签集（所有次划分共享同一无标签集）
        X_labeled, X_unlabeled, Y_labeled, Y_unlabeled = (
            train_test_split(X, Y, test_size=args.unlabeled_ratio, random_state=args.random_state_each))
        os.makedirs(os.path.join(os.path.dirname(file), args.outcome_output_dir_date), exist_ok=True)
        # 保存有标签集和无标签集（各一份，后续每次划分共享）
        X_labeled.to_csv(os.path.join(
            os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                            + f"_X_labeled.csv")), index=False, encoding='gbk')
        Y_labeled.to_csv(os.path.join(
            os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                            + f"_Y_labeled.csv")), index=False, encoding='gbk')
        X_unlabeled.to_csv(os.path.join(
            os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                            + f"_X_unlabeled.csv")), index=False, encoding='gbk')
        Y_unlabeled.to_csv(os.path.join(
            os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                            + f"_Y_unlabeled.csv")), index=False, encoding='gbk')
        dict = {}
        dict[file] = {}
        # 循环进行 times 次随机训练/测试划分，每次使用不同随机种子确保独立性
        for i in range(times):
            X_labeled_train, X_labeled_test, Y_labeled_train, Y_labeled_test = \
                train_test_split(X_labeled, Y_labeled, test_size=args.test_ratio, random_state=i+1)
            dict[file]['X_labeled_train'], dict[file]['X_labeled_test'], dict[file]['Y_labeled_train'], dict[file][
                'Y_labeled_test'] = \
                X_labeled_train, X_labeled_test, Y_labeled_train, Y_labeled_test
            # 保存第 i+1 次划分结果，文件名后缀含编号
            os.makedirs(os.path.join(os.path.dirname(file), args.outcome_output_dir_date), exist_ok=True)
            dict[file]['X_labeled_train'].to_csv(os.path.join(
                os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                             + f"_X_labeled_train_{i + 1}.csv")), index=False, encoding='gbk')
            dict[file]['X_labeled_test'].to_csv(os.path.join(
                os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                             + f"_X_labeled_test_{i + 1}.csv")), index=False, encoding='gbk')
            dict[file]['Y_labeled_train'].to_csv(os.path.join(
                os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                             + f"_Y_labeled_train_{i + 1}.csv")), index=False, encoding='gbk')
            dict[file]['Y_labeled_test'].to_csv(os.path.join(
                os.path.join(os.path.dirname(file), args.outcome_output_dir_date, os.path.basename(file)[0:3]
                             + f"_Y_labeled_test_{i + 1}.csv")), index=False, encoding='gbk')
            print(f"训练测试划分结果{i+1}保存完成")


def main():
    """
    主函数：完整的步态数据 FPCA 压缩与数据集准备流程。

    执行步骤：
    1. 打印命令行参数，设置随机种子；
    2. 检测计算设备（GPU/CPU）；
    3. 批量读取角度序列 CSV 文件；
    4. 按关节进行 FPCA 压缩，提取主成分得分；
    5. 保存各关节的压缩结果 CSV；
    6. 从压缩 CSV 抽取并整合为受试者级宽表特征矩阵；
    7. 读取结局变量和协变量，按受试者 ID 匹配；
    8. 合并特征与结局，按指定关节列筛选，保存整合结果；
    9. 划分有标签/无标签/训练/测试集并保存。

    返回
    ----
    None
    """
    print(args)
    print("忽略的未知参数：", unknown)

    # 设置随机种子以确保可重复性
    torch.manual_seed(42)   # PyTorch 随机种子
    np.random.seed(42)      # NumPy 随机种子

    # 检查是否有可用的GPU，优先使用 GPU 加速
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 获取所有数据集文件夹（支持 glob 通配符路径）
    dataset_dirs = glob.glob(args.dataset_dirs)

    # 步骤 1：收集所有匹配的角度序列数据
    all_vectors, file_info = collect_all_data(dataset_dirs, args.pattern)

    # 步骤 2：创建 FPCA 模型实例（实际拟合在 compress_vectors 内部完成）
    input_dim = all_vectors[0].shape[0]
    model = FPCA1D(args.n_components, args.pve)

    # 步骤 3：对所有关节的序列进行 FPCA 压缩，得到低维主成分得分
    print("开始压缩向量...")
    compressed_vectors, file_info = compress_vectors(model, all_vectors, device, file_info, args.joint)

    # 步骤 4：保存压缩后的主成分得分 CSV
    output_dir = args.output_dir
    print("保存压缩结果...")
    save_compressed_results(file_info, compressed_vectors, output_dir)

    # 步骤 5：从压缩 CSV 中抽取数据，构建按受试者分组的宽表特征矩阵
    (dataframe_name, vector_single_store, vector_double_store,
     index_single_store, index_double_store) = extraction(file_info, output_dir, args.n_components)

    # 步骤 6：读取结局变量 CSV，按受试者 ID 匹配各协变量
    outcome_dir = args.outcome_dir
    # outcome_name = '副本步态视频结局重赋值.csv'
    outcome_name = args.outcome_name
    (Y_single, Y_double, Year_single, Year_double, Gender_single, Gender_double, Height_single, Height_double,
     Weight_single, Weight_double, Speed_single, Speed_double, Power_single, Power_double, Seat_single, Seat_double) \
        = read_outcome(outcome_dir, outcome_name, index_single_store, index_double_store)

    # 步骤 7：将结局变量、协变量和关节主成分特征横向拼接为整合数据框
    Outcome_single = pd.concat([Y_single, Year_single, Gender_single, Height_single,
                                Weight_single, Speed_single, Power_single, Seat_single, vector_single_store], axis=1)
    # Outcome_single.dropna(inplace=True)
    Outcome_double = pd.concat([Y_double, Year_double, Gender_double, Height_double,
                                Weight_double, Speed_double, Power_double, Seat_double, vector_double_store], axis=1)
    # Outcome_double.dropna(inplace=True)

    # 步骤 8：根据 args.consider_joint 筛选需要纳入分析的列
    # # 剔除不相关的关节点变量！！！！！！！！！！！！！！！！
    Outcome_single = Outcome_single.loc[:, args.consider_joint]
    Outcome_double = Outcome_double.loc[:, args.consider_joint]

    # num_cols = Outcome_single.select_dtypes(include='number').columns
    # num_cols = [x for x in num_cols if x not in ['if', '冠-左肩', '冠-右肩', '冠-左肘', '冠-右肘', '冠-左膝', '冠-右膝',
    #                                                   '冠-左踝', '冠-右踝']]
    # ## num_cols = [x for x in num_cols if x not in ['if', '性别']]
    # scaler = StandardScaler()
    # Outcome_single[num_cols] = scaler.fit_transform(Outcome_single[num_cols])

    # df = Outcome_single['六米步速']
    # r = Outcome_single.loc[:,'if'].corr(df, method='pearson')  # 或 'spearman' / 'kendall'
    # print(r)

    # 步骤 9：保存整合后的单/双任务数据文件
    outcome_output_dir = args.outcome_output_dir
    save_Outcome_result(Outcome_single, Outcome_double, outcome_output_dir)

    # 步骤 10：读取整合结果，进行训练集和测试集划分
    dict = split(outcome_output_dir, test_ratio=args.test_ratio, unlabeled_ratio=args.unlabeled_ratio,
                 random_state_each=args.random_state_each)
    # 保存本次划分结果
    save_combine(dict)

    # # 多次随机划分训练集测试集并保存结果（如需多次重复实验可取消注释）
    # save_multi_datasets(outcome_output_dir)

    print("处理完成!")


if __name__ == "__main__":
    # 解析命令行参数，定义数据处理流程的所有可配置项
    parser = argparse.ArgumentParser(description='使用FPCA压缩序列并进行数据预处理')
    parser.add_argument('--pattern', type=str, default='矢*摆动角度.csv', help='文件匹配模式')
    parser.add_argument('--n_components', type=int, default=1, help='序列降维后的维数')
    parser.add_argument('--pve', type=float, default=0.95, help='累计方差贡献标准')
    parser.add_argument('--joint', type=str, default=['鼻子', '左眼', '右眼', '左耳', '右耳',
                                                      '左肩', '右肩', '左肘', '右肘', '左腕',
                                                      '右腕', '左髋', '右髋', '左膝', '右膝',
                                                      '左踝', '右踝'], help='所有关节名')
    # parser.add_argument('--consider_joint', type=str, default=['if', '年龄', '性别', '身高', '体重', '六米步速', '最大握力','起坐时间',
    #                                                   '冠-左肩', '冠-右肩', '冠-左肘', '冠-右肘', '冠-左膝', '冠-右膝', '冠-左踝', '冠-右踝'], help='考虑纳入的关节名')
    # parser.add_argument('--consider_joint', type=str, default=['if',
    #                                                  '冠-左肩', '冠-右肩', '冠-左踝', '冠-右踝'], help='考虑纳入的关节名')
    # parser.add_argument('--consider_joint', type=str, default=['if', '冠-左肩', '冠-右肩', '冠-左踝', '冠-右踝'], help='考虑纳入的关节名')
    # parser.add_argument('--consider_joint', type=str, default=['if', '年龄', '性别', '身高', '体重', '六米步速', '最大握力','起坐时间',
    #                                                   '矢左-鼻子', '矢左-左眼', '矢左-右眼', '矢左-左耳', '矢左-右耳',
    #                                                   '矢左-左肩', '矢左-右肩', '矢左-左肘', '矢左-右肘', '矢左-左腕',
    #                                                   '矢左-右腕', '矢左-左髋', '矢左-右髋', '矢左-左膝', '矢左-右膝',
    #                                                   '矢左-左踝', '矢左-右踝',
    #                                                   '矢右-鼻子', '矢右-左眼', '矢右-右眼', '矢右-左耳',
    #                                                   '矢右-右耳', '矢右-左肩', '矢右-右肩', '矢右-左肘', '矢右-右肘',
    #                                                   '矢右-左腕', '矢右-右腕', '矢右-左髋', '矢右-右髋', '矢右-左膝',
    #                                                   '矢右-右膝', '矢右-左踝', '矢右-右踝'], help='考虑纳入的关节名')
    parser.add_argument('--consider_joint', type=str, default=['if',
                                                      '矢左-左肩', '矢左-右肩', '矢左-左肘', '矢左-右肘', '矢左-左腕',
                                                      '矢左-右腕', '矢左-左髋', '矢左-右髋', '矢左-左膝', '矢左-右膝',
                                                      '矢左-左踝', '矢左-右踝'], help='考虑纳入的关节名')
    # parser.add_argument('--consider_joint', type=str, default=['if',
    #                                                   '矢右-左肩', '矢右-右肩', '矢右-左肘', '矢右-右肘', '矢右-左腕',
    #                                                   '矢右-右腕', '矢右-左髋', '矢右-右髋', '矢右-左膝', '矢右-右膝',
    #                                                   '矢右-左踝', '矢右-右踝'], help='考虑纳入的关节名')
    parser.add_argument('--unlabeled_ratio', type=float, default=0.6, help='无标签数据比例')  # 0.6 0.45
    parser.add_argument('--test_ratio', type=float, default=0.3, help='有标签数据测试集比例')  # 0.3 0.3
    parser.add_argument('--num_split', type=int, default=100, help='随机划分数据集的次数')  # 100
    parser.add_argument('--random_state_each', type=int, default=40, help='默认一次划分种子数')  # 42 40
    parser.add_argument(
        '--dataset_dirs',
        type=str,
        default=os.environ.get('GAIT_FPCA_DATASET_DIRS', ''),
        help='摆动角度 CSV 的目录或 glob 模式。公开代码不包含真实步态数据，请通过参数或环境变量 GAIT_FPCA_DATASET_DIRS 指定。'
    )
    parser.add_argument('--output_dir', type=str, default='compressed_vectors', help='压缩结果csv保存文件夹名')
    parser.add_argument('--outcome_dir', type=str, default='outcome', help='响应文件夹名')
    parser.add_argument('--outcome_name', type=str, default='北京步态视频结局重赋值20250904.csv', help='响应csv文件名')
    parser.add_argument('--outcome_output_dir', type=str, default='outcome_output_dir', help='整合后文件夹名')
    parser.add_argument('--outcome_output_dir_date', type=str, default='2025-09-21', help='训练测试划分文件夹日期名')
    args, unknown = parser.parse_known_args()
    main()
