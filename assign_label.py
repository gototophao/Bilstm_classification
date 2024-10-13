# -*- coding: utf-8 -*-
"""AE_automated

This file is designed to derive the optimum configuration of a neural network
using Keras hypertune and Keras sequence generators.

"""
import pdb

import pandas as pd
import numpy as np
import glob
import os


import random as python_random

# The below is necessary for starting Numpy generated random numbers
# in a well-defined initial state.
np.random.seed(123)

# The below is necessary for starting core Python generated random numbers
# in a well-defined state.
python_random.seed(123)




# 指定数据路径
data_path='/nobackup/projects/bdman08/PLAIR_HK/Processed/'
output_folder = '/nobackup/projects/bdman08/PLAIR_HK/labelled_data/'
# 读取聚类标签的HDF文件
labels_path = '/nobackup/projects/bdman08/PLAIR_HK/combiend_genie_bilstm_ae.hdf'
labels_df = pd.read_hdf(labels_path,key='label_value')
pdb.set_trace()

# 获取所有原始数据的HDF文件路径并排序
list_files = sorted(glob.glob(data_path + '*.hdf'))

# 初始化一个计数器，用于追踪在聚类结果中的当前位置
start_index = 0
total_matched_rows = 0
# 遍历所有文件，清除数据，并将聚类标签添加到数据中
for file in list_files:
    # 读取原始数据文件
    original_data = pd.read_hdf(file)

    # 清除前32列全为0的行
    data = original_data.iloc[:, 0:32].to_numpy()
    valid_rows = ~np.all(data == 0, axis=1)
    cleaned_data = original_data[valid_rows].copy()

    # 计算当前文件中有效行的数量
    num_valid_rows = np.sum(valid_rows)
    total_matched_rows += num_valid_rows
    # 从聚类结果中获取对应的行
    current_labels = labels_df.iloc[start_index:start_index + num_valid_rows]

    # 更新下一个起始索引
    start_index += num_valid_rows

    # 为聚类结果设置列名
    for i in range(2, 11):  # 聚类数目从2到14
        cleaned_data[f'AE_Cluster_{i}'] = current_labels.iloc[:, i - 2].values

    # 构造输出文件的完整路径
    output_path = os.path.join(output_folder, os.path.basename(os.path.splitext(file)[0] + '_with_labels.hdf'))
    print(os.path.splitext(file)[0])
    # 保存结果到新的HDF文件中
    cleaned_data.to_hdf(output_path, key='df', mode='w')
    print(start_index)

    print(f"文件 {os.path.basename(file)}: 处理了 {num_valid_rows} 行有效数据.")
    pdb.set_trace()

# 输出总的匹配行数
print(f"处理完成，所有文件总共匹配了 {total_matched_rows} 行有效数据，并保存到指定文件夹.")