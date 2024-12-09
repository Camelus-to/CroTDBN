import os
import numpy as np
from scipy.io import loadmat
from nilearn.connectome import ConnectivityMeasure
import pandas as pd



def calculate_connectivity_matrix(time_segment, kind='correlation'):
    conn_measure = ConnectivityMeasure(kind=kind)
    return conn_measure.fit_transform([time_segment])[0]



def process_time_series_files(input_dir, fc_static_dir, output_dir, df, num_dynamic_matrices=6):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 初始化列表存储最终输出的内容
    FC_dynamic_list = []
    FC_list = []
    ROI_list = []
    Label_list = []
    Site_list = []


    for filename in os.listdir(input_dir):
        if filename.endswith('.mat'):
            file_path = os.path.join(input_dir, filename)
            fc_static = loadmat(os.path.join(fc_static_dir, filename))['fc_matrix']
            fc_static = np.nan_to_num(fc_static, nan=0.0)
            fc_static[np.diag_indices_from(fc_static)] = 0

            yuan = loadmat(file_path)
            time_series = yuan['ROISignals']

            match = df[df['subject'] == str(filename)[:5]]
            if len(match) == 0:
                print(f"No match found for {filename}")
                continue

            y = match.iloc[0]['DX_GROUP']
            site = match.iloc[0]['SITE_ID']


            T, num_rois = time_series.shape


            if T < num_dynamic_matrices:
                print(f"Time series too short for {filename}, skipping...")
                continue
            elif T < num_dynamic_matrices * 2:

                window_size = T // num_dynamic_matrices
                step_size = window_size  # 每段大小等于窗口大小
            else:

                window_size = T // (num_dynamic_matrices + 1)
                step_size = window_size // 2

            correlation_matrices = []

            for i in range(num_dynamic_matrices):
                start_idx = i * step_size
                end_idx = start_idx + window_size
                if end_idx > T:  # 如果索引超过时间序列长度，直接截断
                    end_idx = T
                    start_idx = end_idx - window_size
                time_segment = time_series[start_idx:end_idx, :]

                corr_matrix = calculate_connectivity_matrix(time_segment, kind='correlation')
                correlation_matrices.append(corr_matrix)

            # 保存每个样本的动态功能连接矩阵到列表中
            FC_dynamic_list.append(correlation_matrices)
            FC_list.append(np.array(fc_static))
            ROI_list.append(time_series)
            Label_list.append(y)
            Site_list.append(site)

            print(f"Processed {filename}, generated 6 dynamic matrices.")

    # 转换列表为数组
    FC_dynamic = np.array(FC_dynamic_list, dtype=object)
    FC_array = np.array(FC_list, dtype=object)
    processed_ROI_array = np.array(ROI_list, dtype=object)
    Label_array = np.array(Label_list, dtype=np.int64)
    Site_array = np.array(Site_list, dtype=object)

    # 将功能连接矩阵、时间序列、标签和站点信息保存到字典中
    data_dict = {'corr': FC_array, 'dcorr': FC_dynamic, 'timeseires': processed_ROI_array, 'label': Label_array,
                 'site': Site_array}

    # 保存字典为 .npy 文件
    output_file = os.path.join(output_dir, 'dynamic_functional_connectivity_data.npy')
    np.save(output_file, data_dict)

    print(f"All data saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 输入和输出目录
    input_dir = 'E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\ROISignals'  # 输入样本文件夹
    fc_static_dir = 'E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\FC'  # 样本的静态功能连接矩阵
    output_dir = 'E:\Technolgy_learning\Learning_code\AD\BrainNetworkTransformer-View-Bert\source\data_processed'  # 最终数据输出文件夹

    # 读取 CSV 文件
    df = pd.read_csv(r'E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\Phenotypic_V1_0b_preprocessed1.csv')

    # 调用函数处理样本文件并保存 我们要生成的是6个动态连接矩阵
    process_time_series_files(input_dir, fc_static_dir, output_dir, df, num_dynamic_matrices=6)
