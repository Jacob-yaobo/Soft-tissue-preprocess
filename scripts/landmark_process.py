'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-05-19 17:29:54
 # @ Modified by: Yaobo Jia
 # @ Modified time: 2025-06-04 18:30:55
 # @ Description: 对landmark完成归一化；
    # 选择pre nose tip在这个未知RAS坐标系下的 coordinate；
    # 计算出将pre nose tip移到（0，0，0）的transform vector；
    # 对post pre同时应用，使得变换之后的post pre处于pre nose tip在原点的RAS系下；
    # 转换为LPS坐标系
 '''


import os 
import pandas as pd

INPUT_DIR = 'raw_data/48_excel'
OUTPUT_DIR = 'data/landmark'
os.makedirs(OUTPUT_DIR, exist_ok=True)

for excel_file in os.listdir(INPUT_DIR):
    if excel_file.endswith('.xlsx'):
        pid = excel_file.split('.')[0]
        raw_file = os.path.join(INPUT_DIR, excel_file)
        # 1.读取原数据，完成列名的整理
        # 读取原始的landmark excel表格
        raw_landmark_df= pd.read_excel(raw_file, usecols="A:E,H:J", header=None, skiprows=4, nrows=43)
        # 原始Excel文件中的列名
        raw_landmark_df.columns = ['landmark_name', 'landmark_num', 'R_post', 'S_post', 'A_post', 'R_pre', 'S_pre', 'A_pre']
        RAS_landmark_df = raw_landmark_df.copy()
        # 计算 dR, dA, dS
        RAS_landmark_df['dR'] = (RAS_landmark_df['R_post'] - RAS_landmark_df['R_pre'])
        RAS_landmark_df['dA'] = (RAS_landmark_df['A_post'] - RAS_landmark_df['A_pre'])
        RAS_landmark_df['dS'] = (RAS_landmark_df['S_post'] - RAS_landmark_df['S_pre'])
        # 调整列顺序为RAS
        column_order = ['landmark_name', 'landmark_num', 'R_post', 'A_post', 'S_post', 'R_pre', 'A_pre', 'S_pre', 'dR', 'dA', 'dS']
        pre_cols = ['R_pre', 'A_pre', 'S_pre']
        post_cols = ['R_post', 'A_post', 'S_post']
        disp_cols = ['dR', 'dA', 'dS']
        RAS_landmark_df = RAS_landmark_df[column_order]
        
        # 2.根据pre的鼻尖点完成坐标系的归一化（统一到pre的鼻尖点上）
        # 获取raw表格中的pre鼻尖点位置
        nosetip_pre_ras = RAS_landmark_df.loc[RAS_landmark_df['landmark_num']=='Landmark 22', 'R_pre': 'S_pre'].values

        landmark_transform_df = RAS_landmark_df.copy()
        landmark_transform_df[pre_cols] = RAS_landmark_df[pre_cols] - nosetip_pre_ras 
        landmark_transform_df[post_cols] = RAS_landmark_df[post_cols] - nosetip_pre_ras
        
        # 3.转换坐标系到LPS，以配合Point cloud/Mesh的坐标系
        # 转换为LPS坐标系
        lps_df = landmark_transform_df.copy()
        # R -> L : R取负
        lps_df['R_pre'] = -lps_df['R_pre']
        lps_df['R_post'] = -lps_df['R_post']
        lps_df['dR'] = -lps_df['dR'] # dL = -dR
        # A -> P : A取负
        lps_df['A_pre'] = -lps_df['A_pre']
        lps_df['A_post'] = -lps_df['A_post']
        lps_df['dA'] = -lps_df['dA'] # dP = -dA
        # 重命名列以反映 LPS 坐标系
        lps_column_map = {
            'R_post': 'L_post', 
            'A_post': 'P_post', 
            'S_post': 'S_post',
            'R_pre':  'L_pre',  
            'A_pre':  'P_pre',  
            'S_pre':  'S_pre',
            'dR':     'dL',     
            'dA':     'dP',     
            'dS':     'dS'
        }
        lps_df.rename(columns=lps_column_map, inplace=True)
        print(f"  PID {pid}: 坐标已转换为 LPS。")

        # 调整最终的列顺序为 LPS
        lps_column_order = ['landmark_name', 'landmark_num', 
                            'L_post', 'P_post', 'S_post', 
                            'L_pre', 'P_pre', 'S_pre', 
                            'dL', 'dP', 'dS']
        lps_df = lps_df[lps_column_order]

        # 3: 保存 LPS 结果
        output_file_path = os.path.join(OUTPUT_DIR, f'{pid}_LPS_normalized.xlsx')
        lps_df.to_excel(output_file_path, index=False)
        print(f"  成功保存 LPS 数据到: {output_file_path}")
