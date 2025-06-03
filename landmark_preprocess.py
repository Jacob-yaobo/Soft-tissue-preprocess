'''
 # @ Author: Jacob
 # @ Create Time: 2025-05-19 17:29:54
 # @ Modified by: Jacob
 # @ Modified time: 2025-05-19 17:30:19
 # @ Description:
 '''

import os
import pandas as pd
import numpy as np

# 读入新的鼻尖点表格
landmark22_df = pd.read_excel('Nose Tip Slicer Coordinates.xlsx', usecols="A, E:G", skiprows=2,header=0)

raw_landmark_dir = 'data/landmark_raw'
# 读取原始的landmark数据
for excel in os.listdir(raw_landmark_dir):
    if excel.endswith('.xlsx'):
        pid = excel.split('.')[0]
        print(f'Processing {pid}...')
        landmark_file_xlsx = os.path.join(raw_landmark_dir, excel)
        landmark_raw_df= pd.read_excel(landmark_file_xlsx, usecols="A:E,H:J", header=None, skiprows=4, nrows=43)
        # 原始Excel文件中的列名，按照RSA顺序排列
        landmark_raw_df.columns = ['name', 'landmark', 'R_post', 'S_post', 'A_post', 'R_pre', 'S_pre', 'A_pre']
        landmark_df = landmark_raw_df.copy()
        # 计算 dR, dA, dS
        landmark_df['dR'] = (landmark_df['R_post'] - landmark_df['R_pre'])
        landmark_df['dA'] = (landmark_df['A_post'] - landmark_df['A_pre'])
        landmark_df['dS'] = (landmark_df['S_post'] - landmark_df['S_pre'])
        # 调整列顺序
        column_order = ['name', 'landmark', 'R_post', 'A_post', 'S_post', 'R_pre', 'A_pre', 'S_pre', 'dR', 'dA', 'dS']
        pre_cols = ['R_pre', 'A_pre', 'S_pre']
        post_cols = ['R_post', 'A_post', 'S_post']
        disp_cols = ['dR', 'dA', 'dS']
        landmark_df = landmark_df[column_order]

        # 读取新的鼻尖点数据
        row = landmark22_df.loc[landmark22_df['pid'] == pid, ['Right.1', 'Anterior.1', 'Superior.1']]
        if row.empty:
            print(f"{pid} not found in landmark22_df")
            continue
        pre_landmark22 = row.values[0]

        # 计算坐标系变换
        old_pre_landmark22 = landmark_df.loc[21, pre_cols].values
        pre_transform_vector = (pre_landmark22 - old_pre_landmark22) 

        # 对数据进行坐标系变换
        landmark_transform_df = landmark_df.copy()
        landmark_transform_df[pre_cols] = landmark_transform_df[pre_cols] + pre_transform_vector 
        landmark_transform_df[post_cols] = landmark_transform_df[post_cols] + pre_transform_vector # 只用pre的坐标系变换关系
        
        # 保存数据
        out_dir = 'landmark_transform'
        os.makedirs(out_dir, exist_ok=True)
        landmark_transform_file = f'landmark_transform/{pid}.xlsx'
        landmark_transform_df.to_excel(landmark_transform_file, index=False)
        print(f'Saved transformed landmark data to {landmark_transform_file}')
