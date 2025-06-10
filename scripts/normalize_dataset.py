'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-06-10 12:00:00
 # @ Modified by: Yaobo Jia & Claude Sonnet 3.7 Assistant
 # @ Modified time: 2025-06-10 12:00:00
 # @ Description: 对点云和landmark数据同步归一化处理
 # 处理步骤：
 # 1. 计算所有点云的全局中心点和缩放因子
 # 2. 对点云数据进行归一化处理
 # 3. 对landmark数据使用相同的归一化参数
 # 4. 将数据保存为便于构建Dataset的格式
 '''

import os
import json
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
from tqdm import tqdm

def normalize_point_cloud(points, center, scale):
    """
    对点云进行归一化处理
    
    Args:
        points: 点云坐标数组 (N, 3)
        center: 中心点坐标 (3,)
        scale: 缩放因子
        
    Returns:
        normalized_points: 归一化后的点云坐标
    """
    # 平移和缩放
    normalized_points = (points - center) * scale
    return normalized_points

def calculate_global_params(point_dir):
    """
    计算所有点云的全局归一化参数
    
    Args:
        point_dir: 点云数据目录
        
    Returns:
        global_center: 全局中心点
        global_scale: 全局缩放因子
    """
    print("计算全局归一化参数...")
    all_points = []
    
    # 收集所有点云数据
    for pid_dir in tqdm(list(Path(point_dir).iterdir())):
        if not pid_dir.is_dir():
            continue
            
        # 处理Pre和Post点云
        for phase in ['Pre', 'Post']:
            phase_dir = pid_dir / phase
            if phase_dir.exists():
                for pcd_file in phase_dir.glob('*.ply'):
                    pcd = o3d.io.read_point_cloud(str(pcd_file))
                    points = np.asarray(pcd.points)
                    all_points.append(points)
    
    # 合并所有点云
    if all_points:
        all_points_array = np.vstack(all_points)
        
        # 计算全局中心
        global_center = np.mean(all_points_array, axis=0)
        
        # 计算到全局中心的最大距离
        centered_points = all_points_array - global_center
        max_distance = np.max(np.linalg.norm(centered_points, axis=1))
        
        # 计算全局缩放因子，确保所有点都在单位球内
        global_scale = 0.9 / max_distance  # 使用0.9来留一些余量
        
        print(f"全局中心点: {global_center}")
        print(f"全局最大距离: {max_distance}")
        print(f"全局缩放因子: {global_scale}")
        
        return global_center, global_scale
    
    return np.zeros(3), 1.0

def normalize_landmark_data(landmark_df, center, scale):
    """
    对landmark数据进行归一化处理
    
    Args:
        landmark_df: landmark数据DataFrame
        center: 中心点坐标
        scale: 缩放因子
        
    Returns:
        normalized_df: 归一化后的landmark数据
    """
    # 复制数据框
    normalized_df = landmark_df.copy()
    
    # 归一化pre坐标
    pre_cols = ['L_pre', 'P_pre', 'S_pre']
    for i, row in normalized_df.iterrows():
        pre_coords = np.array([row['L_pre'], row['P_pre'], row['S_pre']])
        normalized_pre = normalize_point_cloud(pre_coords, center, scale)
        normalized_df.loc[i, pre_cols] = normalized_pre
    
    # 归一化post坐标
    post_cols = ['L_post', 'P_post', 'S_post']
    for i, row in normalized_df.iterrows():
        post_coords = np.array([row['L_post'], row['P_post'], row['S_post']])
        normalized_post = normalize_point_cloud(post_coords, center, scale)
        normalized_df.loc[i, post_cols] = normalized_post
    
    # 重新计算位移
    normalized_df['dL'] = normalized_df['L_post'] - normalized_df['L_pre']
    normalized_df['dP'] = normalized_df['P_post'] - normalized_df['P_pre']
    normalized_df['dS'] = normalized_df['S_post'] - normalized_df['S_pre']
    
    return normalized_df

def create_dataset_format(normalized_point_dir, normalized_landmark_dir, output_dir):
    """
    将归一化后的数据组织成便于训练的格式
    
    Args:
        normalized_point_dir: 归一化后的点云目录
        normalized_landmark_dir: 归一化后的landmark目录
        output_dir: 输出目录
    """
    print("创建训练数据集格式...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建训练集、验证集和测试集目录
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    test_dir = output_dir / 'test'
    
    for dir_path in [train_dir, val_dir, test_dir]:
        dir_path.mkdir(exist_ok=True)
    
    # 获取所有患者ID
    patient_ids = [d.name for d in Path(normalized_point_dir).iterdir() if d.is_dir()]
    
    # 随机划分数据集（70% 训练，15% 验证，15% 测试）
    np.random.seed(42)  # 固定随机种子以确保可重复性
    np.random.shuffle(patient_ids)
    
    n_patients = len(patient_ids)
    n_train = int(n_patients * 0.7)
    n_val = int(n_patients * 0.15)
    
    train_ids = patient_ids[:n_train]
    val_ids = patient_ids[n_train:n_train+n_val]
    test_ids = patient_ids[n_train+n_val:]
    
    print(f"训练集: {len(train_ids)} 患者")
    print(f"验证集: {len(val_ids)} 患者")
    print(f"测试集: {len(test_ids)} 患者")
    
    # 保存数据集划分
    split_info = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    with open(output_dir / 'dataset_split.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    # 处理训练集
    process_dataset_split(train_ids, normalized_point_dir, normalized_landmark_dir, train_dir)
    
    # 处理验证集
    process_dataset_split(val_ids, normalized_point_dir, normalized_landmark_dir, val_dir)
    
    # 处理测试集
    process_dataset_split(test_ids, normalized_point_dir, normalized_landmark_dir, test_dir)
    
    print("数据集创建完成!")

def process_dataset_split(patient_ids, point_dir, landmark_dir, output_dir):
    """
    处理数据集的一个子集（训练/验证/测试）
    
    Args:
        patient_ids: 患者ID列表
        point_dir: 点云数据目录
        landmark_dir: landmark数据目录
        output_dir: 输出目录
    """
    for pid in tqdm(patient_ids, desc=f"处理 {output_dir.name} 集"):
        # 创建患者目录
        pid_dir = output_dir / pid
        pid_dir.mkdir(exist_ok=True)
        
        # 复制点云数据
        src_point_dir = Path(point_dir) / pid
        if src_point_dir.exists():
            # 处理Pre点云
            pre_dir = src_point_dir / 'Pre'
            if pre_dir.exists():
                dest_pre_dir = pid_dir / 'Pre'
                dest_pre_dir.mkdir(exist_ok=True)
                
                for pcd_file in pre_dir.glob('*.ply'):
                    os.system(f"cp {pcd_file} {dest_pre_dir}")
            
            # 处理Post点云
            post_dir = src_point_dir / 'Post'
            if post_dir.exists():
                dest_post_dir = pid_dir / 'Post'
                dest_post_dir.mkdir(exist_ok=True)
                
                for pcd_file in post_dir.glob('*.ply'):
                    os.system(f"cp {pcd_file} {dest_post_dir}")
        
        # 复制landmark数据
        landmark_file = Path(landmark_dir) / f"{pid}_LPS_normalized.xlsx"
        if landmark_file.exists():
            os.system(f"cp {landmark_file} {pid_dir}")

def main():
    """
    主函数
    """
    # 设置路径
    point_dir = 'data/point'
    landmark_dir = 'data/landmark'
    
    normalized_point_dir = 'data/normalized_point'
    normalized_landmark_dir = 'data/normalized_landmark'
    dataset_dir = 'data/dataset'
    
    # 创建输出目录
    Path(normalized_point_dir).mkdir(parents=True, exist_ok=True)
    Path(normalized_landmark_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 计算全局归一化参数
    global_center, global_scale = calculate_global_params(point_dir)
    
    # 保存归一化参数
    normalization_params = {
        'global_center': global_center.tolist(),
        'global_scale': float(global_scale)
    }
    
    with open('data/normalization_params.json', 'w') as f:
        json.dump(normalization_params, f, indent=2)
    
    # 2. 对点云数据进行归一化处理
    print("对点云数据进行归一化处理...")
    for pid_dir in tqdm(list(Path(point_dir).iterdir()), desc="处理点云"):
        if not pid_dir.is_dir():
            continue
            
        pid = pid_dir.name
        
        # 创建输出目录
        pid_output_dir = Path(normalized_point_dir) / pid
        pid_output_dir.mkdir(exist_ok=True)
        
        # 处理Pre点云
        pre_dir = pid_dir / 'Pre'
        pre_output_dir = pid_output_dir / 'Pre'
        pre_output_dir.mkdir(exist_ok=True)
        
        if pre_dir.exists():
            for pcd_file in pre_dir.glob('*.ply'):
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                points = np.asarray(pcd.points)
                
                # 归一化点云
                normalized_points = normalize_point_cloud(points, global_center, global_scale)
                
                # 创建新的点云对象
                normalized_pcd = o3d.geometry.PointCloud()
                normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
                
                # 如果原点云有法向量，复制过来
                if pcd.has_normals():
                    normalized_pcd.normals = pcd.normals
                
                # 保存归一化后的点云
                output_file = pre_output_dir / pcd_file.name
                o3d.io.write_point_cloud(str(output_file), normalized_pcd)
        
        # 处理Post点云
        post_dir = pid_dir / 'Post'
        post_output_dir = pid_output_dir / 'Post'
        post_output_dir.mkdir(exist_ok=True)
        
        if post_dir.exists():
            for pcd_file in post_dir.glob('*.ply'):
                pcd = o3d.io.read_point_cloud(str(pcd_file))
                points = np.asarray(pcd.points)
                
                # 归一化点云
                normalized_points = normalize_point_cloud(points, global_center, global_scale)
                
                # 创建新的点云对象
                normalized_pcd = o3d.geometry.PointCloud()
                normalized_pcd.points = o3d.utility.Vector3dVector(normalized_points)
                
                # 如果原点云有法向量，复制过来
                if pcd.has_normals():
                    normalized_pcd.normals = pcd.normals
                
                # 保存归一化后的点云
                output_file = post_output_dir / pcd_file.name
                o3d.io.write_point_cloud(str(output_file), normalized_pcd)
    
    # 3. 对landmark数据进行归一化处理
    print("对landmark数据进行归一化处理...")
    for landmark_file in tqdm(list(Path(landmark_dir).glob('*_LPS_normalized.xlsx')), desc="处理landmark"):
        pid = landmark_file.stem.split('_LPS_normalized')[0]
        
        # 读取landmark数据
        landmark_df = pd.read_excel(landmark_file)
        
        # 归一化landmark数据
        normalized_df = normalize_landmark_data(landmark_df, global_center, global_scale)
        
        # 保存归一化后的landmark数据
        output_file = Path(normalized_landmark_dir) / f"{pid}_normalized.xlsx"
        normalized_df.to_excel(output_file, index=False)
    
    # 4. 创建数据集格式
    create_dataset_format(normalized_point_dir, normalized_landmark_dir, dataset_dir)

if __name__ == "__main__":
    main()
