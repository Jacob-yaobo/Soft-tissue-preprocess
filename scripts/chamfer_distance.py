'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-06-04 12:36:14
 # @ Modified by: Yaobo Jia
 # @ Modified time: 2025-06-04 12:36:16
 # @ Description:
 '''
import numpy as np
import open3d as o3d
from pathlib import Path

def compute_chamfer_distance(source_pcd, target_pcd):
    """
    计算两个点云之间的 Chamfer Distance
    
    Args:
        source_pcd: Open3D PointCloud 对象
        target_pcd: Open3D PointCloud 对象
        
    Returns:
        float: Chamfer Distance
    """
    source_points = np.asarray(source_pcd.points)
    target_points = np.asarray(target_pcd.points)
    
    # 构建 KDTree
    target_tree = o3d.geometry.KDTreeFlann(target_pcd)
    source_tree = o3d.geometry.KDTreeFlann(source_pcd)
    
    # source -> target
    source_to_target = 0
    for point in source_points:
        [_, idx, dist2] = target_tree.search_knn_vector_3d(point, 1)
        source_to_target += np.sqrt(dist2[0])
    source_to_target /= len(source_points)
    
    # target -> source
    target_to_source = 0
    for point in target_points:
        [_, idx, dist2] = source_tree.search_knn_vector_3d(point, 1)
        target_to_source += np.sqrt(dist2[0])
    target_to_source /= len(target_points)
    
    # 计算双向距离的平均值
    chamfer_dist = (source_to_target + target_to_source) / 2
    
    return chamfer_dist


if __name__ == '__main__':
    # 设置点云文件路径
    base_dir = Path('img/pointcloud')
    pre_dir = base_dir / 'Pre'
    post_dir = base_dir / 'Post'
    
    # 读取点云文件
    pre_pcds = []
    post_pcds = []
    
    # 读取Pre点云
    for i in range(10):
        pre_path = pre_dir / f'Pre_{i}.ply'
        if pre_path.exists():
            pcd = o3d.io.read_point_cloud(str(pre_path))
            pre_pcds.append(pcd)
    
    # 读取Post点云
    for i in range(10):
        post_path = post_dir / f'Post_{i}.ply'
        if post_path.exists():
            pcd = o3d.io.read_point_cloud(str(post_path))
            post_pcds.append(pcd)
    
    print(f"Loaded {len(pre_pcds)} pre-op point clouds")
    print(f"Loaded {len(post_pcds)} post-op point clouds")
    
    # 计算并打印不同采样之间的 Chamfer Distance
    print("\nChecking sampling consistency:")
    for i in range(len(pre_pcds)):
        for j in range(i+1, len(pre_pcds)):
            # 比较 pre 的不同采样
            cd_pre = compute_chamfer_distance(pre_pcds[i], pre_pcds[j])
            print(f"Pre-op samples {i} vs {j}: CD = {cd_pre:.6f}")
            
            # 比较 post 的不同采样
            cd_post = compute_chamfer_distance(post_pcds[i], post_pcds[j])
            print(f"Post-op samples {i} vs {j}: CD = {cd_post:.6f}")