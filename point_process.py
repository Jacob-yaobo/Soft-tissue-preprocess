'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-06-04 12:09:49
 # @ Modified by: Yaobo Jia
 # @ Modified time: 2025-06-04 12:09:50
 # @ Description:
 '''
import os
import open3d as o3d
import numpy as np
from pathlib import Path

from tqdm import tqdm

def mesh_to_pointcloud(mesh, n_points=4096, method='uniform'):
    """
    将mesh转换为点云，并进行降采样
    
    Args:
        mesh: Open3D TriangleMesh对象
        n_points: 目标点数
        method: 采样方法 ('uniform' 或 'poisson')
        
    Returns:
        pcd: Open3D PointCloud对象
    """
    # 计算mesh的法向量
    mesh.compute_vertex_normals()
    
    if method == 'uniform':
        # 均匀采样
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
    else:
        # 泊松盘采样 - 根据面积和曲率分布采样
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)
    
    return pcd


def load_mesh(mesh_path):
    """
    读取mesh文件并进行基本处理
    
    Args:
        mesh_path: mesh文件路径(.ply格式)
        
    Returns:
        mesh: Open3D TriangleMesh对象
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found: {mesh_path}")
        
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    # 基本检查
    if not mesh.has_vertices():
        raise ValueError(f"Loaded mesh has no vertices: {mesh_path}")
    if not mesh.has_triangles():
        raise ValueError(f"Loaded mesh has no triangles: {mesh_path}")
        
    # 计算法向量
    mesh.compute_vertex_normals()
    
    return mesh

def generate_multiple_samples(mesh, n_samples=10, n_points=4096):
    """
    从同一个mesh生成多组点云样本
    
    Args:
        mesh: Open3D TriangleMesh对象
        n_samples: 要生成的点云样本数量
        n_points: 每个点云的点数
    
    Returns:
        pcds: 点云列表
    """
    pcds = []
    for i in range(n_samples):
        pcd = mesh_to_pointcloud(mesh, n_points, method='poisson')
        pcd.estimate_normals()
        pcds.append(pcd)
    return pcds

# 在主程序中添加点云生成和保存的代码
if __name__ == '__main__':
    root_dir = Path('data/mesh')
    post_mesh_files = list(root_dir.rglob("Post.ply"))

    with tqdm(total=len(post_mesh_files), desc="Initialising...") as pbar:
        for post_mesh_file in post_mesh_files:
            pid = post_mesh_file.parts[-2]
            pre_mesh_file = post_mesh_file.with_name('Pre.ply')

            try:
                pre_mesh = load_mesh(pre_mesh_file)
                post_mesh = load_mesh(post_mesh_file)
                
                # 生成多组样本
                n_samples = 10
                pre_pcds = generate_multiple_samples(pre_mesh, n_samples)
                post_pcds = generate_multiple_samples(post_mesh, n_samples)
                
                # 保存点云
                out_dir = Path('data/point') / pid
                out_dir_pcd_post = out_dir / 'Post'
                out_dir_pcd_post.mkdir(parents=True, exist_ok=True)
                out_dir_pcd_pre = out_dir / 'Pre'
                out_dir_pcd_pre.mkdir(parents=True, exist_ok=True)
                
                # 保存每组采样结果
                for i, (pre_pcd, post_pcd) in enumerate(zip(pre_pcds, post_pcds)):
                    o3d.io.write_point_cloud(str(out_dir_pcd_pre / f'Pre_{i}.ply'), pre_pcd)
                    o3d.io.write_point_cloud(str(out_dir_pcd_post / f'Post_{i}.ply'), post_pcd)

                pbar.set_postfix({'pid': f'Processing {pid}', 'Vertices':f'post {len(post_mesh.vertices)} and pre {len(pre_mesh.vertices)}'})
                
            except Exception as e:
                print(f"Error occurred: {str(e)}")

            pbar.update(1)