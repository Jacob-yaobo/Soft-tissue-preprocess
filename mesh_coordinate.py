'''
 # @ Author: Jacob
 # @ Create Time: 2025-05-27 16:34:54
 # @ Modified by: Jacob
 # @ Modified time: 2025-05-27 16:34:55
 # @ Description:完成基础的seg到mesh的生成过程；同时完成坐标轴的统一转换
    # 1. 利用matching cube算法，实现从seg到mesh的生成；
    # 2. 完成坐标轴的统一转换：SimpleITK基于LPS坐标系，将mesh生成的ijk坐标系转换为LPS；同时将landmark基于的RAS转换为LPS坐标系；
    # 3. 根据鼻尖点坐标完成所有LPS坐标系的变换，将原点放置于鼻尖点处。同步将landmark的坐标系也改变。
 '''
import os
import numpy as np
import SimpleITK as sitk
import open3d as o3d
import pandas as pd

from skimage import measure

def generate_mesh(seg_file_path, nose_tip_coord):
    # ------1. 完成mesh的生成，此时是体素索引：ijk坐标------
    # 读取分割图像
    seg_image = sitk.ReadImage(seg_file_path)
    seg_array = sitk.GetArrayFromImage(seg_image)
    # 使用 marching cubes 算法生成 Mesh
    verts_zyx, faces, normals, values = measure.marching_cubes(seg_array, level=0.5)

    # ------2. mesh，坐标变换为RAS------
    # ijk 坐标系转换为 LPS 坐标系
    verts_lps = []
    for vz, vy, vx in verts_zyx:
        physical_point = seg_image.TransformContinuousIndexToPhysicalPoint((float(vx), float(vy), float(vz)))
        verts_lps.append(physical_point)
    # 转换为 NumPy 数组方便后续处理
    verts_lps = np.array(verts_lps)

    # ------3. 坐标归一化，根据输入的鼻尖点坐标，归一化到鼻尖点为坐标原点------
    verts_lps_norm = verts_lps - nose_tip_coord

    # 4. 生成mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_lps_norm)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    return verts_lps_norm, faces, mesh


# ------0. 读入新标记的鼻尖点的坐标------
nosetip_df = pd.read_excel('raw_data/Nose Tip Slicer Coordinates.xlsx', usecols="A, E:G", skiprows=2,header=0)

for batch in os.listdir('raw_data/2_2_review_seg'):
    if not batch.startswith('.'):
        for seg_folder in os.listdir(os.path.join('raw_data/2_2_review_seg', batch)):
            if not seg_folder.startswith('.'):
                print(seg_folder)
                # 读入post、pre的seg图像
                for file_name in os.listdir(os.path.join('raw_data/2_2_review_seg', batch, seg_folder)):
                    # print(file_name.split("_"))
                    if file_name.split('_')[-1] == 'PostOp.nrrd':
                        post_seg = os.path.join('raw_data/2_2_review_seg', batch, seg_folder, file_name)
                    elif file_name.split('_')[-1] == 'PreOp.nrrd':
                        pre_seg = os.path.join('raw_data/2_2_review_seg', batch, seg_folder, file_name) 

                # 得到pid后通过读入的nose文件，得到每个pid下的nose tip坐标（RAS坐标系）
                pid = f'{seg_folder.split("_")[0].zfill(3)}_{seg_folder.split("_")[1]}'
                
                new_nosetip_pre_ras = nosetip_df.loc[nosetip_df['pid'] == pid, ['Right.1', 'Anterior.1', 'Superior.1']].values[0]
                # RAS坐标转换为LPS坐标
                new_nosetip_pre_lps = np.array([-new_nosetip_pre_ras[0], 
                                            -new_nosetip_pre_ras[1],
                                            new_nosetip_pre_ras[2]])

                post_verts_lps_norm, post_faces, post_mesh = generate_mesh(post_seg, new_nosetip_pre_lps)
                pre_verts_lps_norm, pre_faces, pre_mesh = generate_mesh(pre_seg, new_nosetip_pre_lps)

                outdir = f'data/mesh/{pid}'
                os.makedirs(outdir, exist_ok=True)

                o3d.io.write_triangle_mesh(os.path.join(outdir, f'Post_mesh_norm.ply'), post_mesh)
                o3d.io.write_triangle_mesh(os.path.join(outdir, f'Pre_mesh_norm.ply'), pre_mesh)
                    
