'''
 # @ Author: Yaobo Jia
 # @ Create Time: 2025-05-30 14:31:55
 # @ Modified by: Yaobo Jia
 # @ Modified time: 2025-06-04 18:28:17
 # @ Description:通过手动分割的结果生成软组织mesh
1. 生成精细分割，只包含面部在内的薄层。用于mesh生成的mask，也可以用于后续的分割label；
2. mask设置ROI区域，这里采用了硬编码；
3. matching cube算法提取mesh；
4. 坐标变换，ijk转换为LPS坐标系；同时完成归一化；（pre和post的nose tip并不一样）
5. 保存mesh并进行laplacian平滑
 '''


import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import SimpleITK as sitk
import open3d as o3d
import pandas as pd
from scipy.ndimage import binary_dilation, generate_binary_structure
from skimage import measure


def generate_anterior_mask(seg_array):
    """
    从3D分割图像中生成前表面mask
    
    Args:
        seg_array: 3D numpy数组，分割图像数据
        
    Returns:
        tuple: (cropped_mask, new_segmentation)
            - cropped_mask: 裁剪后的mask，仅包含前表面区域
            - new_segmentation: 裁剪后的分割，与原始分割的交集
    """
    seg_array_bool = seg_array > 0 # 将分割图像转为布尔型
    # --- 1. 识别最前体素薄层 (LPS: Y值最小为最前) ---
    # 通过筛选x-z平面中，最小的y值看作 facial tissue
    mask_anterior_thin = np.zeros_like(seg_array_bool, dtype=bool)
    for x_coord in range(seg_array_bool.shape[2]):  # X (Left-Right)
        for z_coord in range(seg_array_bool.shape[0]):  # Z (Superior-Inferior)
            ys_indices = np.where(seg_array_bool[z_coord, :, x_coord])[0]
            if len(ys_indices) > 0:
                y_anterior = ys_indices[-1]  # Y轴索引最小的点为最前点
                mask_anterior_thin[z_coord, y_anterior, x_coord] = True

    # --- 2. 约束性形态学膨胀以包含倾斜表面并“加厚” ---
    # structuring_element 定义了膨胀时的邻域（3x3x3, 18或26连通）
    # struct_elem = generate_binary_structure(rank=3, connectivity=2) # 3D, 18-connectivity
    struct_elem = np.ones((3,3,3), dtype=bool) # 26-connectivity

    dilation_iterations=4
    grown_anterior_mask = binary_dilation(mask_anterior_thin, 
                                            structure=struct_elem,
                                            iterations=dilation_iterations)
    
    # --- 3. 裁剪ROI区域，去掉鼻根以上区域，去掉两侧多余区域 ---
    # 只保留 y > 100 且 z < 255 的区域
    cropped_mask = np.zeros_like(grown_anterior_mask, dtype=bool)
    indices = np.where(grown_anterior_mask)
    z_indices, y_indices, x_indices = indices

    # 找到满足条件的索引
    valid = (y_indices > 100) & (z_indices < 255)
    cropped_mask[z_indices[valid], y_indices[valid], x_indices[valid]] = True

    new_segmentation = cropped_mask & seg_array_bool # 确保膨胀不超出原始分割区域

    return cropped_mask, new_segmentation


def generate_mesh(seg_img, mask, nose_tip_coord):
    """
    从分割图像和mask生成3D mesh
    
    Args:
        seg_img: SimpleITK图像对象，原始分割图像
        mask: numpy布尔数组，用于提取表面的mask
        nose_tip_coord: numpy数组，鼻尖坐标点，用于坐标归一化
        
    Returns:
        mesh_smoothed: Open3D TriangleMesh对象，平滑处理后的mesh
    """
    # 从SimpleITK图像中获取数组数据
    seg_array = sitk.GetArrayFromImage(seg_img)
    # 使用marching cubes算法生成mesh, 将上一步生成的mask在这里获取对应面部区域
    verts_zyx, faces, normals, values = measure.marching_cubes(seg_array, level=0.5, mask=mask)

    # 坐标变换：从IJK(图像)坐标系转换为LPS(物理)坐标系
    verts_lps = []
    for vz, vy, vx in verts_zyx:
        physical_point = seg_img.TransformContinuousIndexToPhysicalPoint((float(vx), float(vy), float(vz)))
        verts_lps.append(physical_point)
    verts_lps = np.array(verts_lps)

    # 坐标归一化：将坐标原点移动到鼻尖点
    verts_lps_norm = verts_lps - nose_tip_coord

    # 生成mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts_lps_norm)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # 应用Laplacian平滑
    mesh_smoothed = mesh.filter_smooth_laplacian(number_of_iterations=20)
    mesh_smoothed.compute_vertex_normals()

    return mesh_smoothed


if __name__ == '__main__':
    """
    主程序流程：
    1. 读取所有手术前后的分割文件
    2. 对每个病例：
        - 提取前表面
        - 生成mask
        - 创建归一化的mesh
        - 保存处理结果
    """
    root_dir = Path('raw_data/2_2_review_seg')
    nose_df = pd.read_excel('data/nose_tip_coordinates_LPS.xlsx')
    for post_file_path in tqdm(root_dir.rglob("*_PostOp.nrrd")):
        # 读取file
        post_filename = post_file_path.name
        pre_filename = post_filename.replace("_PostOp.nrrd", "_PreOp.nrrd")
        pre_file_path = post_file_path.with_name(pre_filename)

        # 对post、pre提取前表面：mask、new segmentation
        # 其中：mask用于后续的matching cube算法；
        # new segmentation是mask & segmentation计算得到的，用于之后的分割模型训练
        post_seg_img = sitk.ReadImage(str(post_file_path))
        post_seg_array = sitk.GetArrayFromImage(post_seg_img)
        post_mask, post_segmentation = generate_anterior_mask(post_seg_array)
        pre_seg_img = sitk.ReadImage(str(pre_file_path))
        pre_seg_array = sitk.GetArrayFromImage(pre_seg_img)
        pre_mask, pre_segmentation = generate_anterior_mask(pre_seg_array)
        # file process：Pid
        post_relative_path = post_file_path.relative_to(root_dir)
        pid = post_relative_path.parts[1]
        pid = f'{pid.split("_")[0].zfill(3)}_{pid.split("_")[1]}'
        # 保存生成的new segmentation
        out_dir_seg = Path('data/segmentation') / pid
        out_dir_seg.mkdir(parents=True, exist_ok=True)
        sitk.WriteImage(sitk.GetImageFromArray(post_segmentation.astype(np.uint8)), str(out_dir_seg / 'Post.nii.gz'))
        sitk.WriteImage(sitk.GetImageFromArray(pre_segmentation.astype(np.uint8)), str(out_dir_seg / 'Pre.nii.gz'))

        # 获取nose_tip_coord
        pre_nose_tip_coord = nose_df.loc[nose_df['pid'] == pid, ['pre_L', 'pre_P', 'pre_S']].values[0]
        post_nose_tip_coord = nose_df.loc[nose_df['pid'] == pid, ['post_L', 'post_P', 'post_S']].values[0]

        # generate mesh，并完成归一化
        out_dir_mesh = Path('data/mesh') / pid
        out_dir_mesh.mkdir(parents=True, exist_ok=True)
        post_mesh = generate_mesh(post_seg_img, post_mask, post_nose_tip_coord)
        o3d.io.write_triangle_mesh(str(out_dir_mesh / 'Post.ply'), post_mesh)
        pre_mesh = generate_mesh(pre_seg_img, pre_mask, pre_nose_tip_coord)
        o3d.io.write_triangle_mesh(str(out_dir_mesh / 'Pre.ply'), pre_mesh)