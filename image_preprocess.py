'''
 # @ Author: Jacob
 # @ Create Time: 2025-05-21 16:38:40
 # @ Modified by: Jacob
 # @ Modified time: 2025-05-21 16:38:41
 # @ Description:
 '''
import numpy as np
import SimpleITK as sitk

from skimage import measure
import open3d as o3d

seg_path = '2_2_review_seg/54/001_WC/001_WC_PostOp.nrrd'

def get_surface(seg_path):
    seg_img = sitk.ReadImage(seg_path)
    seg_array = sitk.GetArrayFromImage(seg_img)  # shape: [z, y, x]
    spacing = seg_img.GetSpacing()  # (x, y, z)
    origin = seg_img.GetOrigin()
    direction = seg_img.GetDirection()

    # 新建只含前表面的mask
    mask_surface = np.zeros_like(seg_array, dtype=bool)
    for x in range(seg_array.shape[2]):
        for z in range(seg_array.shape[0]):
            ys = np.where(seg_array[z, :, x] > 0)[0]
            if len(ys) > 0:
                y_front = ys[-1]
                mask_surface[z-3: z+3, y_front-3: y_front+3, x-3: x+3] = True

    # 增加鼻翼区域
    mask_x = np.zeros_like(seg_array, dtype=bool)
    for y in range(130, seg_array.shape[1]):
        for z in range(100, seg_array.shape[0]):
            xs = np.where(seg_array[z, y, :] > 0)[0]
            if len(xs) > 0:
                x_left = xs[0]
                x_right = xs[-1]
                mask_x[z, y, x_left-2: x_left+2] = True
                mask_x[z, y, x_right-2: x_right+2] = True

    mask_surface = mask_x | mask_surface

    # 增加ROI
    mask_roi = np.zeros_like(seg_array, dtype=bool)

    z_idxs = np.where(mask_surface.any(axis=(1,2)))[0]
    z_min, z_max = z_idxs[0], z_idxs[-1]
    y_idxs = np.where(mask_surface.any(axis=(0,2)))[0]
    y_min, y_max = y_idxs[0], y_idxs[-1]
    x_idxs = np.where(mask_surface.any(axis=(0,1)))[0]
    x_min, x_max = x_idxs[0], x_idxs[-1]
    mask_roi[z_min+15:z_max-20, y_min+80:, x_min+10:x_max-10] = True

    # 结合mask_surface和mask_roi
    mask_roi = mask_surface & mask_roi
    verts, faces, normals, values = measure.marching_cubes(seg_array, level=0.5, spacing=spacing, mask=mask_roi)

    faces = faces[:, ::-1] 

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_vertex_normals()

    # 计算 mesh 的连通片段
    triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)

    # 只保留最大面积的连通区域
    largest_cluster_idx = cluster_area.argmax()
    tri_mask = triangle_clusters == largest_cluster_idx
    mesh_clean = mesh.select_by_index(np.where(tri_mask)[0])


    # 拉普拉斯平滑，number_of_iterations根据效果调整，10-50次都可
    mesh_smoothed = mesh_clean.filter_smooth_laplacian(number_of_iterations=20)
    mesh_smoothed.compute_vertex_normals()

    return mesh_smoothed, np.asarray(mesh_smoothed.vertices)

def farthest_point_sampling_np(points, num_samples):
    N, _ = points.shape
    indices = np.zeros(num_samples, dtype=np.int64)
    distances = np.full(N, np.inf)  # 初始化每个点到已采样点的最小距离为无穷大

    # 随机选择第一个采样点索引
    farthest = np.random.randint(0, N)
    for i in range(num_samples):
        indices[i] = farthest
        # 计算当前采样点到所有点的欧式距离平方
        centroid = points[farthest]
        dist = np.sum((points - centroid) ** 2, axis=1)
        # 更新所有点的最小距离
        distances = np.minimum(distances, dist)
        # 选择距离当前所有采样点中最远的点作为下一个采样点
        farthest = np.argmax(distances)

    sampled_points = points[indices]

    return indices, sampled_points


pre_mesh, pre_verts = get_surface('2_2_review_seg/54/001_WC/001_WC_PreOp.nrrd')
post_mesh, post_verts = get_surface('2_2_review_seg/54/001_WC/001_WC_PostOp.nrrd')

_, pre_points = farthest_point_sampling_np(pre_verts, 4096)
_, post_points = farthest_point_sampling_np(post_verts, 4096)

o3d.io.write_triangle_mesh('pre_mesh.ply', pre_mesh)
o3d.io.write_triangle_mesh('post_mesh.ply', post_mesh)


pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(pre_points)
o3d.io.write_point_cloud('pre_points.ply', pcd)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(post_points)
o3d.io.write_point_cloud('post_points.ply', pcd)


# o3d.visualization.draw_geometries([mesh_smoothed])
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(pre_verts)
# o3d.visualization.draw_geometries([pcd])