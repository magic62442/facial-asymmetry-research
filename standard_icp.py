import open3d as o3d
import numpy as np
import os
import csv
from scipy.spatial import KDTree


def mirror_and_register_manual_icp(obj_path, output_csv='mirror_registration.csv', output_dir='registration_mirror'):
    """
    手动实现的多尺度标准 ICP 算法。

    特点:
    1. 不使用固定点对 (pairs.csv)，完全基于几何最近邻。
    2. 手动实现迭代循环，因此可以精确输出每次迭代的 MSE、Fitness 和总迭代次数。
    3. 采用 Coarse-to-Fine (由粗到精) 策略，通过逐步降低距离阈值来保证收敛。
    """
    print("=" * 80)
    print("镜像配准 - 手动实现标准 ICP (Manual Standard ICP)")
    print("=" * 80)

    # 1. 加载原始模型
    print(f"\n加载模型: {obj_path}")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"找不到文件: {obj_path}")

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices_original = np.asarray(mesh.vertices).copy()
    print(f"顶点数: {len(vertices_original)}")

    # 2. 创建镜像点云（关于x=0平面）
    print("\n创建镜像点云...")
    vertices_mirrored = vertices_original.copy()
    vertices_mirrored[:, 0] = -vertices_mirrored[:, 0]  # 翻转X坐标

    # 创建 Open3D 点云对象
    pcd_target = o3d.geometry.PointCloud()
    pcd_target.points = o3d.utility.Vector3dVector(vertices_original)

    pcd_source = o3d.geometry.PointCloud()
    pcd_source.points = o3d.utility.Vector3dVector(vertices_mirrored)

    # 3. ICP 参数配置
    # 多尺度阈值序列 (mm): 从宽松到严格
    thresholds = [64.0, 32.0, 16.0, 8.0, 4.0, 2.0]

    # 收敛条件
    max_iteration_per_scale = 100
    relative_fitness_threshold = 1e-6
    relative_mse_threshold = 1e-7

    # 初始化变换矩阵 (单位矩阵)
    current_transformation = np.eye(4)

    # 估计器 (使用 Point-to-Point)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    print(f"\n开始多尺度 ICP 循环，阈值序列: {thresholds}")
    total_iteration_count = 0

    # 预构建目标点云的 KDTree (加速最近邻搜索)
    target_tree = KDTree(vertices_original)

    # ================= 多尺度循环 =================
    for scale_idx, threshold in enumerate(thresholds):
        print(f"\n=== 尺度 {scale_idx + 1}/{len(thresholds)}: threshold = {threshold} mm ===")

        prev_fitness = 0.0
        prev_inlier_mse = float('inf')

        # ================= 迭代循环 =================
        for iteration in range(max_iteration_per_scale):
            # A. 变换源点云
            # 注意: 这里我们只变换坐标用于计算，不修改 pcd_source 本身，直到最后
            source_points = np.asarray(pcd_source.points)
            # 齐次变换: (R * p + t)
            # 构造 (N, 4)
            ones = np.ones((len(source_points), 1))
            points_homogeneous = np.hstack([source_points, ones])
            # 变换
            source_transformed = (current_transformation @ points_homogeneous.T).T[:, :3]

            # B. 寻找最近邻 (Nearest Neighbor)
            # 这是标准 ICP 的核心: 每一轮都重新找对应点
            distances, indices = target_tree.query(source_transformed)

            # C. 筛选有效点对 (Valid Mask)
            # 根据当前 threshold 剔除离群点
            valid_mask = distances < threshold

            # 提取有效对应关系
            # source_indices: [0, 1, 2, ...] 中符合条件的索引
            source_indices_valid = np.where(valid_mask)[0]
            target_indices_valid = indices[valid_mask]
            valid_distances = distances[valid_mask]

            # D. 计算指标
            if len(source_indices_valid) == 0:
                print("  警告: 未找到有效配对点，停止当前尺度。")
                break

            current_fitness = len(source_indices_valid) / len(source_points)
            current_inlier_mse = np.mean(valid_distances ** 2)
            current_overall_mse = np.mean(distances ** 2)  # 用户要求的总体 MSE

            # E. 检查收敛
            if iteration > 0:
                fitness_change = abs(current_fitness - prev_fitness)
                mse_change = abs(current_inlier_mse - prev_inlier_mse)

                # 防止除以零
                rel_fitness_change = fitness_change / (prev_fitness + 1e-9)
                rel_mse_change = mse_change / (prev_inlier_mse + 1e-9)

                if rel_fitness_change < relative_fitness_threshold and rel_mse_change < relative_mse_threshold:
                    print(f"  收敛于第 {iteration} 次迭代")
                    print(f"    Fitness: {current_fitness:.6f}")
                    print(f"    Inlier MSE: {current_inlier_mse:.6f} mm²")
                    print(f"    Overall MSE: {current_overall_mse:.6f} mm²")
                    total_iteration_count += iteration
                    break

            # F. 计算新的变换 (SVD)
            # 构建 Open3D 需要的对应对向量 [[src_id, tgt_id], ...]
            # 注意: 这里需要的是点云中的索引
            corres_np = np.stack([source_indices_valid, target_indices_valid], axis=1).astype(np.int32)
            corres_vec = o3d.utility.Vector2iVector(corres_np)

            # 创建临时点云用于计算
            pcd_source_temp = o3d.geometry.PointCloud()
            pcd_source_temp.points = o3d.utility.Vector3dVector(source_transformed)

            # 计算这一步的微调变换 (Delta Transformation)
            delta_trans = estimation.compute_transformation(
                pcd_source_temp, pcd_target, corres_vec
            )

            # G. 更新全局变换矩阵
            # New_T = Delta * Current_T
            current_transformation = delta_trans @ current_transformation

            # 更新历史值
            prev_fitness = current_fitness
            prev_inlier_mse = current_inlier_mse

            # 如果是最后一次迭代，打印状态
            if iteration == max_iteration_per_scale - 1:
                print(f"  达到最大迭代次数 ({max_iteration_per_scale})")
                total_iteration_count += max_iteration_per_scale

    # 4. 最终结果统计
    print(f"\n{'=' * 60}")
    print(f"手动 ICP 配准完成")
    print(f"  总迭代次数: {total_iteration_count}")
    print(f"  最终变换矩阵:\n{current_transformation}")
    print(f"{'=' * 60}")

    # 5. 应用最终变换并生成结果数据
    print("\n应用变换并计算最终几何偏差...")
    pcd_mirrored_final = pcd_source.transform(current_transformation)
    vertices_final = np.asarray(pcd_mirrored_final.points)

    # KDTree 查询最终距离
    final_distances, nearest_indices = target_tree.query(vertices_final)

    mean_dist = final_distances.mean()
    median_dist = np.median(final_distances)
    max_dist = final_distances.max()

    print(f"  全表面平均距离 (Mean Distance): {mean_dist:.6f} mm")
    print(f"  中位数距离 (Median Distance): {median_dist:.6f} mm")

    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_vertex_id', 'target_vertex_id', 'distance'])
        for i in range(len(vertices_final)):
            writer.writerow([i, nearest_indices[i], f"{final_distances[i]:.6f}"])

    print(f"\n结果已保存到: {output_path}")

    return {
        'transformation': current_transformation,
        'mean_distance': mean_dist,
        'output_path': output_path
    }


if __name__ == "__main__":
    obj_file = "displaced_directional/25_0.3.obj"

    try:
        result = mirror_and_register_manual_icp(
            obj_path=obj_file,
            output_csv='25_0.3_manual_icp.csv',
            output_dir='manual_icp_result'
        )
    except Exception as e:
        print(f"程序运行出错: {e}")