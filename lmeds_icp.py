import open3d as o3d
import numpy as np
import os
import csv
from scipy.spatial import KDTree


def lmeds_icp_registration(obj_path, output_csv='lmeds_registration.csv', output_dir='lmeds_icp_result'):
    """
    基于 LMedS (Least Median of Squares) 的鲁棒 ICP 配准算法。

    核心思想:
    1. 不使用固定点对，完全基于几何最近邻。
    2. 使用 LMedS 准则选择内点：最小化残差的中位数而不是均值。
    3. 对离群点和噪声更加鲁棒。
    4. 多尺度策略：从粗到精逐步收敛。

    与传统 ICP 的区别:
    - 传统 ICP: 最小化所有点对的平方误差和（或使用固定阈值）
    - LMedS ICP: 最小化平方误差的中位数，自适应选择内点

    参数:
        obj_path: OBJ文件路径
        output_csv: 输出CSV文件名
        output_dir: 输出目录

    返回:
        字典包含变换矩阵、统计信息等
    """
    print("=" * 80)
    print("LMedS ICP 配准 (Least Median of Squares ICP)")
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

    # 3. LMedS ICP 参数配置
    # 多尺度策略：从宽松到严格
    scales = [
        {'name': 'Coarse', 'median_scale': 3.0, 'max_iter': 50},
        {'name': 'Medium', 'median_scale': 2.5, 'max_iter': 50},
        {'name': 'Fine', 'median_scale': 2.0, 'max_iter': 50},
        {'name': 'Very Fine', 'median_scale': 1.5, 'max_iter': 100},
    ]

    # 收敛条件
    relative_fitness_threshold = 1e-6
    relative_rmse_threshold = 1e-7

    # 初始化变换矩阵 (单位矩阵)
    current_transformation = np.eye(4)

    # 估计器 (使用 Point-to-Point)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    print(f"\n开始多尺度 LMedS ICP 循环")
    total_iteration_count = 0

    # 预构建目标点云的 KDTree (加速最近邻搜索)
    target_tree = KDTree(vertices_original)

    # ================= 多尺度循环 =================
    for scale_idx, scale_params in enumerate(scales):
        scale_name = scale_params['name']
        median_scale = scale_params['median_scale']
        max_iteration = scale_params['max_iter']

        print(f"\n=== 尺度 {scale_idx + 1}/{len(scales)}: {scale_name} (median_scale={median_scale}) ===")

        prev_fitness = 0.0
        prev_rmse = float('inf')

        # ================= 迭代循环 =================
        for iteration in range(max_iteration):
            # A. 变换源点云
            source_points = np.asarray(pcd_source.points)
            ones = np.ones((len(source_points), 1))
            points_homogeneous = np.hstack([source_points, ones])
            source_transformed = (current_transformation @ points_homogeneous.T).T[:, :3]

            # B. 寻找最近邻
            distances, indices = target_tree.query(source_transformed)

            # C. LMedS 核心：计算残差的中位数
            squared_distances = distances ** 2
            median_squared_error = np.median(squared_distances)

            # D. 自适应阈值：基于中位数
            # 阈值 = median_scale * sqrt(median_squared_error)
            # median_scale 是一个倍数因子，控制内点选择的宽松程度
            adaptive_threshold_squared = (median_scale ** 2) * median_squared_error

            # E. 选择内点 (Inliers)
            # 内点定义：平方误差 < adaptive_threshold_squared
            inlier_mask = squared_distances < adaptive_threshold_squared

            source_indices_inliers = np.where(inlier_mask)[0]
            target_indices_inliers = indices[inlier_mask]
            inlier_distances = distances[inlier_mask]
            inlier_count = len(source_indices_inliers)

            # F. 计算指标
            if inlier_count == 0:
                print("  警告: 未找到内点，停止当前尺度。")
                break

            current_fitness = inlier_count / len(source_points)
            current_rmse = np.sqrt(median_squared_error)  # LMedS 使用中位数的 RMSE
            inlier_mean_distance = np.mean(inlier_distances)
            outlier_count = len(source_points) - inlier_count

            # 打印详细信息（每10次迭代）
            if iteration % 10 == 0 or iteration < 5:
                print(f"  Iter {iteration:3d}: Fitness={current_fitness:.4f}, "
                      f"RMSE={current_rmse:.6f} mm, "
                      f"Inliers={inlier_count}/{len(source_points)}, "
                      f"Outliers={outlier_count}")

            # G. 检查收敛
            if iteration > 0:
                fitness_change = abs(current_fitness - prev_fitness)
                rmse_change = abs(current_rmse - prev_rmse)

                rel_fitness_change = fitness_change / (prev_fitness + 1e-9)
                rel_rmse_change = rmse_change / (prev_rmse + 1e-9)

                if rel_fitness_change < relative_fitness_threshold and rel_rmse_change < relative_rmse_threshold:
                    print(f"  收敛于第 {iteration} 次迭代")
                    print(f"    Fitness: {current_fitness:.6f}")
                    print(f"    RMSE (median-based): {current_rmse:.6f} mm")
                    print(f"    Inlier mean distance: {inlier_mean_distance:.6f} mm")
                    print(f"    Adaptive threshold: {np.sqrt(adaptive_threshold_squared):.6f} mm")
                    total_iteration_count += iteration
                    break

            # H. 计算新的变换 (使用内点)
            corres_np = np.stack([source_indices_inliers, target_indices_inliers], axis=1).astype(np.int32)
            corres_vec = o3d.utility.Vector2iVector(corres_np)

            pcd_source_temp = o3d.geometry.PointCloud()
            pcd_source_temp.points = o3d.utility.Vector3dVector(source_transformed)

            delta_trans = estimation.compute_transformation(
                pcd_source_temp, pcd_target, corres_vec
            )

            # I. 更新全局变换矩阵
            current_transformation = delta_trans @ current_transformation

            # 更新历史值
            prev_fitness = current_fitness
            prev_rmse = current_rmse

            # 如果是最后一次迭代
            if iteration == max_iteration - 1:
                print(f"  达到最大迭代次数 ({max_iteration})")
                print(f"    最终 Fitness: {current_fitness:.6f}")
                print(f"    最终 RMSE: {current_rmse:.6f} mm")
                total_iteration_count += max_iteration

    # 4. 最终结果统计
    print(f"\n{'=' * 60}")
    print(f"LMedS ICP 配准完成")
    print(f"  总迭代次数: {total_iteration_count}")
    print(f"  最终变换矩阵:")
    print(current_transformation)
    print(f"{'=' * 60}")

    # 5. 应用最终变换并计算最终统计
    print("\n应用变换并计算最终几何偏差...")
    pcd_mirrored_final = pcd_source.transform(current_transformation)
    vertices_final = np.asarray(pcd_mirrored_final.points)

    # 最终最近邻查询
    final_distances, nearest_indices = target_tree.query(vertices_final)

    # 计算各种统计量
    mean_dist = final_distances.mean()
    median_dist = np.median(final_distances)
    std_dist = final_distances.std()
    max_dist = final_distances.max()
    min_dist = final_distances.min()

    # 计算 MSE 和 RMSE
    mse = np.mean(final_distances ** 2)
    rmse = np.sqrt(mse)

    # 百分位数
    percentiles = [50, 75, 90, 95, 99]
    percentile_values = {p: np.percentile(final_distances, p) for p in percentiles}

    print(f"\n最终配准质量统计:")
    print(f"  顶点总数: {len(final_distances)}")
    print(f"  平均距离 (Mean): {mean_dist:.6f} mm")
    print(f"  中位数距离 (Median): {median_dist:.6f} mm")
    print(f"  标准差 (Std): {std_dist:.6f} mm")
    print(f"  最小距离 (Min): {min_dist:.6f} mm")
    print(f"  最大距离 (Max): {max_dist:.6f} mm")
    print(f"  MSE: {mse:.6f} mm²")
    print(f"  RMSE: {rmse:.6f} mm")

    print(f"\n距离百分位数:")
    for p in percentiles:
        print(f"  {p}%: {percentile_values[p]:.6f} mm")

    # 6. 保存结果
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    print(f"\n保存配对结果到: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_vertex_id', 'target_vertex_id', 'distance'])
        for i in range(len(vertices_final)):
            writer.writerow([i, nearest_indices[i], f"{final_distances[i]:.6f}"])

    print(f"✓ 结果已保存")

    return {
        'transformation': current_transformation,
        'total_iterations': total_iteration_count,
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'mse': mse,
        'rmse': rmse,
        'percentiles': percentile_values,
        'output_path': output_path
    }


def batch_process_lmeds_icp(input_dir='displaced_directional', output_dir='lmeds_icp_result'):
    """
    批量处理目录下的所有 OBJ 文件，使用 LMedS ICP 进行配准。

    参数:
        input_dir: 输入目录（包含变形后的 OBJ 文件）
        output_dir: 输出目录（保存配准结果）
    """
    print("=" * 80)
    print("批量 LMedS ICP 配准")
    print("=" * 80)

    if not os.path.exists(input_dir):
        print(f"错误: 找不到输入目录 {input_dir}")
        return

    # 获取所有 OBJ 文件
    obj_files = [f for f in os.listdir(input_dir) if f.endswith('.obj')]
    obj_files.sort()

    print(f"\n找到 {len(obj_files)} 个 OBJ 文件")

    results = []

    for idx, obj_file in enumerate(obj_files):
        print(f"\n{'=' * 80}")
        print(f"处理文件 {idx + 1}/{len(obj_files)}: {obj_file}")
        print(f"{'=' * 80}")

        obj_path = os.path.join(input_dir, obj_file)

        # 生成输出文件名（去掉 .obj 后缀，加上 _lmeds_icp.csv）
        base_name = os.path.splitext(obj_file)[0]
        output_csv = f"{base_name}_lmeds_icp.csv"

        try:
            result = lmeds_icp_registration(
                obj_path=obj_path,
                output_csv=output_csv,
                output_dir=output_dir
            )

            results.append({
                'file': obj_file,
                'success': True,
                'mean_distance': result['mean_distance'],
                'median_distance': result['median_distance'],
                'rmse': result['rmse'],
                'total_iterations': result['total_iterations']
            })

        except Exception as e:
            print(f"\n错误: 处理 {obj_file} 时出错: {e}")
            results.append({
                'file': obj_file,
                'success': False,
                'error': str(e)
            })

    # 输出汇总
    print("\n" + "=" * 80)
    print("批量处理完成 - 汇总")
    print("=" * 80)

    success_count = sum(1 for r in results if r['success'])
    print(f"\n成功: {success_count}/{len(results)}")

    if success_count > 0:
        print("\n成功处理的文件:")
        print(f"{'文件名':<30} {'Mean(mm)':<12} {'Median(mm)':<12} {'RMSE(mm)':<12} {'迭代次数':<10}")
        print("-" * 80)
        for r in results:
            if r['success']:
                print(f"{r['file']:<30} {r['mean_distance']:<12.6f} {r['median_distance']:<12.6f} "
                      f"{r['rmse']:<12.6f} {r['total_iterations']:<10}")

    if success_count < len(results):
        print("\n失败的文件:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}: {r['error']}")

    return results


if __name__ == "__main__":
    import sys

    # 支持命令行参数
    if len(sys.argv) > 1:
        if sys.argv[1] == 'batch':
            # 批量处理模式
            input_dir = sys.argv[2] if len(sys.argv) > 2 else 'displaced_directional'
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'lmeds_icp_result'

            print(f"批量处理模式")
            print(f"  输入目录: {input_dir}")
            print(f"  输出目录: {output_dir}")

            batch_process_lmeds_icp(input_dir=input_dir, output_dir=output_dir)
        else:
            # 单文件模式
            obj_file = sys.argv[1]
            output_csv = sys.argv[2] if len(sys.argv) > 2 else None
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'lmeds_icp_result'

            if output_csv is None:
                base_name = os.path.splitext(os.path.basename(obj_file))[0]
                output_csv = f"{base_name}_lmeds_icp.csv"

            result = lmeds_icp_registration(
                obj_path=obj_file,
                output_csv=output_csv,
                output_dir=output_dir
            )
    else:
        # 默认示例
        print("使用方法:")
        print("  单文件: python lmeds_icp.py <obj_file> [output_csv] [output_dir]")
        print("  批量:   python lmeds_icp.py batch [input_dir] [output_dir]")
        print()
        print("运行默认示例...")

        obj_file = "displaced_directional/25_0.3.obj"

        if os.path.exists(obj_file):
            result = lmeds_icp_registration(
                obj_path=obj_file,
                output_csv='25_0.3.csv',
                output_dir='lmeds_icp_result'
            )
        else:
            print(f"错误: 找不到测试文件 {obj_file}")
            print("请提供有效的 OBJ 文件路径")
