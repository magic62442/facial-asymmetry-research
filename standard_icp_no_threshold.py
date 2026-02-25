import open3d as o3d
import numpy as np
import os
import csv
from scipy.spatial import KDTree


def mirror_and_register_icp_no_threshold(obj_path, output_csv='mirror_registration.csv', output_dir='no_threshold_icp_result'):
    """
    不使用阈值的标准 ICP 算法。

    特点:
    1. 不使用固定点对 (pairs.csv)，完全基于几何最近邻。
    2. 不使用任何距离阈值筛选，所有点对都参与变换计算。
    3. 手动实现迭代循环，可以精确输出每次迭代的 MSE 和总迭代次数。
    4. 更接近经典 ICP 算法的原始形式。

    与多尺度 ICP 的区别:
    - 多尺度 ICP: 使用阈值序列筛选内点，逐步提高精度
    - 本算法: 所有点对都参与计算，无筛选
    """
    print("=" * 80)
    print("镜像配准 - 无阈值标准 ICP (Standard ICP without Threshold)")
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
    max_iteration = 200  # 最大迭代次数
    relative_mse_threshold = 1e-9  # MSE 相对变化阈值
    absolute_mse_threshold = 1e-10  # MSE 绝对变化阈值

    # 初始化变换矩阵 (单位矩阵)
    current_transformation = np.eye(4)

    # 估计器 (使用 Point-to-Point)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    print(f"\n开始无阈值 ICP 迭代")
    print(f"  最大迭代次数: {max_iteration}")
    print(f"  相对 MSE 阈值: {relative_mse_threshold}")
    print(f"  绝对 MSE 阈值: {absolute_mse_threshold}")

    # 预构建目标点云的 KDTree (加速最近邻搜索)
    target_tree = KDTree(vertices_original)

    prev_mse = float('inf')

    # ================= 主迭代循环 =================
    for iteration in range(max_iteration):
        # A. 变换源点云
        source_points = np.asarray(pcd_source.points)
        ones = np.ones((len(source_points), 1))
        points_homogeneous = np.hstack([source_points, ones])
        source_transformed = (current_transformation @ points_homogeneous.T).T[:, :3]

        # B. 寻找最近邻 (Nearest Neighbor)
        # 这是标准 ICP 的核心: 每一轮都重新找对应点
        distances, indices = target_tree.query(source_transformed)

        # C. 计算 MSE（使用所有点）
        current_mse = np.mean(distances ** 2)
        current_rmse = np.sqrt(current_mse)
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)

        # 打印进度（每10次迭代或前5次）
        if iteration % 10 == 0 or iteration < 5:
            print(f"  Iter {iteration:3d}: MSE={current_mse:.8f} mm², "
                  f"RMSE={current_rmse:.6f} mm, "
                  f"Mean={mean_distance:.6f} mm, "
                  f"Max={max_distance:.6f} mm")

        # D. 检查收敛
        if iteration > 0:
            mse_change = abs(current_mse - prev_mse)
            rel_mse_change = mse_change / (prev_mse + 1e-12)

            # 两个收敛条件（满足任一即可）
            if rel_mse_change < relative_mse_threshold:
                print(f"\n  收敛于第 {iteration} 次迭代 (相对变化)")
                print(f"    最终 MSE: {current_mse:.8f} mm²")
                print(f"    最终 RMSE: {current_rmse:.6f} mm")
                print(f"    相对 MSE 变化: {rel_mse_change:.2e}")
                break

            if mse_change < absolute_mse_threshold:
                print(f"\n  收敛于第 {iteration} 次迭代 (绝对变化)")
                print(f"    最终 MSE: {current_mse:.8f} mm²")
                print(f"    最终 RMSE: {current_rmse:.6f} mm")
                print(f"    绝对 MSE 变化: {mse_change:.2e}")
                break

        # E. 使用所有点对计算新的变换 (SVD)
        # 注意：这里使用全部 N 个点对，不进行筛选
        source_indices_all = np.arange(len(source_points), dtype=np.int32)
        target_indices_all = indices.astype(np.int32)

        corres_np = np.stack([source_indices_all, target_indices_all], axis=1)
        corres_vec = o3d.utility.Vector2iVector(corres_np)

        # 创建临时点云用于计算
        pcd_source_temp = o3d.geometry.PointCloud()
        pcd_source_temp.points = o3d.utility.Vector3dVector(source_transformed)

        # 计算这一步的微调变换 (Delta Transformation)
        delta_trans = estimation.compute_transformation(
            pcd_source_temp, pcd_target, corres_vec
        )

        # F. 更新全局变换矩阵
        current_transformation = delta_trans @ current_transformation

        # 更新历史值
        prev_mse = current_mse

        # 如果是最后一次迭代
        if iteration == max_iteration - 1:
            print(f"\n  达到最大迭代次数 ({max_iteration})")
            print(f"    最终 MSE: {current_mse:.8f} mm²")
            print(f"    最终 RMSE: {current_rmse:.6f} mm")

    total_iterations = iteration + 1

    # 4. 最终结果统计
    print(f"\n{'=' * 60}")
    print(f"无阈值 ICP 配准完成")
    print(f"  总迭代次数: {total_iterations}")
    print(f"  最终变换矩阵:")
    print(current_transformation)
    print(f"{'=' * 60}")

    # 5. 应用最终变换并计算最终几何偏差
    print("\n应用变换并计算最终几何偏差...")
    pcd_mirrored_final = pcd_source.transform(current_transformation)
    vertices_final = np.asarray(pcd_mirrored_final.points)

    # KDTree 查询最终距离
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

    print(f"\n最终配准质量统计:")
    print(f"  顶点总数: {len(final_distances)}")
    print(f"  平均距离 (Mean): {mean_dist:.6f} mm")
    print(f"  中位数距离 (Median): {median_dist:.6f} mm")
    print(f"  标准差 (Std): {std_dist:.6f} mm")
    print(f"  最小距离 (Min): {min_dist:.6f} mm")
    print(f"  最大距离 (Max): {max_dist:.6f} mm")
    print(f"  MSE: {mse:.6f} mm²")
    print(f"  RMSE: {rmse:.6f} mm")

    # 百分位数
    percentiles = [50, 75, 90, 95, 99]
    print(f"\n距离百分位数:")
    for p in percentiles:
        val = np.percentile(final_distances, p)
        print(f"  {p}%: {val:.6f} mm")

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
        'total_iterations': total_iterations,
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'mse': mse,
        'rmse': rmse,
        'output_path': output_path
    }


def batch_process_no_threshold_icp(input_dir='displaced_directional', output_dir='no_threshold_icp_result'):
    """
    批量处理目录下的所有 OBJ 文件，使用无阈值 ICP 进行配准。

    参数:
        input_dir: 输入目录（包含变形后的 OBJ 文件）
        output_dir: 输出目录（保存配准结果）
    """
    print("=" * 80)
    print("批量无阈值 ICP 配准")
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

        # 生成输出文件名
        base_name = os.path.splitext(obj_file)[0]
        output_csv = f"{base_name}_no_threshold_icp.csv"

        try:
            result = mirror_and_register_icp_no_threshold(
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
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'no_threshold_icp_result'

            print(f"批量处理模式")
            print(f"  输入目录: {input_dir}")
            print(f"  输出目录: {output_dir}")

            batch_process_no_threshold_icp(input_dir=input_dir, output_dir=output_dir)
        else:
            # 单文件模式
            obj_file = sys.argv[1]
            output_csv = sys.argv[2] if len(sys.argv) > 2 else None
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'no_threshold_icp_result'

            if output_csv is None:
                base_name = os.path.splitext(os.path.basename(obj_file))[0]
                output_csv = f"{base_name}_no_threshold_icp.csv"

            result = mirror_and_register_icp_no_threshold(
                obj_path=obj_file,
                output_csv=output_csv,
                output_dir=output_dir
            )
    else:
        # 默认示例
        print("使用方法:")
        print("  单文件: python standard_icp_no_threshold.py <obj_file> [output_csv] [output_dir]")
        print("  批量:   python standard_icp_no_threshold.py batch [input_dir] [output_dir]")
        print()
        print("运行默认示例...")

        obj_file = "displaced_directional/25_0.3.obj"

        if os.path.exists(obj_file):
            result = mirror_and_register_icp_no_threshold(
                obj_path=obj_file,
                output_csv='25_0.3_no_threshold_icp.csv',
                output_dir='no_threshold_icp_result'
            )
        else:
            print(f"错误: 找不到测试文件 {obj_file}")
            print("请提供有效的 OBJ 文件路径")
