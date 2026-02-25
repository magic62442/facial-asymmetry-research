import open3d as o3d
import numpy as np
import os
import csv
from scipy.spatial import KDTree
import trimesh


def mirror_and_register_icp_point_to_surface(obj_path, output_csv='mirror_registration_p2s.csv', output_dir='point_to_surface_icp_result'):
    """
    使用点到面距离的真正 ICP 算法。

    特点:
    1. ICP配准过程使用点到mesh表面的精确距离（trimesh）
    2. 每次迭代都计算点到表面的距离，而不是点到顶点
    3. 不使用固定点对 (pairs.csv)，完全基于几何最近面
    4. 不使用任何距离阈值筛选，所有点对都参与变换计算
    5. 输出CSV包含点到表面的精确距离和最近三角形信息

    与标准 ICP 的区别:
    - 标准 ICP: 点到最近顶点的距离（KDTree）
    - 本算法: 点到mesh表面三角形的精确距离（trimesh）

    注意：由于每次迭代都要计算点到面距离，速度会比标准ICP慢
    """
    print("=" * 80)
    print("镜像配准 - 点到表面距离 ICP (Point-to-Surface ICP)")
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
    max_iteration = 100  # 最大迭代次数
    relative_mse_threshold = 1e-6  # MSE 相对变化阈值
    absolute_mse_threshold = 1e-6  # MSE 绝对变化阈值

    # 初始化变换矩阵 (单位矩阵)
    current_transformation = np.eye(4)

    # 估计器 (使用 Point-to-Point)
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    print(f"\n开始 ICP 迭代 (使用点到表面距离)")
    print(f"  最大迭代次数: {max_iteration}")
    print(f"  相对 MSE 阈值: {relative_mse_threshold}")
    print(f"  绝对 MSE 阈值: {absolute_mse_threshold}")

    # 使用trimesh加载目标mesh（用于ICP迭代中的点到面距离计算）
    print(f"  加载目标mesh (trimesh)...")
    mesh_target_tm = trimesh.load(obj_path, force='mesh')
    print(f"    目标mesh: {len(mesh_target_tm.vertices)} 个顶点, {len(mesh_target_tm.faces)} 个面")

    prev_mse = float('inf')

    # ================= 主迭代循环 (使用点到表面距离) =================
    for iteration in range(max_iteration):
        # A. 变换源点云
        source_points = np.asarray(pcd_source.points)
        ones = np.ones((len(source_points), 1))
        points_homogeneous = np.hstack([source_points, ones])
        source_transformed = (current_transformation @ points_homogeneous.T).T[:, :3]

        # B. 计算点到mesh表面的距离（使用trimesh）
        closest_points_on_surface, distances, triangle_ids = trimesh.proximity.closest_point(
            mesh_target_tm, source_transformed
        )

        # C. 计算 MSE（使用点到表面的距离）
        current_mse = np.mean(distances ** 2)
        current_rmse = np.sqrt(current_mse)
        mean_distance = np.mean(distances)
        max_distance = np.max(distances)

        # 打印进度（每10次迭代或前5次）
        if iteration % 10 == 0 or iteration < 5:
            print(f"  Iter {iteration:3d}: MSE={current_mse:.8f} mm² (点到面), "
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
                print(f"    最终 MSE (点到面): {current_mse:.8f} mm²")
                print(f"    最终 RMSE (点到面): {current_rmse:.6f} mm")
                print(f"    相对 MSE 变化: {rel_mse_change:.2e}")
                break

            if mse_change < absolute_mse_threshold:
                print(f"\n  收敛于第 {iteration} 次迭代 (绝对变化)")
                print(f"    最终 MSE (点到面): {current_mse:.8f} mm²")
                print(f"    最终 RMSE (点到面): {current_rmse:.6f} mm")
                print(f"    绝对 MSE 变化: {mse_change:.2e}")
                break

        # E. 建立点对对应关系：源点 -> 表面最近点
        # 注意：closest_points_on_surface 是表面上的连续点，不是顶点索引
        # 我们需要构建对应关系用于变换估计
        source_indices_all = np.arange(len(source_points), dtype=np.int32)

        # 创建临时点云：源点变换后的位置 和 表面最近点
        pcd_source_temp = o3d.geometry.PointCloud()
        pcd_source_temp.points = o3d.utility.Vector3dVector(source_transformed)

        pcd_target_temp = o3d.geometry.PointCloud()
        pcd_target_temp.points = o3d.utility.Vector3dVector(closest_points_on_surface)

        # 建立一对一对应关系（每个源点对应其在表面的最近点）
        corres_np = np.stack([source_indices_all, source_indices_all], axis=1)
        corres_vec = o3d.utility.Vector2iVector(corres_np)

        # F. 计算这一步的微调变换 (SVD)
        delta_trans = estimation.compute_transformation(
            pcd_source_temp, pcd_target_temp, corres_vec
        )

        # G. 更新全局变换矩阵
        current_transformation = delta_trans @ current_transformation

        # 更新历史值
        prev_mse = current_mse

        # 如果是最后一次迭代
        if iteration == max_iteration - 1:
            print(f"\n  达到最大迭代次数 ({max_iteration})")
            print(f"    最终 MSE (点到面): {current_mse:.8f} mm²")
            print(f"    最终 RMSE (点到面): {current_rmse:.6f} mm")

    total_iterations = iteration + 1

    # 4. 最终结果统计
    print(f"\n{'=' * 60}")
    print(f"ICP 配准完成")
    print(f"  总迭代次数: {total_iterations}")
    print(f"  最终变换矩阵:")
    print(current_transformation)
    print(f"{'=' * 60}")

    # 6. 计算最终统计（使用最后一次迭代的结果）
    print("\n计算最终统计...")

    # 应用最终变换
    pcd_mirrored_final = pcd_source.transform(current_transformation)
    vertices_final = np.asarray(pcd_mirrored_final.points)

    # 使用最后一次迭代的距离结果（已经是点到面的距离）
    closest_points_final, final_distances, triangle_ids_final = trimesh.proximity.closest_point(
        mesh_target_tm, vertices_final
    )

    # 计算各种统计量
    mean_dist = final_distances.mean()
    median_dist = np.median(final_distances)
    std_dist = final_distances.std()
    max_dist = final_distances.max()
    min_dist = final_distances.min()

    # 计算 MSE 和 RMSE
    mse = np.mean(final_distances ** 2)
    rmse = np.sqrt(mse)

    print(f"\n最终配准质量统计 (点到表面距离):")
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

    # 7. 保存结果（包含点到表面的距离和最近三角形ID）
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_csv)

    print(f"\n保存配对结果到: {output_path}")
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_vertex_id', 'distance_to_surface'])
        for i in range(len(vertices_final)):
            writer.writerow([
                i,
                f"{final_distances[i]:.6f}",
            ])

    print(f"✓ 结果已保存")

    # 8. 保存变换后的mesh（在原始顶点上应用变换）
    transformed_mesh_path = os.path.join(output_dir, output_csv.replace('.csv', '.obj'))

    # 对原始顶点应用变换矩阵
    ones = np.ones((len(vertices_original), 1))
    vertices_original_homogeneous = np.hstack([vertices_original, ones])
    vertices_transformed = (current_transformation @ vertices_original_homogeneous.T).T[:, :3]

    # 直接使用原始mesh并更新顶点坐标，保持顶点顺序和拓扑结构不变
    mesh.vertices = o3d.utility.Vector3dVector(vertices_transformed)
    mesh.compute_vertex_normals()  # 重新计算法向量

    o3d.io.write_triangle_mesh(transformed_mesh_path, mesh)
    print(f"✓ 变换后的mesh已保存到: {transformed_mesh_path}")

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
        'output_path': output_path,
        'transformed_mesh_path': transformed_mesh_path
    }


def batch_process_point_to_surface_icp(input_dir='displaced_directional', output_dir='point_to_surface_icp_result'):
    """
    批量处理目录下的所有 OBJ 文件，使用点到表面距离评估。

    参数:
        input_dir: 输入目录（包含变形后的 OBJ 文件）
        output_dir: 输出目录（保存配准结果）
    """
    print("=" * 80)
    print("批量点到表面距离 ICP 配准")
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
        output_csv = f"{base_name}_point_to_surface.csv"

        try:
            result = mirror_and_register_icp_point_to_surface(
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
            import traceback
            traceback.print_exc()
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
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'point_to_surface_icp_result'

            print(f"批量处理模式")
            print(f"  输入目录: {input_dir}")
            print(f"  输出目录: {output_dir}")

            batch_process_point_to_surface_icp(input_dir=input_dir, output_dir=output_dir)
        else:
            # 单文件模式
            obj_file = sys.argv[1]
            output_csv = sys.argv[2] if len(sys.argv) > 2 else None
            output_dir = sys.argv[3] if len(sys.argv) > 3 else 'point_to_surface_icp_result'

            if output_csv is None:
                base_name = os.path.splitext(os.path.basename(obj_file))[0]
                output_csv = f"{base_name}_point_to_surface.csv"

            result = mirror_and_register_icp_point_to_surface(
                obj_path=obj_file,
                output_csv=output_csv,
                output_dir=output_dir
            )
    else:
        # 默认示例
        print("使用方法:")
        print("  单文件: python standard_icp_point_to_surface.py <obj_file> [output_csv] [output_dir]")
        print("  批量:   python standard_icp_point_to_surface.py batch [input_dir] [output_dir]")
        print()
        print("运行默认示例...")

        obj_file = "kedian/12_40_directional.obj"

        if os.path.exists(obj_file):
            result = mirror_and_register_icp_point_to_surface(
                obj_path=obj_file,
                output_csv='12_40_point_to_surface.csv',
                output_dir='point_to_surface_icp_result'
            )
        else:
            print(f"错误: 找不到测试文件 {obj_file}")
            print("请提供有效的 OBJ 文件路径")