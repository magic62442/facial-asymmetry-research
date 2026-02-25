import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

def analyze_symmetry(obj_path, plane='yz'):
    """
    分析3D模型的对称性

    参数:
        obj_path: OBJ文件路径
        plane: 对称面类型 ('yz' 表示x=0平面, 'xz' 表示y=0平面, 'xy' 表示z=0平面)

    返回:
        对称性分析结果
    """
    print(f"正在加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if not mesh.has_vertices():
        print("错误: 无法加载模型")
        return None

    vertices = np.asarray(mesh.vertices)
    print(f"顶点数: {len(vertices)}")

    # 计算模型的中心和边界
    center = vertices.mean(axis=0)
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)

    print(f"\n模型中心: {center}")
    print(f"边界范围:")
    print(f"  X: [{min_bound[0]:.3f}, {max_bound[0]:.3f}]")
    print(f"  Y: [{min_bound[1]:.3f}, {max_bound[1]:.3f}]")
    print(f"  Z: [{min_bound[2]:.3f}, {max_bound[2]:.3f}]")

    # 根据选择的平面进行镜像
    if plane == 'yz':
        # 关于YZ平面（x=0）镜像
        axis_idx = 0
        axis_name = "X"
        print(f"\n分析关于YZ平面（x=0）的对称性...")
    elif plane == 'xz':
        # 关于XZ平面（y=0）镜像
        axis_idx = 1
        axis_name = "Y"
        print(f"\n分析关于XZ平面（y=0）的对称性...")
    elif plane == 'xy':
        # 关于XY平面（z=0）镜像
        axis_idx = 2
        axis_name = "Z"
        print(f"\n分析关于XY平面（z=0）的对称性...")
    else:
        print(f"不支持的平面类型: {plane}")
        return None

    # 创建镜像顶点
    mirrored_vertices = vertices.copy()
    mirrored_vertices[:, axis_idx] = -mirrored_vertices[:, axis_idx]

    # 使用KDTree找到最近邻
    tree = KDTree(vertices)
    distances, indices = tree.query(mirrored_vertices)

    # 计算对称性度量
    mean_distance = distances.mean()
    median_distance = np.median(distances)
    max_distance = distances.max()
    std_distance = distances.std()

    print(f"\n对称性分析结果:")
    print(f"  平均距离: {mean_distance:.6f}")
    print(f"  中位数距离: {median_distance:.6f}")
    print(f"  最大距离: {max_distance:.6f}")
    print(f"  标准差: {std_distance:.6f}")

    # 判断对称性（阈值可调整）
    tolerance = 0.01  # 1mm
    symmetric_points = np.sum(distances < tolerance)
    symmetry_ratio = symmetric_points / len(vertices)

    print(f"\n对称点比例（阈值={tolerance}）: {symmetry_ratio*100:.2f}%")

    if symmetry_ratio > 0.95:
        print(f"结论: 模型关于{axis_name}轴高度对称")
    elif symmetry_ratio > 0.80:
        print(f"结论: 模型关于{axis_name}轴基本对称")
    elif symmetry_ratio > 0.50:
        print(f"结论: 模型关于{axis_name}轴部分对称")
    else:
        print(f"结论: 模型关于{axis_name}轴不对称")

    return {
        'vertices': vertices,
        'mirrored_vertices': mirrored_vertices,
        'distances': distances,
        'mean_distance': mean_distance,
        'median_distance': median_distance,
        'max_distance': max_distance,
        'symmetry_ratio': symmetry_ratio,
        'center': center,
        'axis_idx': axis_idx
    }

def find_best_symmetry_plane(obj_path):
    """
    尝试找到最佳对称面
    """
    print("=" * 60)
    print("搜索最佳对称平面...")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    # 尝试不同的对称平面
    planes = {
        'YZ平面 (X轴镜像)': 0,
        'XZ平面 (Y轴镜像)': 1,
        'XY平面 (Z轴镜像)': 2
    }

    results = {}

    for plane_name, axis_idx in planes.items():
        print(f"\n测试 {plane_name}:")
        print("-" * 40)

        # 创建镜像顶点
        mirrored_vertices = vertices.copy()
        mirrored_vertices[:, axis_idx] = -mirrored_vertices[:, axis_idx]

        # 使用KDTree找到最近邻
        tree = KDTree(vertices)
        distances, _ = tree.query(mirrored_vertices)

        mean_distance = distances.mean()
        median_distance = np.median(distances)

        tolerance = 0.01
        symmetric_points = np.sum(distances < tolerance)
        symmetry_ratio = symmetric_points / len(vertices)

        print(f"  平均距离: {mean_distance:.6f}")
        print(f"  对称点比例: {symmetry_ratio*100:.2f}%")

        results[plane_name] = {
            'mean_distance': mean_distance,
            'median_distance': median_distance,
            'symmetry_ratio': symmetry_ratio
        }

    # 找到最佳对称面
    best_plane = min(results.items(), key=lambda x: x[1]['mean_distance'])

    print("\n" + "=" * 60)
    print(f"最佳对称面: {best_plane[0]}")
    print(f"  平均距离: {best_plane[1]['mean_distance']:.6f}")
    print(f"  对称点比例: {best_plane[1]['symmetry_ratio']*100:.2f}%")
    print("=" * 60)

    return results

def visualize_symmetry(obj_path, result):
    """
    可视化对称性分析结果
    """
    print("\n准备可视化...")

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    # 创建距离的颜色映射（蓝色=对称，红色=不对称）
    distances = result['distances']
    max_dist = np.percentile(distances, 95)  # 使用95百分位数避免异常值

    # 归一化距离到[0, 1]
    normalized_dist = np.clip(distances / max_dist, 0, 1)

    # 创建颜色映射：蓝色(对称) -> 绿色 -> 红色(不对称)
    colors = np.zeros((len(vertices), 3))
    colors[:, 0] = normalized_dist  # 红色通道
    colors[:, 2] = 1 - normalized_dist  # 蓝色通道

    # 创建带颜色的mesh
    colored_mesh = o3d.geometry.TriangleMesh()
    colored_mesh.vertices = mesh.vertices
    colored_mesh.triangles = mesh.triangles
    colored_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    colored_mesh.compute_vertex_normals()

    # 创建对称面
    axis_idx = result['axis_idx']
    bounds_size = 100  # 对称面的大小

    if axis_idx == 0:  # YZ平面
        plane_points = np.array([
            [0, -bounds_size, -bounds_size],
            [0, bounds_size, -bounds_size],
            [0, bounds_size, bounds_size],
            [0, -bounds_size, bounds_size]
        ])
    elif axis_idx == 1:  # XZ平面
        plane_points = np.array([
            [-bounds_size, 0, -bounds_size],
            [bounds_size, 0, -bounds_size],
            [bounds_size, 0, bounds_size],
            [-bounds_size, 0, bounds_size]
        ])
    else:  # XY平面
        plane_points = np.array([
            [-bounds_size, -bounds_size, 0],
            [bounds_size, -bounds_size, 0],
            [bounds_size, bounds_size, 0],
            [-bounds_size, bounds_size, 0]
        ])

    # 创建平面mesh
    plane_mesh = o3d.geometry.TriangleMesh()
    plane_mesh.vertices = o3d.utility.Vector3dVector(plane_points)
    plane_mesh.triangles = o3d.utility.Vector3iVector([[0, 1, 2], [0, 2, 3]])
    plane_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    plane_mesh.compute_vertex_normals()

    print("\n颜色说明:")
    print("  蓝色 = 高度对称")
    print("  绿色 = 中等对称")
    print("  红色 = 不对称")
    print("  灰色平面 = 对称面")

    # 可视化
    o3d.visualization.draw_geometries(
        [colored_mesh, plane_mesh],
        window_name="对称性分析 (颜色表示对称程度)",
        width=1024,
        height=768,
        mesh_show_back_face=True
    )

def plot_distance_histogram(result):
    """
    绘制距离分布直方图
    """
    distances = result['distances']

    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(result['mean_distance'], color='red', linestyle='--', linewidth=2, label=f'Mean: {result["mean_distance"]:.4f}')
    plt.axvline(result['median_distance'], color='green', linestyle='--', linewidth=2, label=f'Median: {result["median_distance"]:.4f}')
    plt.xlabel('Distance from Mirrored Point to Nearest Point')
    plt.ylabel('Number of Vertices')
    plt.title('Symmetry Distance Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('symmetry_analysis.pdf', format='pdf')
    print("\n距离分布图已保存到: symmetry_analysis.pdf")

def generate_symmetry_pairs(obj_path, plane='yz', tolerance=0.01):
    """
    生成对称顶点配对（不写文件）

    参数:
        obj_path: OBJ文件路径
        plane: 对称面类型 ('yz' 表示x=0平面)
        tolerance: 配对容忍距离（mm）

    返回:
        配对信息（包含pairs, distances, nearest_indices等）
    """
    print("=" * 60)
    print("生成对称顶点配对")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    print(f"\n顶点总数: {len(vertices)}")

    # 确定镜像轴
    if plane == 'yz':
        axis_idx = 0  # X轴
        axis_name = "X"
    elif plane == 'xz':
        axis_idx = 1  # Y轴
        axis_name = "Y"
    elif plane == 'xy':
        axis_idx = 2  # Z轴
        axis_name = "Z"
    else:
        print(f"不支持的平面类型: {plane}")
        return None

    print(f"对称平面: {plane.upper()} (沿{axis_name}轴镜像)")

    # 创建镜像顶点
    mirrored_vertices = vertices.copy()
    mirrored_vertices[:, axis_idx] = -mirrored_vertices[:, axis_idx]

    # 使用KDTree找到最近邻
    tree = KDTree(vertices)
    distances, nearest_indices = tree.query(mirrored_vertices)

    # 创建配对列表
    pairs = []
    paired_vertices = set()
    conflicts = []

    for i in range(len(vertices)):
        j = nearest_indices[i]
        dist = distances[i]

        # 检查是否在容忍范围内
        if dist > tolerance:
            continue

        # 验证双向配对
        # 如果i->j是一对，那么j->i也应该是一对
        j_mirror_nearest = nearest_indices[j]

        if j_mirror_nearest == i:
            # 完美双向配对，记录 i->j
            pairs.append({
                'vertex_id_1': i,
                'vertex_id_2': j,
                'distance': dist
            })
            paired_vertices.add(i)
            if i != j:
                paired_vertices.add(j)
        else:
            # 配对冲突
            conflicts.append({
                'vertex': i,
                'nearest': j,
                'reverse_nearest': j_mirror_nearest,
                'distance': dist
            })

    # 统计信息
    print(f"\n配对统计:")
    print(f"  成功配对数量: {len(pairs)}")
    print(f"  已配对顶点数: {len(paired_vertices)}")
    print(f"  未配对顶点数: {len(vertices) - len(paired_vertices)}")
    print(f"  配对冲突数: {len(conflicts)}")

    # 检查每个顶点是否只有一个配对
    vertex_pair_count = {}
    for pair in pairs:
        v1, v2 = pair['vertex_id_1'], pair['vertex_id_2']
        vertex_pair_count[v1] = vertex_pair_count.get(v1, 0) + 1
        if v1 != v2:  # 非自对称点
            vertex_pair_count[v2] = vertex_pair_count.get(v2, 0) + 1

    multiple_pairs = {v: count for v, count in vertex_pair_count.items() if count > 1}

    if multiple_pairs:
        print(f"\n警告: 发现 {len(multiple_pairs)} 个顶点有多个配对:")
        for v, count in list(multiple_pairs.items())[:5]:  # 只显示前5个
            print(f"  顶点 {v}: {count} 个配对")
    else:
        print("\n✓ 验证通过: 每个顶点最多只有一个配对")

    # 分析配对距离
    pair_distances = [pair['distance'] for pair in pairs]
    if pair_distances:
        mean_dist = np.mean(pair_distances)
        max_dist = np.max(pair_distances)
        print(f"\n配对距离统计:")
        print(f"  平均距离: {mean_dist:.6f} mm")
        print(f"  最大距离: {max_dist:.6f} mm")

    # 按照vertex_id_1排序
    pairs.sort(key=lambda x: x['vertex_id_1'])
    print(f"\n✓ 配对列表已按vertex_id_1排序")

    return {
        'pairs': pairs,
        'paired_vertices': paired_vertices,
        'conflicts': conflicts,
        'total_pairs': len(pairs),
        'total_paired_vertices': len(paired_vertices),
        'distances': distances,
        'nearest_indices': nearest_indices,
        'vertex_count': len(vertices)
    }

def write_symmetry_pairs_csv(pair_result, output_csv='pairs.csv'):
    """
    如果模型完全对称，写CSV文件
    第一列包含所有顶点ID（每个顶点一行）

    参数:
        pair_result: generate_symmetry_pairs的返回结果
        output_csv: 输出CSV文件名

    返回:
        是否成功写入
    """
    import csv

    vertex_count = pair_result['vertex_count']
    paired_vertices = pair_result['paired_vertices']
    nearest_indices = pair_result['nearest_indices']
    distances = pair_result['distances']

    # 检查是否所有顶点都配对了
    if len(paired_vertices) != vertex_count:
        print(f"\n警告: 模型不是完全对称的")
        print(f"  总顶点数: {vertex_count}")
        print(f"  已配对顶点数: {len(paired_vertices)}")
        print(f"  未配对顶点数: {vertex_count - len(paired_vertices)}")
        print(f"  不写入CSV文件")
        return False

    print("\n模型完全对称，写入CSV文件...")

    # 写入CSV：第一列包含所有顶点ID（0到vertex_count-1）
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['source_vertex_id', 'target_vertex_id', 'distance'])

        for i in range(vertex_count):
            j = nearest_indices[i]
            dist = distances[i]
            writer.writerow([i, j, f"{dist:.6f}"])

    print(f"✓ CSV文件已保存: {output_csv}")
    print(f"  包含所有 {vertex_count} 个顶点的配对信息")

    return True

if __name__ == "__main__":
    obj_file = "Template.obj"

    # 分析关于YZ平面（x=0，左右对称）的对称性
    result = analyze_symmetry(obj_file, plane='yz')

    if result:
        # 绘制距离分布图
        plot_distance_histogram(result)

        # 如果不够对称，搜索最佳对称面
        if result['symmetry_ratio'] < 0.95:
            print("\n模型不是完全对称的，搜索其他可能的对称面...")
            find_best_symmetry_plane(obj_file)

        # 生成对称顶点配对
        print("\n")
        pair_result = generate_symmetry_pairs(obj_file, plane='yz', tolerance=0.01)

        # 如果完全对称，写入CSV文件
        if pair_result:
            write_symmetry_pairs_csv(pair_result, output_csv='pairs.csv')

        # 可视化对称性
        # visualize_symmetry(obj_file, result)