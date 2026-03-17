import open3d as o3d
import numpy as np
from scipy.spatial import KDTree
import os
from view_template import mirror_obj_file


def local_laplacian_smooth(mesh, center_y, y_range=10.0, iterations=5, lambda_factor=0.5):
    """
    对网格进行局部拉普拉斯平滑，只平滑 y 值在 [center_y - y_range, center_y + y_range] 范围内的顶点

    参数:
        mesh: Open3D TriangleMesh 对象
        center_y: 中心 y 坐标（如颏点的 y 值）
        y_range: y 方向的平滑范围（±y_range）
        iterations: 平滑迭代次数
        lambda_factor: 平滑强度（0-1），越大平滑越强

    返回:
        平滑后的 mesh
    """
    vertices = np.asarray(mesh.vertices).copy()
    triangles = np.asarray(mesh.triangles)

    # 构建邻接表：每个顶点的邻居顶点列表
    n_vertices = len(vertices)
    adjacency = [set() for _ in range(n_vertices)]

    for tri in triangles:
        v0, v1, v2 = tri
        adjacency[v0].add(v1)
        adjacency[v0].add(v2)
        adjacency[v1].add(v0)
        adjacency[v1].add(v2)
        adjacency[v2].add(v0)
        adjacency[v2].add(v1)

    # 找出需要平滑的顶点（y 值在范围内）
    smooth_mask = np.abs(vertices[:, 1] - center_y) <= y_range
    smooth_indices = np.where(smooth_mask)[0]

    print(f"  局部平滑: {len(smooth_indices)} / {n_vertices} 个顶点 (y ∈ [{center_y - y_range:.1f}, {center_y + y_range:.1f}])")

    # 迭代平滑
    for iteration in range(iterations):
        new_vertices = vertices.copy()

        for i in smooth_indices:
            neighbors = list(adjacency[i])
            if len(neighbors) == 0:
                continue

            # 计算邻居的平均位置
            neighbor_avg = vertices[neighbors].mean(axis=0)

            # 拉普拉斯平滑：向邻居平均位置移动
            new_vertices[i] = vertices[i] + lambda_factor * (neighbor_avg - vertices[i])

        vertices = new_vertices

    # 更新 mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh


def load_landmarks_from_csv(csv_path):
    """
    从CSV文件读取landmark坐标点
    """
    landmarks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = [float(x.strip()) for x in line.split(',')]
            if len(coords) == 3:
                landmarks.append(coords)
    return np.array(landmarks)

def gaussian_normal_displacement(obj_path, csv_path, landmark_idx, A, r, sigma=1, output_dir="output"):
    """
    对3D模型进行高斯法向位移

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径（landmark坐标）
        landmark_idx: landmark点的索引（从0开始）
        A: 幅度（mm）
        r: 核半径（mm）
        sigma: 位移符号（+1=外鼓，-1=内凹），默认+1
        cutoff_factor: 距离截断系数，只影响距离 <= cutoff_factor*r 的点，默认3.0
        output_dir: 输出目录

    返回:
        输出文件路径
    """
    # 1. 读取landmark点
    landmarks = load_landmarks_from_csv(csv_path)
    if landmark_idx >= len(landmarks):
        print(f"错误: landmark索引{landmark_idx}超出范围（共{len(landmarks)}个点）")
        return None

    c = landmarks[landmark_idx]  # 中心点
    print(f"中心点 c (landmark {landmark_idx}): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")

    # 2. 加载3D模型
    print(f"加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 3. 计算顶点法向量
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices).copy()
    normals = np.asarray(mesh.vertex_normals)

    print(f"模型顶点数: {len(vertices)}")

    # 4. 计算参数
    cutoff_distance = 60.0  # 截断距离

    print(f"\n参数:")
    print(f"  A (幅度): {A} mm")
    print(f"  r (核半径): {r} mm")
    print(f"  sigma (符号): {sigma:+d}")
    print(f"  截断距离: {cutoff_distance:.3f} mm (只影响距离 <= {cutoff_distance:.1f}mm 的点)")

    # 5. 逐点计算位移（添加距离截断）
    num_affected = 0
    num_skipped = 0
    max_displacement = 0.0

    for i in range(len(vertices)):
        v_i = vertices[i]
        n_i = normals[i]

        # 计算径向距离
        d_i = np.linalg.norm(v_i - c)

        # 距离截断：跳过距离超过截断距离的点
        if d_i > cutoff_distance:
            num_skipped += 1
            continue

        # 计算高斯权重
        w_i = np.exp(-0.5 * (d_i / r) ** 2)

        # 计算位移
        delta_v_i = sigma * A * w_i * n_i

        # 更新顶点
        vertices[i] = v_i + delta_v_i

        # 统计
        displacement_magnitude = np.linalg.norm(delta_v_i)
        if displacement_magnitude > 0.001:  # 大于0.001mm才算有效位移
            num_affected += 1
        max_displacement = max(max_displacement, displacement_magnitude)

    print(f"\n位移统计:")
    print(f"  截断距离内的顶点数: {len(vertices) - num_skipped} / {len(vertices)}")
    print(f"  跳过的顶点数: {num_skipped} / {len(vertices)}")
    print(f"  实际受影响的顶点数: {num_affected} / {len(vertices)} ({num_affected/len(vertices)*100:.2f}%)")
    print(f"  最大位移量: {max_displacement:.6f} mm")

    # 6. 更新mesh并保存
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # 重新计算法向量

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    output_filename = f"{A}_{r}.obj"
    output_path = os.path.join(output_dir, output_filename)

    # 保存OBJ文件
    success = o3d.io.write_triangle_mesh(output_path, mesh)

    if success:
        print(f"\n已保存到: {output_path}")

        # 同时生成镜像文件
        name, ext = os.path.splitext(output_path)
        mirrored_path = f"{name}_mirrored{ext}"
        mirror_obj_file(output_path, mirrored_path, plane='x')

        return output_path
    else:
        print(f"\n保存失败!")
        return None

def gaussian_directional_displacement(obj_path, csv_path, landmark_idx, A, r, sigma=1,
                                      direction=None, output_dir="output", cutoff_distance=40.0):
    """
    对3D模型进行高斯方向性位移（所有点沿固定方向位移）

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径（landmark坐标）
        landmark_idx: landmark点的索引（从0开始）
        A: 幅度（mm）
        r: 核半径（mm）
        sigma: 位移符号（+1=外鼓，-1=内凹），默认+1
        direction: 位移方向，默认(1, 0, 0)
        cutoff_factor: 距离截断系数，只影响距离 <= cutoff_factor*r 的点，默认3.0
        output_dir: 输出目录

    返回:
        输出文件路径
    """
    # 默认参数
    if direction is None:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)  # 归一化

    # 1. 读取landmark点
    landmarks = load_landmarks_from_csv(csv_path)
    if landmark_idx >= len(landmarks):
        print(f"错误: landmark索引{landmark_idx}超出范围（共{len(landmarks)}个点）")
        return None

    c = landmarks[landmark_idx]  # 中心点
    print(f"中心点 c (landmark {landmark_idx}): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")

    # 2. 加载3D模型
    print(f"加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 3. 获取顶点（不需要计算法向量）
    vertices = np.asarray(mesh.vertices).copy()

    print(f"模型顶点数: {len(vertices)}")

    # 4. 计算参数
    print(f"\n参数:")
    print(f"  A (幅度): {A} mm")
    print(f"  r (核半径): {r} mm")
    print(f"  sigma (符号): {sigma:+d}")
    print(f"  方向: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
    print(f"  截断距离: {cutoff_distance:.3f} mm (只影响距离 <= {cutoff_distance:.1f}mm 的点)")

    # 5. 逐点计算位移（添加距离截断）
    num_affected = 0
    num_skipped = 0
    max_weight = 0.0

    for i in range(len(vertices)):
        v_i = vertices[i]

        # 计算径向距离
        d_i = np.linalg.norm(v_i - c)

        # 距离截断：跳过距离超过截断距离的点
        if d_i > cutoff_distance:
            num_skipped += 1
            continue

        # 计算高斯权重
        w_i = np.exp(-0.5 * (d_i / r) ** 2)

        # 计算位移：权重 * 幅度 * 方向
        delta_v_i = sigma * w_i * A * direction

        # 更新顶点
        vertices[i] = v_i + delta_v_i

        # 统计
        num_affected += 1
        max_weight = max(max_weight, w_i)

    print(f"\n位移统计:")
    print(f"  截断距离内的顶点数: {num_affected} / {len(vertices)} ({num_affected/len(vertices)*100:.2f}%)")
    print(f"  跳过的顶点数: {num_skipped} / {len(vertices)} ({num_skipped/len(vertices)*100:.2f}%)")
    print(f"  最大权重: {max_weight:.6f}")
    print(f"  最大实际位移: {max_weight * A:.6f} mm")

    # 6. 更新mesh并保存
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # 重新计算法向量

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    output_filename = f"{A}_{int(cutoff_distance)}_directional.obj"
    output_path = os.path.join(output_dir, output_filename)

    # 保存OBJ文件
    success = o3d.io.write_triangle_mesh(output_path, mesh)

    if success:
        print(f"\n已保存到: {output_path}")

        # 同时生成镜像文件
        name, ext = os.path.splitext(output_path)
        mirrored_path = f"{name}_mirrored{ext}"
        mirror_obj_file(output_path, mirrored_path, plane='x')

        return output_path
    else:
        print(f"\n保存失败!")
        return None

def batch_generate(obj_path, csv_path, landmark_idx, A_values, r_values, output_dir="output"):
    """
    批量生成多个配置的位移模型（法向位移）

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径
        landmark_idx: landmark点的索引
        A_values: 幅度列表
        r_values: 核半径列表
        output_dir: 输出目录
    """
    print("=" * 80)
    print("批量生成高斯法向位移模型")
    print("=" * 80)

    total = len(A_values) * len(r_values)
    count = 0

    results = []

    for A in A_values:
        for r in r_values:
            count += 1
            print(f"\n{'='*80}")
            print(f"进度: {count}/{total} - A={A}, r={r}")
            print(f"{'='*80}")

            output_path = gaussian_normal_displacement(
                obj_path=obj_path,
                csv_path=csv_path,
                landmark_idx=landmark_idx,
                A=A,
                r=r,
                sigma=1,  # 默认外鼓
                output_dir=output_dir
            )

            if output_path:
                results.append({
                    'A': A,
                    'r': r,
                    'file': output_path
                })

    print("\n" + "=" * 80)
    print(f"批量生成完成! 共生成 {len(results)} 个文件")
    print("=" * 80)

    print("\n生成的文件列表:")
    for result in results:
        print(f"  A={result['A']}, r={result['r']} -> {result['file']}")

    return results

def gaussian_directional_displacement_y_distance(obj_path, csv_path, landmark_idx, A, r, sigma=1,
                                                  direction=None, output_dir="output", cutoff_distance=40.0,
                                                  x_decay_radius=15.0, apply_x_decay=True):
    """
    对3D模型进行高斯方向性位移（所有点沿固定方向位移）
    距离计算只使用y方向的距离
    x > 0 的点会有额外的衰减（可选）

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径（landmark坐标）
        landmark_idx: landmark点的索引（从0开始）
        A: 幅度（mm）
        r: 核半径（mm）
        sigma: 位移符号（+1=外鼓，-1=内凹），默认+1
        direction: 位移方向，默认(1, 0, 0)
        cutoff_distance: 距离截断（只针对y方向距离）
        output_dir: 输出目录
        x_decay_radius: x方向衰减半径（mm），x>0时的高斯衰减参数
        apply_x_decay: 是否应用x>0的衰减，默认True

    返回:
        输出文件路径
    """
    # 默认参数
    if direction is None:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)  # 归一化

    # 1. 读取landmark点
    landmarks = load_landmarks_from_csv(csv_path)
    if landmark_idx >= len(landmarks):
        print(f"错误: landmark索引{landmark_idx}超出范围（共{len(landmarks)}个点）")
        return None

    c = landmarks[landmark_idx]  # 中心点
    print(f"中心点 c (landmark {landmark_idx}): [{c[0]:.6f}, {c[1]:.6f}, {c[2]:.6f}]")

    # 2. 加载3D模型
    print(f"加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 3. 获取顶点（不需要计算法向量）
    vertices = np.asarray(mesh.vertices).copy()

    print(f"模型顶点数: {len(vertices)}")

    # 4. 计算参数
    print(f"\n参数:")
    print(f"  A (幅度): {A} mm")
    print(f"  r (核半径): {r} mm")
    print(f"  sigma (符号): {sigma:+d}")
    print(f"  方向: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
    print(f"  截断距离 (y方向): {cutoff_distance:.3f} mm")
    print(f"  x方向衰减: {'启用' if apply_x_decay else '禁用'}")
    if apply_x_decay:
        print(f"  x方向衰减半径: {x_decay_radius:.3f} mm (x>0时生效)")

    # 5. 逐点计算位移（只使用y方向距离，x>0时额外衰减）
    num_affected = 0
    num_skipped = 0
    max_weight = 0.0

    for i in range(len(vertices)):
        v_i = vertices[i]

        # 只计算y方向距离
        d_i = abs(v_i[1] - c[1])

        # 距离截断：跳过y方向距离超过截断距离的点
        if d_i > cutoff_distance:
            num_skipped += 1
            continue

        # 计算高斯权重（基于y方向距离）
        w_i = np.exp(-0.5 * (d_i / r) ** 2)

        # x > 0 时添加额外的高斯衰减（如果启用）
        if apply_x_decay:
            x_val = v_i[0]
            if x_val > 0:
                x_decay = np.exp(-0.5 * (x_val / x_decay_radius) ** 2)
                w_i = w_i * x_decay

        # 计算位移：权重 * 幅度 * 方向
        delta_v_i = sigma * w_i * A * direction

        # 更新顶点
        vertices[i] = v_i + delta_v_i

        # 统计
        num_affected += 1
        max_weight = max(max_weight, w_i)

    print(f"\n位移统计 (y方向距离模式):")
    print(f"  截断距离内的顶点数: {num_affected} / {len(vertices)} ({num_affected/len(vertices)*100:.2f}%)")
    print(f"  跳过的顶点数: {num_skipped} / {len(vertices)} ({num_skipped/len(vertices)*100:.2f}%)")
    print(f"  最大权重: {max_weight:.6f}")
    print(f"  最大实际位移: {max_weight * A:.6f} mm")

    # 6. 更新mesh并保存
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # 重新计算法向量

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    output_filename = f"{A}_{int(cutoff_distance)}_directional.obj"
    output_path = os.path.join(output_dir, output_filename)

    # 保存OBJ文件
    success = o3d.io.write_triangle_mesh(output_path, mesh)

    if success:
        print(f"\n已保存到: {output_path}")

        # 同时生成镜像文件
        name, ext = os.path.splitext(output_path)
        mirrored_path = f"{name}_mirrored{ext}"
        mirror_obj_file(output_path, mirrored_path, plane='x')

        return output_path
    else:
        print(f"\n保存失败!")
        return None


def batch_generate_directional_y_distance(obj_path, csv_path, landmark_idx, A_values, r_values, output_dir="output",
                                          cutoff_distance=40.0, x_decay_radius=15.0, apply_x_decay=True):
    """
    批量生成多个配置的方向性位移模型（使用y方向距离）

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径
        landmark_idx: landmark点的索引
        A_values: 幅度列表
        r_values: 核半径列表
        output_dir: 输出目录
        cutoff_distance: y方向距离截断
        x_decay_radius: x方向衰减半径（mm），x>0时的高斯衰减参数
        apply_x_decay: 是否应用x>0的衰减，默认True
    """
    print("=" * 80)
    print("批量生成高斯方向性位移模型 (y方向距离模式)")
    print(f"  x方向衰减: {'启用' if apply_x_decay else '禁用'}")
    print("=" * 80)

    total = len(A_values) * len(r_values)
    count = 0

    results = []

    for A in A_values:
        for r in r_values:
            count += 1
            print(f"\n{'='*80}")
            print(f"进度: {count}/{total} - A={A}, r={r}")
            print(f"{'='*80}")

            output_path = gaussian_directional_displacement_y_distance(
                obj_path=obj_path,
                csv_path=csv_path,
                landmark_idx=landmark_idx,
                A=A,
                r=r,
                sigma=1,  # 默认外鼓
                direction=[1.0, 0.0, 0.0],  # 默认+x方向
                output_dir=output_dir,
                cutoff_distance=cutoff_distance,
                x_decay_radius=A*1.2,
                apply_x_decay=apply_x_decay
            )

            if output_path:
                results.append({
                    'A': A,
                    'r': r,
                    'file': output_path
                })

    print("\n" + "=" * 80)
    print(f"批量生成完成! 共生成 {len(results)} 个文件")
    print("=" * 80)

    print("\n生成的文件列表:")
    for result in results:
        print(f"  A={result['A']}, r={result['r']} -> {result['file']}")

    return results


def batch_generate_directional(obj_path, csv_path, landmark_idx, A_values, r_values, output_dir="output",
                               cutoff_distance=40.0):
    """
    批量生成多个配置的方向性位移模型

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径
        landmark_idx: landmark点的索引
        A_values: 幅度列表
        r_values: 核半径列表
        output_dir: 输出目录
    """
    print("=" * 80)
    print("批量生成高斯方向性位移模型")
    print("=" * 80)

    total = len(A_values) * len(r_values)
    count = 0

    results = []

    for A in A_values:
        for r in r_values:
            count += 1
            print(f"\n{'='*80}")
            print(f"进度: {count}/{total} - A={A}, r={r}")
            print(f"{'='*80}")

            output_path = gaussian_directional_displacement(
                obj_path=obj_path,
                csv_path=csv_path,
                landmark_idx=landmark_idx,
                A=A,
                r=r,
                sigma=1,  # 默认外鼓
                direction=[1.0, 0.0, 0.0],  # 默认+x方向
                output_dir=output_dir,
                cutoff_distance=cutoff_distance
            )

            if output_path:
                results.append({
                    'A': A,
                    'r': r,
                    'file': output_path
                })

    print("\n" + "=" * 80)
    print(f"批量生成完成! 共生成 {len(results)} 个文件")
    print("=" * 80)

    print("\n生成的文件列表:")
    for result in results:
        print(f"  A={result['A']}, r={result['r']} -> {result['file']}")

    return results

if __name__ == "__main__":
    # 输入文件
    obj_file = "Template.obj"
    csv_file = "bijian.csv"
    landmark_index = 0
    cutoff_distance = 50.0
    output_directory = "bijian"

    # 参数配置
    # A_values = [1, 2, 3, 4, 5, 6]  # 幅度
    if csv_file == "kedian.csv":
        A_values = [5, 7]  # 幅度
        r_values = [15]  # 核半径
        cutoff_distance = 40.0
        x_decay_radius = 10.0  # x方向衰减半径，越小衰减越快
        landmark_index = 0

        # kedian 使用 y 方向距离模式（带x>0衰减）
        output_directory = "kedian"
        results = batch_generate_directional_y_distance(
            obj_path=obj_file,
            csv_path=csv_file,
            landmark_idx=landmark_index,
            A_values=A_values,
            r_values=r_values,
            output_dir=output_directory,
            cutoff_distance=cutoff_distance,
            x_decay_radius=x_decay_radius,
            apply_x_decay=True
        )

        # kedian2 使用 y 方向距离模式（不带x>0衰减）
        # output_directory = "kedian2"
        # results = batch_generate_directional_y_distance(
        #     obj_path=obj_file,
        #     csv_path=csv_file,
        #     landmark_idx=landmark_index,
        #     A_values=A_values,
        #     r_values=r_values,
        #     output_dir=output_directory,
        #     cutoff_distance=cutoff_distance,
        #     x_decay_radius=x_decay_radius,
        #     apply_x_decay=False  # 不使用x>0衰减
        # )
    elif csv_file == "xiahedian.csv":
        A_values = [8]  # 幅度
        r_values = [20]  # 核半径
        cutoff_distance = 70.0
        output_directory = "xiahedian"
        landmark_index = 0
        results = batch_generate_directional(
            obj_path=obj_file,
            csv_path=csv_file,
            landmark_idx=landmark_index,
            A_values=A_values,
            r_values=r_values,
            output_dir=output_directory,
            cutoff_distance=cutoff_distance
        )
    else:
        # 默认使用普通方向性位移
        A_values = [7, 8]
        r_values = [15]
        results = batch_generate_directional(
            obj_path=obj_file,
            csv_path=csv_file,
            landmark_idx=landmark_index,
            A_values=A_values,
            r_values=r_values,
            output_dir=output_directory,
            cutoff_distance=cutoff_distance
        )
    print(f"\n所有文件已保存到目录: {output_directory}")