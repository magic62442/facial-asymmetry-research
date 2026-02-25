import open3d as o3d
import numpy as np
import os

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

def rbf_wendland_displacement(obj_path, csv_path, landmark_idx, r, k, sigma=1, direction=None, output_dir="output"):
    """
    对3D模型进行基于RBF Wendland C2函数的形变

    Wendland C2权重函数: w(r) = (1 - r)^4 * (4r + 1)
    - 当 r=0 (中心) 时 w=1
    - 当 r=1 (边界) 时 w=0
    - 平滑过渡，C2连续

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径（landmark坐标）
        landmark_idx: landmark点的索引（从0开始）
        r: 影响半径（mm）
        k: 位移比例
        sigma: 位移符号（+1=右移，-1=左移），默认+1
        direction: 位移方向，默认(1, 0, 0)
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

    center = landmarks[landmark_idx]  # 中心点
    print(f"中心点 (landmark {landmark_idx}): [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")

    # 2. 加载3D模型
    print(f"加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 3. 获取顶点
    vertices = np.asarray(mesh.vertices).copy()

    print(f"模型顶点数: {len(vertices)}")

    # 4. 计算参数
    displacement_magnitude = r * k  # 位移长度 = r * k

    print(f"\n参数:")
    print(f"  r (影响半径): {r} mm")
    print(f"  k (位移比例): {k}")
    print(f"  位移长度: {displacement_magnitude:.3f} mm")
    print(f"  sigma (符号): {sigma:+d}")
    print(f"  方向: [{direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f}]")
    print(f"  权重函数: Wendland C2")

    # 5. 计算每个点到中心的距离
    diffs = vertices - center
    dists = np.linalg.norm(diffs, axis=1)

    # 6. 筛选在影响半径内的点
    mask = dists < r
    num_affected = np.sum(mask)

    if num_affected == 0:
        print(f"\n警告: 没有顶点在影响半径内")
        return None

    print(f"  影响范围内的顶点数: {num_affected} / {len(vertices)}")

    # 7. 计算归一化距离 (0到1)
    r_normalized = dists[mask] / r

    # 8. 计算Wendland C2权重
    # w(r) = (1 - r)^4 * (4r + 1)
    weights = np.power(1 - r_normalized, 4) * (4 * r_normalized + 1)

    max_weight = weights.max()
    mean_weight = weights.mean()
    print(f"  最大权重: {max_weight:.6f}")
    print(f"  平均权重: {mean_weight:.6f}")

    # 9. 应用位移
    # 变形量 = sigma * 权重 * 位移长度 * 方向
    displacement = sigma * weights[:, np.newaxis] * displacement_magnitude * direction

    # 更新顶点位置
    vertices[mask] += displacement

    print(f"\n位移统计:")
    print(f"  受影响的顶点数: {num_affected} / {len(vertices)}")
    print(f"  最大实际位移: {max_weight * displacement_magnitude:.6f} mm")
    print(f"  平均实际位移: {mean_weight * displacement_magnitude:.6f} mm")

    # 10. 更新mesh并保存
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()  # 重新计算法向量

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 生成文件名
    output_filename = f"{r}_{k}.obj"
    output_path = os.path.join(output_dir, output_filename)

    # 保存OBJ文件
    success = o3d.io.write_triangle_mesh(output_path, mesh)

    if success:
        print(f"\n已保存到: {output_path}")
        return output_path
    else:
        print(f"\n保存失败!")
        return None

def batch_generate_rbf(obj_path, csv_path, landmark_idx, r_values, k_values, output_dir="output"):
    """
    批量生成多个配置的RBF Wendland形变模型

    参数:
        obj_path: 输入OBJ文件路径
        csv_path: CSV文件路径
        landmark_idx: landmark点的索引
        r_values: 影响半径列表
        k_values: 位移比例列表
        output_dir: 输出目录
    """
    print("=" * 80)
    print("批量生成RBF Wendland形变模型")
    print("=" * 80)

    total = len(r_values) * len(k_values)
    count = 0

    results = []

    for r in r_values:
        for k in k_values:
            count += 1
            print(f"\n{'='*80}")
            print(f"进度: {count}/{total} - r={r}, k={k}")
            print(f"{'='*80}")

            output_path = rbf_wendland_displacement(
                obj_path=obj_path,
                csv_path=csv_path,
                landmark_idx=landmark_idx,
                r=r,
                k=k,
                sigma=1,  # 默认向右
                direction=[1.0, 0.0, 0.0],  # 默认+x方向
                output_dir=output_dir
            )

            if output_path:
                results.append({
                    'r': r,
                    'k': k,
                    'file': output_path
                })

    print("\n" + "=" * 80)
    print(f"批量生成完成! 共生成 {len(results)} 个文件")
    print("=" * 80)

    print("\n生成的文件列表:")
    for result in results:
        print(f"  r={result['r']}, k={result['k']} -> {result['file']}")

    return results

if __name__ == "__main__":
    # 输入文件
    obj_file = "Template.obj"
    csv_file = "template landmark.csv"
    landmark_index = 2  # 第三个点（索引从0开始）

    # 参数配置
    r_values = [10, 15, 20, 25]  # 影响半径
    k_values = [0.1, 0.15, 0.2, 0.25, 0.3]  # 位移比例

    # 输出目录
    output_directory = "displaced_rbf"

    # 批量生成
    results = batch_generate_rbf(
        obj_path=obj_file,
        csv_path=csv_file,
        landmark_idx=landmark_index,
        r_values=r_values,
        k_values=k_values,
        output_dir=output_directory
    )

    print(f"\n所有文件已保存到目录: {output_directory}")