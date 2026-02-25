import open3d as o3d
import numpy as np
from scipy.spatial import KDTree

def check_displaced_symmetry(original_obj, displaced_obj):
    """
    比较原始模型和位移后模型的对称性
    """
    print("=" * 60)
    print("对比原始模型和位移后模型的对称性")
    print("=" * 60)

    # 加载模型
    mesh_orig = o3d.io.read_triangle_mesh(original_obj)
    mesh_disp = o3d.io.read_triangle_mesh(displaced_obj)

    vertices_orig = np.asarray(mesh_orig.vertices)
    vertices_disp = np.asarray(mesh_disp.vertices)

    print(f"\n原始模型: {original_obj}")
    print(f"位移模型: {displaced_obj}")

    # 分析原始模型的对称性
    print("\n--- 原始模型对称性 ---")
    analyze_x_symmetry(vertices_orig)

    # 分析位移后模型的对称性
    print("\n--- 位移后模型对称性 ---")
    analyze_x_symmetry(vertices_disp)

    # 比较位移量
    print("\n--- 位移分析 ---")
    analyze_displacement(vertices_orig, vertices_disp)

def analyze_x_symmetry(vertices):
    """
    分析关于x=0平面的对称性
    """
    # 镜像顶点（关于x=0）
    mirrored = vertices.copy()
    mirrored[:, 0] = -mirrored[:, 0]

    # 找最近邻
    tree = KDTree(vertices)
    distances, _ = tree.query(mirrored)

    mean_dist = distances.mean()
    median_dist = np.median(distances)
    max_dist = distances.max()

    tolerance = 0.01
    symmetric_count = np.sum(distances < tolerance)
    symmetry_ratio = symmetric_count / len(vertices)

    print(f"  顶点数: {len(vertices)}")
    print(f"  平均镜像距离: {mean_dist:.6f} mm")
    print(f"  中位数距离: {median_dist:.6f} mm")
    print(f"  最大距离: {max_dist:.6f} mm")
    print(f"  对称点比例: {symmetry_ratio*100:.2f}%")

def analyze_displacement(vertices_orig, vertices_disp):
    """
    分析位移的分布
    """
    displacements = vertices_disp - vertices_orig
    displacement_magnitudes = np.linalg.norm(displacements, axis=1)

    # 按X坐标分组
    left_mask = vertices_orig[:, 0] < -1.0  # 左侧
    right_mask = vertices_orig[:, 0] > 1.0  # 右侧
    center_mask = np.abs(vertices_orig[:, 0]) <= 1.0  # 中间

    print(f"  左侧顶点平均位移: {displacement_magnitudes[left_mask].mean():.6f} mm")
    print(f"  右侧顶点平均位移: {displacement_magnitudes[right_mask].mean():.6f} mm")
    print(f"  中间顶点平均位移: {displacement_magnitudes[center_mask].mean():.6f} mm")
    print(f"  总体平均位移: {displacement_magnitudes.mean():.6f} mm")
    print(f"  最大位移: {displacement_magnitudes.max():.6f} mm")

    # X方向的位移
    x_displacements = displacements[:, 0]
    print(f"\n  X方向位移统计:")
    print(f"    平均: {x_displacements.mean():.6f} mm")
    print(f"    左侧平均: {x_displacements[left_mask].mean():.6f} mm")
    print(f"    右侧平均: {x_displacements[right_mask].mean():.6f} mm")

if __name__ == "__main__":
    original = "Template.obj"

    # 测试几个位移后的模型
    test_files = [
        "displaced_models/1_10_0.1.obj",
        "displaced_models/2_10_0.3.obj",
        "displaced_models/4_10_0.3.obj",
        "displaced_models/6_10_0.3.obj",
        "displaced_models/6_25_0.3.obj",
    ]

    for disp_file in test_files:
        check_displaced_symmetry(original, disp_file)
        print("\n")