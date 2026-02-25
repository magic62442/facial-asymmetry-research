import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_face_normal(v1, v2, v3):
    """
    计算三角形面的法向量

    参数:
        v1, v2, v3: 三角形的三个顶点坐标

    返回:
        归一化的法向量
    """
    # 计算两条边向量
    edge1 = v2 - v1
    edge2 = v3 - v1

    # 叉积得到垂直于面的向量
    normal = np.cross(edge1, edge2)

    # 归一化（变成单位向量）
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm

    return normal

def compute_vertex_normals_manual(vertices, faces):
    """
    手动计算每个顶点的法向量

    方法：每个顶点的法向量 = 所有相邻面的法向量的平均
    """
    # 初始化顶点法向量数组
    vertex_normals = np.zeros_like(vertices)
    vertex_counts = np.zeros(len(vertices))

    # 遍历每个三角形面
    for face in faces:
        i0, i1, i2 = face
        v0, v1, v2 = vertices[i0], vertices[i1], vertices[i2]

        # 计算这个面的法向量
        face_normal = compute_face_normal(v0, v1, v2)

        # 将面法向量加到三个顶点上
        vertex_normals[i0] += face_normal
        vertex_normals[i1] += face_normal
        vertex_normals[i2] += face_normal

        # 记录每个顶点被多少个面共享
        vertex_counts[i0] += 1
        vertex_counts[i1] += 1
        vertex_counts[i2] += 1

    # 对每个顶点的法向量取平均并归一化
    for i in range(len(vertices)):
        if vertex_counts[i] > 0:
            vertex_normals[i] /= vertex_counts[i]
            norm = np.linalg.norm(vertex_normals[i])
            if norm > 0:
                vertex_normals[i] /= norm

    return vertex_normals

def visualize_normals(obj_path, num_normals=50):
    """
    可视化部分顶点的法向量

    参数:
        obj_path: OBJ文件路径
        num_normals: 显示多少个法向量（避免太密集）
    """
    print(f"加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 计算顶点法向量
    mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    print(f"顶点数: {len(vertices)}")
    print(f"法向量数: {len(normals)}")

    # 随机选择一些顶点来显示法向量
    indices = np.random.choice(len(vertices), size=min(num_normals, len(vertices)), replace=False)

    # 创建3D图
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制选中的顶点
    selected_vertices = vertices[indices]
    selected_normals = normals[indices]

    ax.scatter(selected_vertices[:, 0],
               selected_vertices[:, 1],
               selected_vertices[:, 2],
               c='blue', s=50, alpha=0.6, label='Vertices')

    # 绘制法向量（作为箭头）
    scale = 5.0  # 法向量的显示长度
    for i in range(len(selected_vertices)):
        v = selected_vertices[i]
        n = selected_normals[i]
        ax.quiver(v[0], v[1], v[2],
                 n[0]*scale, n[1]*scale, n[2]*scale,
                 color='red', arrow_length_ratio=0.3, linewidth=2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Vertex Normals Visualization ({num_normals} samples)')
    ax.legend()

    # 设置相同的坐标轴比例
    max_range = np.array([selected_vertices[:, 0].max() - selected_vertices[:, 0].min(),
                          selected_vertices[:, 1].max() - selected_vertices[:, 1].min(),
                          selected_vertices[:, 2].max() - selected_vertices[:, 2].min()]).max() / 2.0

    mid_x = (selected_vertices[:, 0].max() + selected_vertices[:, 0].min()) * 0.5
    mid_y = (selected_vertices[:, 1].max() + selected_vertices[:, 1].min()) * 0.5
    mid_z = (selected_vertices[:, 2].max() + selected_vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    plt.savefig('vertex_normals_visualization.pdf', format='pdf')
    print("\n法向量可视化已保存到: vertex_normals_visualization.pdf")
    plt.show()

def compare_computation_methods(obj_path):
    """
    比较Open3D自动计算和手动计算的法向量
    """
    print("=" * 60)
    print("比较法向量计算方法")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # 方法1: Open3D自动计算
    print("\n方法1: Open3D自动计算")
    mesh.compute_vertex_normals()
    normals_o3d = np.asarray(mesh.vertex_normals)
    print(f"计算完成，法向量数量: {len(normals_o3d)}")

    # 方法2: 手动计算
    print("\n方法2: 手动计算（面法向量平均）")
    normals_manual = compute_vertex_normals_manual(vertices, faces)
    print(f"计算完成，法向量数量: {len(normals_manual)}")

    # 比较结果
    difference = np.abs(normals_o3d - normals_manual)
    mean_diff = difference.mean()
    max_diff = difference.max()

    print(f"\n两种方法的差异:")
    print(f"  平均差异: {mean_diff:.6f}")
    print(f"  最大差异: {max_diff:.6f}")

    # 检查几个示例
    print(f"\n示例对比（前5个顶点）:")
    for i in range(min(5, len(vertices))):
        print(f"\n顶点 {i}:")
        print(f"  Open3D:  [{normals_o3d[i][0]:.4f}, {normals_o3d[i][1]:.4f}, {normals_o3d[i][2]:.4f}]")
        print(f"  Manual:  [{normals_manual[i][0]:.4f}, {normals_manual[i][1]:.4f}, {normals_manual[i][2]:.4f}]")
        print(f"  差异:    {np.linalg.norm(normals_o3d[i] - normals_manual[i]):.6f}")

def analyze_single_face(obj_path, face_idx=0):
    """
    分析单个三角形面的法向量计算
    """
    print("\n" + "=" * 60)
    print(f"分析单个三角形面 (索引: {face_idx})")
    print("=" * 60)

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    # 获取三角形的三个顶点
    face = faces[face_idx]
    v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

    print(f"\n三角形顶点索引: {face}")
    print(f"\n顶点坐标:")
    print(f"  v0 (索引 {face[0]}): [{v0[0]:.4f}, {v0[1]:.4f}, {v0[2]:.4f}]")
    print(f"  v1 (索引 {face[1]}): [{v1[0]:.4f}, {v1[1]:.4f}, {v1[2]:.4f}]")
    print(f"  v2 (索引 {face[2]}): [{v2[0]:.4f}, {v2[1]:.4f}, {v2[2]:.4f}]")

    # 计算边向量
    edge1 = v1 - v0
    edge2 = v2 - v0

    print(f"\n边向量:")
    print(f"  edge1 (v1-v0): [{edge1[0]:.4f}, {edge1[1]:.4f}, {edge1[2]:.4f}]")
    print(f"  edge2 (v2-v0): [{edge2[0]:.4f}, {edge2[1]:.4f}, {edge2[2]:.4f}]")

    # 计算叉积
    cross_product = np.cross(edge1, edge2)
    print(f"\n叉积结果: [{cross_product[0]:.4f}, {cross_product[1]:.4f}, {cross_product[2]:.4f}]")
    print(f"叉积长度: {np.linalg.norm(cross_product):.4f}")

    # 归一化得到法向量
    face_normal = compute_face_normal(v0, v1, v2)
    print(f"\n归一化后的面法向量: [{face_normal[0]:.4f}, {face_normal[1]:.4f}, {face_normal[2]:.4f}]")
    print(f"法向量长度: {np.linalg.norm(face_normal):.4f} (应该为1.0)")

def get_landmark_normal(obj_path, csv_path, landmark_idx):
    """
    获取指定landmark点的法向量

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径
        landmark_idx: landmark点的索引（从0开始）

    返回:
        法向量
    """
    from scipy.spatial import KDTree

    # 1. 读取landmark点
    landmarks = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            coords = [float(x.strip()) for x in line.split(',')]
            if len(coords) == 3:
                landmarks.append(coords)

    landmarks = np.array(landmarks)

    if landmark_idx >= len(landmarks):
        print(f"错误: landmark索引{landmark_idx}超出范围（共{len(landmarks)}个点）")
        return None

    landmark_point = landmarks[landmark_idx]
    print(f"Landmark点{landmark_idx}的坐标: [{landmark_point[0]:.6f}, {landmark_point[1]:.6f}, {landmark_point[2]:.6f}]")

    # 2. 加载3D模型
    print(f"\n加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 3. 计算顶点法向量
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    print(f"模型顶点数: {len(vertices)}")

    # 4. 找到离landmark点最近的顶点
    tree = KDTree(vertices)
    distance, nearest_vertex_idx = tree.query(landmark_point)

    nearest_vertex = vertices[nearest_vertex_idx]
    normal = normals[nearest_vertex_idx]

    print(f"\n最近的模型顶点:")
    print(f"  索引: {nearest_vertex_idx}")
    print(f"  坐标: [{nearest_vertex[0]:.6f}, {nearest_vertex[1]:.6f}, {nearest_vertex[2]:.6f}]")
    print(f"  与landmark点的距离: {distance:.6f}")

    print(f"\n该点的法向量:")
    print(f"  [{normal[0]:.6f}, {normal[1]:.6f}, {normal[2]:.6f}]")
    print(f"  法向量长度: {np.linalg.norm(normal):.6f} (应该接近1.0)")

    return normal

if __name__ == "__main__":
    obj_file = "Template.obj"

    print("法向量计算演示\n")

    # 1. 分析单个三角形面
    analyze_single_face(obj_file, face_idx=0)

    # 2. 比较计算方法
    compare_computation_methods(obj_file)

    # 3. 可视化法向量
    print("\n生成法向量可视化...")
    visualize_normals(obj_file, num_normals=100)

    # 4. 获取第三个landmark点的法向量
    print("\n" + "=" * 60)
    print("获取第3个landmark点的法向量")
    print("=" * 60 + "\n")
    csv_file = "template landmark.csv"
    normal = get_landmark_normal(obj_file, csv_file, landmark_idx=2)