import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

def load_landmarks_from_csv(csv_path):
    """
    从CSV文件读取landmark坐标点

    参数:
        csv_path: CSV文件路径

    返回:
        numpy数组，形状为(N, 3)，包含N个3D坐标点
    """
    landmarks = []

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # 解析格式："x,y,z"
                coords = [float(x.strip()) for x in line.split(',')]
                if len(coords) == 3:
                    landmarks.append(coords)

        return np.array(landmarks)
    except Exception as e:
        print(f"读取CSV文件出错: {e}")
        return np.array([])

def rotation_matrix_to_euler_angles(R):
    """
    将旋转矩阵转换为欧拉角 (度)

    参数:
        R: 3x3 旋转矩阵

    返回:
        (rx, ry, rz): 绕X、Y、Z轴的旋转角度（度）
    """
    import math

    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)

    singular = sy < 1e-6

    if not singular:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0

    # 转换为度
    return (math.degrees(rx), math.degrees(ry), math.degrees(rz))


def view_3d_model(obj_path, csv_path=None):
    """
    加载并可视化3D模型和landmark点

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径（可选），包含landmark坐标

    交互操作:
        - 鼠标左键拖动: 旋转模型
        - 鼠标右键拖动: 平移模型
        - 滚轮: 缩放
        - R: 重置视角
        - P: 打印当前视角参数（旋转角度、相机位置等）
        - S: 保存当前视角截图
        - Q/ESC: 退出
    """
    print(f"正在加载模型: {obj_path}")

    # 加载OBJ文件
    mesh = o3d.io.read_triangle_mesh(obj_path)

    # 检查是否成功加载
    if not mesh.has_vertices():
        print("错误: 无法加载模型或模型为空")
        return

    print(f"模型加载成功!")
    print(f"顶点数: {len(mesh.vertices)}")
    print(f"三角面数: {len(mesh.triangles)}")

    # 计算顶点法线
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # 参考MATLAB: light('Position', [0 0 1000]) - 单光源从正前方
    light_dir = np.array([0.0, 0.0, 1.0])

    # 计算光照强度 (material dull - 无高光，只有漫反射)
    ambient = 0.3  # 环境光
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot

    base_color = np.array([1.0, 1.0, 1.0])
    colors = np.outer(intensities, base_color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 创建要显示的几何对象列表
    geometries = [mesh]

    # 如果提供了CSV文件，加载并显示landmark点
    if csv_path:
        print(f"\n正在加载landmark点: {csv_path}")
        landmarks = load_landmarks_from_csv(csv_path)

        if len(landmarks) > 0:
            print(f"Landmark点加载成功! 共 {len(landmarks)} 个点")

            # 创建点云对象
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(landmarks)

            # 设置所有点为红色
            colors = np.array([[1.0, 0.0, 0.0]] * len(landmarks))
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            # 添加到可视化列表
            geometries.append(point_cloud)
        else:
            print("未能从CSV文件中加载任何landmark点")

    # 创建可视化窗口
    print("\n可视化窗口控制:")
    print("  - 鼠标左键拖动: 旋转模型")
    print("  - 鼠标右键拖动: 平移模型")
    print("  - 滚轮: 缩放")
    print("  - R: 重置视角")
    print("  - P: 打印当前视角参数（旋转角度）")
    print("  - S: 保存当前视角截图")
    print("  - Q/ESC: 退出")

    # 使用 VisualizerWithKeyCallback 以便注册按键回调
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="3D Face Template Viewer", width=1024, height=768)

    for geom in geometries:
        vis.add_geometry(geom)

    # 设置渲染选项
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

    # 设置正交投影
    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(step=-90)  # 减小FOV接近正交投影

    # 定义打印视角参数的回调函数
    def print_view_params(vis):
        """按P键时打印当前视角参数"""
        import math

        view_ctrl = vis.get_view_control()

        # 临时切换到透视投影以获取相机参数
        # 保存当前缩放级别
        view_ctrl.change_field_of_view(step=90)  # 恢复透视视图

        try:
            cam_params = view_ctrl.convert_to_pinhole_camera_parameters()
            extrinsic = np.asarray(cam_params.extrinsic).copy()
            rotation_matrix = extrinsic[:3, :3]
            translation = extrinsic[:3, 3]

            # 从旋转矩阵计算欧拉角
            rx, ry, rz = rotation_matrix_to_euler_angles(rotation_matrix)

            # 计算相机的front向量（视线方向）= 旋转矩阵的第三行的负值
            front = -rotation_matrix[2, :]

            # 计算方位角和仰角
            azimuth = math.degrees(math.atan2(front[0], front[2]))
            elevation = math.degrees(math.asin(np.clip(-front[1], -1, 1)))

            print("\n" + "=" * 50)
            print("当前视角参数:")
            print("=" * 50)
            print(f"欧拉角 (度):")
            print(f"  绕X轴旋转 (Pitch): {rx:.2f}°")
            print(f"  绕Y轴旋转 (Yaw):   {ry:.2f}°")
            print(f"  绕Z轴旋转 (Roll):  {rz:.2f}°")
            print(f"\n视角角度:")
            print(f"  方位角 (水平): {azimuth:.2f}°")
            print(f"  仰角 (垂直):   {elevation:.2f}°")
            print(f"\n相机位置 (平移):")
            print(f"  X: {translation[0]:.4f}")
            print(f"  Y: {translation[1]:.4f}")
            print(f"  Z: {translation[2]:.4f}")
            print(f"\n视线方向 (Front): [{front[0]:.4f}, {front[1]:.4f}, {front[2]:.4f}]")
            print(f"\n旋转矩阵:")
            print(rotation_matrix)
            print("=" * 50)
        except Exception as e:
            print(f"获取视角参数失败: {e}")
        finally:
            # 恢复正交投影
            view_ctrl.change_field_of_view(step=-90)

        return False  # 返回False以继续运行

    # 定义保存截图的回调函数
    def save_screenshot(vis):
        """按S键时保存截图"""
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"screenshot_{timestamp}.png"
        vis.capture_screen_image(filename, do_render=True)
        print(f"\n截图已保存: {filename}")
        return False

    # 注册按键回调 (P键 = ASCII 80, S键 = ASCII 83)
    vis.register_key_callback(80, print_view_params)  # P键
    vis.register_key_callback(83, save_screenshot)    # S键

    vis.run()
    vis.destroy_window()

def mirror_obj_file(obj_path, output_path, plane='x'):
    """
    将OBJ文件按照指定平面镜像并保存

    此函数手动解析OBJ文件，保持原始的纹理坐标映射关系，
    避免Open3D/trimesh因(v,vt)组合不同导致的顶点重复问题。

    参数:
        obj_path: 输入OBJ文件路径
        output_path: 输出OBJ文件路径
        plane: 镜像平面 ('x' = x=0平面, 'y' = y=0平面, 'z' = z=0平面)

    返回:
        是否成功保存
    """
    print("=" * 60)
    print(f"镜像OBJ文件: {obj_path}")
    print("=" * 60)

    # 确定镜像轴
    if plane == 'x':
        axis_idx = 0
        axis_name = "X轴 (YZ平面, x=0)"
    elif plane == 'y':
        axis_idx = 1
        axis_name = "Y轴 (XZ平面, y=0)"
    elif plane == 'z':
        axis_idx = 2
        axis_name = "Z轴 (XY平面, z=0)"
    else:
        print(f"错误: 不支持的镜像平面 '{plane}'")
        return False

    print(f"\n镜像平面: {axis_name}")

    # 读取并处理OBJ文件
    output_lines = []
    vertex_count = 0
    normal_count = 0
    face_count = 0

    with open(obj_path, 'r') as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith('v '):
                # 顶点行：镜像坐标
                parts = stripped.split()
                coords = [float(parts[1]), float(parts[2]), float(parts[3])]
                coords[axis_idx] = -coords[axis_idx]
                output_lines.append(f"v {coords[0]} {coords[1]} {coords[2]}\n")
                vertex_count += 1

            elif stripped.startswith('vn '):
                # 法向量行：镜像法向量
                parts = stripped.split()
                normal = [float(parts[1]), float(parts[2]), float(parts[3])]
                normal[axis_idx] = -normal[axis_idx]
                output_lines.append(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
                normal_count += 1

            elif stripped.startswith('f '):
                # 面行：翻转顶点顺序以保持正确的法向量方向
                parts = stripped.split()
                # 原始: f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3
                # 翻转: f v1/vt1/vn1 v3/vt3/vn3 v2/vt2/vn2
                if len(parts) == 4:  # 三角面
                    output_lines.append(f"f {parts[1]} {parts[3]} {parts[2]}\n")
                else:
                    # 处理其他多边形面（保持第一个顶点，翻转其余顶点顺序）
                    reversed_parts = [parts[0], parts[1]] + parts[2:][::-1]
                    output_lines.append(' '.join(reversed_parts) + '\n')
                face_count += 1

            else:
                # 其他行（注释、mtllib、vt等）保持不变
                output_lines.append(line)

    print(f"\n原始模型信息:")
    print(f"  顶点数: {vertex_count}")
    print(f"  法向量数: {normal_count}")
    print(f"  三角面数: {face_count}")

    # 写入输出文件
    with open(output_path, 'w') as f:
        f.writelines(output_lines)

    print(f"\n✓ 镜像模型已保存到: {output_path}")
    return True

def visualize_sphere_region(
        obj_path="./Template.obj",
        csv_path="template landmark.csv",
        landmark_index=2,
        radius=25.0,
        output_pdf="sphere_region.pdf"
):
    """
    可视化球形区域内的点，并保存为PDF。

    参数:
        obj_path: OBJ文件路径
        csv_path: landmark CSV文件路径
        landmark_index: landmark索引（默认2，即第3个点）
        radius: 球体半径（mm，默认25）
        output_pdf: 输出PDF文件路径
    """
    print("=" * 80)
    print(f"可视化球形区域内的点")
    print("=" * 80)

    # 1. 加载OBJ文件
    print(f"\n加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if not mesh.has_vertices():
        print("错误: 无法加载模型或模型为空")
        return

    vertices = np.asarray(mesh.vertices)
    print(f"  顶点数: {len(vertices)}")

    # 2. 加载landmarks
    print(f"\n加载landmarks: {csv_path}")
    landmarks = load_landmarks_from_csv(csv_path)

    if len(landmarks) == 0:
        print("错误: 无法加载landmarks")
        return

    print(f"  Landmark数量: {len(landmarks)}")

    if landmark_index >= len(landmarks):
        print(f"错误: landmark索引 {landmark_index} 超出范围")
        return

    # 3. 获取球心坐标
    center = landmarks[landmark_index]
    print(f"\n球心位置 (Landmark {landmark_index}):")
    print(f"  坐标: [{center[0]:.6f}, {center[1]:.6f}, {center[2]:.6f}]")
    print(f"  半径: {radius:.2f} mm")

    # 4. 计算所有顶点到球心的距离
    print(f"\n计算顶点到球心的距离...")
    distances = np.linalg.norm(vertices - center, axis=1)

    # 5. 分类顶点
    inside_mask = distances <= radius
    num_inside = np.sum(inside_mask)
    num_outside = len(vertices) - num_inside

    print(f"  球内顶点数: {num_inside}")
    print(f"  球外顶点数: {num_outside}")
    print(f"  球内比例: {num_inside/len(vertices)*100:.2f}%")

    # 统计球内点的距离
    if num_inside > 0:
        inside_distances = distances[inside_mask]
        print(f"\n球内顶点距离统计:")
        print(f"  平均距离: {inside_distances.mean():.2f} mm")
        print(f"  最小距离: {inside_distances.min():.2f} mm")
        print(f"  最大距离: {inside_distances.max():.2f} mm")

    # 6. 为mesh着色
    print(f"\n为模型着色...")
    colors = np.zeros((len(vertices), 3))
    # 球外: 灰色
    colors[~inside_mask] = [0.7, 0.7, 0.7]
    # 球内: 红色到黄色渐变（根据距离）
    if num_inside > 0:
        # 归一化距离 [0, radius] -> [0, 1]
        normalized_dist = distances[inside_mask] / radius
        # 红色(1,0,0) -> 黄色(1,1,0)
        colors[inside_mask, 0] = 1.0  # R通道固定为1
        colors[inside_mask, 1] = normalized_dist  # G通道随距离增加
        colors[inside_mask, 2] = 0.0  # B通道固定为0

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    # 7. 创建球心标记（大红点）
    sphere_marker = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
    sphere_marker.translate(center)
    sphere_marker.paint_uniform_color([1.0, 0.0, 0.0])
    sphere_marker.compute_vertex_normals()

    # 8. 使用Open3D渲染
    print(f"\n渲染3D模型...")
    vis_capture = o3d.visualization.Visualizer()
    vis_capture.create_window(window_name="Sphere Region", width=1200, height=900, visible=False)
    vis_capture.add_geometry(mesh)
    vis_capture.add_geometry(sphere_marker)

    render_option = vis_capture.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = True

    view_control = vis_capture.get_view_control()
    view_control.set_zoom(0.8)

    vis_capture.poll_events()
    vis_capture.update_renderer()

    temp_image = "temp_sphere_region.png"
    vis_capture.capture_screen_image(temp_image, do_render=True)
    vis_capture.destroy_window()

    # 9. 生成PDF
    print(f"\n生成PDF报告...")
    with PdfPages(output_pdf) as pdf:
        # 第1页: 3D可视化
        fig1 = plt.figure(figsize=(12, 10))

        ax1 = plt.subplot(111)
        img = plt.imread(temp_image)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title(f'Sphere Region Visualization\nCenter: Landmark {landmark_index}, Radius: {radius:.1f} mm',
                     fontsize=14, fontweight='bold', pad=20)

        # 添加颜色说明
        textstr = f'Gray: Outside sphere ({num_outside} vertices)\n'
        textstr += f'Red→Yellow: Inside sphere ({num_inside} vertices)\n'
        textstr += f'Red dot: Sphere center'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax1.text(0.02, 0.98, textstr, transform=ax1.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)

        plt.tight_layout()
        pdf.savefig(fig1, dpi=150, bbox_inches='tight')
        plt.close(fig1)

        # 第2页: 统计信息
        fig2 = plt.figure(figsize=(12, 10))

        # 2.1 距离分布直方图
        ax2 = plt.subplot(2, 2, 1)
        if num_inside > 0:
            ax2.hist(inside_distances, bins=30, color='orangered', alpha=0.7, edgecolor='black')
            ax2.axvline(inside_distances.mean(), color='blue', linestyle='--', linewidth=2,
                       label=f'Mean: {inside_distances.mean():.2f} mm')
            ax2.axvline(radius, color='red', linestyle='--', linewidth=2,
                       label=f'Radius: {radius:.2f} mm')
        ax2.set_xlabel('Distance from Center (mm)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax2.set_title('Distance Distribution (Inside Sphere)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 2.2 饼图
        ax3 = plt.subplot(2, 2, 2)
        sizes = [num_inside, num_outside]
        labels = [f'Inside\n({num_inside})', f'Outside\n({num_outside})']
        colors_pie = ['orangered', 'lightgray']
        explode = (0.1, 0)
        ax3.pie(sizes, explode=explode, labels=labels, colors=colors_pie,
               autopct='%1.1f%%', shadow=True, startangle=90)
        ax3.set_title('Vertex Distribution', fontsize=12, fontweight='bold')

        # 2.3 统计表格
        ax4 = plt.subplot(2, 2, 3)
        ax4.axis('off')

        stats_data = [
            ['Metric', 'Value'],
            ['Total Vertices', f'{len(vertices)}'],
            ['Sphere Center', f'Landmark {landmark_index}'],
            ['Center Coords', f'[{center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f}]'],
            ['Sphere Radius', f'{radius:.2f} mm'],
            ['Vertices Inside', f'{num_inside}'],
            ['Vertices Outside', f'{num_outside}'],
            ['Inside Ratio', f'{num_inside/len(vertices)*100:.2f}%'],
        ]

        if num_inside > 0:
            stats_data.extend([
                ['Mean Distance (Inside)', f'{inside_distances.mean():.2f} mm'],
                ['Min Distance (Inside)', f'{inside_distances.min():.2f} mm'],
                ['Max Distance (Inside)', f'{inside_distances.max():.2f} mm'],
            ])

        table = ax4.table(cellText=stats_data, loc='center', cellLoc='left',
                         colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2.2)

        # 表头样式
        for i in range(2):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')

        # 交替行颜色
        for i in range(1, len(stats_data)):
            if i % 2 == 0:
                for j in range(2):
                    table[(i, j)].set_facecolor('#E7E6E6')

        ax4.set_title('Statistics Summary', fontsize=12, fontweight='bold', pad=20)

        # 2.4 所有顶点距离分布
        ax5 = plt.subplot(2, 2, 4)
        ax5.hist(distances, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax5.axvline(radius, color='red', linestyle='--', linewidth=2,
                   label=f'Sphere Radius: {radius:.2f} mm')
        ax5.set_xlabel('Distance from Center (mm)', fontsize=11, fontweight='bold')
        ax5.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax5.set_title('Distance Distribution (All Vertices)', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

        plt.tight_layout()
        pdf.savefig(fig2, dpi=150, bbox_inches='tight')
        plt.close(fig2)

    # 清理临时文件
    import os
    if os.path.exists(temp_image):
        os.remove(temp_image)

    print(f"\n✓ PDF已保存: {output_pdf}")
    print(f"  包含2页: 3D可视化 + 统计分析")

    return {
        'center': center,
        'radius': radius,
        'total_vertices': len(vertices),
        'inside_vertices': num_inside,
        'outside_vertices': num_outside,
        'inside_ratio': num_inside/len(vertices),
        'mean_distance_inside': inside_distances.mean() if num_inside > 0 else 0,
        'output_pdf': output_pdf
    }


def batch_mirror_obj_files(input_dir, output_dir=None, plane='x'):
    """
    批量镜像目录下的所有OBJ文件

    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径（默认为 input_dir + '_mirrored'）
        plane: 镜像平面 ('x' = x=0平面)

    返回:
        成功镜像的文件数量
    """
    import glob

    # 设置输出目录
    if output_dir is None:
        output_dir = input_dir.rstrip('/') + '_mirrored'

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 查找所有OBJ文件
    obj_files = glob.glob(os.path.join(input_dir, '*.obj'))

    if len(obj_files) == 0:
        print(f"在 {input_dir} 中未找到OBJ文件")
        return 0

    print("=" * 60)
    print(f"批量镜像OBJ文件")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  找到文件数: {len(obj_files)}")
    print("=" * 60)

    success_count = 0
    for obj_path in sorted(obj_files):
        # 获取文件名
        filename = os.path.basename(obj_path)
        name, ext = os.path.splitext(filename)

        # 生成输出路径
        output_path = os.path.join(output_dir, f"{name}_mirrored{ext}")

        print(f"\n处理: {filename}")

        # 执行镜像
        if mirror_obj_file(obj_path, output_path, plane):
            success_count += 1

    print("\n" + "=" * 60)
    print(f"批量镜像完成!")
    print(f"  成功: {success_count} / {len(obj_files)}")
    print("=" * 60)

    return success_count


def find_closest_vertex(query_point, obj_path="Template.obj"):
    """
    在OBJ文件中查找距离给定坐标最近的顶点

    参数:
        query_point: 查询坐标，可以是列表或numpy数组 [x, y, z]
        obj_path: OBJ文件路径（默认为Template.obj）

    返回:
        dict: 包含最近顶点的信息
            - 'index': 顶点索引（从0开始）
            - 'coord': 顶点坐标 [x, y, z]
            - 'distance': 到查询点的距离
    """
    from scipy.spatial import KDTree

    # 加载OBJ文件
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if not mesh.has_vertices():
        print(f"错误: 无法加载模型 {obj_path}")
        return None

    vertices = np.asarray(mesh.vertices)
    query_point = np.array(query_point)

    # 使用KDTree进行快速最近邻查找
    tree = KDTree(vertices)
    distance, index = tree.query(query_point)

    closest_coord = vertices[index]

    print(f"查询点: [{query_point[0]:.4f}, {query_point[1]:.4f}, {query_point[2]:.4f}]")
    print(f"最近顶点索引: {index}")
    print(f"最近顶点坐标: [{closest_coord[0]:.4f}, {closest_coord[1]:.4f}, {closest_coord[2]:.4f}]")
    print(f"距离: {distance:.6f} mm")

    return {
        'index': index,
        'coord': closest_coord.tolist(),
        'distance': distance
    }


if __name__ == "__main__":
    # 测试 find_closest_vertex 函数
    # query = [0.0000, -61.2906, 43.4027]
    # result = find_closest_vertex(query)

    obj_path = 'bijian/8_50_directional.obj'
    # csv_path = 'kedian.csv'
    view_3d_model(obj_path)


    # 批量镜像kedian和bijian目录下的所有OBJ文件
    # print("\n" + "=" * 80)
    # print("批量镜像kedian和bijian目录下的OBJ文件")
    # print("=" * 80)

    # 镜像kedian目录（保存在原目录）
    # batch_mirror_obj_files('kedian13', 'kedian', plane='x')
    #
    # # 镜像bijian目录（保存在原目录）
    # batch_mirror_obj_files('bijian', 'bijian', plane='x')