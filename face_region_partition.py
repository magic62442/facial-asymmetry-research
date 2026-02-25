#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Face Region Partition Script

根据CSV文件中的7个点将面部网格分成6个区域。
只使用x,y坐标进行分区判断。

区域定义:
- 区域1: l1以上的所有点 (y > v1.y)
- 区域2: 剩余部分中l6以上的点 (y > v7.y)
- 区域3: 剩余部分中l6以下的点 (y <= v7.y)
- 区域4: l2, l3, l4, l5, l7之间的区域
- 区域5: l7, l8, l4, l5之间的区域
- 区域6: l8以下的点 (y < v5.y)

其中:
- l1: y = v1.y
- l2: v1到v2的线段
- l3: v1到v3的线段
- l4: v2到v5的连线，延伸到脸的下方
- l5: v3到(-v5.x, v5.y, v5.z)的连线，延伸到脸的下方
- l6: y = v7.y
- l7: y = v4.y
- l8: y = v5.y
"""

import open3d as o3d
import numpy as np
import argparse
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


def point_side_of_line(px, py, x1, y1, x2, y2):
    """
    判断点(px, py)相对于线段(x1,y1)-(x2,y2)的位置

    返回:
        > 0: 点在线段的左侧
        < 0: 点在线段的右侧
        = 0: 点在线段上
    """
    return (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)


def is_point_in_region_4(px, py, v1, v2, v3, v4, v5):
    """
    判断点是否在区域4内
    区域4由l2, l3, l4, l5, l7围成

    条件:
    - y <= v1.y (在l1以下)
    - y >= v4.y (在l7以上)
    - 在l2 (v1-v2) 的右侧 (对于x>0的点)
    - 在l3 (v1-v3) 的左侧 (对于x<0的点)
    - 在l4 (v2-v5) 和 l5 (v3-v5_mirror) 之间
    """
    # 在l1以下
    if py > v1[1]:
        return False

    # 在l7以上
    if py < v4[1]:
        return False

    # v5的镜像点
    v5_mirror = np.array([-v5[0], v5[1], v5[2]])

    # 对于x >= 0的点，检查是否在l2和l4的右侧
    # 对于x < 0的点，检查是否在l3和l5的左侧

    if px >= 0:
        # 右半边脸
        # 检查是否在l2 (v1-v2)的右侧或线上
        side_l2 = point_side_of_line(px, py, v1[0], v1[1], v2[0], v2[1])
        if side_l2 > 0:  # 在l2左侧
            return False

        # 检查是否在l4 (v2-v5)的右侧或线上
        side_l4 = point_side_of_line(px, py, v2[0], v2[1], v5[0], v5[1])
        if side_l4 > 0:  # 在l4左侧
            return False
    else:
        # 左半边脸
        # 检查是否在l3 (v1-v3)的左侧或线上
        side_l3 = point_side_of_line(px, py, v1[0], v1[1], v3[0], v3[1])
        if side_l3 < 0:  # 在l3右侧
            return False

        # 检查是否在l5 (v3-v5_mirror)的左侧或线上
        side_l5 = point_side_of_line(px, py, v3[0], v3[1], v5_mirror[0], v5_mirror[1])
        if side_l5 < 0:  # 在l5右侧
            return False

    return True


def is_point_in_region_5(px, py, v4, v5, v7, v3):
    """
    判断点是否在区域5内
    区域5由l7, l8, l4, l5围成

    条件:
    - y < v4.y (在l7以下)
    - y >= v7.y (在l8以上)
    - 在l4和l5之间
    """
    # 在l7以下
    if py >= v4[1]:
        return False

    # 在l8以上
    if py < v7[1]:
        return False

    # v5的镜像点
    v5_mirror = np.array([-v5[0], v5[1], v5[2]])

    # 需要在l4和l5之间
    # l4是v2-v5的延长线，但这里我们用v5的位置来判断
    # l5是v3-v5_mirror的延长线

    # 简化判断：检查x坐标是否在v5_mirror.x和v5.x之间
    # 同时需要考虑线的斜率

    if px >= 0:
        # 右半边脸 - 需要在l4的右侧
        # l4从v2延伸到v5再往下
        # 计算l4在当前y值处的x坐标
        # 假设v2和v5确定了l4的斜率
        if v5[1] != v4[1]:
            # 使用v2-v5的斜率来外推
            # 但这里我们需要v2，所以使用传入的v4来近似
            # 实际上区域5的边界是l4的延长部分
            pass

        # 简化：检查是否在v5.x的左侧
        if px > v5[0]:
            return False
    else:
        # 左半边脸 - 需要在l5的左侧
        if px < v5_mirror[0]:
            return False

    return True


def partition_face(obj_path, csv_path, output_txt="region_labels.txt", reorder=False):
    """
    根据CSV中的7个点将面部网格分成6个区域

    参数:
        obj_path: OBJ文件路径
        csv_path: 包含7个点坐标的CSV文件路径
        output_txt: 输出文本文件路径
        reorder: 是否将结果从Open3D顺序重新排列为原始OBJ顺序（默认False）

    返回:
        region_labels: 每个顶点的区域标签数组
                      如果reorder=False，返回Open3D顺序
                      如果reorder=True，返回原始OBJ顺序
    """
    print("=" * 80)
    print("Face Region Partition")
    print("=" * 80)

    # 1. 加载OBJ文件
    print(f"\n加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)

    if not mesh.has_vertices():
        print("错误: 无法加载模型或模型为空")
        return None

    vertices = np.asarray(mesh.vertices)
    print(f"  顶点数: {len(vertices)}")

    # 2. 加载7个点
    print(f"\n加载分区点: {csv_path}")
    landmarks = load_landmarks_from_csv(csv_path)

    if len(landmarks) != 7:
        print(f"错误: 需要7个点，但读取到 {len(landmarks)} 个点")
        return None

    # v1到v7
    v1, v2, v3, v4, v5, v6, v7 = landmarks

    print("\n分区点坐标 (只使用x, y):")
    for i, v in enumerate(landmarks):
        print(f"  v{i+1}: ({v[0]:.4f}, {v[1]:.4f})")

    # v5的镜像点
    v5_mirror = np.array([-v5[0], v5[1], v5[2]])
    print(f"  v5_mirror: ({v5_mirror[0]:.4f}, {v5_mirror[1]:.4f})")

    # 3. 定义分界线
    print("\n分界线:")
    print(f"  l1: y = {v1[1]:.4f}")
    print(f"  l2: 线段 v1({v1[0]:.2f}, {v1[1]:.2f}) - v2({v2[0]:.2f}, {v2[1]:.2f})")
    print(f"  l3: 线段 v1({v1[0]:.2f}, {v1[1]:.2f}) - v3({v3[0]:.2f}, {v3[1]:.2f})")
    print(f"  l4: v2({v2[0]:.2f}, {v2[1]:.2f}) - v5({v5[0]:.2f}, {v5[1]:.2f}) 延伸到下方")
    print(f"  l5: v3({v3[0]:.2f}, {v3[1]:.2f}) - v5_mirror({v5_mirror[0]:.2f}, {v5_mirror[1]:.2f}) 延伸到下方")
    print(f"  l6: y = {v7[1]:.4f}")
    print(f"  l7: y = {v4[1]:.4f}")
    print(f"  l8: y = {v5[1]:.4f}")

    # 4. 分区逻辑
    print("\n开始分区...")
    region_labels = np.zeros(len(vertices), dtype=int)

    for i, vertex in enumerate(vertices):
        px, py = vertex[0], vertex[1]  # 只使用x, y坐标

        # 区域1: y > v1.y (l1以上)
        if py > v1[1]:
            region_labels[i] = 1
            continue

        # 区域6: y < v5.y (l8以下)
        if py < v5[1]:
            region_labels[i] = 6
            continue

        # 检查区域4: l2, l3, l4, l5, l7之间
        # 条件: v4.y <= y <= v1.y, 且在鼻梁区域内
        # 注意: v2在左侧(x<0), v3在右侧(x>0)
        # l2(v1-v2)和l4(v2-v5)是左边界, l3(v1-v3)和l5(v3-v5_mirror)是右边界
        in_region_4 = False
        if v4[1] <= py <= v1[1]:
            # 计算左边界x值 (l2或l4)
            if py >= v2[1]:
                # 在v2以上，用l2 (v1-v2)
                if v2[1] != v1[1]:
                    t2 = (py - v1[1]) / (v2[1] - v1[1])
                    x_left = v1[0] + t2 * (v2[0] - v1[0])
                else:
                    x_left = v2[0]
            else:
                # 在v2以下，用l4 (v2-v5)
                if v5[1] != v2[1]:
                    t4 = (py - v2[1]) / (v5[1] - v2[1])
                    x_left = v2[0] + t4 * (v5[0] - v2[0])
                else:
                    x_left = v2[0]

            # 计算右边界x值 (l3或l5)
            if py >= v3[1]:
                # 在v3以上，用l3 (v1-v3)
                if v3[1] != v1[1]:
                    t3 = (py - v1[1]) / (v3[1] - v1[1])
                    x_right = v1[0] + t3 * (v3[0] - v1[0])
                else:
                    x_right = v3[0]
            else:
                # 在v3以下，用l5 (v3-v5_mirror)
                if v5_mirror[1] != v3[1]:
                    t5 = (py - v3[1]) / (v5_mirror[1] - v3[1])
                    x_right = v3[0] + t5 * (v5_mirror[0] - v3[0])
                else:
                    x_right = v3[0]

            # 点在左右边界之间
            if x_left <= px <= x_right:
                in_region_4 = True

        if in_region_4:
            region_labels[i] = 4
            continue

        # 检查区域5: l7, l8, l4, l5之间
        # 条件: v5.y <= y < v4.y, 且在l4和l5之间
        in_region_5 = False
        if v5[1] <= py < v4[1]:
            # 计算左边界x值 (l4: v2-v5)
            if v5[1] != v2[1]:
                t4 = (py - v2[1]) / (v5[1] - v2[1])
                x_left = v2[0] + t4 * (v5[0] - v2[0])
            else:
                x_left = v2[0]

            # 计算右边界x值 (l5: v3-v5_mirror)
            if v5_mirror[1] != v3[1]:
                t5 = (py - v3[1]) / (v5_mirror[1] - v3[1])
                x_right = v3[0] + t5 * (v5_mirror[0] - v3[0])
            else:
                x_right = v3[0]

            # 点在左右边界之间
            if x_left <= px <= x_right:
                in_region_5 = True

        if in_region_5:
            region_labels[i] = 5
            continue

        # 剩余部分
        # 区域2: l6以上 (y >= v7.y，但排除区域1,4,5后的剩余部分)
        # 区域3: l6以下 (y < v7.y，但排除区域6后的剩余部分)
        # 注意：此时已经排除了区域1,4,5,6
        # 根据y与v7.y的关系判断区域2还是3
        if py >= v7[1]:
            region_labels[i] = 2
        else:
            region_labels[i] = 3

    # 5. 统计各区域顶点数
    print("\n分区结果:")
    for region in range(1, 7):
        count = np.sum(region_labels == region)
        percentage = count / len(vertices) * 100
        print(f"  区域{region}: {count} 个顶点 ({percentage:.2f}%)")

    # 6. 如果需要重排序，将Open3D顺序转换为原始OBJ顺序
    if reorder:
        print("\n重排序: Open3D顺序 -> 原始OBJ顺序")
        from vertex_reorder import compute_vertex_mapping, reorder_values_o3d_to_orig

        # 计算顶点映射
        o3d_to_orig, orig_to_o3d = compute_vertex_mapping(obj_path)

        # 将region_labels从Open3D顺序转换为原始OBJ顺序
        region_labels = reorder_values_o3d_to_orig(region_labels, o3d_to_orig)
        print("  重排序完成")

    # 7. 输出到文本文件
    print(f"\n保存分区结果到: {output_txt}")
    with open(output_txt, 'w') as f:
        f.write("# Face Region Partition Results\n")
        f.write(f"# Total vertices: {len(vertices)}\n")
        if reorder:
            f.write("# Vertex order: Original OBJ file order\n")
        else:
            f.write("# Vertex order: Open3D order (may differ from OBJ file)\n")
        f.write("# Format: vertex_index, region_label\n")
        f.write("#\n")
        for i, label in enumerate(region_labels):
            f.write(f"{i},{label}\n")

    print(f"  完成!")

    return region_labels


def visualize_regions(obj_path, region_labels, output_pdf="region_partition.pdf"):
    """
    可视化分区结果，每个区域使用不同颜色

    参数:
        obj_path: OBJ文件路径
        region_labels: 区域标签数组
        output_pdf: 输出PDF路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    print("\n" + "=" * 80)
    print("Visualize Region Partition")
    print("=" * 80)

    # 1. 加载mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # 2. 定义6种颜色
    region_colors = {
        1: [1.0, 0.0, 0.0],    # 红色 - 区域1 (额头)
        2: [0.0, 1.0, 0.0],    # 绿色 - 区域2
        3: [0.0, 0.0, 1.0],    # 蓝色 - 区域3
        4: [1.0, 1.0, 0.0],    # 黄色 - 区域4 (鼻梁上部)
        5: [1.0, 0.0, 1.0],    # 洋红 - 区域5 (鼻梁下部)
        6: [0.0, 1.0, 1.0],    # 青色 - 区域6 (下巴)
    }

    # 3. 为每个顶点着色
    colors = np.zeros((len(vertices), 3))
    for i, label in enumerate(region_labels):
        if label in region_colors:
            colors[i] = region_colors[label]
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色 (未分类)

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 4. 渲染并保存
    print("\n渲染3D模型...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Face Region Partition", width=1200, height=900, visible=False)
    vis.add_geometry(mesh)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = True

    view_control = vis.get_view_control()
    view_control.set_zoom(0.8)

    vis.poll_events()
    vis.update_renderer()

    temp_image = "temp_region_partition.png"
    vis.capture_screen_image(temp_image, do_render=True)
    vis.destroy_window()

    # 5. 生成PDF
    print(f"\n生成PDF报告: {output_pdf}")
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(14, 10))

        # 主图像
        ax1 = plt.subplot(1, 2, 1)
        img = plt.imread(temp_image)
        ax1.imshow(img)
        ax1.axis('off')
        ax1.set_title('Face Region Partition', fontsize=14, fontweight='bold')

        # 图例和统计
        ax2 = plt.subplot(1, 2, 2)
        ax2.axis('off')

        # 创建颜色图例
        region_names = {
            1: "Region 1 (Forehead)",
            2: "Region 2 (Upper Cheeks)",
            3: "Region 3 (Lower Cheeks)",
            4: "Region 4 (Upper Nose Bridge)",
            5: "Region 5 (Lower Nose Bridge)",
            6: "Region 6 (Chin)",
        }

        y_pos = 0.9
        for region in range(1, 7):
            count = np.sum(region_labels == region)
            percentage = count / len(region_labels) * 100
            color = region_colors[region]

            # 颜色方块
            ax2.add_patch(plt.Rectangle((0.1, y_pos - 0.02), 0.08, 0.06,
                                        facecolor=color, edgecolor='black', linewidth=1))
            # 文字说明
            ax2.text(0.22, y_pos, f"{region_names[region]}", fontsize=11, va='center')
            ax2.text(0.7, y_pos, f"{count} vertices ({percentage:.1f}%)", fontsize=10, va='center')

            y_pos -= 0.12

        # 总计
        ax2.text(0.1, y_pos - 0.05, f"Total: {len(region_labels)} vertices",
                fontsize=12, fontweight='bold')

        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_title('Legend & Statistics', fontsize=14, fontweight='bold')

        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight')
        plt.close(fig)

    # 清理临时文件
    if os.path.exists(temp_image):
        os.remove(temp_image)

    print(f"  PDF已保存: {output_pdf}")


def visualize_boundaries(obj_path, csv_path, output_pdf="region_boundaries.pdf"):
    """
    可视化分区边界线和点，使用正交投影（和view_template一致）

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径（7个点）
        output_pdf: 输出PDF路径
    """
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    print("\n" + "=" * 80)
    print("Visualize Region Boundaries")
    print("=" * 80)

    # 1. 加载mesh
    print(f"\n加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)
    normals = np.asarray(mesh.vertex_normals)

    # 2. 使用和view_template相同的光照设置
    # 单光源从正前方
    light_dir = np.array([0.0, 0.0, 1.0])
    ambient = 0.3
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot
    base_color = np.array([1.0, 1.0, 1.0])
    colors = np.outer(intensities, base_color)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 3. 加载7个点
    print(f"加载分区点: {csv_path}")
    landmarks = load_landmarks_from_csv(csv_path)
    if len(landmarks) != 7:
        print(f"错误: 需要7个点，但读取到 {len(landmarks)} 个点")
        return

    v1, v2, v3, v4, v5, v6, v7 = landmarks
    v5_mirror = np.array([-v5[0], v5[1], v5[2]])

    # 4. 创建几何对象列表
    geometries = [mesh]

    # 5. 创建点（小球）- 7个原始点 + v5_mirror
    sphere_radius = 1.0
    # 原始7个点
    for i, v in enumerate(landmarks):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
        sphere.translate(v)
        sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    # v5_mirror点
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=sphere_radius)
    sphere.translate(v5_mirror)
    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # 红色
    sphere.compute_vertex_normals()
    geometries.append(sphere)

    # 6. 渲染（使用正交投影）
    print("\n渲染3D模型（正交投影）...")
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Region Boundaries", width=1200, height=900, visible=False)

    for geom in geometries:
        vis.add_geometry(geom)

    # 设置渲染选项（和view_template一致）
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

    # 设置正交投影
    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(step=-90)  # 减小FOV接近正交投影

    vis.poll_events()
    vis.update_renderer()

    temp_image = "temp_boundaries.png"
    vis.capture_screen_image(temp_image, do_render=True)
    vis.destroy_window()

    # 7. 生成PDF（只有一张图）
    print(f"\n生成PDF: {output_pdf}")
    with PdfPages(output_pdf) as pdf:
        fig = plt.figure(figsize=(10, 12))
        ax = plt.subplot(111)
        img = plt.imread(temp_image)
        ax.imshow(img)
        ax.axis('off')
        plt.tight_layout()
        pdf.savefig(fig, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    # 清理临时文件
    if os.path.exists(temp_image):
        os.remove(temp_image)

    print(f"  PDF已保存: {output_pdf}")


def interactive_view(obj_path, region_labels):
    """
    交互式查看分区结果
    """
    print("\n" + "=" * 80)
    print("Interactive View")
    print("=" * 80)

    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    vertices = np.asarray(mesh.vertices)

    # 定义6种颜色
    region_colors = {
        1: [1.0, 0.0, 0.0],    # 红色
        2: [0.0, 1.0, 0.0],    # 绿色
        3: [0.0, 0.0, 1.0],    # 蓝色
        4: [1.0, 1.0, 0.0],    # 黄色
        5: [1.0, 0.0, 1.0],    # 洋红
        6: [0.0, 1.0, 1.0],    # 青色
    }

    colors = np.zeros((len(vertices), 3))
    for i, label in enumerate(region_labels):
        if label in region_colors:
            colors[i] = region_colors[label]
        else:
            colors[i] = [0.5, 0.5, 0.5]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    print("\n可视化窗口控制:")
    print("  - 鼠标左键拖动: 旋转模型")
    print("  - 鼠标右键拖动: 平移模型")
    print("  - 滚轮: 缩放")
    print("  - R: 重置视角")
    print("  - Q/ESC: 退出")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Face Region Partition - Interactive", width=1024, height=768)
    vis.add_geometry(mesh)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = True

    view_ctrl = vis.get_view_control()
    view_ctrl.change_field_of_view(step=-90)

    vis.run()
    vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description='Face Region Partition')
    parser.add_argument('--csv_file', default='region.csv', help='CSV file with 7 landmark points')
    parser.add_argument('--obj', default='Template.obj', help='OBJ file path (default: Template.obj)')
    parser.add_argument('--output', default='region_labels.txt', help='Output text file (default: region_labels.txt)')
    parser.add_argument('--pdf', default='region_partition.pdf', help='Output PDF file (default: region_partition.pdf)')
    parser.add_argument('--boundaries_pdf', default='region_boundaries.pdf', help='Boundaries PDF file (default: region_boundaries.pdf)')
    parser.add_argument('--interactive', action='store_true', help='Show interactive 3D viewer')
    parser.add_argument('--reorder', action='store_true', help='Reorder output from Open3D order to original OBJ order')

    args = parser.parse_args()

    # 检查文件是否存在
    if not os.path.exists(args.csv_file):
        print(f"错误: CSV文件不存在: {args.csv_file}")
        return

    if not os.path.exists(args.obj):
        print(f"错误: OBJ文件不存在: {args.obj}")
        return

    # 执行分区
    region_labels = partition_face(args.obj, args.csv_file, args.output, reorder=args.reorder)

    if region_labels is None:
        return

    # 可视化分区结果
    visualize_regions(args.obj, region_labels, args.pdf)

    # 可视化分界线
    visualize_boundaries(args.obj, args.csv_file, args.boundaries_pdf)

    # 交互式查看
    if args.interactive:
        interactive_view(args.obj, region_labels)


if __name__ == "__main__":
    main()
