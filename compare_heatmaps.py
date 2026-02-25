"""
生成比较热力图的大图
每个大图包含 3x6=18 个小热力图
行: Ground Truth, MeshMonk, ICP
列: 不同的位移量
"""

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import trimesh
import os
import re
import csv
from vertex_reorder import load_mapping


def extract_max_from_filename(filename):
    """从文件名中提取第一个数字，乘以2作为colorbar最大值"""
    match = re.search(r'(\d+)', filename)
    if match:
        first_number = int(match.group(1))
        return first_number * 2
    return None


def load_obj_vertices_only(obj_path):
    """手动解析OBJ文件，只提取顶点坐标"""
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)


def create_mini_colorbar(fig, cax, cmap, local_max, cutoff_distance=None):
    """
    创建小热力图的 colorbar（无label，刻度字体加大）
    最小值（0）写在colorbar下方，最大值写在colorbar上方
    """
    cb_height = 256
    cb_data = np.linspace(0, local_max, cb_height)
    cb_colors = np.zeros((cb_height, 3))

    if cutoff_distance is not None and cutoff_distance > 0:
        cutoff_idx = int(cutoff_distance / local_max * cb_height)
        cutoff_idx = min(cutoff_idx, cb_height - 1)
        cb_colors[:cutoff_idx, :] = [0.85, 0.85, 0.85]
        for i in range(cutoff_idx, cb_height):
            normalized_val = (cb_data[i] - cutoff_distance) / (local_max - cutoff_distance)
            cb_colors[i, :] = cmap(normalized_val)[:3]
    else:
        for i in range(cb_height):
            normalized_val = cb_data[i] / local_max
            cb_colors[i, :] = cmap(normalized_val)[:3]

    cb_image = cb_colors[:, np.newaxis, :]
    cax.clear()
    cax.imshow(cb_image, aspect='auto', origin='lower', extent=[0, 1, 0, local_max])
    cax.set_xlim (0, 1)
    cax.set_ylim(0, local_max)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_xticks([])

    # 设置刻度间隔
    if local_max <= 2:
        tick_interval = 0.5
    elif local_max <= 5:
        tick_interval = 1
    elif local_max <= 15:
        tick_interval = 2
    else:
        tick_interval = 5

    ticks = np.arange(0, local_max + tick_interval * 0.1, tick_interval)
    ticks = ticks[ticks <= local_max]

    # 添加cutoff_distance到刻度
    if cutoff_distance is not None and cutoff_distance > 0 and cutoff_distance not in ticks:
        ticks = np.sort(np.append(ticks, cutoff_distance))

    # 移除0和最大值，它们将显示在colorbar上下方
    ticks = ticks[(ticks > 0) & (ticks < local_max)]

    cax.set_yticks(ticks)
    cax.tick_params(labelsize=16)

    # 在colorbar下方添加0
    cax.text(0.5, -0.02, '0', transform=cax.transAxes,
            ha='center', va='top', fontsize=16)

    # 在colorbar上方添加最大值
    cax.text(0.5, 1.02, f'{local_max:.1f}' if local_max != int(local_max) else f'{int(local_max)}',
            transform=cax.transAxes, ha='center', va='bottom', fontsize=16)

    # 分界线
    if cutoff_distance is not None and cutoff_distance > 0 and cutoff_distance < local_max:
        cax.axhline(y=cutoff_distance, color='black', linewidth=1.0, zorder=10)

    return cax


def render_heatmap_to_image(obj_path, distances, local_max, cutoff_distance, colormap='jet'):
    """
    渲染热力图并返回图像数组
    """
    # 加载mesh用于可视化
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    cmap = plt.get_cmap(colormap)

    # 为每个顶点分配颜色
    colors = np.zeros((len(vertices), 3))
    for i in range(len(vertices)):
        if i < len(distances):
            dist = distances[i]
            if cutoff_distance is not None and dist <= cutoff_distance:
                colors[i] = [0.85, 0.85, 0.85]
            else:
                if cutoff_distance is not None and local_max > cutoff_distance:
                    normalized_val = (dist - cutoff_distance) / (local_max - cutoff_distance)
                else:
                    normalized_val = dist / local_max
                normalized_val = min(1.0, max(0.0, normalized_val))
                colors[i] = cmap(normalized_val)[:3]
        else:
            colors[i] = [0.5, 0.5, 0.5]

    # 应用光照
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    light_dir = np.array([0.0, 0.0, 1.0])
    ambient = 0.3
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot
    colors = colors * intensities[:, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 渲染
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Capture", width=800, height=600, visible=False)
    vis.add_geometry(mesh)
    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = False
    view_control = vis.get_view_control()
    view_control.change_field_of_view(step=-90)
    vis.poll_events()
    vis.update_renderer()

    temp_image = f"temp_heatmap_{os.getpid()}.png"
    vis.capture_screen_image(temp_image, do_render=True)
    vis.destroy_window()

    img = Image.open(temp_image)
    img_array = np.array(img)
    os.remove(temp_image)

    # 裁剪空白
    if len(img_array.shape) == 3:
        non_white = np.any(img_array < 250, axis=2)
    else:
        non_white = img_array < 250
    rows = np.any(non_white, axis=1)
    cols = np.any(non_white, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        margin = 5
        rmin = max(0, rmin - margin)
        rmax = min(img_array.shape[0], rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(img_array.shape[1], cmax + margin)
        img_array = img_array[rmin:rmax, cmin:cmax]

    return img_array


def compute_ground_truth_distances(obj_path, pairs_csv='pairs.csv'):
    """计算 Ground Truth 距离（使用pairs.csv的对称点对）"""
    vertices = load_obj_vertices_only(obj_path)

    # 加载 pairs.csv
    source_ids = []
    target_ids = []
    with open(pairs_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            source_ids.append(int(row[0]))
            target_ids.append(int(row[1]))

    # 镜像顶点
    vertices_mirrored = vertices.copy()
    vertices_mirrored[:, 0] = -vertices_mirrored[:, 0]

    # 计算距离
    distances = np.zeros(len(vertices))
    for src_id, tgt_id in zip(source_ids, target_ids):
        src_vertex = vertices[src_id]
        tgt_vertex = vertices_mirrored[tgt_id]
        distances[src_id] = np.linalg.norm(src_vertex - tgt_vertex)

    return distances


def compute_icp_distances(obj_path, mirrored_aligned_path):
    """计算 ICP 距离（点到表面距离）"""
    vertices = load_obj_vertices_only(obj_path)
    mesh_mirrored = trimesh.load(mirrored_aligned_path, force='mesh')
    closest_points, distances, _ = trimesh.proximity.closest_point(mesh_mirrored, vertices)
    return distances


def load_meshmonk_distances(csv_path, num_vertices, mapping_path='vertex_mapping.npz'):
    """加载 MeshMonk 距离，并进行顶点索引重排序"""
    # 先按原始OBJ顺序加载
    distances_orig = np.zeros(num_vertices)
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            vertex_id = int(row[0])
            dist = float(row[1])
            if vertex_id < num_vertices:
                distances_orig[vertex_id] = dist

    # 重排序到Open3D顶点顺序
    if os.path.exists(mapping_path):
        o3d_to_orig, orig_to_o3d = load_mapping(mapping_path)
        distances = np.zeros(num_vertices)
        for orig_idx in range(num_vertices):
            if orig_idx < len(orig_to_o3d):
                o3d_idx = orig_to_o3d[orig_idx]
                distances[o3d_idx] = distances_orig[orig_idx]
        return distances

    return distances_orig


def generate_comparison_figure(input_dir, output_pdf, displacement_values, file_pattern,
                                pairs_csv='pairs.csv', cutoff_distance=0.5, colormap='jet'):
    """
    生成一个大的比较图

    参数:
        input_dir: 输入目录 (kedian 或 bijian)
        output_pdf: 输出PDF路径
        displacement_values: 位移量列表 (如 [2, 4, 6, 8, 10, 12])
        file_pattern: 文件名模式 (如 "{}_40_directional")
        pairs_csv: 对称点对文件路径
        cutoff_distance: 截断值
        colormap: 颜色映射
    """
    print("=" * 80)
    print(f"生成比较热力图: {output_pdf}")
    print(f"  输入目录: {input_dir}")
    print(f"  位移量: {displacement_values}")
    print("=" * 80)

    n_rows = 3  # Ground Truth, MeshMonk, ICP
    n_cols = len(displacement_values)
    row_labels = ['Ground Truth', 'MeshMonk', 'ICP']

    # 创建大图，使用固定尺寸确保两个图一致
    fig_width = 4 * n_cols + 2  # 每个小图4英寸宽，加上边距
    fig_height = 12  # 固定高度，确保kedian和bijian一致
    fig = plt.figure(figsize=(fig_width, fig_height))

    # 计算子图位置
    left_margin = 0.04
    right_margin = 0.02
    top_margin = 0.08
    bottom_margin = 0.05
    row_label_width = 0.03
    col_header_height = 0.04
    hspace = 0.06  # 增加垂直间距
    wspace = 0.04
    colorbar_width = 0.015

    plot_width = (1 - left_margin - right_margin - row_label_width) / n_cols - wspace
    plot_height = (1 - top_margin - bottom_margin - col_header_height) / n_rows - hspace

    # 添加列标题（位移量）
    for col_idx, disp in enumerate(displacement_values):
        x_center = left_margin + row_label_width + col_idx * (plot_width + wspace) + plot_width / 2
        # 将列标题放在更靠近图片的位置
        y_pos = 1 - top_margin - col_header_height * 0.3
        fig.text(x_center, y_pos, f'{disp} mm',
                 ha='center', va='center', fontsize=24, fontweight='bold')

    # 添加行标题
    for row_idx, label in enumerate(row_labels):
        y_center = 1 - top_margin - col_header_height - row_idx * (plot_height + hspace) - plot_height / 2
        # 将行标题放在更靠近图片的位置
        x_pos = left_margin + row_label_width * 0.85
        fig.text(x_pos, y_center, label,
                 ha='center', va='center', fontsize=24, fontweight='bold', rotation=90)

    # 生成每个小热力图
    for col_idx, disp in enumerate(displacement_values):
        filename = file_pattern.format(disp)
        obj_path = os.path.join(input_dir, f"{filename}.obj")
        mirrored_aligned_path = os.path.join(input_dir, f"{filename}_mirrored_aligned.obj")
        meshmonk_csv = os.path.join(input_dir, 'meshmonk', f"{filename}_fa_values.csv")

        # 从文件名提取 max_distance
        local_max = extract_max_from_filename(filename)
        if local_max is None:
            local_max = disp * 2

        print(f"\n处理位移量 {disp}mm (colorbar max: {local_max}mm)")

        # 检查文件是否存在
        if not os.path.exists(obj_path):
            print(f"  警告: 找不到 {obj_path}")
            continue

        vertices = load_obj_vertices_only(obj_path)
        num_vertices = len(vertices)

        # 三种方法的距离
        distances_list = []

        # 1. Ground Truth
        print(f"  计算 Ground Truth...")
        try:
            gt_distances = compute_ground_truth_distances(obj_path, pairs_csv)
            distances_list.append(gt_distances)
        except Exception as e:
            print(f"    错误: {e}")
            distances_list.append(np.zeros(num_vertices))

        # 2. MeshMonk
        print(f"  加载 MeshMonk...")
        try:
            if os.path.exists(meshmonk_csv):
                mm_distances = load_meshmonk_distances(meshmonk_csv, num_vertices)
            else:
                print(f"    警告: 找不到 {meshmonk_csv}")
                mm_distances = np.zeros(num_vertices)
            distances_list.append(mm_distances)
        except Exception as e:
            print(f"    错误: {e}")
            distances_list.append(np.zeros(num_vertices))

        # 3. ICP
        print(f"  计算 ICP...")
        try:
            if os.path.exists(mirrored_aligned_path):
                icp_distances = compute_icp_distances(obj_path, mirrored_aligned_path)
            else:
                print(f"    警告: 找不到 {mirrored_aligned_path}")
                icp_distances = np.zeros(num_vertices)
            distances_list.append(icp_distances)
        except Exception as e:
            print(f"    错误: {e}")
            distances_list.append(np.zeros(num_vertices))

        # 渲染并添加到图中
        for row_idx, distances in enumerate(distances_list):
            print(f"  渲染 {row_labels[row_idx]}...")

            # 渲染热力图
            img_array = render_heatmap_to_image(obj_path, distances, local_max,
                                                 cutoff_distance, colormap)

            # 计算子图位置
            x_pos = left_margin + row_label_width + col_idx * (plot_width + wspace)
            y_pos = 1 - top_margin - col_header_height - (row_idx + 1) * (plot_height + hspace) + hspace

            # 热力图占据的宽度（减去colorbar空间）
            img_width = plot_width - colorbar_width - 0.01
            ax = fig.add_axes([x_pos, y_pos, img_width, plot_height])
            ax.imshow(img_array)
            ax.axis('off')

            # 添加 colorbar
            cax = fig.add_axes([x_pos + img_width + 0.005, y_pos, colorbar_width, plot_height])
            cmap_obj = plt.get_cmap(colormap)
            create_mini_colorbar(fig, cax, cmap_obj, local_max, cutoff_distance)

    # 保存
    plt.savefig(output_pdf, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ 保存到: {output_pdf}")


if __name__ == "__main__":
    # 生成 kedian 比较图
    print("\n" + "=" * 80)
    print("生成 kedian 比较图")
    print("=" * 80)
    generate_comparison_figure(
        input_dir='kedian',
        output_pdf='kedian_comparison.pdf',
        displacement_values=[2, 4, 6, 8, 10, 12],
        file_pattern="{}_40_directional",
        pairs_csv='pairs.csv',
        cutoff_distance=0.5,
        colormap='jet'
    )

    # 生成 bijian 比较图
    print("\n" + "=" * 80)
    print("生成 bijian 比较图")
    print("=" * 80)
    generate_comparison_figure(
        input_dir='bijian',
        output_pdf='bijian_comparison.pdf',
        displacement_values=[1, 2, 3, 4, 5, 6],
        file_pattern="{}_50_directional",
        pairs_csv='pairs.csv',
        cutoff_distance=0.5,
        colormap='jet'
    )

    print("\n" + "=" * 80)
    print("所有比较图生成完成!")
    print("=" * 80)
