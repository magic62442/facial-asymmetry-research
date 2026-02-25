import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image
import trimesh
import os
import glob


def create_truncated_colorbar(fig, cax, cmap, local_max, cutoff_distance=None, label='Distance (mm)'):
    """
    创建一个带底部截断的 colorbar

    显示 [0, local_max] 范围
    如果指定了 cutoff_distance，则 [0, cutoff_distance] 区间显示灰色
    [cutoff_distance, local_max] 区间使用完整的色谱（从蓝到红）

    参数:
        fig: matplotlib figure 对象
        cax: colorbar 的 axes 对象
        cmap: colormap 对象
        local_max: 当前图像的最大值（用于 colorbar 范围）
        cutoff_distance: 截断值，低于此值显示灰色
        label: colorbar 标签

    返回:
        cax: colorbar axes 对象
    """
    # 创建 colorbar 图像数据
    cb_height = 256
    cb_data = np.linspace(0, local_max, cb_height)

    # 创建颜色数组
    cb_colors = np.zeros((cb_height, 3))

    if cutoff_distance is not None and cutoff_distance > 0:
        # 计算截断点的索引
        cutoff_idx = int(cutoff_distance / local_max * cb_height)
        cutoff_idx = min(cutoff_idx, cb_height - 1)

        # 0 到 cutoff_distance: 灰色
        cb_colors[:cutoff_idx, :] = [0.85, 0.85, 0.85]

        # cutoff_distance 到 local_max: 完整色谱
        # 将 [cutoff_distance, local_max] 映射到 [0, 1] 的色谱
        color_norm = mcolors.Normalize(vmin=0, vmax=1)
        for i in range(cutoff_idx, cb_height):
            # 将当前值映射到 [0, 1] 范围
            normalized_val = (cb_data[i] - cutoff_distance) / (local_max - cutoff_distance)
            cb_colors[i, :] = cmap(normalized_val)[:3]
    else:
        # 没有截断，整个范围使用完整色谱
        for i in range(cb_height):
            normalized_val = cb_data[i] / local_max
            cb_colors[i, :] = cmap(normalized_val)[:3]

    # 创建 2D 图像数据
    cb_image = cb_colors[:, np.newaxis, :]  # Shape: (cb_height, 1, 3) RGB

    # 清除 cax 并绘制 colorbar 图像
    cax.clear()
    cax.imshow(cb_image, aspect='auto', origin='lower', extent=[0, 1, 0, local_max])

    # 设置刻度
    cax.set_xlim(0, 1)
    cax.set_ylim(0, local_max)
    cax.yaxis.tick_right()
    cax.yaxis.set_label_position('right')
    cax.set_xticks([])

    # 设置合适的刻度间隔
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
    # 确保 cutoff_distance 也在刻度中
    if cutoff_distance is not None and cutoff_distance > 0 and cutoff_distance not in ticks:
        ticks = np.sort(np.append(ticks, cutoff_distance))
    # 确保最大值也在刻度中
    if local_max not in ticks:
        ticks = np.sort(np.append(ticks, local_max))
    cax.set_yticks(ticks)
    cax.tick_params(labelsize=14)

    # 设置标签
    cax.set_ylabel(label, rotation=270, labelpad=20, fontsize=16)

    # 如果有截断值，在截断处添加分界线
    if cutoff_distance is not None and cutoff_distance > 0 and cutoff_distance < local_max:
        cax.axhline(y=cutoff_distance, color='black', linewidth=1.0, zorder=10)

    return cax


def extract_max_from_filename(filename):
    """
    从文件名中提取第一个数字，乘以2作为colorbar最大值

    例如: 2_40_directional_fa_values.csv -> 2 * 2 = 4

    参数:
        filename: 文件名

    返回:
        colorbar最大值，如果无法提取则返回None
    """
    import re
    # 提取文件名中的第一个数字
    match = re.search(r'(\d+)', filename)
    if match:
        first_number = int(match.group(1))
        return first_number * 2
    return None


def load_obj_vertices_only(obj_path):
    """
    手动解析OBJ文件，只提取顶点坐标（忽略纹理坐标映射问题）

    这个函数解决了Open3D/trimesh在加载含有纹理坐标的OBJ文件时，
    由于(v,vt)组合不同导致顶点被重复的问题。

    参数:
        obj_path: OBJ文件路径

    返回:
        vertices: numpy数组，形状为(N, 3)的顶点坐标
    """
    vertices = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                vertices.append([x, y, z])
    return np.array(vertices)


def compute_stats(distances, threshold=0.5):
    """计算统计指标"""
    n = len(distances)
    mean_d = distances.mean()
    max_d = distances.max()
    mse = np.mean(distances ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    mad = np.mean(np.abs(distances - mean_d))  # Mean Absolute Deviation
    pct_within = np.sum(distances <= threshold) / n * 100  # % within threshold
    return {
        'count': n,
        'Mean': mean_d,
        'Max': max_d,
        'MSE': mse,
        'RMSE': rmse,
        'MAD': mad,
        'pct_within': pct_within
    }


def visualize_asymmetry(original_obj, mirrored_registered_obj, output_pdf='asymmetry_heatmap.pdf',
                       colormap='jet', show_interactive=False, distance_method='point_to_point',
                       region_labels_path='region_labels.txt', stats_output_path=None,
                       max_distance=None, cutoff_distance=None):
    """
    可视化面部不对称性热力图

    不对称性定义：对于原始模型的每个点，计算到镜像配准模型的距离。

    参数:
        original_obj: 原始OBJ文件路径
        mirrored_registered_obj: 镜像并配准后的OBJ文件路径
        output_pdf: 输出PDF文件名
        colormap: 颜色映射方案 ('jet', 'coolwarm', 'viridis', 'hot', 'plasma')
        show_interactive: 是否显示交互式窗口
        distance_method: 距离计算方法
            - 'point_to_surface': 点到mesh表面的精确距离（推荐，更准确）
            - 'point_to_point': 点到最近顶点的距离（快速，但可能不够精确）
        region_labels_path: 分区标签文件路径
        stats_output_path: 统计信息输出文件路径（默认与PDF同名但扩展名为.txt）
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色

    返回:
        统计信息字典（包含MSE和RMSE）
    """
    print("=" * 80)
    print("面部不对称性可视化")
    print("=" * 80)

    # 1. 加载原始模型
    print(f"\n加载原始模型: {original_obj}")
    if not os.path.exists(original_obj):
        raise FileNotFoundError(f"找不到文件: {original_obj}")

    # 使用手动解析获取正确的顶点坐标（避免纹理坐标映射导致的顶点重复问题）
    vertices_original = load_obj_vertices_only(original_obj)
    print(f"  顶点数: {len(vertices_original)}")

    # 使用Open3D加载mesh用于可视化（可能有顶点重复，但只用于渲染）
    mesh_original = o3d.io.read_triangle_mesh(original_obj)

    # 2. 加载镜像配准后的模型
    print(f"\n加载镜像配准模型: {mirrored_registered_obj}")
    if not os.path.exists(mirrored_registered_obj):
        raise FileNotFoundError(f"找不到文件: {mirrored_registered_obj}")

    # 使用手动解析获取正确的顶点坐标
    vertices_mirrored = load_obj_vertices_only(mirrored_registered_obj)
    print(f"  顶点数: {len(vertices_mirrored)}")

    # 验证顶点数是否一致（通常应该一致）
    if len(vertices_original) != len(vertices_mirrored):
        print(f"\n警告: 两个模型的顶点数不一致!")
        print(f"  原始模型: {len(vertices_original)}")
        print(f"  镜像模型: {len(vertices_mirrored)}")
        print("  将继续处理，但结果可能不准确...")

    # 3. 根据选择的方法计算距离
    print(f"\n计算不对称性距离 (方法: {distance_method})...")

    if distance_method == 'point_to_surface':
        # 方法1: 点到mesh表面的精确距离（使用trimesh）
        print("  使用trimesh库计算点到表面的精确距离...")
        mesh_mirrored_tm = trimesh.load(mirrored_registered_obj, force='mesh')

        # 使用trimesh的proximity模块计算点到mesh表面的距离
        closest_points, distances, triangle_ids = trimesh.proximity.closest_point(
            mesh_mirrored_tm, vertices_original
        )
        print(f"    ✓ 点到表面距离计算完成")

    elif distance_method == 'point_to_point':
        # 方法2: 点到最近顶点的距离（使用KDTree）
        print("  使用KDTree计算点到最近顶点的距离...")
        from scipy.spatial import KDTree
        tree = KDTree(vertices_mirrored)
        distances, nearest_indices = tree.query(vertices_original)
        print(f"    ✓ 点到点距离计算完成")

    else:
        raise ValueError(f"未知的距离计算方法: {distance_method}. 请使用 'point_to_surface' 或 'point_to_point'")

    print(f"  计算完成: {len(distances)} 个距离值")

    # 4. 统计分析
    print(f"\n不对称性统计:")
    print(f"  总顶点数: {len(distances)}")
    print(f"  平均距离 (Mean): {distances.mean():.6f} mm")
    print(f"  中位数距离 (Median): {np.median(distances):.6f} mm")
    print(f"  标准差 (Std): {distances.std():.6f} mm")
    print(f"  最小距离 (Min): {distances.min():.6f} mm")
    print(f"  最大距离 (Max): {distances.max():.6f} mm")

    # 计算MSE和RMSE
    mse = np.mean(distances ** 2)
    rmse = np.sqrt(mse)

    print(f"\n误差度量:")
    print(f"  MSE (Mean Squared Error): {mse:.6f} mm²")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f} mm")

    # 5. 加载分区标签
    region_labels = None
    if os.path.exists(region_labels_path):
        print(f"\n加载分区标签: {region_labels_path}")
        region_labels = np.zeros(len(vertices_original), dtype=int)
        with open(region_labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    idx, label = int(parts[0]), int(parts[1])
                    if 0 <= idx < len(vertices_original):
                        region_labels[idx] = label
        print(f"  加载完成，共6个区域")

    # 6. 创建颜色映射
    print(f"\n创建热力图 (颜色方案: {colormap})...")

    # 计算colorbar范围
    # 如果指定了max_distance，使用它作为colorbar范围
    # 否则使用数据本身的最大值
    if max_distance is not None:
        local_max = max_distance
    else:
        local_max = distances.max()

    print(f"  colorbar范围: 0 - {local_max:.6f} mm")
    if cutoff_distance is not None:
        print(f"  截断值: {cutoff_distance:.6f} mm (以下显示浅灰色)")

    cmap = plt.get_cmap(colormap)

    # 为每个顶点分配颜色
    # cutoff以下用灰色，cutoff到local_max使用完整色谱
    colors = np.zeros((len(vertices_original), 3))

    for i in range(len(vertices_original)):
        if cutoff_distance is not None and distances[i] <= cutoff_distance:
            # 低于截断值 - 浅灰色
            colors[i] = [0.85, 0.85, 0.85]
        else:
            # 有效距离 - 使用完整色谱 [cutoff_distance, local_max] -> [0, 1]
            if cutoff_distance is not None and local_max > cutoff_distance:
                normalized_val = (distances[i] - cutoff_distance) / (local_max - cutoff_distance)
            else:
                normalized_val = distances[i] / local_max
            normalized_val = min(1.0, max(0.0, normalized_val))  # 限制在 [0, 1]
            colors[i] = cmap(normalized_val)[:3]  # 取RGB，丢弃alpha

    # 7. 应用颜色到mesh（使用原始mesh进行可视化）
    mesh_original.compute_vertex_normals()
    normals = np.asarray(mesh_original.vertex_normals)

    # 参考MATLAB风格: 单光源从正前方，material dull
    light_dir = np.array([0.0, 0.0, 1.0])
    ambient = 0.3
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot

    # 将光照强度应用到颜色上
    colors = colors * intensities[:, np.newaxis]
    mesh_original.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 8. 交互式可视化（可选）
    if show_interactive:
        print("\n打开交互式可视化窗口...")
        print("  提示: 可以旋转、缩放查看不对称性分布")
        print("  关闭窗口继续生成PDF报告...")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Asymmetry Heatmap", width=1200, height=900)
        vis.add_geometry(mesh_original)

        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

        # 设置正交投影
        view_control = vis.get_view_control()
        view_control.change_field_of_view(step=-90)

        # 运行可视化
        vis.run()
        vis.destroy_window()

    # 9. 使用Open3D捕获渲染图像
    print("\n生成热力图...")
    print("  使用Open3D渲染...")

    # 创建一个隐藏的可视化窗口用于捕获图像
    vis_capture = o3d.visualization.Visualizer()
    vis_capture.create_window(window_name="Capture", width=1200, height=900, visible=False)
    vis_capture.add_geometry(mesh_original)

    # 设置渲染选项
    render_option_capture = vis_capture.get_render_option()
    render_option_capture.mesh_show_back_face = True
    render_option_capture.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

    # 设置正交投影
    view_control_capture = vis_capture.get_view_control()
    view_control_capture.change_field_of_view(step=-90)

    # 更新渲染
    vis_capture.poll_events()
    vis_capture.update_renderer()

    # 捕获图像
    temp_image = "temp_asymmetry_heatmap.png"
    vis_capture.capture_screen_image(temp_image, do_render=True)
    vis_capture.destroy_window()

    # 验证图像
    if os.path.exists(temp_image):
        file_size = os.path.getsize(temp_image)
        print(f"    ✓ 图像已保存 ({file_size} bytes)")
    else:
        print(f"    ✗ 图像保存失败!")
        raise RuntimeError("热力图生成失败")

    # 10. 生成PDF报告
    print(f"\n生成PDF报告: {output_pdf}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")

    with PdfPages(output_pdf) as pdf:
        # ========== 热力图（只有一页，无统计信息） ==========
        img = Image.open(temp_image)
        img_array = np.array(img)

        # 裁剪空白区域：找到非白色像素的边界
        if len(img_array.shape) == 3:
            non_white = np.any(img_array < 250, axis=2)
        else:
            non_white = img_array < 250

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 添加少量边距
        margin = 10
        rmin = max(0, rmin - margin)
        rmax = min(img_array.shape[0], rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(img_array.shape[1], cmax + margin)

        img_cropped = img_array[rmin:rmax, cmin:cmax]

        # 根据裁剪后图像的宽高比调整figure大小
        h, w = img_cropped.shape[:2]
        aspect = w / h
        fig_height = 8
        fig_width = fig_height * aspect + 1.5  # 额外空间给colorbar

        fig1 = plt.figure(figsize=(fig_width, fig_height))
        ax1 = fig1.add_axes([0.02, 0.05, 0.80, 0.90])
        ax1.imshow(img_cropped)
        ax1.axis('off')
        ax1.set_title('ICP Heatmap', fontsize=20, pad=10)

        # 添加colorbar，高度与图像匹配
        # 使用截断的colorbar：底部截断显示灰色，cutoff以上使用完整色谱
        cax = fig1.add_axes([0.85, 0.05, 0.03, 0.90])
        create_truncated_colorbar(fig1, cax, cmap, local_max,
                                  cutoff_distance=cutoff_distance, label='Distance (mm)')

        pdf.savefig(fig1, dpi=300)
        plt.close(fig1)

    # 11. 保存统计信息到文件
    if stats_output_path is None:
        stats_output_path = output_pdf.replace('.pdf', '_stats.txt')

    print(f"\n保存统计信息: {stats_output_path}")

    with open(stats_output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Overall Statistics\n")
        f.write("=" * 60 + "\n")

        overall_stats = compute_stats(distances)
        f.write(f"Max: {overall_stats['Max']:.6f} mm\n")
        f.write(f"MSE: {overall_stats['MSE']:.6f} mm²\n")
        f.write(f"RMSE: {overall_stats['RMSE']:.6f} mm\n")
        f.write(f"MAD: {overall_stats['MAD']:.6f} mm\n")
        f.write(f"% within 0.5mm: {overall_stats['pct_within']:.2f}%\n")

        # 分区统计
        if region_labels is not None:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Statistics by Region\n")
            f.write("=" * 60 + "\n")

            for region in range(1, 7):
                region_mask = (region_labels == region)
                region_distances = distances[region_mask]

                if len(region_distances) > 0:
                    region_stats = compute_stats(region_distances)

                    f.write(f"\nRegion {region}:\n")
                    f.write(f"  Vertices: {len(region_distances)}\n")
                    f.write(f"  Max: {region_stats['Max']:.6f} mm\n")
                    f.write(f"  MSE: {region_stats['MSE']:.6f} mm²\n")
                    f.write(f"  RMSE: {region_stats['RMSE']:.6f} mm\n")
                    f.write(f"  MAD: {region_stats['MAD']:.6f} mm\n")
                    f.write(f"  % within 0.5mm: {region_stats['pct_within']:.2f}%\n")
                else:
                    f.write(f"\nRegion {region}: No valid vertices\n")

    # 12. 清理临时文件
    print("\n清理临时文件...")
    if os.path.exists(temp_image):
        os.remove(temp_image)

    print(f"\n✓ PDF报告已保存: {output_pdf}")
    print(f"✓ 统计信息已保存: {stats_output_path}")

    # 计算分区统计用于返回
    region_stats_dict = {}
    if region_labels is not None:
        for region in range(1, 7):
            region_mask = (region_labels == region)
            region_distances = distances[region_mask]

            if len(region_distances) > 0:
                region_stats_dict[region] = compute_stats(region_distances)
            else:
                # 空区域返回零值
                region_stats_dict[region] = {
                    'count': 0,
                    'Mean': 0.0,
                    'Median': 0.0,
                    'Std': 0.0,
                    'Min': 0.0,
                    'Max': 0.0,
                    'MSE': 0.0,
                    'RMSE': 0.0,
                    'MAD': 0.0,
                    'pct_within': 0.0
                }

    # 计算整体统计
    overall_stats = compute_stats(distances)

    return {
        'vertex_count': len(distances),
        'mean_distance': distances.mean(),
        'median_distance': np.median(distances),
        'std_distance': distances.std(),
        'min_distance': distances.min(),
        'max_distance': distances.max(),
        'mse': mse,
        'rmse': rmse,
        'pct_within': overall_stats['pct_within'],
        'output_pdf': output_pdf,
        'stats_file': stats_output_path,
        'region_stats': region_stats_dict
    }


def visualize_asymmetry_from_csv(obj_path, csv_path, output_pdf='asymmetry_heatmap_csv.pdf',
                                  colormap='jet', show_interactive=False, reorder=False,
                                  region_labels_path='region_labels.txt', stats_output_path=None,
                                  max_distance=None, cutoff_distance=None):
    """
    从CSV文件读取距离数据并可视化不对称性热力图

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径 (第一列: vertex_id, 第二列: distance)
        output_pdf: 输出PDF文件名
        colormap: 颜色映射方案 ('jet', 'coolwarm', 'viridis', 'hot', 'plasma')
        show_interactive: 是否显示交互式窗口
        reorder: 是否重新排序CSV中的顶点索引
                 当使用Open3D读取OBJ时顶点顺序可能被打乱，设为True可自动修正
        region_labels_path: 分区标签文件路径
        stats_output_path: 统计信息输出文件路径（默认与PDF同名但扩展名为.txt）
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色

    返回:
        统计信息字典（包含MSE和RMSE）
    """
    import csv as csv_module
    from vertex_reorder import load_mapping

    print("=" * 80)
    print("从CSV文件可视化不对称性")
    print("=" * 80)

    # 1. 加载OBJ文件
    print(f"\n加载模型: {obj_path}")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"找不到文件: {obj_path}")

    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    print(f"  顶点数: {len(vertices)}")

    # 2. 加载CSV文件
    print(f"\n加载距离数据: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    vertex_ids = []
    distances_list = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv_module.reader(f)
        header = next(reader)  # 跳过表头
        print(f"  CSV表头: {header}")

        for row in reader:
            vertex_ids.append(int(row[0]))
            distances_list.append(float(row[1]))

    vertex_ids = np.array(vertex_ids)
    distances_from_csv = np.array(distances_list)
    print(f"  读取距离数据: {len(distances_from_csv)} 条记录")

    # 3. 如果需要重新排序顶点索引
    if reorder:
        print(f"\n加载顶点对应关系 (reorder=True)...")
        o3d_to_orig, orig_to_o3d = load_mapping('vertex_mapping.npz')

        # CSV中的vertex_id是原始OBJ顺序的索引
        # 需要将其转换为Open3D顺序的索引
        # 创建原始顺序的距离数组
        distances_orig_order = np.full(len(vertices), np.nan)
        for vid, dist in zip(vertex_ids, distances_from_csv):
            if 0 <= vid < len(vertices):
                distances_orig_order[vid] = dist

        # 重新排序为Open3D顺序
        # orig_to_o3d[orig_idx] = o3d_idx
        # 所以 distances_o3d[o3d_idx] = distances_orig[orig_idx]
        distances = np.full(len(vertices), np.nan)
        for orig_idx in range(len(vertices)):
            o3d_idx = orig_to_o3d[orig_idx]
            distances[o3d_idx] = distances_orig_order[orig_idx]

        print(f"  顶点索引已重新排序")
    else:
        # 4. 验证数据完整性
        if len(vertex_ids) != len(vertices):
            print(f"\n警告: CSV记录数 ({len(vertex_ids)}) 与OBJ顶点数 ({len(vertices)}) 不匹配!")

        # 创建完整的距离数组
        distances = np.full(len(vertices), np.nan)
        for vid, dist in zip(vertex_ids, distances_from_csv):
            if 0 <= vid < len(vertices):
                distances[vid] = dist

    # 过滤有效距离
    valid_mask = ~np.isnan(distances)
    valid_distances = distances[valid_mask]

    print(f"  有效距离记录: {len(valid_distances)} / {len(vertices)}")

    # 4. 统计分析
    print(f"\n不对称性统计:")
    print(f"  总顶点数: {len(vertices)}")
    print(f"  有效顶点数: {len(valid_distances)}")
    print(f"  平均距离 (Mean): {valid_distances.mean():.6f} mm")
    print(f"  中位数距离 (Median): {np.median(valid_distances):.6f} mm")
    print(f"  标准差 (Std): {valid_distances.std():.6f} mm")
    print(f"  最小距离 (Min): {valid_distances.min():.6f} mm")
    print(f"  最大距离 (Max): {valid_distances.max():.6f} mm")

    # 计算MSE和RMSE
    mse = np.mean(valid_distances ** 2)
    rmse = np.sqrt(mse)

    print(f"\n误差度量:")
    print(f"  MSE (Mean Squared Error): {mse:.6f} mm²")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f} mm")

    # 5. 加载分区标签
    region_labels = None
    if os.path.exists(region_labels_path):
        print(f"\n加载分区标签: {region_labels_path}")
        region_labels = np.zeros(len(vertices), dtype=int)
        with open(region_labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    idx, label = int(parts[0]), int(parts[1])
                    if 0 <= idx < len(vertices):
                        region_labels[idx] = label
        print(f"  加载完成，共6个区域")

    # 6. 创建颜色映射
    print(f"\n创建热力图 (颜色方案: {colormap})...")

    # 计算colorbar范围
    # 如果指定了max_distance，使用它作为colorbar范围
    # 否则使用数据本身的最大值
    if max_distance is not None:
        local_max = max_distance
    else:
        local_max = valid_distances.max()

    print(f"  colorbar范围: 0 - {local_max:.6f} mm")
    if cutoff_distance is not None:
        print(f"  截断值: {cutoff_distance:.6f} mm (以下显示浅灰色)")

    cmap = plt.get_cmap(colormap)

    # 为每个顶点分配颜色
    # cutoff以下用灰色，cutoff到local_max使用完整色谱
    colors = np.zeros((len(vertices), 3))

    for i in range(len(vertices)):
        if valid_mask[i]:
            if cutoff_distance is not None and distances[i] <= cutoff_distance:
                # 低于截断值 - 浅灰色
                colors[i] = [0.85, 0.85, 0.85]
            else:
                # 有效距离 - 使用完整色谱 [cutoff_distance, local_max] -> [0, 1]
                if cutoff_distance is not None and local_max > cutoff_distance:
                    normalized_val = (distances[i] - cutoff_distance) / (local_max - cutoff_distance)
                else:
                    normalized_val = distances[i] / local_max
                normalized_val = min(1.0, max(0.0, normalized_val))  # 限制在 [0, 1]
                colors[i] = cmap(normalized_val)[:3]  # 取RGB，丢弃alpha
        else:
            colors[i] = [0.5, 0.5, 0.5]  # 灰色表示无效数据

    # 7. 应用颜色到mesh
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # 参考MATLAB风格: 单光源从正前方，material dull
    light_dir = np.array([0.0, 0.0, 1.0])
    ambient = 0.3
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot

    # 将光照强度应用到颜色上
    colors = colors * intensities[:, np.newaxis]
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 8. 交互式可视化（可选）
    if show_interactive:
        print("\n打开交互式可视化窗口...")
        print("  提示: 可以旋转、缩放查看不对称性分布")
        print("  关闭窗口继续生成PDF报告...")

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="Asymmetry Heatmap (from CSV)", width=1200, height=900)
        vis.add_geometry(mesh)

        # 设置渲染选项
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

        # 设置正交投影
        view_control = vis.get_view_control()
        view_control.change_field_of_view(step=-90)

        # 运行可视化
        vis.run()
        vis.destroy_window()

    # 9. 使用Open3D捕获渲染图像
    print("\n生成热力图...")
    print("  使用Open3D渲染...")

    # 创建一个隐藏的可视化窗口用于捕获图像
    vis_capture = o3d.visualization.Visualizer()
    vis_capture.create_window(window_name="Capture", width=1200, height=900, visible=False)
    vis_capture.add_geometry(mesh)

    # 设置渲染选项
    render_option_capture = vis_capture.get_render_option()
    render_option_capture.mesh_show_back_face = True
    render_option_capture.light_on = False  # 关闭Open3D光照，使用预计算的顶点颜色

    # 设置正交投影
    view_control_capture = vis_capture.get_view_control()
    view_control_capture.change_field_of_view(step=-90)

    # 更新渲染
    vis_capture.poll_events()
    vis_capture.update_renderer()

    # 捕获图像
    temp_image = "temp_asymmetry_heatmap_csv.png"
    vis_capture.capture_screen_image(temp_image, do_render=True)
    vis_capture.destroy_window()

    # 验证图像
    if os.path.exists(temp_image):
        file_size = os.path.getsize(temp_image)
        print(f"    ✓ 图像已保存 ({file_size} bytes)")
    else:
        print(f"    ✗ 图像保存失败!")
        raise RuntimeError("热力图生成失败")

    # 10. 生成PDF报告
    print(f"\n生成PDF报告: {output_pdf}")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_pdf)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"  创建输出目录: {output_dir}")

    with PdfPages(output_pdf) as pdf:
        # ========== 热力图（只有一页，无统计信息） ==========
        img = Image.open(temp_image)
        img_array = np.array(img)

        # 裁剪空白区域：找到非白色像素的边界
        if len(img_array.shape) == 3:
            non_white = np.any(img_array < 250, axis=2)
        else:
            non_white = img_array < 250

        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # 添加少量边距
        margin = 10
        rmin = max(0, rmin - margin)
        rmax = min(img_array.shape[0], rmax + margin)
        cmin = max(0, cmin - margin)
        cmax = min(img_array.shape[1], cmax + margin)

        img_cropped = img_array[rmin:rmax, cmin:cmax]

        # 根据裁剪后图像的宽高比调整figure大小
        h, w = img_cropped.shape[:2]
        aspect = w / h
        fig_height = 8
        fig_width = fig_height * aspect + 1.5  # 额外空间给colorbar

        fig1 = plt.figure(figsize=(fig_width, fig_height))
        ax1 = fig1.add_axes([0.02, 0.05, 0.80, 0.90])
        ax1.imshow(img_cropped)
        ax1.axis('off')
        ax1.set_title('Asymmetry Heatmap (from CSV)', fontsize=20, pad=10)

        # 添加colorbar，高度与图像匹配
        # 使用截断的colorbar：底部截断显示灰色，cutoff以上使用完整色谱
        cax = fig1.add_axes([0.85, 0.05, 0.03, 0.90])
        create_truncated_colorbar(fig1, cax, cmap, local_max,
                                  cutoff_distance=cutoff_distance, label='Distance (mm)')

        pdf.savefig(fig1, dpi=300)
        plt.close(fig1)

    # 11. 保存统计信息到文件
    if stats_output_path is None:
        stats_output_path = output_pdf.replace('.pdf', '_stats.txt')

    print(f"\n保存统计信息: {stats_output_path}")

    with open(stats_output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Overall Statistics\n")
        f.write("=" * 60 + "\n")

        overall_stats = compute_stats(valid_distances)
        f.write(f"Max: {overall_stats['Max']:.6f} mm\n")
        f.write(f"MSE: {overall_stats['MSE']:.6f} mm²\n")
        f.write(f"RMSE: {overall_stats['RMSE']:.6f} mm\n")
        f.write(f"MAD: {overall_stats['MAD']:.6f} mm\n")
        f.write(f"% within 0.5mm: {overall_stats['pct_within']:.2f}%\n")

        # 分区统计
        if region_labels is not None:
            f.write("\n" + "=" * 60 + "\n")
            f.write("Statistics by Region\n")
            f.write("=" * 60 + "\n")

            for region in range(1, 7):
                region_mask = (region_labels == region) & valid_mask
                region_distances = distances[region_mask]

                if len(region_distances) > 0:
                    region_stats = compute_stats(region_distances)

                    f.write(f"\nRegion {region}:\n")
                    f.write(f"  Vertices: {len(region_distances)}\n")
                    f.write(f"  Max: {region_stats['Max']:.6f} mm\n")
                    f.write(f"  MSE: {region_stats['MSE']:.6f} mm²\n")
                    f.write(f"  RMSE: {region_stats['RMSE']:.6f} mm\n")
                    f.write(f"  MAD: {region_stats['MAD']:.6f} mm\n")
                    f.write(f"  % within 0.5mm: {region_stats['pct_within']:.2f}%\n")
                else:
                    f.write(f"\nRegion {region}: No valid vertices\n")

    # 12. 清理临时文件
    print("\n清理临时文件...")
    if os.path.exists(temp_image):
        os.remove(temp_image)

    print(f"\n✓ PDF报告已保存: {output_pdf}")
    print(f"✓ 统计信息已保存: {stats_output_path}")

    # 计算分区统计用于返回
    region_stats_dict = {}
    if region_labels is not None:
        for region in range(1, 7):
            region_mask = (region_labels == region) & valid_mask
            region_distances = distances[region_mask]

            if len(region_distances) > 0:
                region_stats_dict[region] = compute_stats(region_distances)
            else:
                # 空区域返回零值
                region_stats_dict[region] = {
                    'count': 0,
                    'Mean': 0.0,
                    'Median': 0.0,
                    'Std': 0.0,
                    'Min': 0.0,
                    'Max': 0.0,
                    'MSE': 0.0,
                    'RMSE': 0.0,
                    'MAD': 0.0,
                    'pct_within': 0.0
                }

    return {
        'vertex_count': len(vertices),
        'valid_count': len(valid_distances),
        'mean_distance': valid_distances.mean(),
        'median_distance': np.median(valid_distances),
        'std_distance': valid_distances.std(),
        'min_distance': valid_distances.min(),
        'max_distance': valid_distances.max(),
        'mse': mse,
        'rmse': rmse,
        'pct_within': overall_stats['pct_within'],
        'output_pdf': output_pdf,
        'stats_file': stats_output_path,
        'region_stats': region_stats_dict
    }


def batch_visualize_asymmetry(
        input_dir,
        output_dir=None,
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,
        cutoff_distance=None,
        distance_method='point_to_point'
):
    """
    批量处理目录下的所有OBJ文件对，为每一对生成不对称性热力图PDF

    文件命名约定：
    - 原始文件: xxx.obj
    - 镜像配准文件: xxx_mirrored_aligned.obj

    参数:
        input_dir: 输入目录路径，包含原始OBJ文件和对应的镜像配准OBJ文件
        output_dir: 输出目录路径（默认为 input_dir/icp_result）
        colormap: 颜色映射方案
        region_labels_path: 分区标签文件路径
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色
        distance_method: 距离计算方法 ('point_to_point' 或 'point_to_surface')

    返回:
        处理结果列表，包含每个文件对的统计信息
    """
    # 设置输出目录：默认保存到 input_dir/icp_result
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'icp_result')

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 查找所有原始OBJ文件（排除 *_mirrored.obj 和 *_mirrored_aligned.obj）
    all_obj_files = glob.glob(os.path.join(input_dir, '*.obj'))
    original_files = [f for f in all_obj_files
                      if not f.endswith('_mirrored.obj')
                      and not f.endswith('_mirrored_aligned.obj')]

    if len(original_files) == 0:
        print(f"在 {input_dir} 中未找到原始OBJ文件")
        return []

    print("=" * 80)
    print(f"批量生成ICP不对称性热力图")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  找到原始文件数: {len(original_files)}")
    print(f"  距离计算方法: {distance_method}")
    print("=" * 80)

    results = []
    success_count = 0
    skip_count = 0

    for obj_path in sorted(original_files):
        # 获取文件名
        filename = os.path.basename(obj_path)
        name, ext = os.path.splitext(filename)

        # 构建镜像配准文件路径
        mirrored_aligned_path = os.path.join(input_dir, f"{name}_mirrored_aligned{ext}")

        # 检查镜像配准文件是否存在
        if not os.path.exists(mirrored_aligned_path):
            print(f"\n跳过: {filename} (未找到对应的镜像配准文件 {name}_mirrored_aligned{ext})")
            skip_count += 1
            continue

        # 构建输出PDF路径
        output_pdf = os.path.join(output_dir, f"{name}_icp_heatmap.pdf")

        print(f"\n处理: {filename}")
        print(f"  原始文件: {obj_path}")
        print(f"  镜像配准文件: {mirrored_aligned_path}")
        print(f"  输出PDF: {output_pdf}")

        # 从文件名提取max_distance（第一个数字 * 2）
        file_max_distance = extract_max_from_filename(filename)
        if file_max_distance is not None:
            print(f"  从文件名提取的colorbar最大值: {file_max_distance} mm")
        else:
            file_max_distance = max_distance  # 回退到传入的默认值

        try:
            # 调用可视化函数
            result = visualize_asymmetry(
                original_obj=obj_path,
                mirrored_registered_obj=mirrored_aligned_path,
                output_pdf=output_pdf,
                colormap=colormap,
                show_interactive=False,
                distance_method=distance_method,
                region_labels_path=region_labels_path,
                max_distance=file_max_distance,
                cutoff_distance=cutoff_distance
            )
            result['original_file'] = filename
            results.append(result)
            success_count += 1

        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'original_file': filename,
                'error': str(e)
            })

    print("\n" + "=" * 80)
    print(f"批量处理完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  总计: {len(original_files)}")
    print("=" * 80)

    return results


def batch_visualize_asymmetry_from_csv(
        input_dir,
        csv_subdir='meshmonk',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,
        cutoff_distance=None,
        reorder=False
):
    """
    批量处理目录下的所有CSV文件，为每个CSV生成不对称性热力图PDF

    文件命名约定：
    - CSV文件: meshmonk/XXX_fa_values.csv
    - OBJ文件: XXX.obj (在input_dir下)
    - 输出PDF: meshmonk/XXX_meshmonk_heatmap.pdf

    参数:
        input_dir: 输入目录路径，包含OBJ文件
        csv_subdir: CSV文件所在子目录名（默认为 'meshmonk'）
        colormap: 颜色映射方案
        region_labels_path: 分区标签文件路径
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色
        reorder: 是否重新排序顶点索引

    返回:
        处理结果列表，包含每个文件的统计信息
    """
    # CSV目录路径
    csv_dir = os.path.join(input_dir, csv_subdir)

    if not os.path.exists(csv_dir):
        print(f"CSV目录不存在: {csv_dir}")
        return []

    # 查找所有CSV文件
    csv_files = glob.glob(os.path.join(csv_dir, '*_fa_values.csv'))

    if len(csv_files) == 0:
        print(f"在 {csv_dir} 中未找到 *_fa_values.csv 文件")
        return []

    print("=" * 80)
    print(f"批量生成MeshMonk不对称性热力图 (从CSV)")
    print(f"  输入目录: {input_dir}")
    print(f"  CSV目录: {csv_dir}")
    print(f"  找到CSV文件数: {len(csv_files)}")
    print("=" * 80)

    results = []
    success_count = 0
    skip_count = 0

    for csv_path in sorted(csv_files):
        # 获取CSV文件名
        csv_filename = os.path.basename(csv_path)
        # 从 XXX_fa_values.csv 提取 XXX
        name = csv_filename.replace('_fa_values.csv', '')

        # 构建OBJ文件路径
        obj_path = os.path.join(input_dir, f"{name}.obj")

        # 检查OBJ文件是否存在
        if not os.path.exists(obj_path):
            print(f"\n跳过: {csv_filename} (未找到对应的OBJ文件 {name}.obj)")
            skip_count += 1
            continue

        # 构建输出PDF路径（保存在CSV同目录下）
        output_pdf = os.path.join(csv_dir, f"{name}_meshmonk_heatmap.pdf")

        print(f"\n处理: {csv_filename}")
        print(f"  OBJ文件: {obj_path}")
        print(f"  CSV文件: {csv_path}")
        print(f"  输出PDF: {output_pdf}")

        # 从文件名提取max_distance（第一个数字 * 2）
        file_max_distance = extract_max_from_filename(csv_filename)
        if file_max_distance is not None:
            print(f"  从文件名提取的colorbar最大值: {file_max_distance} mm")
        else:
            file_max_distance = max_distance  # 回退到传入的默认值

        try:
            # 调用可视化函数
            result = visualize_asymmetry_from_csv(
                obj_path=obj_path,
                csv_path=csv_path,
                output_pdf=output_pdf,
                colormap=colormap,
                show_interactive=False,
                reorder=reorder,
                region_labels_path=region_labels_path,
                max_distance=file_max_distance,
                cutoff_distance=cutoff_distance
            )
            result['csv_file'] = csv_filename
            results.append(result)
            success_count += 1

        except Exception as e:
            print(f"  错误: {e}")
            results.append({
                'csv_file': csv_filename,
                'error': str(e)
            })

    print("\n" + "=" * 80)
    print(f"批量处理完成!")
    print(f"  成功: {success_count}")
    print(f"  跳过: {skip_count}")
    print(f"  总计: {len(csv_files)}")
    print("=" * 80)

    return results


def save_summary_statistics(results, output_csv, method_name='MeshMonk'):
    """
    保存批量处理结果的汇总统计到CSV文件

    参数:
        results: batch_visualize_asymmetry_from_csv 返回的结果列表
        output_csv: 输出CSV文件路径
        method_name: 方法名称（用于日志输出）
    """
    import csv

    print(f"\n保存 {method_name} 汇总统计到: {output_csv}")

    # 过滤掉错误结果
    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) == 0:
        print("  警告: 没有有效结果可保存")
        return

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        header = ['Filename', 'Total_Vertices', 'Mean', 'Median', 'Std', 'Min', 'Max', 'RMSE', 'Pct_Within_Cutoff']

        # 添加分区统计列
        for region in range(1, 7):
            header.extend([
                f'R{region}_Count', f'R{region}_Mean', f'R{region}_Max', f'R{region}_RMSE', f'R{region}_Pct_Within'
            ])

        writer.writerow(header)

        # 写入每个文件的数据
        for result in valid_results:
            # 获取文件名（优先使用 csv_file，其次使用 output_pdf）
            if 'csv_file' in result:
                filename = result['csv_file']
            elif 'output_pdf' in result:
                filename = os.path.basename(result['output_pdf'])
            else:
                filename = 'unknown'

            row = [
                filename,
                result.get('vertex_count', 0),
                f"{result.get('mean_distance', 0.0):.6f}",
                f"{result.get('median_distance', 0.0):.6f}",
                f"{result.get('std_distance', 0.0):.6f}",
                f"{result.get('min_distance', 0.0):.6f}",
                f"{result.get('max_distance', 0.0):.6f}",
                f"{result.get('rmse', 0.0):.6f}",
                f"{result.get('pct_within', 0.0):.2f}"
            ]

            # 添加分区统计
            region_stats = result.get('region_stats', {})
            for region in range(1, 7):
                if region in region_stats:
                    stats = region_stats[region]
                    row.extend([
                        stats.get('count', 0),
                        f"{stats.get('Mean', 0.0):.6f}",
                        f"{stats.get('Max', 0.0):.6f}",
                        f"{stats.get('RMSE', 0.0):.6f}",
                        f"{stats.get('pct_within', 0.0):.2f}"
                    ])
                else:
                    # 没有分区数据，填充零值
                    row.extend([0, '0.000000', '0.000000', '0.000000', '0.00'])

            writer.writerow(row)

    print(f"  完成! 保存了 {len(valid_results)} 个文件的统计信息")


if __name__ == "__main__":
    # ========== ICP批量处理 ==========
    print("\n" + "=" * 80)
    print("批量生成ICP不对称性热力图 (point-to-surface)")
    print("=" * 80)

    # 处理kedian目录 - ICP方法
    results_kedian_icp = batch_visualize_asymmetry(
        input_dir='kedian',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        distance_method='point_to_surface'
    )
    # 保存ICP汇总统计
    save_summary_statistics(
        results_kedian_icp,
        output_csv='kedian/icp_result/icp_summary_statistics.csv',
        method_name='ICP'
    )

    # 处理bijian目录 - ICP方法
    results_bijian_icp = batch_visualize_asymmetry(
        input_dir='bijian',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        distance_method='point_to_surface'
    )
    # 保存ICP汇总统计
    save_summary_statistics(
        results_bijian_icp,
        output_csv='bijian/icp_result/icp_summary_statistics.csv',
        method_name='ICP'
    )

    # ========== MeshMonk CSV批量处理 ==========
    print("\n" + "=" * 80)
    print("批量生成MeshMonk不对称性热力图 (从CSV)")
    print("=" * 80)

    # 处理kedian目录 - MeshMonk CSV方法
    results_kedian_csv = batch_visualize_asymmetry_from_csv(
        input_dir='kedian',
        csv_subdir='meshmonk',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        reorder=True
    )
    # 保存MeshMonk汇总统计
    save_summary_statistics(
        results_kedian_csv,
        output_csv='kedian/meshmonk/meshmonk_summary_statistics.csv',
        method_name='MeshMonk'
    )

    # 处理bijian目录 - MeshMonk CSV方法
    results_bijian_csv = batch_visualize_asymmetry_from_csv(
        input_dir='bijian',
        csv_subdir='meshmonk',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        reorder=True
    )
    # 保存MeshMonk汇总统计
    save_summary_statistics(
        results_bijian_csv,
        output_csv='bijian/meshmonk/meshmonk_summary_statistics.csv',
        method_name='MeshMonk'
    )

    print("\n" + "=" * 80)
    print("所有批处理完成!")
    print("=" * 80)

