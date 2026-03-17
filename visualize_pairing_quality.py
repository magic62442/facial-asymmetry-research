import open3d as o3d
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image


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


def extract_max_from_filename(filename, multiplier=2):
    """
    从文件名中提取第一个数字，乘以multiplier作为colorbar最大值

    例如: 2_40_directional_fa_values.csv, multiplier=2 -> 2 * 2 = 4

    参数:
        filename: 文件名
        multiplier: 乘数（默认2，xiahedian使用1）

    返回:
        colorbar最大值，如果无法提取则返回None
    """
    import re
    # 提取文件名中的第一个数字
    match = re.search(r'(\d+)', filename)
    if match:
        first_number = int(match.group(1))
        return first_number * multiplier
    return None


def get_region_ids(region_labels):
    """返回标签数组中所有有效区域ID（忽略0）。"""
    if region_labels is None:
        return []
    return sorted(int(region) for region in np.unique(region_labels) if region > 0)


def load_pairing_csv(csv_path):
    """
    从CSV文件读取配对结果

    参数:
        csv_path: CSV文件路径

    返回:
        source_ids: 源顶点ID列表
        target_ids: 目标顶点ID列表
        distances: 配对距离列表
    """
    source_ids = []
    target_ids = []
    distances = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        two_col = len(header) < 3  # 兼容2列格式（source_vertex_id, distance_to_surface）

        for row in reader:
            source_ids.append(int(row[0]))
            if two_col:
                target_ids.append(int(row[0]))  # dummy，2列格式无target_id
                distances.append(float(row[1]))
            else:
                target_ids.append(int(row[1]))
                distances.append(float(row[2]))

    return np.array(source_ids), np.array(target_ids), np.array(distances)

def recompute_distances_from_pairing(obj_path, csv_path):
    """
    根据 CSV 文件中的顶点对，在 OBJ 文件上重新计算实际的几何距离。

    不使用 CSV 文件中的第三列 distance，而是根据顶点索引重新计算。

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径（只使用前两列：source_id, target_id）

    返回:
        元组 (source_ids, target_ids, recomputed_distances)
        - source_ids: 源顶点索引数组
        - target_ids: 目标顶点索引数组
        - recomputed_distances: 重新计算的距离数组
    """
    print("=" * 80)
    print("根据顶点对重新计算距离")
    print("=" * 80)

    # 1. 加载 OBJ 文件
    print(f"\n加载 OBJ 文件: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)
    print(f"  顶点数: {len(vertices)}")

    # 2. 读取 CSV 文件（只读取前两列）
    print(f"\n读取 CSV 文件: {csv_path}")
    source_ids = []
    target_ids = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)  # 跳过表头
        print(f"  CSV 表头: {header}")

        for row in reader:
            source_ids.append(int(row[0]))
            target_ids.append(int(row[1]))
            # 注意：不读取第三列的距离

    source_ids = np.array(source_ids)
    target_ids = np.array(target_ids)
    print(f"  配对数量: {len(source_ids)}")

    # 3. 验证索引有效性
    max_source_id = source_ids.max()
    max_target_id = target_ids.max()

    if max_source_id >= len(vertices):
        raise ValueError(f"源顶点索引超出范围: {max_source_id} >= {len(vertices)}")
    if max_target_id >= len(vertices):
        raise ValueError(f"目标顶点索引超出范围: {max_target_id} >= {len(vertices)}")

    print(f"  源顶点索引范围: [{source_ids.min()}, {max_source_id}]")
    print(f"  目标顶点索引范围: [{target_ids.min()}, {max_target_id}]")
    print(f"  ✓ 索引有效")

    # 4. 重新计算距离
    print(f"\n重新计算距离...")
    recomputed_distances = np.zeros(len(source_ids))

    for i in range(len(source_ids)):
        src_id = source_ids[i]
        tgt_id = target_ids[i]

        # 获取顶点坐标
        src_vertex = vertices[src_id]
        tgt_vertex = vertices[tgt_id]

        # 计算欧氏距离
        distance = np.linalg.norm(src_vertex - tgt_vertex)
        recomputed_distances[i] = distance

    # 5. 统计信息
    print(f"\n重新计算的距离统计:")
    print(f"  平均值: {recomputed_distances.mean():.6f} mm")
    print(f"  中位数: {np.median(recomputed_distances):.6f} mm")
    print(f"  标准差: {recomputed_distances.std():.6f} mm")
    print(f"  最小值: {recomputed_distances.min():.6f} mm")
    print(f"  最大值: {recomputed_distances.max():.6f} mm")

    mse = np.mean(recomputed_distances ** 2)
    rmse = np.sqrt(mse)
    print(f"  MSE: {mse:.6f} mm²")
    print(f"  RMSE: {rmse:.6f} mm")

    print(f"\n{'=' * 80}")
    print("距离重新计算完成")
    print(f"{'=' * 80}\n")

    return source_ids, target_ids, recomputed_distances

def visualize_pairing_quality(obj_path, csv_path, output_pdf='pairing_quality.pdf', colormap='jet',
                               region_labels_path='region_labels.txt', stats_output_path=None,
                               max_distance=None, cutoff_distance=None, title='Ground Truth Heatmap',
                               generate_visualization=True):
    """
    可视化配对质量热力图

    参数:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径（包含配对和距离）
        output_pdf: 输出PDF文件名
        colormap: 颜色映射方案 ('jet', 'coolwarm', 'viridis', 'hot')
        region_labels_path: 分区标签文件路径
        stats_output_path: 统计信息输出文件路径（默认与PDF同名但扩展名为.txt）
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色
        generate_visualization: 是否生成热力图/PDF，False时仅输出统计结果

    返回:
        统计信息字典
    """
    print("=" * 80)
    print("配对质量可视化")
    print("=" * 80)

    # 1. 加载OBJ文件
    print(f"\n加载模型: {obj_path}")
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    print(f"顶点数: {len(vertices)}")

    # 2. 加载配对CSV
    print(f"\n加载配对数据: {csv_path}")
    source_ids, target_ids, distances = load_pairing_csv(csv_path)

    print(f"配对数量: {len(source_ids)}")

    # 验证第一列包含所有顶点
    if len(source_ids) != len(vertices):
        print(f"\n警告: CSV第一列数量 ({len(source_ids)}) 与OBJ顶点数 ({len(vertices)}) 不匹配!")
        print("请确保第一列包含所有顶点的配对信息")

    # 检查source_ids是否覆盖所有顶点
    unique_sources = set(source_ids)
    missing_vertices = set(range(len(vertices))) - unique_sources
    if missing_vertices:
        print(f"\n警告: 缺少 {len(missing_vertices)} 个顶点的配对信息")
        print(f"  缺失的顶点ID: {sorted(list(missing_vertices))[:10]}...")  # 显示前10个

    # 3. 创建距离数组（每个顶点对应一个距离值）
    # 初始化为NaN
    vertex_distances = np.full(len(vertices), np.nan)

    # 填充距离值
    for src, dist in zip(source_ids, distances):
        if 0 <= src < len(vertices):
            vertex_distances[src] = dist

    # 4. 统计分析
    valid_distances = vertex_distances[~np.isnan(vertex_distances)]

    print(f"\n配对距离统计:")
    print(f"  有效配对数: {len(valid_distances)} / {len(vertices)}")
    print(f"  平均距离: {valid_distances.mean():.6f} mm")
    print(f"  中位数距离: {np.median(valid_distances):.6f} mm")
    print(f"  标准差: {valid_distances.std():.6f} mm")
    print(f"  最小距离: {valid_distances.min():.6f} mm")
    print(f"  最大距离: {valid_distances.max():.6f} mm")

    # 计算MSE和RMSE
    mse = np.mean(valid_distances ** 2)
    rmse = np.sqrt(mse)

    print(f"\n误差度量:")
    print(f"  MSE (Mean Squared Error): {mse:.6f} mm²")
    print(f"  RMSE (Root Mean Squared Error): {rmse:.6f} mm")

    # 5. 加载分区标签
    region_labels = None
    region_ids = []
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
        region_ids = get_region_ids(region_labels)
        print(f"  加载完成，共{len(region_ids)}个区域")

    # 6. 创建颜色映射并生成热力图（可选）
    if generate_visualization:
        print(f"\n创建热力图 (颜色方案: {colormap})...")

        if max_distance is not None:
            local_max = max_distance
        else:
            local_max = valid_distances.max()

        print(f"  colorbar范围: 0 - {local_max:.6f} mm")
        if cutoff_distance is not None:
            print(f"  截断值: {cutoff_distance:.6f} mm (以下显示浅灰色)")

        cmap = plt.get_cmap(colormap)
        colors = np.zeros((len(vertices), 3))

        for i in range(len(vertices)):
            if not np.isnan(vertex_distances[i]):
                if cutoff_distance is not None and vertex_distances[i] <= cutoff_distance:
                    colors[i] = [0.85, 0.85, 0.85]
                else:
                    if cutoff_distance is not None and local_max > cutoff_distance:
                        normalized_val = (vertex_distances[i] - cutoff_distance) / (local_max - cutoff_distance)
                    else:
                        normalized_val = vertex_distances[i] / local_max
                    normalized_val = min(1.0, max(0.0, normalized_val))
                    colors[i] = cmap(normalized_val)[:3]
            else:
                colors[i] = [0.5, 0.5, 0.5]

        mesh.compute_vertex_normals()
        normals = np.asarray(mesh.vertex_normals)

        light_dir = np.array([0.0, 0.0, 1.0])
        ambient = 0.3
        dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
        intensities = ambient + (1 - ambient) * dot

        colors = colors * intensities[:, np.newaxis]
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        print("\n生成热力图...")
        print("  使用Open3D渲染...")

        temp_image = "temp_pairing_quality_heatmap.png"
        vis_capture = o3d.visualization.Visualizer()
        vis_capture.create_window(window_name="Capture", width=1200, height=900, visible=False)
        vis_capture.add_geometry(mesh)

        render_option_capture = vis_capture.get_render_option()
        render_option_capture.mesh_show_back_face = True
        render_option_capture.light_on = False

        view_control_capture = vis_capture.get_view_control()
        view_control_capture.change_field_of_view(step=-90)

        vis_capture.poll_events()
        vis_capture.update_renderer()
        vis_capture.capture_screen_image(temp_image, do_render=True)
        vis_capture.destroy_window()

        if os.path.exists(temp_image):
            file_size = os.path.getsize(temp_image)
            print(f"    ✓ 图像已保存 ({file_size} bytes)")
        else:
            print(f"    ✗ 图像保存失败!")
            raise RuntimeError("热力图生成失败")

        print(f"\n生成PDF报告: {output_pdf}")

        output_dir = os.path.dirname(output_pdf)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  创建输出目录: {output_dir}")

        with PdfPages(output_pdf) as pdf:
            img = Image.open(temp_image)
            img_array = np.array(img)

            if len(img_array.shape) == 3:
                non_white = np.any(img_array < 250, axis=2)
            else:
                non_white = img_array < 250

            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            margin = 10
            rmin = max(0, rmin - margin)
            rmax = min(img_array.shape[0], rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(img_array.shape[1], cmax + margin)

            img_cropped = img_array[rmin:rmax, cmin:cmax]

            h, w = img_cropped.shape[:2]
            aspect = w / h
            fig_height = 8
            fig_width = fig_height * aspect + 1.5

            fig1 = plt.figure(figsize=(fig_width, fig_height))
            ax1 = fig1.add_axes([0.02, 0.05, 0.80, 0.90])
            ax1.imshow(img_cropped)
            ax1.axis('off')
            ax1.set_title(title, fontsize=20, pad=10)

            cax = fig1.add_axes([0.85, 0.05, 0.03, 0.90])
            create_truncated_colorbar(fig1, cax, cmap, local_max,
                                      cutoff_distance=cutoff_distance, label='Distance (mm)')

            pdf.savefig(fig1, dpi=300)
            plt.close(fig1)

        print("\n清理临时文件...")
        if os.path.exists(temp_image):
            os.remove(temp_image)

        print(f"\n✓ PDF报告已保存: {output_pdf}")
    else:
        print("\n跳过热力图和PDF生成（generate_visualization=False）")

    # 10. 保存统计信息到文件
    if stats_output_path is None:
        stats_output_path = output_pdf.replace('.pdf', '_stats.txt')

    print(f"\n保存统计信息: {stats_output_path}")

    # 定义计算统计指标的函数
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
            'Mean': mean_d,
            'Max': max_d,
            'MSE': mse,
            'RMSE': rmse,
            'MAD': mad,
            'pct_within': pct_within
        }

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

            for region in region_ids:
                region_mask = (region_labels == region) & (~np.isnan(vertex_distances))
                region_distances = vertex_distances[region_mask]

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

    print(f"✓ 统计信息已保存: {stats_output_path}")

    return {
        'mean_distance': valid_distances.mean(),
        'median_distance': np.median(valid_distances),
        'std_distance': valid_distances.std(),
        'min_distance': valid_distances.min(),
        'max_distance': valid_distances.max(),
        'mse': mse,
        'rmse': rmse,
        'output_pdf': output_pdf if generate_visualization else None,
        'stats_file': stats_output_path,
        'visualization_generated': generate_visualization
    }

def visualize_mirrored_registration_pairing(
        obj1_path="displaced_directional/25_0.3.obj",
        obj2_path=None,
        csv_path="pairs.csv",
        output_pdf="mirrored_registration_heatmap.pdf",
        colormap='jet',
        region_labels_path='region_labels.txt',
        stats_output_path=None,
        max_distance=None,
        cutoff_distance=None,
        generate_visualization=True
):
    """
    使用两个OBJ文件进行镜像距离可视化。

    流程：
    1. 加载obj1和obj2 (如果提供了obj2_path)
    2. 如果提供了obj2_path：
       - 将obj1关于x=0平面镜像
       - 根据pairs.csv计算distance(obj2[src_id], mirrored_obj1[tgt_id])
       - 在obj2上可视化距离热图
    3. 如果未提供obj2_path（默认行为）：
       - 将obj1关于x=0平面镜像
       - 根据pairs.csv计算distance(obj1[src_id], mirrored_obj1[tgt_id])
       - 在obj1上可视化距离热图

    参数：
        obj1_path: 第一个OBJ文件路径（用于镜像）
        obj2_path: 第二个OBJ文件路径（可选，用于source顶点）
        csv_path: 顶点配对CSV文件路径
        output_pdf: 输出PDF文件路径
        colormap: 颜色映射方案

    返回：
        包含统计信息的字典
    """
    print("=" * 80)
    print("镜像距离可视化")
    print("=" * 80)

    # 1. 加载OBJ文件
    print(f"\n加载OBJ文件...")
    print(f"  OBJ1 (用于镜像): {obj1_path}")

    mesh1 = o3d.io.read_triangle_mesh(obj1_path)
    vertices1 = np.asarray(mesh1.vertices)
    print(f"  OBJ1顶点数: {len(vertices1)}")

    # 如果提供了obj2_path，加载obj2
    if obj2_path is not None:
        print(f"  OBJ2 (用于source): {obj2_path}")
        mesh2 = o3d.io.read_triangle_mesh(obj2_path)
        vertices2 = np.asarray(mesh2.vertices)
        print(f"  OBJ2顶点数: {len(vertices2)}")

        # 验证顶点数是否一致
        if len(vertices1) != len(vertices2):
            print(f"\n警告: OBJ1和OBJ2的顶点数不一致!")
            print(f"  OBJ1: {len(vertices1)}")
            print(f"  OBJ2: {len(vertices2)}")

        # 使用obj2作为source
        source_vertices = vertices2
        visualization_mesh = mesh2
    else:
        # 使用obj1作为source
        source_vertices = vertices1
        visualization_mesh = mesh1

    # 2. 加载配对CSV
    print(f"\n加载配对文件: {csv_path}")
    source_ids = []
    target_ids = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            source_ids.append(int(row[0]))
            target_ids.append(int(row[1]))

    source_ids = np.array(source_ids)
    target_ids = np.array(target_ids)
    print(f"  配对数量: {len(source_ids)}")

    # 3. 对obj1应用镜像变换
    print(f"\n对OBJ1应用镜像变换（x=0平面）...")
    vertices1_mirrored = vertices1.copy()
    vertices1_mirrored[:, 0] = -vertices1_mirrored[:, 0]  # 翻转x坐标
    print(f"  镜像完成")

    # 4. 计算距离：distance(source[src_id], mirrored_obj1[tgt_id])
    print(f"\n计算配对距离...")
    if obj2_path is not None:
        print(f"  计算: obj2[source] <-> mirrored_obj1[target]")
    else:
        print(f"  计算: obj1[source] <-> mirrored_obj1[target]")

    distances = np.zeros(len(source_ids))

    for i in range(len(source_ids)):
        src_id = source_ids[i]
        tgt_id = target_ids[i]

        # source顶点（来自obj2或obj1）
        src_vertex = source_vertices[src_id]
        # 镜像后的目标点（来自obj1）
        tgt_vertex = vertices1_mirrored[tgt_id]

        # 计算欧氏距离
        distance = np.linalg.norm(src_vertex - tgt_vertex)
        distances[i] = distance

    # 5. 统计信息
    mean_dist = distances.mean()
    median_dist = np.median(distances)
    std_dist = distances.std()
    max_dist = distances.max()
    min_dist = distances.min()
    mse = np.mean(distances ** 2)
    rmse = np.sqrt(mse)

    print(f"\n距离统计:")
    print(f"  配对数量: {len(distances)}")
    print(f"  平均距离: {mean_dist:.6f} mm")
    print(f"  中位数距离: {median_dist:.6f} mm")
    print(f"  标准差: {std_dist:.6f} mm")
    print(f"  最小距离: {min_dist:.6f} mm")
    print(f"  最大距离: {max_dist:.6f} mm")
    print(f"  MSE: {mse:.6f} mm²")
    print(f"  RMSE: {rmse:.6f} mm")

    percentiles = [50, 75, 90, 95, 99]
    print(f"\n距离百分位数:")
    for p in percentiles:
        val = np.percentile(distances, p)
        print(f"  {p}%: {val:.6f} mm")

    # 6. 加载分区标签
    region_labels = None
    region_ids = []
    if os.path.exists(region_labels_path):
        print(f"\n加载分区标签: {region_labels_path}")
        region_labels = np.zeros(len(source_vertices), dtype=int)
        with open(region_labels_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                parts = line.split(',')
                if len(parts) == 2:
                    idx, label = int(parts[0]), int(parts[1])
                    if 0 <= idx < len(source_vertices):
                        region_labels[idx] = label
        region_ids = get_region_ids(region_labels)
        print(f"  加载完成，共{len(region_ids)}个区域")

    # 7. 创建颜色映射到所有顶点
    vertex_distances = np.zeros(len(source_vertices))
    vertex_distances[:] = np.nan  # 默认无效

    # 将配对距离映射到源顶点
    for i in range(len(source_ids)):
        src_id = source_ids[i]
        vertex_distances[src_id] = distances[i]

    # 处理有效距离
    valid_mask = ~np.isnan(vertex_distances)
    valid_distances = vertex_distances[valid_mask]

    print(f"  有效顶点数: {valid_mask.sum()}/{len(source_vertices)}")

    if generate_visualization:
        print(f"\n准备可视化...")

        if max_distance is not None:
            local_max = max_distance
        else:
            local_max = valid_distances.max()

        print(f"  colorbar范围: 0 - {local_max:.6f} mm")
        if cutoff_distance is not None:
            print(f"  截断值: {cutoff_distance:.6f} mm (以下显示浅灰色)")

        cmap = plt.get_cmap(colormap)
        colors = np.zeros((len(source_vertices), 3))
        for i in range(len(source_vertices)):
            if valid_mask[i]:
                dist = vertex_distances[i]
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

        visualization_mesh.compute_vertex_normals()
        normals = np.asarray(visualization_mesh.vertex_normals)

        light_dir = np.array([0.0, 0.0, 1.0])
        ambient = 0.3
        dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
        intensities = ambient + (1 - ambient) * dot

        colors = colors * intensities[:, np.newaxis]
        visualization_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

        print(f"\n生成热力图...")
        print("  使用Open3D渲染...")

        temp_image = "temp_mirrored_registration_heatmap.png"
        vis_capture = o3d.visualization.Visualizer()
        vis_capture.create_window(window_name="Capture", width=1200, height=900, visible=False)
        vis_capture.add_geometry(visualization_mesh)

        render_option_capture = vis_capture.get_render_option()
        render_option_capture.mesh_show_back_face = True
        render_option_capture.light_on = False

        view_control_capture = vis_capture.get_view_control()
        view_control_capture.change_field_of_view(step=-90)

        vis_capture.poll_events()
        vis_capture.update_renderer()
        vis_capture.capture_screen_image(temp_image, do_render=True)
        vis_capture.destroy_window()

        if os.path.exists(temp_image):
            file_size = os.path.getsize(temp_image)
            print(f"    ✓ 图像已保存 ({file_size} bytes)")
        else:
            print(f"    ✗ 图像保存失败!")
            raise RuntimeError("热力图生成失败")

        print(f"\n生成PDF报告: {output_pdf}")

        output_dir = os.path.dirname(output_pdf)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"  创建输出目录: {output_dir}")

        with PdfPages(output_pdf) as pdf:
            img = Image.open(temp_image)
            img_array = np.array(img)

            if len(img_array.shape) == 3:
                non_white = np.any(img_array < 250, axis=2)
            else:
                non_white = img_array < 250

            rows = np.any(non_white, axis=1)
            cols = np.any(non_white, axis=0)
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            margin = 10
            rmin = max(0, rmin - margin)
            rmax = min(img_array.shape[0], rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(img_array.shape[1], cmax + margin)

            img_cropped = img_array[rmin:rmax, cmin:cmax]

            h, w = img_cropped.shape[:2]
            aspect = w / h
            fig_height = 8
            fig_width = fig_height * aspect + 1.5

            fig1 = plt.figure(figsize=(fig_width, fig_height))
            ax1 = fig1.add_axes([0.02, 0.05, 0.80, 0.90])
            ax1.imshow(img_cropped)
            ax1.axis('off')
            ax1.set_title('Ground Truth Heatmap', fontsize=20, pad=10)

            cax = fig1.add_axes([0.85, 0.05, 0.03, 0.90])
            create_truncated_colorbar(fig1, cax, cmap, local_max,
                                      cutoff_distance=cutoff_distance, label='Distance (mm)')

            pdf.savefig(fig1, dpi=300)
            plt.close(fig1)

        print("\n清理临时文件...")
        if os.path.exists(temp_image):
            os.remove(temp_image)

        print(f"\n✓ PDF报告已保存: {output_pdf}")
    else:
        print("\n跳过热力图和PDF生成（generate_visualization=False）")

    # 11. 保存统计信息到文件
    if stats_output_path is None:
        stats_output_path = output_pdf.replace('.pdf', '_stats.txt')

    print(f"\n保存统计信息: {stats_output_path}")

    # 定义计算统计指标的函数
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
            'Mean': mean_d,
            'Max': max_d,
            'MSE': mse,
            'RMSE': rmse,
            'MAD': mad,
            'pct_within': pct_within
        }

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

            for region in region_ids:
                region_mask = (region_labels == region) & (~np.isnan(vertex_distances))
                region_distances = vertex_distances[region_mask]

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

    print(f"✓ 统计信息已保存: {stats_output_path}")

    # 计算分区统计用于返回
    region_stats_dict = {}
    if region_labels is not None:
        for region in region_ids:
            region_mask = (region_labels == region) & (~np.isnan(vertex_distances))
            region_distances = vertex_distances[region_mask]

            if len(region_distances) > 0:
                region_stats_dict[region] = compute_stats(region_distances)
                # 添加count字段
                region_stats_dict[region]['count'] = len(region_distances)
            else:
                # 空区域返回零值
                region_stats_dict[region] = {
                    'count': 0,
                    'Max': 0.0,
                    'MSE': 0.0,
                    'RMSE': 0.0,
                    'MAD': 0.0,
                    'pct_within': 0.0
                }

    # 计算整体统计
    overall_stats = compute_stats(valid_distances)

    return {
        'vertex_count': len(vertex_distances),
        'mean_distance': mean_dist,
        'median_distance': median_dist,
        'std_distance': std_dist,
        'min_distance': min_dist,
        'max_distance': max_dist,
        'mse': mse,
        'rmse': rmse,
        'pct_within': overall_stats['pct_within'],
        'output_pdf': output_pdf if generate_visualization else None,
        'stats_file': stats_output_path,
        'region_stats': region_stats_dict,
        'visualization_generated': generate_visualization
    }


def batch_visualize_mirror_distance(
        input_dir,
        csv_path="pairs.csv",
        output_dir=None,
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,
        cutoff_distance=None,
        generate_visualization=True
):
    """
    批量处理目录下的所有原始和镜像OBJ文件对，为每一对生成热力图PDF

    文件命名约定：
    - 原始文件: xxx.obj
    - 镜像文件: xxx_mirrored.obj

    参数:
        input_dir: 输入目录路径，包含原始OBJ文件和对应的镜像OBJ文件
        csv_path: 顶点配对CSV文件路径（用于所有文件对）
        output_dir: 输出目录路径（默认为 input_dir/ground_truth）
        colormap: 颜色映射方案
        region_labels_path: 分区标签文件路径
        max_distance: colorbar最大值（None则自动计算）
        cutoff_distance: 截断值，distance <= cutoff时显示浅灰色
        generate_visualization: 是否生成热力图/PDF，False时仅输出统计结果

    返回:
        处理结果列表，包含每个文件对的统计信息
    """
    import glob

    # 设置输出目录：默认保存到 input_dir/ground_truth
    if output_dir is None:
        output_dir = os.path.join(input_dir, 'ground_truth')

    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 查找所有非镜像OBJ文件（排除 *_mirrored.obj）
    all_obj_files = glob.glob(os.path.join(input_dir, '*.obj'))
    original_files = [f for f in all_obj_files if not f.endswith('_mirrored.obj')]

    if len(original_files) == 0:
        print(f"在 {input_dir} 中未找到原始OBJ文件")
        return []

    print("=" * 80)
    print(f"批量生成镜像距离热力图")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")
    print(f"  配对文件: {csv_path}")
    print(f"  找到原始文件数: {len(original_files)}")
    print("=" * 80)

    results = []
    success_count = 0
    skip_count = 0

    for obj_path in sorted(original_files):
        # 获取文件名
        filename = os.path.basename(obj_path)
        name, ext = os.path.splitext(filename)

        # 构建镜像文件路径
        mirrored_path = os.path.join(input_dir, f"{name}_mirrored{ext}")

        # 检查镜像文件是否存在
        if not os.path.exists(mirrored_path):
            print(f"\n跳过: {filename} (未找到对应的镜像文件 {name}_mirrored{ext})")
            skip_count += 1
            continue

        # 构建输出PDF路径
        output_pdf = os.path.join(output_dir, f"{name}_heatmap.pdf")

        print(f"\n处理: {filename}")
        print(f"  原始文件: {obj_path}")
        print(f"  镜像文件: {mirrored_path}")
        print(f"  输出PDF: {output_pdf}")

        # 从文件名提取max_distance（xiahedian用offset，其他用offset*2）
        multiplier = 1 if 'xiahedian' in input_dir else 2
        file_max_distance = extract_max_from_filename(filename, multiplier=multiplier)
        if file_max_distance is not None:
            print(f"  从文件名提取的colorbar最大值: {file_max_distance} mm")
        else:
            file_max_distance = max_distance  # 回退到传入的默认值

        try:
            # 调用可视化函数
            result = visualize_mirrored_registration_pairing(
                obj1_path=obj_path,
                obj2_path=obj_path,
                csv_path=csv_path,
                output_pdf=output_pdf,
                colormap=colormap,
                region_labels_path=region_labels_path,
                max_distance=file_max_distance,
                cutoff_distance=cutoff_distance,
                generate_visualization=generate_visualization
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


def save_summary_statistics(results, output_csv, method_name='Ground Truth'):
    """
    保存批量处理结果的汇总统计到CSV文件

    参数:
        results: batch_visualize_mirror_distance 返回的结果列表
        output_csv: 输出CSV文件路径
        method_name: 方法名称（用于日志输出）
    """
    print(f"\n保存 {method_name} 汇总统计到: {output_csv}")

    # 过滤掉错误结果
    valid_results = [r for r in results if 'error' not in r]

    if len(valid_results) == 0:
        print("  警告: 没有有效结果可保存")
        return

    region_ids = sorted({
        int(region)
        for result in valid_results
        for region in result.get('region_stats', {}).keys()
        if int(region) > 0
    })

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        header = ['Filename', 'Total_Vertices', 'Mean', 'Median', 'Std', 'Min', 'Max', 'RMSE', 'Pct_Within_Cutoff']

        # 添加分区统计列
        for region in region_ids:
            header.extend([
                f'R{region}_Count', f'R{region}_Mean', f'R{region}_Max', f'R{region}_RMSE', f'R{region}_Pct_Within'
            ])

        writer.writerow(header)

        # 写入每个文件的数据
        for result in valid_results:
            # 获取文件名
            if result.get('original_file'):
                filename = result['original_file']
            elif result.get('output_pdf'):
                filename = os.path.basename(result['output_pdf'])
            elif result.get('stats_file'):
                filename = os.path.basename(result['stats_file'])
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
            for region in region_ids:
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
    # 批量处理kedian和bijian目录
    print("\n" + "=" * 80)
    print("批量生成Ground Truth热力图")
    print("=" * 80)

    # 处理kedian目录
    results_kedian = batch_visualize_mirror_distance(
        input_dir='kedian',
        csv_path='pairs.csv',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        generate_visualization=False
    )
    # 保存汇总统计
    save_summary_statistics(
        results_kedian,
        output_csv='kedian/ground_truth/ground_truth_summary_statistics.csv',
        method_name='Ground Truth'
    )

    #处理bijian目录
    results_bijian = batch_visualize_mirror_distance(
        input_dir='bijian',
        csv_path='pairs.csv',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,      # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,    # 设置截断值
        generate_visualization=False
    )
    # 保存汇总统计
    save_summary_statistics(
        results_bijian,
        output_csv='bijian/ground_truth/ground_truth_summary_statistics.csv',
        method_name='Ground Truth'
    )

    # 处理xiahedian目录
    results_xiahedian = batch_visualize_mirror_distance(
        input_dir='xiahedian',
        csv_path='pairs.csv',
        colormap='jet',
        region_labels_path='region_labels.txt',
        max_distance=None,  # None表示自动计算每个文件的最大值
        cutoff_distance=0.5,  # 设置截断值
        generate_visualization=False
    )
    # 保存汇总统计
    save_summary_statistics(
        results_xiahedian,
        output_csv='xiahedian/ground_truth/ground_truth_summary_statistics.csv',
        method_name='Ground Truth'
    )

    print("\n" + "=" * 80)
    print("所有批处理完成!")
    print("=" * 80)
