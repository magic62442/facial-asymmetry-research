#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch Asymmetry Analysis with Region Partition (CSV-based)

批量处理output_ppdh和output_headspace目录下的OBJ文件及其对应的CSV不对称性数据，
为每个文件生成热力图并计算整体和分区的统计值。

处理流程：
1. 读取OBJ文件（用于可视化）
2. 读取对应的FA结果CSV文件（包含每个顶点的距离）
3. 重新排序顶点（适配Open3D的顶点顺序）
4. 使用region_labels.txt中的分区标签（基于顶点ID）
5. 计算整体统计值和各分区统计值（基于CSV数据）
6. 生成热力图PDF（使用CSV中的距离值）
7. 保存所有结果到CSV汇总文件
"""

import os
import glob
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.backends.backend_pdf import PdfPages
import csv
from PIL import Image
from vertex_reorder import load_mapping


def load_region_labels_from_file(region_labels_path, n_vertices=7160):
    """
    从region_labels.txt文件读取分区标签

    Parameters:
        region_labels_path: region_labels.txt文件路径
        n_vertices: 顶点总数

    Returns:
        region_labels: numpy array of region labels for each vertex
    """
    region_labels = np.zeros(n_vertices, dtype=int)

    if not os.path.exists(region_labels_path):
        print(f"  警告: 分区文件不存在: {region_labels_path}")
        return region_labels

    with open(region_labels_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split(',')
            if len(parts) == 2:
                try:
                    idx = int(parts[0])
                    label = int(parts[1])
                    if 0 <= idx < n_vertices:
                        region_labels[idx] = label
                except ValueError:
                    continue

    return region_labels


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


def compute_statistics(distances, cutoff_distance=0.5):
    """
    计算统计值

    Parameters:
        distances: numpy array of distances
        cutoff_distance: cutoff threshold in mm (用于计算 pct_within)

    Returns:
        stats: dictionary of statistics
    """
    # 移除 NaN 值
    valid_mask = ~np.isnan(distances)
    clean_distances = distances[valid_mask]

    if len(clean_distances) == 0:
        return {
            'count': 0,
            'mean': 0.0,
            'median': 0.0,
            'std': 0.0,
            'min': 0.0,
            'max': 0.0,
            'mse': 0.0,
            'rmse': 0.0,
            'percentage': 0.0,
            'pct_within': 0.0,
            'mad': 0.0
        }

    # 计算cutoff内的比例
    within_cutoff = np.sum(clean_distances <= cutoff_distance)
    pct_within = within_cutoff / len(clean_distances) * 100
    mean_d = np.mean(clean_distances)

    stats = {
        'count': len(clean_distances),
        'mean': mean_d,
        'median': np.median(clean_distances),
        'std': np.std(clean_distances),
        'min': np.min(clean_distances),
        'max': np.max(clean_distances),
        'mse': np.mean(clean_distances ** 2),
        'rmse': np.sqrt(np.mean(clean_distances ** 2)),
        'percentage': len(clean_distances) / len(distances) * 100 if len(distances) > 0 else 0,
        'pct_within': pct_within,
        'mad': np.mean(np.abs(clean_distances - mean_d))
    }

    return stats


def compute_region_statistics(distances, region_labels, cutoff_distance=0.5):
    """
    计算每个区域的统计值

    Parameters:
        distances: numpy array of distances
        region_labels: numpy array of region labels
        cutoff_distance: cutoff threshold in mm

    Returns:
        region_stats: dictionary mapping region number to statistics
    """
    region_stats = {}

    for region in range(1, 7):
        mask = (region_labels == region)
        region_distances = distances[mask]

        stats = compute_statistics(region_distances, cutoff_distance)
        region_stats[region] = stats

    return region_stats


def generate_heatmap_pdf(obj_path, distances, output_pdf,
                         overall_stats, cutoff_distance=0.5, use_custom_view=False):
    """
    生成热力图PDF报告

    Parameters:
        obj_path: OBJ文件路径
        distances: 距离数组
        output_pdf: 输出PDF路径
        overall_stats: 整体统计值字典
        cutoff_distance: cutoff阈值
        use_custom_view: 是否使用自定义视角（用于headspace数据）
    """
    # 加载mesh
    mesh = o3d.io.read_triangle_mesh(obj_path)
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)

    # 颜色映射 - 使用当前对象的最大值
    local_max = overall_stats['max']
    cmap = plt.get_cmap('jet')

    # 为顶点着色
    colors = np.zeros((len(distances), 3))

    for i in range(len(distances)):
        if distances[i] <= cutoff_distance:
            # 低于截断值 - 浅灰色
            colors[i] = [0.85, 0.85, 0.85]
        else:
            # 有效距离 - 使用完整色谱 [cutoff_distance, local_max] -> [0, 1]
            if local_max > cutoff_distance:
                normalized_val = (distances[i] - cutoff_distance) / (local_max - cutoff_distance)
            else:
                normalized_val = distances[i] / local_max
            normalized_val = min(1.0, max(0.0, normalized_val))  # 限制在 [0, 1]
            colors[i] = cmap(normalized_val)[:3]

    # 添加光照效果
    light_dir = np.array([0.0, 0.0, 1.0])
    ambient = 0.3
    dot = np.clip(np.sum(normals * light_dir, axis=1), 0, 1)
    intensities = ambient + (1 - ambient) * dot
    colors = colors * intensities[:, np.newaxis]

    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    # 如果使用自定义视角，先旋转mesh
    if use_custom_view:
        # 从view_template.py记录的旋转矩阵
        rotation_matrix = np.array([
            [0.73778493, 0.48020384, -0.47442351],
            [-0.08306618, -0.63288673, -0.76977555],
            [-0.66990552, 0.60733735, -0.4270456]
        ])

        # 绕mesh中心旋转
        center = mesh.get_center()
        mesh.rotate(rotation_matrix, center=center)

        # 上下颠倒：绕X轴旋转180度
        flip_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        mesh.rotate(flip_matrix, center=center)

    # 渲染3D模型
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Asymmetry Heatmap", width=1200, height=900, visible=False)
    vis.add_geometry(mesh)

    render_option = vis.get_render_option()
    render_option.mesh_show_back_face = True
    render_option.light_on = False

    view_control = vis.get_view_control()
    view_control.change_field_of_view(step=-90)  # 正交投影

    vis.poll_events()
    vis.update_renderer()

    temp_image = "temp_heatmap_batch.png"
    vis.capture_screen_image(temp_image, do_render=True)
    vis.destroy_window()

    # 生成PDF
    with PdfPages(output_pdf) as pdf:
        img = Image.open(temp_image)
        img_array = np.array(img)

        # 裁剪空白
        if len(img_array.shape) == 3:
            non_white = np.any(img_array < 250, axis=2)
        else:
            non_white = img_array < 250
            
        rows = np.any(non_white, axis=1)
        cols = np.any(non_white, axis=0)
        
        if np.any(rows) and np.any(cols):
            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]
            
            # Add margin
            margin = 10
            rmin = max(0, rmin - margin)
            rmax = min(img_array.shape[0], rmax + margin)
            cmin = max(0, cmin - margin)
            cmax = min(img_array.shape[1], cmax + margin)
            
            img_cropped = img_array[rmin:rmax, cmin:cmax]
        else:
            img_cropped = img_array

        # 根据裁剪后图像的宽高比调整figure大小
        h, w = img_cropped.shape[:2]
        aspect = w / h
        fig_height = 8
        fig_width = fig_height * aspect + 1.5  # 额外空间给colorbar

        fig1 = plt.figure(figsize=(fig_width, fig_height))
        ax1 = fig1.add_axes([0.02, 0.05, 0.80, 0.90])
        ax1.imshow(img_cropped)
        ax1.axis('off')
        
        # 移除之前的标题
        # ax1.set_title('Asymmetry Heatmap', fontsize=20, pad=10)

        # 添加colorbar，高度与图像匹配
        # 使用截断的colorbar：底部截断显示灰色，cutoff以上使用完整色谱
        cax = fig1.add_axes([0.85, 0.05, 0.03, 0.90])
        create_truncated_colorbar(fig1, cax, cmap, local_max,
                                  cutoff_distance=cutoff_distance)

        pdf.savefig(fig1, dpi=300)
        plt.close(fig1)

    # 清理临时文件
    if os.path.exists(temp_image):
        os.remove(temp_image)


def save_individual_stats(stats_path, overall_stats, region_stats):
    """
    保存单个文件的详细统计信息到TXT文件
    """
    with open(stats_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Overall Statistics\n")
        f.write("=" * 60 + "\n")

        f.write(f"Max: {overall_stats['max']:.6f} mm\n")
        f.write(f"MSE: {overall_stats['mse']:.6f} mm²\n")
        f.write(f"RMSE: {overall_stats['rmse']:.6f} mm\n")
        f.write(f"MAD: {overall_stats['mad']:.6f} mm\n")
        f.write(f"% within 0.5mm: {overall_stats['pct_within']:.2f}%\n")
        f.write(f"Mean: {overall_stats['mean']:.6f} mm\n")
        f.write(f"Median: {overall_stats['median']:.6f} mm\n")
        f.write(f"Std: {overall_stats['std']:.6f} mm\n")

        # 分区统计
        f.write("\n" + "=" * 60 + "\n")
        f.write("Statistics by Region\n")
        f.write("=" * 60 + "\n")

        region_names = {
            1: "Region 1 (Forehead)",
            2: "Region 2 (Upper Cheeks)",
            3: "Region 3 (Lower Cheeks)",
            4: "Region 4 (Upper Nose Bridge)",
            5: "Region 5 (Lower Nose Bridge)",
            6: "Region 6 (Chin)",
        }

        for region in range(1, 7):
            stats = region_stats[region]
            f.write(f"\n{region_names[region]}:\n")
            
            if stats['count'] > 0:
                f.write(f"  Vertices: {stats['count']}\n")
                f.write(f"  Max: {stats['max']:.6f} mm\n")
                f.write(f"  MSE: {stats['mse']:.6f} mm²\n")
                f.write(f"  RMSE: {stats['rmse']:.6f} mm\n")
                f.write(f"  MAD: {stats['mad']:.6f} mm\n")
                f.write(f"  % within 0.5mm: {stats['pct_within']:.2f}%\n")
            else:
                f.write(f"  No valid vertices in this region\n")


def process_single_pair(obj_path, csv_path, region_labels_orig, output_pdf,
                       orig_to_o3d, cutoff_distance=0.5, use_custom_view=False):
    """
    处理单个文件对 (OBJ + CSV)

    Parameters:
        obj_path: OBJ文件路径
        csv_path: CSV文件路径
        region_labels_orig: 原始顶点顺序的分区标签 (从region_labels.txt加载)
        output_pdf: 输出PDF路径
        orig_to_o3d: 顶点索引映射
        cutoff_distance: cutoff阈值
        use_custom_view: 是否使用自定义视角

    Returns:
        result: 包含统计结果的字典
    """
    # 1. 加载OBJ
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    # 2. 加载CSV并读取距离
    distances_orig_order = np.full(len(vertices), np.nan)

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) >= 2:
                try:
                    vid = int(row[0])
                    dist = float(row[1])
                    if 0 <= vid < len(vertices):
                        distances_orig_order[vid] = dist
                except ValueError:
                    continue

    # 3. 重新排序距离 (Original -> Open3D)
    distances = np.full(len(vertices), np.nan)
    for orig_idx in range(len(vertices)):
        if orig_idx < len(orig_to_o3d):
            o3d_idx = orig_to_o3d[orig_idx]
            if 0 <= o3d_idx < len(vertices):
                distances[o3d_idx] = distances_orig_order[orig_idx]

    # 4. 重新排序分区标签 (Original -> Open3D)
    region_labels = np.zeros(len(vertices), dtype=int)
    for orig_idx in range(len(vertices)):
        if orig_idx < len(orig_to_o3d) and orig_idx < len(region_labels_orig):
            o3d_idx = orig_to_o3d[orig_idx]
            if 0 <= o3d_idx < len(vertices):
                region_labels[o3d_idx] = region_labels_orig[orig_idx]

    # 5. 计算统计值
    overall_stats = compute_statistics(distances, cutoff_distance)
    region_stats = compute_region_statistics(distances, region_labels, cutoff_distance)

    # 6. 生成PDF (只包含图像和colorbar)
    generate_heatmap_pdf(obj_path, distances, output_pdf,
                        overall_stats, cutoff_distance, use_custom_view)
                        
    # 7. 保存统计信息到TXT
    stats_path = output_pdf.replace('.pdf', '_stats.txt')
    save_individual_stats(stats_path, overall_stats, region_stats)
    print(f"  统计信息已保存: {stats_path}")

    return {
        'filename': os.path.basename(obj_path),
        'overall': overall_stats,
        'regions': region_stats
    }


def batch_process_directory(input_dir, region_labels_path, output_dir, cutoff_distance=0.5, use_custom_view=False):
    """
    批量处理目录下的所有OBJ文件和对应的CSV文件

    Parameters:
        input_dir: 输入目录
        region_labels_path: region_labels.txt文件路径
        output_dir: 输出目录
        cutoff_distance: cutoff阈值
        use_custom_view: 是否使用自定义视角
    """
    print("=" * 80)
    print(f"批量处理目录: {input_dir}")
    print("=" * 80)

    # 确定OBJ目录和CSV目录
    obj_dir = os.path.join(input_dir, 'mapped_templates')
    csv_dir = os.path.join(input_dir, 'fa_results')

    if not os.path.exists(obj_dir) or not os.path.exists(csv_dir):
        print(f"  警告: 找不到 mapped_templates 或 fa_results 子目录")
        obj_dir = input_dir
        csv_dir = input_dir

    print(f"  OBJ目录: {obj_dir}")
    print(f"  CSV目录: {csv_dir}")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 加载顶点映射
    print(f"  加载顶点映射...")
    try:
        o3d_to_orig, orig_to_o3d = load_mapping('vertex_mapping.npz')
    except Exception as e:
        print(f"  错误: 无法加载 vertex_mapping.npz: {e}")
        return []

    # 加载分区标签
    print(f"  加载分区标签: {region_labels_path}")
    region_labels_orig = load_region_labels_from_file(region_labels_path)
    region_counts = {i: np.sum(region_labels_orig == i) for i in range(1, 7)}
    print(f"  分区顶点统计: {region_counts}")

    # 查找文件
    obj_files = sorted(glob.glob(os.path.join(obj_dir, '*.obj')))
    results = []

    for obj_path in obj_files:
        filename = os.path.basename(obj_path)
        name, _ = os.path.splitext(filename)
        
        # 查找对应的CSV
        # OBJ: xxx_mapped.obj -> CSV: xxx_fa.csv
        base_name = name.replace('_mapped', '')
        csv_filename = f"{base_name}_fa.csv"
        csv_path = os.path.join(csv_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            csv_path = os.path.join(csv_dir, f"{base_name}.csv")
        if not os.path.exists(csv_path):
            print(f"  跳过: {filename} (未找到对应的CSV: {csv_filename})")
            continue
            
        # 输出PDF路径
        pdf_filename = f"{name}_mapped_asymmetry.pdf"
        output_pdf = os.path.join(output_dir, pdf_filename)
        
        print(f"\n处理: {filename}")
        print(f"  CSV: {csv_filename}")
        
        try:
            result = process_single_pair(obj_path, csv_path, region_labels_orig, output_pdf,
                                        orig_to_o3d, cutoff_distance, use_custom_view)
            results.append(result)
            print(f"  完成: {output_pdf}")
        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    return results


def save_results_to_csv(results, output_csv):
    """
    保存所有结果到CSV文件
    """
    print(f"\n保存汇总结果到: {output_csv}")

    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # 写入表头
        header = ['Filename', 'Total_Vertices', 'Mean', 'Median', 'Std', 'Min', 'Max', 'RMSE', 'Pct_Within_Cutoff']
        for region in range(1, 7):
            header.extend([
                f'R{region}_Count', f'R{region}_Mean', f'R{region}_Max', f'R{region}_RMSE', f'R{region}_Pct_Within'
            ])
        writer.writerow(header)

        for result in results:
            overall = result['overall']
            regions = result['regions']
            
            row = [
                result['filename'],
                overall['count'],
                f"{overall['mean']:.6f}",
                f"{overall['median']:.6f}",
                f"{overall['std']:.6f}",
                f"{overall['min']:.6f}",
                f"{overall['max']:.6f}",
                f"{overall['rmse']:.6f}",
                f"{overall['pct_within']:.2f}"
            ]
            
            for region in range(1, 7):
                stats = regions[region]
                row.extend([
                    stats['count'],
                    f"{stats['mean']:.6f}",
                    f"{stats['max']:.6f}",
                    f"{stats['rmse']:.6f}",
                    f"{stats['pct_within']:.2f}"
                ])
                
            writer.writerow(row)
            
    print(f"  完成!")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dirs', nargs='+', default=['output_ppdh', 'output_headspace'])
    parser.add_argument('--region_labels', default='region_labels.txt')
    parser.add_argument('--cutoff', type=float, default=0.5)
    parser.add_argument('--output_prefix', default='analysis')
    args = parser.parse_args()

    for input_dir in args.input_dirs:
        if not os.path.exists(input_dir):
            print(f"跳过不存在的目录: {input_dir}")
            continue

        # 输出目录命名
        dir_name = os.path.basename(input_dir.rstrip('/'))
        output_dir = f"{args.output_prefix}_{dir_name}_mapped_templates"

        # 判断是否是headspace数据，使用自定义视角
        use_custom_view = 'headspace' in dir_name.lower()
        if use_custom_view:
            print(f"检测到headspace数据，将使用自定义视角")

        results = batch_process_directory(input_dir, args.region_labels, output_dir, args.cutoff, use_custom_view)

        if results:
            save_results_to_csv(results, os.path.join(output_dir, 'summary_statistics.csv'))

if __name__ == "__main__":
    main()
