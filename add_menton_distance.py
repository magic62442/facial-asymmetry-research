#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
计算menton点到正中矢状面的距离，并添加到summary统计中

正中矢状面通过双侧配对顶点中点的鲁棒平面拟合估计：
1. 利用pairs.csv中的对称配对，计算每对双侧顶点的中点
2. 中点天然落在正中矢状面附近
3. 迭代加权最小二乘拟合平面，下调离群中点的权重

menton坐标从外部CSV文件读取。
"""

import os
import csv
import numpy as np
import pandas as pd
import open3d as o3d


def load_pairs(pairs_csv):
    """
    从pairs.csv加载对称顶点配对

    Parameters:
        pairs_csv: pairs.csv文件路径

    Returns:
        pairs: dict, {vertex_id: mirror_vertex_id}
    """
    pairs = {}
    with open(pairs_csv, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            pairs[int(row[0])] = int(row[1])
    return pairs


def compute_midsagittal_plane_robust(vertices, pairs, max_iter=100, tol=1e-6):
    """
    通过双侧配对顶点中点的鲁棒平面拟合估计正中矢状面

    原理: vertex i 与 vertex pairs[i] 分别在面部左右两侧，
    它们的中点天然位于正中矢状面附近。对所有中点做加权平面拟合，
    迭代下调偏离平面较远的中点权重（不对称区域），即得到鲁棒的正中矢状面。

    Parameters:
        vertices: (n, 3) 顶点坐标
        pairs: dict, 对称配对映射
        max_iter: 最大迭代次数
        tol: 权重收敛阈值

    Returns:
        normal: 平面法向量 (归一化)
        d: 平面常数 (ax + by + cz + d = 0)
    """
    n = len(vertices)

    # 计算每对双侧顶点的中点
    midpoints = np.zeros((n, 3))
    for i in range(n):
        j = pairs.get(i, i)
        midpoints[i] = (vertices[i] + vertices[j]) / 2.0

    # 迭代加权平面拟合
    W = np.ones(n)

    for it in range(max_iter):
        W_prev = W.copy()

        # 加权平面拟合
        w = W / W.sum()
        centroid = (w[:, None] * midpoints).sum(axis=0)
        centered = midpoints - centroid
        cov = (w[:, None] * centered).T @ centered

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        normal = eigenvectors[:, 0]  # 最小特征值对应的特征向量
        d = -np.dot(normal, centroid)

        # 残差: 每个中点到当前平面的距离
        residuals = np.abs(midpoints @ normal + d)

        # 鲁棒估计内点分布参数σ (基于MAD)
        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        sigma = 1.4826 * mad
        if sigma < 1e-10:
            sigma = 1e-10

        # 外点分布参数λ (均匀分布)
        r_max = residuals.max()
        if r_max < 1e-10:
            break
        lam = 1.0 / r_max

        # 贝叶斯后验权重更新
        inlier_p = np.exp(-0.5 * (residuals / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        W = inlier_p / (inlier_p + lam)

        # 收敛检查
        if np.max(np.abs(W - W_prev)) < tol:
            break

    return normal, d


def point_to_plane_distance(point, normal, d):
    """
    计算点到平面的距离

    平面方程: ax + by + cz + d = 0
    距离 = |ax + by + cz + d| / sqrt(a^2 + b^2 + c^2)
    由于normal已归一化，分母为1
    """
    return abs(np.dot(normal, point) + d)


def get_menton_distance_for_obj(obj_path, menton_coords, pairs):
    """
    计算单个OBJ文件中menton点到正中矢状面的距离

    Parameters:
        obj_path: OBJ文件路径
        menton_coords: menton点坐标 [x, y, z]
        pairs: dict, 对称顶点配对映射

    Returns:
        distance: menton到正中矢状面的距离
    """
    if not os.path.exists(obj_path):
        print(f"  文件不存在: {obj_path}")
        return np.nan

    # 加载OBJ
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    # 鲁棒估计正中矢状面
    normal, d = compute_midsagittal_plane_robust(vertices, pairs)

    # 计算menton到平面的距离
    menton = np.array(menton_coords)
    distance = point_to_plane_distance(menton, normal, d)

    return distance


def process_headspace(obj_dir, csv_path, output_summary_path, pairs):
    """
    处理headspace数据集

    Parameters:
        obj_dir: OBJ文件所在目录 (mapped_templates)
        csv_path: 包含menton坐标的CSV文件
        output_summary_path: 输出summary CSV路径
        pairs: dict, 对称顶点配对映射
    """
    print("=" * 60)
    print("处理 headspace 数据集")
    print("=" * 60)

    # 读取menton坐标CSV
    df_menton = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"读取menton坐标文件: {csv_path}")
    print(f"  共 {len(df_menton)} 条记录")

    # 读取现有的summary统计
    if os.path.exists(output_summary_path):
        df_summary = pd.read_csv(output_summary_path)
        print(f"读取现有summary: {output_summary_path}")
    else:
        print(f"  未找到现有summary: {output_summary_path}")
        return

    # 创建文件名到menton坐标的映射
    menton_map = {}
    for _, row in df_menton.iterrows():
        new_seq = row.get('new sequence')
        if pd.isna(new_seq) or str(new_seq).strip() == '':
            continue

        menton_x = row.get('menton X')
        menton_y = row.get('menton Y')
        menton_z = row.get('menton Z')

        if pd.isna(menton_x) or pd.isna(menton_y) or pd.isna(menton_z):
            continue

        # 文件名格式: {new_sequence}_mapped.obj
        filename = f"{int(new_seq)}_mapped.obj"
        menton_map[filename] = [float(menton_x), float(menton_y), float(menton_z)]

    print(f"  找到 {len(menton_map)} 个有效menton坐标")

    # 计算每个文件的menton距离
    menton_distances = []
    for idx, row in df_summary.iterrows():
        filename = row['Filename']

        if filename in menton_map:
            obj_path = os.path.join(obj_dir, filename)
            menton_coords = menton_map[filename]
            distance = get_menton_distance_for_obj(obj_path, menton_coords, pairs)
            menton_distances.append(distance)
            print(f"  {filename}: menton距离 = {distance:.4f} mm")
        else:
            menton_distances.append(np.nan)
            print(f"  {filename}: 未找到menton坐标")

    # 添加到summary
    df_summary['Menton_Distance'] = menton_distances

    # 保存
    df_summary.to_csv(output_summary_path, index=False)
    print(f"\n已更新summary: {output_summary_path}")

    return df_summary


def process_ppdh(obj_dir, csv_path, output_summary_path, pairs):
    """
    处理ppdh数据集

    Parameters:
        obj_dir: OBJ文件所在目录 (mapped_templates)
        csv_path: 包含menton坐标的CSV文件
        output_summary_path: 输出summary CSV路径
        pairs: dict, 对称顶点配对映射
    """
    print("=" * 60)
    print("处理 ppdh 数据集")
    print("=" * 60)

    # 读取menton坐标CSV
    df_menton = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"读取menton坐标文件: {csv_path}")
    print(f"  共 {len(df_menton)} 条记录")

    # 读取现有的summary统计
    if os.path.exists(output_summary_path):
        df_summary = pd.read_csv(output_summary_path)
        print(f"读取现有summary: {output_summary_path}")
    else:
        print(f"  未找到现有summary: {output_summary_path}")
        return

    # 创建文件名到menton坐标的映射
    menton_map = {}
    for _, row in df_menton.iterrows():
        folder_name = row.get('FolderName')
        if pd.isna(folder_name) or str(folder_name).strip() == '':
            continue

        menton_x = row.get('menton X')
        menton_y = row.get('menton Y')
        menton_z = row.get('menton Z')

        if pd.isna(menton_x) or pd.isna(menton_y) or pd.isna(menton_z):
            continue

        # 文件名格式: {FolderName}_mapped.obj
        filename = f"{folder_name}_mapped.obj"
        menton_map[filename] = [float(menton_x), float(menton_y), float(menton_z)]

    print(f"  找到 {len(menton_map)} 个有效menton坐标")

    # 计算每个文件的menton距离
    menton_distances = []
    for idx, row in df_summary.iterrows():
        filename = row['Filename']

        if filename in menton_map:
            obj_path = os.path.join(obj_dir, filename)
            menton_coords = menton_map[filename]
            distance = get_menton_distance_for_obj(obj_path, menton_coords, pairs)
            menton_distances.append(distance)
            print(f"  {filename}: menton距离 = {distance:.4f} mm")
        else:
            menton_distances.append(np.nan)
            print(f"  {filename}: 未找到menton坐标")

    # 添加到summary
    df_summary['Menton_Distance'] = menton_distances

    # 保存
    df_summary.to_csv(output_summary_path, index=False)
    print(f"\n已更新summary: {output_summary_path}")

    return df_summary


def main():
    # 加载对称配对（只需加载一次，所有mesh共用）
    pairs_csv = os.path.join(os.path.dirname(__file__), 'pairs.csv')
    print(f"加载对称配对: {pairs_csv}")
    pairs = load_pairs(pairs_csv)
    print(f"  共 {len(pairs)} 个配对")

    # headspace
    headspace_obj_dir = 'output_headspace/mapped_templates'
    headspace_menton_csv = '/Users/lqy/Desktop/research/lele/summary.csv'
    headspace_summary = 'analysis_output_headspace_mapped_templates/summary_statistics.csv'

    if os.path.exists(headspace_obj_dir):
        process_headspace(headspace_obj_dir, headspace_menton_csv, headspace_summary, pairs)

    print("\n")

    # ppdh
    ppdh_obj_dir = 'output_ppdh/mapped_templates'
    ppdh_menton_csv = '/Users/lqy/Desktop/research/lele/asymmetry_ppdh_folders.csv'
    ppdh_summary = 'analysis_output_ppdh_mapped_templates/summary_statistics.csv'

    if os.path.exists(ppdh_obj_dir):
        process_ppdh(ppdh_obj_dir, ppdh_menton_csv, ppdh_summary, pairs)


if __name__ == "__main__":
    main()
