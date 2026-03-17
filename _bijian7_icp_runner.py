#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
临时脚本：用 point-to-surface ICP 重新处理 bijian/7_50_directional.obj
输出覆盖 bijian/icp_result/ 下的7号 heatmap、stats、summary CSV 行
"""
import os
import csv
import numpy as np
import sys
sys.path.insert(0, '.')

from standard_icp_point_to_surface import mirror_and_register_icp_point_to_surface
from visualize_pairing_quality import visualize_pairing_quality

OBJ_PATH            = 'bijian/7_50_directional.obj'
ICP_CSV_NAME        = '7_50_directional_icp.csv'
ICP_DIR             = 'bijian/icp_result'
CSV_PATH            = os.path.join(ICP_DIR, ICP_CSV_NAME)
PDF_PATH            = os.path.join(ICP_DIR, '7_50_directional_icp_heatmap.pdf')
STATS_PATH          = os.path.join(ICP_DIR, '7_50_directional_icp_heatmap_stats.txt')
SUMMARY_CSV         = os.path.join(ICP_DIR, 'icp_summary_statistics.csv')
REGION_LABELS_PATH  = 'region_labels.txt'
CUTOFF              = 0.5
MAX_DIST            = 14   # 文件名首数字 7 × 2

# ── Step 1: Point-to-Surface ICP ──────────────────────────────────────────────
print("=" * 70)
print("Step 1: Point-to-Surface ICP")
print("=" * 70)
mirror_and_register_icp_point_to_surface(
    obj_path=OBJ_PATH,
    output_csv=ICP_CSV_NAME,
    output_dir=ICP_DIR,
)

# ── Step 2: 生成 heatmap PDF + stats TXT ─────────────────────────────────────
print("\n" + "=" * 70)
print("Step 2: Generate heatmap & stats")
print("=" * 70)
result = visualize_pairing_quality(
    obj_path=OBJ_PATH,
    csv_path=CSV_PATH,
    output_pdf=PDF_PATH,
    region_labels_path=REGION_LABELS_PATH,
    stats_output_path=STATS_PATH,
    max_distance=MAX_DIST,
    cutoff_distance=CUTOFF,
    title='Standard ICP Heatmap',
)

# ── Step 3: 更新 summary CSV ──────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Step 3: Update summary CSV")
print("=" * 70)

# 读取 ICP CSV 距离（2列格式：source_vertex_id, distance_to_surface）
distances = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        distances.append(float(row[1]))
distances = np.array(distances)

# 读取分区标签
region_labels = np.zeros(len(distances), dtype=int)
if os.path.exists(REGION_LABELS_PATH):
    with open(REGION_LABELS_PATH, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split(',')
            if len(parts) == 2:
                idx, label = int(parts[0]), int(parts[1])
                if 0 <= idx < len(distances):
                    region_labels[idx] = label

def compute_stats(dists, threshold=0.5):
    return {
        'count': len(dists),
        'Mean': float(dists.mean()),
        'Max': float(dists.max()),
        'RMSE': float(np.sqrt(np.mean(dists ** 2))),
        'pct_within': float(np.sum(dists <= threshold) / len(dists) * 100),
    }

region_stats = {}
for r in range(1, 7):
    mask = region_labels == r
    if mask.sum() > 0:
        region_stats[r] = compute_stats(distances[mask])
    else:
        region_stats[r] = {'count': 0, 'Mean': 0.0, 'Max': 0.0, 'RMSE': 0.0, 'pct_within': 0.0}

FILENAME = '7_50_directional_icp_heatmap.pdf'
new_row = [
    FILENAME,
    len(distances),
    f"{result['mean_distance']:.6f}",
    f"{result['median_distance']:.6f}",
    f"{result['std_distance']:.6f}",
    f"{result['min_distance']:.6f}",
    f"{result['max_distance']:.6f}",
    f"{result['rmse']:.6f}",
    f"{np.sum(distances <= CUTOFF) / len(distances) * 100:.2f}",
]
for r in range(1, 7):
    s = region_stats[r]
    new_row.extend([
        s['count'],
        f"{s['Mean']:.6f}",
        f"{s['Max']:.6f}",
        f"{s['RMSE']:.6f}",
        f"{s['pct_within']:.2f}",
    ])

# 替换 summary CSV 中对应行
rows = []
updated = False
with open(SUMMARY_CSV, 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    header = next(reader)
    rows.append(header)
    for row in reader:
        if row and row[0] == FILENAME:
            rows.append(new_row)
            updated = True
        else:
            rows.append(row)
if not updated:
    rows.append(new_row)

with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(rows)

print(f"✓ Summary CSV updated: {SUMMARY_CSV}")
print("\nAll done.")
