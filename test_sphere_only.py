#!/usr/bin/env python3
"""
测试 gaussian_directional_sphere_only 函数
"""

from gaussian_displacement import gaussian_directional_sphere_only

if __name__ == "__main__":
    # 输入文件
    obj_file = "Template.obj"
    csv_file = "template landmark.csv"
    landmark_index = 2  # 第3个点（索引从0开始）

    # 参数
    r = 25      # 球体半径 25mm
    k = 0.3     # 位移比例

    print("=" * 80)
    print("测试 gaussian_directional_sphere_only 函数")
    print("=" * 80)
    print(f"\n输入:")
    print(f"  OBJ文件: {obj_file}")
    print(f"  Landmark文件: {csv_file}")
    print(f"  Landmark索引: {landmark_index}")
    print(f"  球体半径 r: {r} mm")
    print(f"  位移比例 k: {k}")
    print(f"  实际位移: {r * k} mm")

    # 调用函数
    output_path = gaussian_directional_sphere_only(
        obj_path=obj_file,
        csv_path=csv_file,
        landmark_idx=landmark_index,
        r=r,
        k=k,
        sigma=1,  # 外鼓
        direction=[1.0, 0.0, 0.0],  # +x方向
        output_dir="test_sphere_output"
    )

    if output_path:
        print("\n" + "=" * 80)
        print("测试成功!")
        print("=" * 80)
        print(f"生成文件: {output_path}")
    else:
        print("\n测试失败!")
