import open3d as o3d
import numpy as np
import copy
from standard_icp import mirror_and_register_manual_icp  # 假设您把刚才的函数存为了文件

if __name__ == '__main__':
    # 1. 加载模型
    obj_path = "displaced_directional/25_0.3.obj"
    mesh = o3d.io.read_triangle_mesh(obj_path)
    vertices = np.asarray(mesh.vertices)

    # 2. 人为引入一个显著的初始变换（捣乱）
    # 比如：绕 Z 轴旋转 10 度，再平移 [5, 5, 5]
    print("正在人为施加干扰变换...")
    theta = np.radians(10.0)
    perturbation_matrix = np.array([
        [np.cos(theta), -np.sin(theta), 0, 5.0],
        [np.sin(theta),  np.cos(theta), 0, 5.0],
        [0,              0,             1, 5.0],
        [0,              0,             0, 1.0]
    ])

    # 创建一个被干扰的模型文件用于测试
    perturbed_mesh = copy.deepcopy(mesh)
    perturbed_mesh.transform(perturbation_matrix)
    test_obj_path = "test_perturbed.obj"
    o3d.io.write_triangle_mesh(test_obj_path, perturbed_mesh)

    # 3. 对这个被干扰的模型运行您的 ICP 代码
    print("\n开始对干扰后的模型进行配准测试...")
    result = mirror_and_register_manual_icp(
        obj_path=test_obj_path,
        output_csv='test_verify.csv',
        output_dir='test_result'
    )

    # 4. 验证结果
    print("\n" + "="*50)
    print("验证结果分析")
    print("="*50)
    T_recovered = result['transformation']
    print("ICP 计算出的变换矩阵 (T_recovered):\n", T_recovered)
    print("\n原本施加的干扰矩阵的逆 (理论上 T_recovered 应该接近这个):\n", np.linalg.inv(perturbation_matrix))

    # 检查两者差异
    diff = np.abs(T_recovered - np.linalg.inv(perturbation_matrix))
    print("\n差异矩阵 (越接近0说明算法越好):\n", diff)