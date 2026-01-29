#!/usr/bin/env python3
"""加载并打印 grasp .npy 文件内容，分析 data 组成。"""
import os
import numpy as np

NPY_PATH = "src/curobo/content/assets/output/test/graspdata/sem_FoodItem_d7a0fa331fd0c8dfa79a6a16f6a9c559/tabletop_ur10e/scale008_pose004_0_grasp.npy"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ABS_PATH = os.path.join(ROOT, NPY_PATH)

def analyze(obj, indent=0):
    prefix = "  " * indent
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0 and obj.dtype == object:
            analyze(obj.item(), indent)
        else:
            print(f"{prefix}ndarray shape={obj.shape} dtype={obj.dtype}")
            if obj.size <= 4 and obj.dtype in (np.float32, np.float64, np.int32, np.int64):
                print(f"{prefix}  values: {obj}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{prefix}{k}:")
            analyze(v, indent + 1)
    elif isinstance(obj, (list, tuple)):
        print(f"{prefix}{type(obj).__name__} len={len(obj)}")
        if obj and len(obj) <= 3:
            for i, x in enumerate(obj):
                print(f"{prefix}  [{i}]:")
                analyze(x, indent + 2)
        elif obj:
            print(f"{prefix}  [0]:")
            analyze(obj[0], indent + 2)
    else:
        print(f"{prefix}{type(obj).__name__} = {repr(obj)[:100]}")

def print_joint_stats(d):
    """输出每个关节维度的 min/max/mean。"""
    robot_pose = d.get("robot_pose")
    joint_names = d.get("joint_names", [])
    if robot_pose is None:
        return
    # robot_pose: (1, 20, 3, 27) -> 27 个关节/自由度
    pose = np.asarray(robot_pose).reshape(-1, robot_pose.shape[-1])
    n_dof = pose.shape[1]
    n_base = max(0, n_dof - len(joint_names))
    print("--- 每个关节的大小（min / max / mean）---")
    print(f"  共 {n_dof} 维，前 {n_base} 维为手 base 位姿 [x y z qw qx qy qz]，后 {len(joint_names)} 维为 joint_names")
    base_names = ["base_x", "base_y", "base_z", "base_qw", "base_qx", "base_qy", "base_qz"]
    for j in range(n_dof):
        col = pose[:, j]
        if j < min(n_base, len(base_names)):
            name = base_names[j]
        elif j >= n_base and (j - n_base) < len(joint_names):
            name = joint_names[j - n_base]
        else:
            name = f"dim_{j}"
        print(f"  [{j:2d}] {name:24s}  min={col.min():.4f}  max={col.max():.4f}  mean={col.mean():.4f}")

if __name__ == "__main__":
    path = ABS_PATH if os.path.isfile(ABS_PATH) else NPY_PATH
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=True)
    print("path:", path)
    print("shape:", data.shape, "dtype:", data.dtype)
    print("--- 组成分析 ---")
    if data.ndim == 0 and data.dtype == object:
        obj = data.item()
        analyze(obj)
        if isinstance(obj, dict):
            print_joint_stats(obj)
    else:
        analyze(data)
    print("--- 原始 print ---")
    print(data)
