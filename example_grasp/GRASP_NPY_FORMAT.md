# Grasp 结果 NPY 文件格式说明

本仓库中抓取结果以 `.npy` 形式保存，`np.load(..., allow_pickle=True).item()` 得到一个字典。两种典型路径：

- **Tabletop（桌面）**：`.../graspdata/<物体ID>/tabletop_ur10e/scale008_pose004_0_grasp.npy`  
  含桌面场景，`world_cfg` 中有 `mesh` + `cuboid`（桌子）。
- **Floating（浮空）**：`.../graspdata/<物体ID>/floating/scale008_grasp.npy`  
  无桌面，`world_cfg` 中仅有 `mesh`。

二者顶层字段与数组 shape 一致，仅场景相关字段内容不同。

---

## 顶层字段一览

| 键 | 类型 | Shape / 说明 |
|----|------|----------------|
| `robot_pose` | ndarray float32 | `(1, 20, 3, 27)`，见下 |
| `world_cfg` | list[dict] | 长度 1，场景配置（mesh ± cuboid） |
| `manip_name` | list[str] | 长度 1，manipulator 名称 |
| `scene_path` | list[str] | 长度 1，场景 .npy 路径 |
| `contact_point` | ndarray float32 | `(1, 20, 1, 5, 3)` |
| `contact_frame` | ndarray float32 | `(1, 20, 5, 3, 3)` |
| `contact_force` | ndarray float32 | `(1, 20, 6, 5, 3)` |
| `grasp_error` | ndarray float32 | `(1, 20, 6)` |
| `dist_error` | ndarray float32 | `(1, 20, 2)` |
| `joint_names` | list[str] | 长度 20，手部关节名 |

---

## `robot_pose`

- **Shape**：`(1, 20, 3, 27)`
  - `1`：batch（单条记录）
  - `20`：20 个抓取候选
  - `3`：3 个时间步（如：起始 / 中间 / 结束）
  - `27`：**状态维数** = 7（手的 base 位姿） + 20（手关节）
- **最后一维 27 的含义**：
  - 索引 `0..6`：**手的 base 位姿**，格式为 \([x, y, z, qw, qx, qy, qz]\)
  - 索引 `7..26`：与 `joint_names` 一一对应，灵巧手 20 个关节（如 `finger1_joint1`～`finger4_joint4`）
- **单位**：
  - base 位姿：位置（米）+ 四元数（无量纲）
  - 手关节：关节角（弧度）

---

## `world_cfg`

`world_cfg[0]` 为一条场景配置字典。

- **Tabletop**：通常包含
  - `mesh`：物体 mesh，键为场景/物体 ID；值含 `scale`(3,)、`pose`(7,)、`file_path`、`urdf_path`。
    - `file_path` 往往是**相对于 `scene_path[0]` 指向的 scene_cfg 文件所在目录**的相对路径（例如仅为 `xxx.stl`）。
  - `cuboid`：如 `table`，含 `dims`（如 `[5, 5, 0.2]`）、`pose`（7 维）。
- **Floating**：通常仅含
  - `mesh`：同上，无 `cuboid`。

---

## 接触与误差

- **contact_point** `(1, 20, 1, 5, 3)`：每个抓取 5 个接触点的 3D 坐标。
- **contact_frame** `(1, 20, 5, 3, 3)`：每个接触点的 3×3 坐标系。
- **contact_force** `(1, 20, 6, 5, 3)`：接触力相关（6 可能为力/力矩分量）。
- **grasp_error** `(1, 20, 6)`：每个抓取的 6 维误差。
- **dist_error** `(1, 20, 2)`：每个抓取的距离类误差 2 维。

---

## 示例：读取与取 `robot_pose`

```python
import numpy as np

# 任选其一
path_tabletop = ".../graspdata/sem_FoodItem_.../tabletop_ur10e/scale008_pose004_0_grasp.npy"
path_floating = ".../graspdata/core_bottle_.../floating/scale008_grasp.npy"

data = np.load(path_tabletop, allow_pickle=True).item()
robot_pose = data["robot_pose"]   # (1, 20, 3, 27)
joint_names = data["joint_names"] # 20 个手关节名
world_cfg = data["world_cfg"][0] # mesh ± cuboid
```

使用脚本查看内容与每个关节的 min/max/mean：

```bash
# 修改 example_grasp/print_grasp_npy.py 中 NPY_PATH 后运行
python3 example_grasp/print_grasp_npy.py
```
