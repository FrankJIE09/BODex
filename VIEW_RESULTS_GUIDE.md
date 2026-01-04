# BODex 结果查看指南

## 📊 查看运行结果的方法

BODex 运行后会生成 `.npy` 文件，包含抓取姿态数据。以下是查看这些结果的几种方法：

## 方法 1: 使用可视化脚本（推荐）

### 生成 USD 文件进行可视化

**最简单的方法（推荐）：**

```bash
# 使用可视化脚本（自动设置环境变量）
./visualize_results.sh -c sim_shadow/fc.yml -p debug -m grasp

# 带相机设置（方便截图）
./visualize_results.sh -c sim_shadow/fc.yml -p debug -m grasp -s
```

**或者手动设置环境变量：**

```bash
# 激活环境
conda activate bodex

# 设置 PYTHONPATH（重要！）
export PYTHONPATH=$PWD/src:$PYTHONPATH

# 可视化抓取结果（生成 USD 文件，可用 USD Composer 或 Omniverse 打开）
python example_grasp/visualize_npy.py -c sim_shadow/fc.yml -p debug -m grasp

# 带相机设置（方便截图）
python example_grasp/visualize_npy.py -c sim_shadow/fc.yml -p debug -m grasp -s
```

**参数说明：**
- `-c, --manip_cfg_file`: 配置文件路径（如 `sim_shadow/fc.yml`）
- `-p, --path`: 结果文件夹名称（如 `debug`，会自动查找 `graspdata` 子目录）
- `-m, --mode`: 模式，`grasp` 或 `mogen`
- `-s, --set_camera`: 设置相机（可选）

**输出位置：**
- USD 文件会保存在 `sim_shadow/fc/debug/graspdata/` 目录下
- 每个对象会生成对应的 `.usd` 文件

## 方法 2: 使用查看工具脚本

### 查看单个文件

```bash
python view_results.py -f src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata/sem_Planet_65929d262a58d9e8fdaa51cdb5785d22/floating/scale006_grasp.npy
```

### 列出所有结果文件

```bash
python view_results.py -d src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata
```

### 查看统计信息

```bash
python view_results.py -s
```

## 方法 3: 使用 Python 直接查看

```python
import numpy as np

# 加载文件
data = np.load('path/to/grasp.npy', allow_pickle=True).item()

# 查看数据键
print("数据键:", list(data.keys()))

# 查看数据形状
for key, value in data.items():
    if hasattr(value, 'shape'):
        print(f"{key}: shape={value.shape}, dtype={value.dtype}")
```

## 方法 4: 使用 DexGraspBench 评估

使用 [DexGraspBench](https://github.com/JYChen18/DexGraspBench) 进行更详细的评估：

```bash
cd /home/lenovo/Frank/code/DexGraspBench
conda activate DGBench

# 评估 BODex 的抓取结果
bash script/test_BODex_shadow.sh
```

## 📁 结果文件结构

```
src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata/
├── {object_name_1}/
│   └── floating/
│       ├── scale006_grasp.npy
│       ├── scale008_grasp.npy
│       └── scale010_grasp.npy
├── {object_name_2}/
│   └── floating/
│       └── ...
└── ...
```

## 📋 .npy 文件内容说明

每个 `.npy` 文件通常包含以下数据：

- **robot_pose**: 机器人姿态（预抓取、抓取、挤压姿态）
- **world_cfg**: 世界配置（对象信息）
- **其他调试信息**（如果使用 `-debug` 参数）

## 🎨 可视化工具

### USD Composer / Omniverse

1. 安装 [NVIDIA Omniverse](https://www.nvidia.com/en-us/omniverse/)
2. 打开生成的 `.usd` 文件
3. 可以查看 3D 抓取姿态和优化过程

### 其他工具

- **MeshLab**: 可以查看 3D 网格
- **Blender**: 可以导入 USD 文件（需要插件）

## 📊 快速检查结果

```bash
# 统计结果文件数量
find src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata -name "*.npy" | wc -l

# 查看文件大小
du -sh src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata

# 列出所有对象
ls src/curobo/content/assets/output/sim_shadow/fc/debug/graspdata
```

## 💡 提示

1. **首次可视化**: 生成 USD 文件可能需要一些时间
2. **文件大小**: 每个 `.npy` 文件通常几 KB 到几十 KB
3. **调试模式**: 使用 `-debug` 参数运行会保存更多信息（包括优化过程）
4. **批量处理**: 可视化脚本会自动跳过已存在的文件（除非使用 `-k` 参数）

