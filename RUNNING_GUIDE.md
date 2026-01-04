# BODex 运行指南

## ✅ 环境搭建完成

所有依赖已安装，CUDA 扩展编译问题已解决，程序可以正常运行。

## 🚀 快速开始

### 方法 1: 使用运行脚本（最简单）

```bash
# 单 GPU 版本 - 合成抓取姿态
./run_bodex.sh -c sim_shadow/fc.yml -w 40

# 调试模式
CUDA_VISIBLE_DEVICES=0 ./run_bodex.sh -c sim_shadow/fc.yml -w 1 -m usd -debug -d all -i 0 1

# 合成接近轨迹（带机械臂）
./run_bodex.sh -c sim_shadow/tabletop.yml -t grasp_and_mogen
```

### 方法 2: 使用环境设置脚本

```bash
# 设置环境变量
source setup_env.sh

# 运行程序
python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40
```

### 方法 3: 手动设置

```bash
conda activate bodex
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH

python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40
```

## ⚠️ 重要提示

1. **首次运行**: CUDA 扩展会进行 JIT 编译，可能需要几分钟时间。请耐心等待。
2. **对象资产**: 如果看到 "get 0 scene cfgs" 警告，需要下载对象资产（见 README.md）。
3. **GPU 选择**: 使用 `CUDA_VISIBLE_DEVICES` 环境变量选择 GPU。

## 📋 程序参数说明

- `-c, --manip_cfg_file`: 配置文件路径（必需）
- `-w, --parallel_world`: 并行处理的世界数量
- `-m, --save_mode`: 保存模式（usd, npy, usd+npy, none）
- `-d, --save_data`: 保存哪些数据
- `-i, --save_id`: 保存哪些结果
- `-debug`: 调试模式，保存接触法线
- `-k, --skip`: 跳过已存在的文件

## 📁 配置文件位置

- `src/curobo/content/configs/manip/sim_shadow/fc.yml` - 抓取合成
- `src/curobo/content/configs/manip/sim_shadow/tabletop.yml` - 带桌面的轨迹规划

## 🎉 环境信息

- Python: 3.10.19
- PyTorch: 2.2.2+cu121
- CUDA: 12.1
- GPU: NVIDIA GeForce RTX 3090
- Conda 环境: bodex

