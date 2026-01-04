# BODex 环境搭建状态报告

## ✅ 已完成

1. **Conda 环境**: `bodex` (Python 3.10.19) ✓
2. **PyTorch**: 2.2.2 + CUDA 12.1 ✓
3. **GPU**: NVIDIA GeForce RTX 3090 可用 ✓
4. **核心依赖**:
   - NumPy 1.26.4 ✓
   - SciPy 1.15.3 ✓
   - torch-scatter ✓
   - coal 3.0.1 ✓
   - coal-openmp-wrapper ✓
   - usd-core ✓
5. **项目包**: curobo 可导入 ✓

## ✅ 已解决问题

### CUDA 扩展编译

**解决方案**: 设置正确的 CUDA 包含路径：
```bash
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
```

**状态**: ✅ CUDA 扩展可以正常编译（首次运行时会 JIT 编译，需要一些时间）

## ✅ 编译问题已解决

通过设置正确的 CUDA 包含路径，CUDA 扩展现在可以正常编译。所有必要的头文件（包括 `nv/target`）都已找到。

## 📝 运行程序

### 方法 1: 使用运行脚本（推荐）

```bash
# 单 GPU 版本
./run_bodex.sh -c sim_shadow/fc.yml -w 40

# 调试模式
CUDA_VISIBLE_DEVICES=0 ./run_bodex.sh -c sim_shadow/fc.yml -w 1 -m usd -debug -d all -i 0 1
```

### 方法 2: 手动设置环境变量

```bash
# 激活环境并设置变量
source setup_env.sh

# 运行程序
python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40
```

### 方法 3: 直接运行（需要先设置环境变量）

```bash
conda activate bodex
export PYTHONPATH=$PWD/src:$PYTHONPATH
export CUDA_HOME=$CONDA_PREFIX
export CPLUS_INCLUDE_PATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH
python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40
```

**注意**: 首次运行时会 JIT 编译 CUDA 扩展，可能需要几分钟时间。编译完成后会缓存，后续运行会更快。

## 🔍 当前环境信息

- **Python**: 3.10.19
- **PyTorch**: 2.2.2+cu121
- **CUDA Runtime**: 12.1.105
- **CUDA Compiler (nvcc)**: 12.1.105
- **GPU**: NVIDIA GeForce RTX 3090
- **Conda 环境**: bodex

## 📌 下一步

1. ✅ CUDA 扩展编译问题已解决
2. ✅ 主程序可以运行
3. 准备对象资产（如果需要，从 Hugging Face 下载 `DGN_2k_processed.zip`）
4. 开始使用程序进行抓取姿态合成

## 🎉 环境搭建完成！

所有问题已解决，程序可以正常运行。首次运行时会进行 CUDA 扩展的 JIT 编译，请耐心等待。

