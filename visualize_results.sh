#!/bin/bash
# BODex 结果可视化脚本
# 使用方法: ./visualize_results.sh [参数]

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bodex

# 设置环境变量
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

# 运行可视化脚本
cd "$(dirname "$0")"
python example_grasp/visualize_npy.py "$@"

