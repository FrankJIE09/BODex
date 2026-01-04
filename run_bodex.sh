#!/bin/bash
# BODex 运行脚本
# 使用方法: ./run_bodex.sh [参数]

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bodex

# 设置环境变量
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

# 运行主程序
cd "$(dirname "$0")"
python example_grasp/plan_batch_env.py "$@"

