#!/bin/bash
# BODex 环境变量设置脚本
# 使用方法: source setup_env.sh

# 激活 conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate bodex

# 设置必要的环境变量
export PYTHONPATH="$PWD/src:$PYTHONPATH"
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_PATH="$CONDA_PREFIX"
export CPLUS_INCLUDE_PATH="$CONDA_PREFIX/targets/x86_64-linux/include:$CPLUS_INCLUDE_PATH"

echo "✓ BODex 环境已激活"
echo "  - Conda 环境: bodex"
echo "  - PYTHONPATH: $PYTHONPATH"
echo "  - CUDA_HOME: $CUDA_HOME"
echo ""
echo "现在可以运行程序了："
echo "  python example_grasp/plan_batch_env.py -c sim_shadow/fc.yml -w 40"

