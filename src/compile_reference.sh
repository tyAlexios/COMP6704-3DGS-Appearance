#!/bin/bash
set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
export CUDA_HOME=/home/tianyi/data/anaconda3/envs/3dgs
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4
/home/tianyi/data/anaconda3/envs/3dgs/bin/python -m pip install "$P/reference_rasterizer" --target "$P/reference_deps" --no-deps --no-build-isolation
