#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=/deltadisk/dachuang/colcon_ws/src/handfather:$PYTHONPATH

python train.py wandb_activate=false num_envs=1 headless=false test=true task=LeapHandRot  checkpoint=/deltadisk/dachuang/colcon_ws/src/handfather/leapsim/runs/woven.pth
