#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=/deltadisk/dachuang/colcon_ws/src/handfather:$PYTHONPATH
python train.py headless=false force_render=false
