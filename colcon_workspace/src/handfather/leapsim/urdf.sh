#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PYTHONPATH=~/colcon_ws/src/leaphandsim:$PYTHONPATH
python test_is.py
