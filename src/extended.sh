#!/bin/bash
set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
scene=$1
until [ -f "$P/results/${scene}__width512/complete.json" ]; do sleep 5; done
PY=/home/tianyi/data/anaconda3/envs/3dgs/bin/python
"$PY" -u "$P/src/experiment.py" --scene "$scene" --seeds 0 --steps 1000 --methods CG PCG --tag extended
