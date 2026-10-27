#!/bin/bash
set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
scene=$1
until [ -f "$P/results/${scene}__main/complete.json" ]; do sleep 5; done
PY=/home/tianyi/data/anaconda3/envs/3dgs/bin/python
for r in 0.1 1 10; do
  "$PY" -u "$P/src/experiment.py" --scene "$scene" --seeds 0 --ridge "$r" --tag "ridge_$r" --pilot
done
"$PY" -u "$P/src/experiment.py" --scene "$scene" --seeds 0 --views 32 --tag views32 --pilot
"$PY" -u "$P/src/experiment.py" --scene "$scene" --seeds 0 --width 512 --tag width512 --pilot
