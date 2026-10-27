#!/bin/bash
set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
scene=$1
until [ -f "$P/geometry_$scene/point_cloud/iteration_3000/point_cloud.ply" ]; do sleep 10; done
exec /home/tianyi/data/anaconda3/envs/3dgs/bin/python -u "$P/src/experiment.py" --scene "$scene" --pilot
