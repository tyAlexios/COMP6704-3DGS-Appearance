#!/bin/bash
set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
until [ -f "$P/results/train__extended/complete.json" ] && [ -f "$P/results/truck__extended/complete.json" ]; do sleep 5; done
bash "$P/src/export_measurements.sh"
