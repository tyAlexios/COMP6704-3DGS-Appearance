set -e
P=/home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
for scene in train truck; do
  test -f "$P/results/${scene}__extended/complete.json"
done
cd "$P"
tar -czf measurements.tar.gz --exclude='*.pt' results logs environment.txt vendor_sha256.txt reference_rasterizer_commit.txt reference_submodules.txt results_installed_backward_diagnostic/truck__main/operator_checks.json
tar -czf source_snapshot.tar.gz --exclude='*/__pycache__' --exclude='*/.git' --exclude='*/build' --exclude='*/dist' --exclude='*.egg-info' vendor src reference_rasterizer environment.txt vendor_sha256.txt reference_rasterizer_commit.txt reference_submodules.txt
tar -czf geometry_checkpoints.tar.gz geometry_train/point_cloud/iteration_3000/point_cloud.ply geometry_truck/point_cloud/iteration_3000/point_cloud.ply geometry_train/cfg_args geometry_truck/cfg_args geometry_train/cameras.json geometry_truck/cameras.json
ls -lh measurements.tar.gz source_snapshot.tar.gz geometry_checkpoints.tar.gz
pgrep -af 'COMP6704_individual_appearance_20261027/src/(experiment|ablations|extended)' || true
