COMP6704: Fixed-Geometry 3D Gaussian Splatting Appearance Optimization

Scope
  Optimize only one unconstrained RGB triplet per frozen Gaussian. Geometry,
  opacity, camera poses, background and population are fixed. Degree-zero SH
  preparation colors anchor ridge regularization. No clipped or SSIM training
  objective is used in the optimizer comparison.

Objective
  F(x) = (sum_v ||A_v x - (image_v - A_v c0)||^2 + lambda ||x||^2)/(2*m).
  x is an RGB correction, c0 is the common prepared color, and m is the total
  number of fitted scalar RGB observations. H = A^T A + lambda I.
  The renderer forward computes A*x and its cached backward computes A^T*y.

Server project
  /home/tianyi/data/3D_Research/COMP6704_individual_appearance_20261027
  Python: /home/tianyi/data/anaconda3/envs/3dgs/bin/python
  Dataset: /home/tianyi/data/3D_Research/DATASETS/tandt/{train,truck}
  Runtime: Python 3.7.12, PyTorch 1.13.0, CUDA 11.6, RTX 3090.
  See environment.txt for all installed package versions.
  Appearance rendering uses a project-local build of the official Graphdeco
  reference rasterizer at commit 59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d.
  The installed accelerated backward failed a numerical adjoint check; its
  initial color results are excluded and retained only as diagnostic data.

Layout on server
  vendor/           independent snapshot of the installed 3DGS Python source
  vendor_sha256.txt exact source snapshot hashes (no available Git revision)
  geometry_{scene}/ fresh common geometry, cameras and input-point snapshot
  reference_rasterizer/ official CUDA reference source (separate research license)
  reference_deps/   project-local compiled extension, selected by experiment.py
  src/experiment.py five solvers, exact raw-color rendering, operator checks
  src/test_solvers.py dense SPD cross-check
  results/          split/input manifests, traces, metrics and color checkpoints
  logs/             complete geometry and appearance execution logs

Reproduction
  Copy the independent vendor snapshot into a new project root, retaining the
  directory layout above. Install the CUDA rasterizer and other packages from
  the recorded environment. The source dataset must already have cameras.bin,
  images.bin, points3D.bin and points3D.ply in sparse/0; the original loader may
  generate the PLY if it is absent, so verify its presence before running with
  a read-only dataset. Do not run setup_remote.sh on an existing project: it
  intentionally refuses to overwrite an existing target.

  cd PROJECT/vendor
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 PYTHON train.py \
    -s DATASET/train -m PROJECT/geometry_train --eval -r 512 --sh_degree 0 \
    --iterations 3000 --save_iterations 3000 --test_iterations 3000 \
    --densify_until_iter 2000 --opacity_reset_interval 10000 \
    --optimizer_type default --port 6027
  Repeat for truck, using an available GPU and different port if concurrent.
  Replace PYTHON, PROJECT and DATASET with actual paths, not shell variables
  from this explanatory example. The original safe_state sets preparation
  seeds to zero. CUDA atomics can cause minor nondeterminism; exact published
  geometry hashes identify the actual checkpoints used in the measurements.

  Before appearance runs, build the project-local reference extension:
  CUDA_HOME=/home/tianyi/data/anaconda3/envs/3dgs
  PATH="$CUDA_HOME/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=4 \
    PYTHON -m pip install PROJECT/reference_rasterizer \
    --target PROJECT/reference_deps --no-deps --no-build-isolation
  Export CUDA_HOME in the shell as well. This selects the existing CUDA 11.6
  toolkit, rather than system CUDA 12.1, which does not match PyTorch 1.13.
  The original conda environment is not overwritten by the local extension.

  CUDA_VISIBLE_DEVICES=0 PYTHON PROJECT/src/test_solvers.py
  CUDA_VISIBLE_DEVICES=0 PYTHON PROJECT/src/experiment.py --scene train --pilot
  CUDA_VISIBLE_DEVICES=7 PYTHON PROJECT/src/experiment.py --scene truck --pilot

  Each results directory must be new. Main runs use 100 updates, width 256,
  8 fitted views, ridge 0.01, and seeds 0 1 2. Adam pilots use seed 0 and
  40 updates for learning rates 0.003, 0.01 and 0.03; selection uses only
  appearance validation. Method order rotates by seed. Identical x0 is
  regenerated for every method in a seed. No method-specific early stopping
  is applied in the reported equal-update comparison.

  Sensitivity commands (both scenes, seed 0):
    experiment.py --scene train --seeds 0 --ridge 0.1 --tag ridge_0.1 --pilot
    experiment.py --scene train --seeds 0 --ridge 1 --tag ridge_1 --pilot
    experiment.py --scene train --seeds 0 --ridge 10 --tag ridge_10 --pilot
    experiment.py --scene train --seeds 0 --views 32 --tag views32 --pilot
    experiment.py --scene train --seeds 0 --width 512 --tag width512 --pilot
    experiment.py --scene train --seeds 0 --steps 1000 --methods CG PCG --tag extended
  Repeat replacing train with truck. These runs share geometry with main.

Interpretation and limitations
  Test images are every eighth name-sorted view. Geometry trains on all other
  targets. Appearance validation is every eleventh member of that training
  pool and is not used in the color objective. Eight (or 32) uniformly spaced
  remaining views fit colors. COLMAP reconstruction may already use all views.
  Validation therefore selects appearance hyperparameters conditional on
  shared geometry, rather than end-to-end reconstruction hyperparameters.
  Three seeds vary color noise only; geometry is prepared once per scene.
  CG/PCG restart after true residual replacement every 25 updates.
  Row-sum preconditioning is a diagonal majorizer, not exact Jacobi.
  Raw PSNR and clipped PSNR/SSIM are separately labelled. The report's primary
  optimization values are unclipped. Less training loss need not improve test
  quality. Solver timing excludes setup, history diagnostics, evaluation and
  pilots; pilot and setup costs are recorded separately. Hardware is shared.

Results and report generation
  All rows and curves must be regenerated from result JSON/CSV files. Run
  build_report.py locally after obtaining the result archive. Its figures use
  Matplotlib. The Overleaf project keeps the original neurips_2026.sty and
  existing layout. The main source imports report_body.tex; no custom margin,
  font-size, line-spacing, style option, or geometry override is introduced.
  report_body.tex contains the cited literature and actual-results analysis.

Security and ownership
  No credentials are part of the reproducibility archive. Existing datasets,
  previous experiment directories and synced sources/ files were not edited.
  Preserve the copied third-party LICENSE.md with the vendor source. Its
  research/evaluation license is not replaced by a new license for this work.
  The cloud document is a draft research report; no Canvas submission occurs.

Administrative constraint
  The personal assignment must differ from the user's group topic. The group
  topic was not provided in the accessible context, so distinctness cannot be
  independently certified. Personal identifiers were omitted as requested.
