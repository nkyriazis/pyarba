# pyARBA — triangulation-free bundle adjustment for camera pose refinement

Code companion to *Triangulation-Free Bundle Adjustment with Graduated
Non-Convexity for Camera Pose Refinement from Coarse Priors*
([arXiv:2608.21008](https://arxiv.org/abs/2608.21008)), Nikolaos Kyriazis.

A phone hands you a metric camera pose for every frame it captures, accurate
to a fraction of a degree and a few millimetres. It is not accurate enough
for novel-view synthesis, and the usual way to improve it — triangulate
structure from the prior, then bundle-adjust — routinely makes it worse,
because the prior's error is baked into the structure the optimizer then
trusts.

This code refines the prior without ever triangulating a point. Every
keypoint owns one scalar depth along its own back-projected ray, and every
match contributes two symmetric cross-projection residuals, so structure is
re-expressed from scratch at each iterate instead of being committed up
front. Because nothing is committed, the objective can be annealed under
graduated non-convexity, which widens the basin of convergence from the
one or two degrees of prior error where classical refinement collapses to
tens of degrees. And because there is no reconstruction to succeed or fail,
every input pose comes out refined: no frame is ever dropped.

What it costs you is a standard SfM pass. It is a *refiner*: it needs a pose
prior to start from, and a set of verified feature matches. It does not, and
cannot, build a map from nothing.

## Quickstart

Two commands, about a minute, no data to download:

```bash
docker build -t pyarba .
docker run --rm pyarba python3 -m arba.quickstart
```

`arba/quickstart.py` synthesises a toy scene with a known answer — cameras
on an arc, a cloud of points, exact correspondences, then poses perturbed
into a coarse prior — and refines it. Because the objective is gauge-free up
to a similarity, error is reported after Umeyama alignment, the standard
protocol. Expected output:

```
INFO Toy scene: 12 cameras, 250 points, 66 verified pairs, 16500 correspondences
INFO Stage 0: arctan 10000, position prior 0.01
INFO Stage 1: arctan 1000, position prior 0.01

solved in 0.51 s; final Ceres report:
  Initial                          2.133716e-01
  Final                            2.133716e-01
  Change                           0.000000e+00
  Termination:                      CONVERGENCE (Function tolerance reached. ...)

mean absolute pose error against the known answer
(similarity-aligned; the objective is gauge-free)
  prior:       56.00 mm     4.829 deg
  refined:      1.46 mm     0.043 deg
  improved:     38.5x    112.0x

OK
```

It exits non-zero if refinement fails to beat the prior, so it doubles as a
smoke test. `--help` lists the knobs: prior magnitude (`--rot-deg`,
`--trans-mm`), observation noise, the GNC schedule. Push the prior out to
`--rot-deg 16 --trans-mm 160` and it still converges.

## Building without Docker

Everything comes from apt on Ubuntu 24.04. That version is not incidental:
it is the oldest Ubuntu whose packaged Ceres (2.2.0) has the
`ceres::Manifold` API this code uses, so nothing has to be compiled from
source.

```bash
sudo apt-get install -y \
  build-essential cmake \
  libeigen3-dev libceres-dev libsuitesparse-dev libtbb-dev \
  libboost-log-dev libboost-system-dev libboost-filesystem-dev \
  libboost-test-dev \
  python3-dev python3-pip

# pybind11 from pip, NOT from apt: pybind11 2.11 (what Ubuntu 24.04 ships)
# segfaults the moment it converts a NumPy >= 2 array, which is the first
# thing every entry point here does. CMake picks up the pip one on its own.
pip install -r requirements-run.txt

cmake -S arcore_bundle_adjuster -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

./build/tests                                 # Boost.Test unit tests
PYTHONPATH="$PWD:$PWD/build" python3 -m arba.quickstart
```

`cmake --install build` drops `pyARBA` into site-packages.

Build dependencies, in full: Ceres 2.2 (with SuiteSparse for
`SPARSE_SCHUR`), Eigen 3.4, TBB, Boost log / system / filesystem /
unit_test_framework, pybind11 >= 2.12, Python 3 development headers. CUDA
is optional — it is used only for Ceres' `dense_schur_cuda` linear solver,
and nothing here needs it.

Python runtime dependencies, in full: **numpy, scipy, tqdm**. Every module
under `arba/` imports on those three alone, and CI enforces it in a clean
virtualenv on every push. The one exception is optional and explicit:
`refine_core.extract_and_match` imports `pycolmap` inside the function,
because that is the only place the SIFT front end is needed. If you already
have a COLMAP database, you never touch it.

The C++ side logs its per-evaluation trace through Boost.Log; it is off by
default, and `ARBA_LOG_LEVEL=info` (or `trace`, `debug`) turns it back on.

## Layout

| path | what it is |
| --- | --- |
| `arcore_bundle_adjuster/` | the solver. C++20 on Ceres; `bundle_adjuster.{hpp,cpp}` is the ray-time problem, `bundle_adjuster_cross_projection_residuals.cpp` the residuals, `camera_trajectory_smoother.{hpp,cpp}` the temporal prior, `pybind.cpp` the `pyARBA` Python module, `tests.cpp` the unit tests. |
| `arba/` | the Python side. `refine_core.py` is the dataset-agnostic entry point (images + prior poses + intrinsics → matching → refinement → refined poses); `colmap_db.py` reads a COLMAP database; `pose_eval.py` is the evaluation protocol; `quickstart.py` the toy demo; `utils.py` logging helpers. |
| `Dockerfile` | the one image you want. Builds the solver and the Python side. |

The built module is `pyARBA`; the Python package that drives it is `arba`.
The C++ directory keeps the name `arcore_bundle_adjuster` because that is
what it is — bundle adjustment for ARCore/ARKit pose priors.

## What the module exposes

`arba/refine_core.py` is the seam. Give it image file names, a shared
pinhole intrinsic, and camera-to-world prior poses as positions plus xyzw
quaternions in OpenCV axes (+x right, +y down, +z forward — the conventions
are stated at the top of the file and in `arcore_bundle_adjuster/math.hpp`):

- `extract_and_match` runs SIFT extraction, exhaustive matching and
  geometric verification through pycolmap into a COLMAP database. This is
  the only entry point that needs pycolmap, and it is optional: any COLMAP
  database will do, whoever produced it.
- `load_problem_from_db` turns that database into a `RefinementProblem` —
  keypoints, two-view geometries, verified pairs, prior poses, intrinsics.
- `refine` solves it. `gnc_schedule` controls the annealing; it returns the
  refined poses, the refined intrinsics, the Ceres report and wall time.
- `compute_feature_tracks` is the union-find over the match graph that
  backs the anchored-depth mode, where a whole track shares one ray time.

`arba/pose_eval.py` is the scoring side: `evaluate_poses` applies the
similarity (Umeyama) alignment the gauge-free objective requires and
reports mean absolute rotation and translation error. `arba/quickstart.py`
is a complete worked example of all of it in under 300 lines, against a
scene whose answer is known exactly.

## Continuous integration

`.github/workflows/build.yml` builds the C++ from apt dependencies on
ubuntu-24.04, runs the Boost.Test suite, runs the quickstart, checks that
every `arba/` module imports on numpy + scipy + tqdm alone, and repeats the
build and both runs inside the Docker image. If it is green, the
instructions above work.

## Citation

If you use this in academic work, please cite the accompanying paper,
*Triangulation-Free Bundle Adjustment with Graduated Non-Convexity for
Camera Pose Refinement from Coarse Priors*.

## License

Apache License 2.0, see [LICENSE](LICENSE). The licence carries an express
patent grant, so what you receive with the code is unambiguous.
