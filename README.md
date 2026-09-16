# pyARBA — triangulation-free bundle adjustment

Code companion to *Triangulation-Free Bundle Adjustment with Graduated
Non-Convexity for Camera Pose Refinement from Coarse Priors*
([arXiv:2608.21008](https://arxiv.org/abs/2608.21008)), Nikolaos Kyriazis.

An AR capture provides a metric camera pose for every frame, accurate to a
fraction of a degree. That is not accurate enough for novel-view synthesis.
The common approach is to triangulate structure from the prior and
bundle-adjust, which commits the prior's error into the structure the
optimiser then trusts, and which can drop frames that fail to register.

This code refines the prior directly. Each keypoint carries one scalar depth
along its own back-projected ray, and each match contributes two symmetric
cross-projection residuals, so structure is re-expressed at every iterate
rather than fixed in advance. Because no structure is fixed, the objective
can be annealed under graduated non-convexity. Annealing widens the range of
prior error from which the solve converges, and the paper measures that
range. Because there is no reconstruction step, every input pose is refined
and no frame is dropped.

The method requires a pose prior and verified feature matches. It does not
build a map.

## Quickstart

```bash
docker build -t pyarba .
docker run --rm pyarba python3 -m arba.quickstart
```

No data is needed. `arba/quickstart.py` synthesises a scene with a known
answer: cameras on an arc, a cloud of points, exact correspondences, then
poses perturbed into a coarse prior. It refines that prior. The objective is
gauge-free up to a similarity, so error is reported after a Umeyama
alignment. Expected output:

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

The command returns 1 if refinement does not improve on the prior, so it can
be used as a smoke test. `--help` lists the options: prior magnitude
(`--rot-deg`, `--trans-mm`), observation noise, and the GNC schedule. At
`--rot-deg 16 --trans-mm 160` the prior is 146.05 mm and 18.842 deg and the
refined result is 1.46 mm and 0.043 deg.

## Prerequisites

The C++ extension depends on system libraries that pip cannot supply. They
have to be installed first, whichever way the package is then installed.

| dependency | why | floor |
| --- | --- | --- |
| Ceres | the solver | **2.2**. This code uses the `ceres::Manifold` API, which 2.1 does not have |
| SuiteSparse | Ceres' `SPARSE_SCHUR` | — |
| Eigen | every matrix in the solver | 3.4 |
| TBB | parallel residual evaluation | — |
| Boost log / system / filesystem | the solver's tracing | — |
| Boost unit_test_framework | builds `tests` | — |
| pybind11 | the Python bindings | **2.12**. 2.11 segfaults converting a NumPy >= 2 array, which is the first thing every entry point here does |
| Python 3 development headers | the Python bindings | 3.10 |

On Ubuntu 24.04 these come from one apt line. 24.04 is the oldest Ubuntu
whose packaged Ceres is 2.2, so nothing has to be compiled from source.

```bash
sudo apt-get install -y build-essential cmake \
  libeigen3-dev libceres-dev libsuitesparse-dev libtbb-dev \
  libboost-log-dev libboost-system-dev libboost-filesystem-dev \
  libboost-test-dev python3-dev python3-pip
```

pybind11 comes from pip rather than from apt. Ubuntu 24.04 ships pybind11
2.11.1, and pip will install NumPy 2 alongside it. That combination builds
and then crashes on the first call. CMake picks up the pip copy on its own.

If one of these is missing, the build stops with a message naming it and
repeating the apt line, rather than with a CMake module-search dump.

## Installing

```bash
pip install .
```

This builds the CMake project and places the `pyARBA` extension module in
the wheel next to the `arba` package. Both then import from any working
directory:

```bash
python -c "import arba, pyARBA"
python -m arba.quickstart
```

`pip install '.[sift]'` adds `pycolmap`. See *Python runtime dependencies*
below for what needs it.

The build backend is
[scikit-build-core](https://github.com/scikit-build/scikit-build-core).
`pyproject.toml` points it at `arcore_bundle_adjuster/CMakeLists.txt`, the
same CMake project the next section builds by hand.

## Building without Docker

The in-place build is what the companion experiments repository uses, and
what the driver docstrings there assume.

```bash
pip install -r requirements-run.txt

cmake -S arcore_bundle_adjuster -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j "$(nproc)"

./build/tests                                 # Boost.Test unit tests
PYTHONPATH="$PWD:$PWD/build" python3 -m arba.quickstart
```

`cmake --install build` places `pyARBA` in site-packages.

CUDA is optional and unused by default. Ceres picks it up for the
`dense_schur_cuda` linear solver if it is present; nothing here selects that
solver.

The C++ side traces each evaluation through Boost.Log. It is off by default.
`ARBA_LOG_LEVEL=info` (or `trace`, `debug`) enables it.

### Python runtime dependencies

numpy, scipy and tqdm. Every module under `arba/` imports on those three
alone, and CI checks that in a clean virtualenv on every push.

`refine_core.extract_and_match` imports `pycolmap` inside the function,
because the SIFT front end is the only part that needs it. Reading a COLMAP
database, building the problem, solving it and scoring the result all work
without pycolmap installed. To install it alongside the package:

```bash
pip install '.[sift]'
```

## Layout

| path | what it is |
| --- | --- |
| `arcore_bundle_adjuster/` | the solver, C++20 on Ceres. `bundle_adjuster.{hpp,cpp}` is the ray-time problem, `bundle_adjuster_cross_projection_residuals.cpp` the residuals, `pybind.cpp` the `pyARBA` Python module, `tests.cpp` the unit tests. `camera_trajectory_smoother.{hpp,cpp}` is a separate smoother over a pose sequence, exposed as `pyARBA.smooth_camera_trajectory`; the refinement path does not call it. |
| `arba/` | the Python package. `refine_core.py` is the dataset-agnostic entry point, from images, prior poses and intrinsics through matching and refinement to refined poses. `colmap_db.py` reads a COLMAP database. `pose_eval.py` is the evaluation protocol. `quickstart.py` is the synthetic demo. `utils.py` holds logging helpers. |
| `Dockerfile` | builds the solver and the Python package into one image. |
| `pyproject.toml` | the wheel build, through scikit-build-core over the same CMake project. |

The built extension module is `pyARBA`. The Python package that drives it is
`arba`. The C++ directory is named `arcore_bundle_adjuster` after what it
does, bundle adjustment for ARCore and ARKit pose priors.

## What the module exposes

`arba/refine_core.py` is the entry point. It takes image file names, a shared
pinhole intrinsic, and camera-to-world prior poses as positions plus xyzw
quaternions in OpenCV axes (+x right, +y down, +z forward). The conventions
are stated at the top of the file and in
`arcore_bundle_adjuster/math.hpp`.

- `extract_and_match` runs SIFT extraction, exhaustive matching and
  geometric verification through pycolmap into a COLMAP database. It is the
  only function that needs pycolmap. Any COLMAP database can be used
  instead, whatever produced it.
- `load_problem_from_db` reads that database into a `RefinementProblem`:
  keypoints, two-view geometries, verified pairs, prior poses, intrinsics.
- `refine` solves the problem. `gnc_schedule` sets the annealing. It returns
  the refined poses, the refined intrinsics, the Ceres report and the wall
  time.
- `compute_feature_tracks` is the union-find over the match graph used by
  the anchored-depth mode, in which a whole track shares one ray time.

`arba/pose_eval.py` holds the scoring. `evaluate_poses` applies the
similarity (Umeyama) alignment the gauge-free objective requires and reports
mean absolute rotation and translation error.

`arba/quickstart.py` is a worked example of the solve and the scoring. It
builds a `RefinementProblem` directly and calls `refine` and
`evaluate_poses`. It does not call `extract_and_match`, so it needs neither
images nor pycolmap.

## Continuous integration

`.github/workflows/build.yml` runs four jobs on ubuntu-24.04:

- build the C++ from the apt dependencies above, run the Boost.Test suite,
  run the quickstart;
- import every `arba/` module in a clean virtualenv containing only numpy,
  scipy and tqdm;
- `pip install .` into a clean virtualenv, then import `arba` and `pyARBA`
  and run the quickstart from outside the source tree;
- build the Docker image, then run the tests and the quickstart inside it.

## Citation

Please cite the accompanying paper, *Triangulation-Free Bundle Adjustment
with Graduated Non-Convexity for Camera Pose Refinement from Coarse Priors*.

## License

Apache License 2.0, see [LICENSE](LICENSE).
