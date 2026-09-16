"""Dataset-agnostic pose refinement: images + prior camera-to-world poses +
intrinsics -> SIFT matching (pycolmap) -> ray-time bundle adjustment
(pyARBA) -> refined poses.

Conventions (must match arcore_bundle_adjuster/math.hpp):
- poses are camera-to-world: position (3,) and quaternion xyzw, OpenCV axes
  (+x right, +y down, +z forward);
- pixels are OpenCV-style, integer coordinates at pixel centers;
- distortion is OpenCV k1 k2 p1 p2 applied in normalized camera coordinates.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from arba.colmap_db import (
    read_images,
    read_keypoints,
    read_two_view_geometries,
)
from arba.utils import log


def compute_feature_tracks(problem: "RefinementProblem") -> dict:
    """Union-find over the match graph -> per-camera int arrays of track
    ids (-1 for unmatched features). Sharing one ray time per track restores
    multi-view rigidity (anchored-depth mode)."""
    parent = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for (i, j), data in problem.matches.items():
        for k0, k1 in data:
            union((i, int(k0)), (j, int(k1)))

    root_ids = {}
    tracks = {}
    for cam, feats in problem.features.items():
        ids = np.full(len(feats), -1, dtype=np.int32)
        for k in range(len(feats)):
            if (cam, k) in parent:
                root = find((cam, k))
                ids[k] = root_ids.setdefault(root, len(root_ids))
        tracks[cam] = ids.reshape(-1, 1)
    return tracks


@dataclass
class RefinementProblem:
    image_names: list  # sorted image file names, index == camera index
    focal_length: np.ndarray  # (2,)
    principal_point: np.ndarray  # (2,)
    dist_coeffs: np.ndarray  # (4,)
    positions: np.ndarray  # (N, 3) camera-to-world
    quaternions_xyzw: np.ndarray  # (N, 4) camera-to-world
    features: dict = field(default_factory=dict)  # cam idx -> (M, 2) pixels
    matches: dict = field(default_factory=dict)  # (i, j) -> (K, 2) kp indices


def poses_to_matrices(positions, quaternions_xyzw):
    mats = np.tile(np.eye(4), (len(positions), 1, 1))
    mats[:, :3, :3] = Rotation.from_quat(quaternions_xyzw).as_matrix()
    mats[:, :3, 3] = positions
    return mats


def matrices_to_poses(mats):
    positions = mats[:, :3, 3].copy()
    quaternions = Rotation.from_matrix(mats[:, :3, :3]).as_quat()
    return positions, quaternions


def extract_and_match(
    image_dir: Path,
    database_path: Path,
    camera_params: str,
    camera_model: str = "PINHOLE",
    use_gpu: bool = False,
):
    """SIFT extraction + exhaustive matching + geometric verification with
    pycolmap, one shared camera for all frames."""
    import pycolmap

    if database_path.exists():
        log.info(f"Reusing existing database {database_path}")
        return

    device = pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu
    pycolmap.extract_features(
        database_path.as_posix(),
        image_dir.as_posix(),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        reader_options={"camera_model": camera_model, "camera_params": camera_params},
        device=device,
    )
    pycolmap.match_exhaustive(database_path.as_posix(), device=device)


def _prior_epipolar_filter(
    problem: RefinementProblem,
    keypoints0: np.ndarray,
    keypoints1: np.ndarray,
    i: int,
    j: int,
    data: np.ndarray,
    threshold_px: float,
) -> np.ndarray:
    """Keep matches whose symmetric epipolar distance under the *prior*
    poses/intrinsics is below threshold_px. Rejects repeated-texture aliasing
    that in-pair RANSAC verification cannot catch."""
    K = np.array(
        [
            [problem.focal_length[0], 0, problem.principal_point[0]],
            [0, problem.focal_length[1], problem.principal_point[1]],
            [0, 0, 1.0],
        ]
    )
    Ri = Rotation.from_quat(problem.quaternions_xyzw[i]).as_matrix()
    Rj = Rotation.from_quat(problem.quaternions_xyzw[j]).as_matrix()
    ti, tj = problem.positions[i], problem.positions[j]
    # world->cam_j composed with cam_i->world
    R_rel = Rj.T @ Ri
    t_rel = Rj.T @ (ti - tj)
    tx = np.array(
        [
            [0, -t_rel[2], t_rel[1]],
            [t_rel[2], 0, -t_rel[0]],
            [-t_rel[1], t_rel[0], 0],
        ]
    )
    Kinv = np.linalg.inv(K)
    F = Kinv.T @ tx @ R_rel @ Kinv

    p0 = np.hstack([keypoints0[data[:, 0]], np.ones((len(data), 1))])
    p1 = np.hstack([keypoints1[data[:, 1]], np.ones((len(data), 1))])
    Fp0 = p0 @ F.T
    Ftp1 = p1 @ F
    num = np.abs(np.sum(p1 * Fp0, axis=1))
    d0 = num / np.linalg.norm(Fp0[:, :2], axis=1)
    d1 = num / np.linalg.norm(Ftp1[:, :2], axis=1)
    return data[np.maximum(d0, d1) < threshold_px]


def refilter_matches(
    problem: RefinementProblem,
    positions: np.ndarray,
    quaternions_xyzw: np.ndarray,
    threshold_px: float,
    min_pair_inliers: int = 15,
) -> tuple:
    """Residual-driven match refiltering: keep matches whose symmetric
    epipolar distance under the *given* (refined) poses is below
    threshold_px. Returns (filtered matches dict, kept, total)."""
    saved = problem.positions, problem.quaternions_xyzw
    problem.positions, problem.quaternions_xyzw = positions, quaternions_xyzw
    filtered = {}
    kept = total = 0
    try:
        for (i, j), data in problem.matches.items():
            new = _prior_epipolar_filter(
                problem,
                problem.features[i],
                problem.features[j],
                i,
                j,
                data,
                threshold_px,
            )
            total += len(data)
            if len(new) >= min_pair_inliers:
                kept += len(new)
                filtered[(i, j)] = new
    finally:
        problem.positions, problem.quaternions_xyzw = saved
    log.info(
        f"Refilter @ {threshold_px} px: kept {kept}/{total} correspondences, "
        f"{len(filtered)}/{len(problem.matches)} pairs"
    )
    return filtered, kept, total


def load_problem_from_db(
    problem: RefinementProblem,
    database_path: Path,
    min_pair_inliers: int = 15,
    max_pair_matches: int = 0,
    prior_epipolar_px: float = 0.0,
    min_observation_degree: int = 0,
) -> RefinementProblem:
    """Fill features/matches from a COLMAP database, remapping image ids to
    the problem's frame indices (by file name)."""
    name_to_index = {name: i for i, name in enumerate(problem.image_names)}

    id_to_index = {}
    for image in read_images(database_path):
        if image["name"] in name_to_index:
            id_to_index[image["image_id"]] = name_to_index[image["name"]]

    keypoints = {
        kp["image_id"]: kp["data"] for kp in read_keypoints(database_path)
    }

    for tvg in read_two_view_geometries(database_path):
        id0, id1 = tvg["pair_id"]
        if (
            tvg["data"] is None
            or len(tvg["data"]) < min_pair_inliers
            or id0 not in id_to_index
            or id1 not in id_to_index
        ):
            continue
        data = tvg["data"].astype(np.int32)
        if prior_epipolar_px > 0:
            data = _prior_epipolar_filter(
                problem,
                keypoints[id0][:, :2] - 0.5,
                keypoints[id1][:, :2] - 0.5,
                id_to_index[id0],
                id_to_index[id1],
                data,
                prior_epipolar_px,
            )
            if len(data) < min_pair_inliers:
                continue
        if max_pair_matches and len(data) > max_pair_matches:
            # deterministic per-pair subsample
            rng = np.random.default_rng([id0, id1])
            data = data[
                rng.choice(len(data), max_pair_matches, replace=False)
            ]
        problem.matches[(id_to_index[id0], id_to_index[id1])] = data

    if min_observation_degree > 1:
        # drop matches in which BOTH observations appear in fewer than
        # min_observation_degree pairs: such matches contribute purely
        # pairwise epipolar information with no cross-view coupling
        from collections import Counter

        degree = Counter()
        for (i, j), data in problem.matches.items():
            for k0, k1 in data:
                degree[(i, k0)] += 1
                degree[(j, k1)] += 1
        total = kept = 0
        for (i, j), data in list(problem.matches.items()):
            keep = np.array(
                [
                    degree[(i, k0)] >= min_observation_degree
                    or degree[(j, k1)] >= min_observation_degree
                    for k0, k1 in data
                ]
            )
            total += len(data)
            kept += int(keep.sum())
            if keep.sum() < 5:
                del problem.matches[(i, j)]
            else:
                problem.matches[(i, j)] = data[keep]
        log.info(
            f"Degree filter: kept {kept}/{total} correspondences"
        )

    referenced = set()
    for i, j in problem.matches:
        referenced.add(i)
        referenced.add(j)

    for image_id, index in id_to_index.items():
        if index in referenced:
            # COLMAP puts the center of the top-left pixel at (0.5, 0.5);
            # our projection model puts it at (0, 0)
            problem.features[index] = (
                keypoints[image_id][:, :2].astype(np.float64) - 0.5
            )

    log.info(
        f"Problem: {len(problem.features)}/{len(problem.image_names)} cameras "
        f"with features, {len(problem.matches)} verified pairs, "
        f"{sum(len(m) for m in problem.matches.values())} correspondences"
    )
    return problem


def refine(
    problem: RefinementProblem,
    arctan_loss: float = 1e2,
    position_prior_lambda: float = 1e-2,
    orientation_prior_lambda: float = 0.0,
    max_iterations: int = 100,
    function_tolerance: float = 1e-6,
    optimize: str = "camera_poses",
    track_sharing: bool = False,
    ray_time_init: dict = None,
    gnc_schedule: str = "",
    prior_schedule: str = "",
    consensus_schedule: str = "",
    consensus_joint: bool = False,
    lifted_sigma_schedule: str = "",
    rotation_first: bool = False,
    linear_solver: str = "sparse_schur",
    verbose: bool = False,
):
    """Run the ray-time bundle adjuster. Returns (positions, quaternions,
    intrinsics dict, solver report, wall seconds).

    Staged solving (all warm-started on the same problem object):
    - gnc_schedule: comma-separated arctan scales, one solve per stage
      (graduated non-convexity), e.g. "1e4,1e3,1e2";
    - prior_schedule: comma-separated position-prior lambdas relaxed over
      the same stages (prior homotopy);
    - rotation_first: prepend an orientations-only solve (positions and
      ray times still free; positions held constant);
    - consensus_schedule: comma-separated weights for the lifted track
      consensus (alternated): after the main solve, each round recomputes
      per-track consensus points closed-form from the current state and
      re-solves with residuals pulling ray points toward them,
      e.g. "1e4,1e6"."""
    from pyARBA import bundle_adjuster

    camera_poses = {
        i: (problem.positions[i].copy(), problem.quaternions_xyzw[i].copy())
        for i in sorted(problem.features)
    }

    t0 = time.perf_counter()
    ba = bundle_adjuster(
        focal_length=problem.focal_length.copy(),
        principal_point=problem.principal_point.copy(),
        distortion_coefficients=problem.dist_coeffs.copy(),
        features={i: problem.features[i] for i in sorted(problem.features)},
        matches=problem.matches,
        camera_poses=camera_poses,
        arctan_loss=arctan_loss,
        position_prior_lambda=position_prior_lambda,
        orientation_prior_lambda=orientation_prior_lambda,
        feature_tracks=(
            compute_feature_tracks(problem)
            if (track_sharing or consensus_schedule)
            else {}
        ),
        initial_ray_times=(
            {i: t.reshape(-1, 1) for i, t in ray_time_init.items()}
            if ray_time_init
            else {}
        ),
        consensus=bool(consensus_schedule),
        consensus_joint=consensus_joint,
        lifted_weights_sigma=(
            float(lifted_sigma_schedule.split(",")[0])
            if lifted_sigma_schedule
            else 0.0
        ),
    )

    inclusion = bundle_adjuster.optimization_inclusion.none
    for part in optimize.split("+"):
        inclusion = inclusion | getattr(
            bundle_adjuster.optimization_inclusion, part
        )

    if rotation_first:
        log.info("Stage: rotation-first (positions constant)")
        ba.solve(
            max_iterations=max_iterations,
            verbose=verbose,
            inclusion=bundle_adjuster.optimization_inclusion.camera_orientations,
            function_tolerance=function_tolerance,
            linear_solver=linear_solver,
        )

    # lifted-weights sigma annealing (GNC-GM): one warm-started solve per
    # sigma except the last, which the final solve below uses
    lifted_sigmas = [
        float(x) for x in lifted_sigma_schedule.split(",") if x
    ]
    for s in lifted_sigmas[:-1]:
        log.info(f"Lifted-GM stage: sigma {s:g}")
        ba.set_lifted_sigma(s)
        ba.solve(
            max_iterations=max_iterations,
            verbose=verbose,
            inclusion=inclusion,
            function_tolerance=function_tolerance,
            linear_solver=linear_solver,
        )
    if lifted_sigmas:
        ba.set_lifted_sigma(lifted_sigmas[-1])

    gnc = [float(x) for x in gnc_schedule.split(",") if x]
    priors = [float(x) for x in prior_schedule.split(",") if x]
    for k in range(max(len(gnc), len(priors)) - 1 if gnc or priors else 0):
        a = gnc[min(k, len(gnc) - 1)] if gnc else arctan_loss
        lam = (
            priors[min(k, len(priors) - 1)]
            if priors
            else position_prior_lambda
        )
        log.info(f"Stage {k}: arctan {a:g}, position prior {lam:g}")
        ba.set_arctan_loss(a)
        ba.set_position_prior_lambda(lam)
        ba.solve(
            max_iterations=max_iterations,
            verbose=verbose,
            inclusion=inclusion,
            function_tolerance=function_tolerance,
            linear_solver=linear_solver,
        )
    # final stage at the schedule's last values (or the nominal ones)
    ba.set_arctan_loss(gnc[-1] if gnc else arctan_loss)
    ba.set_position_prior_lambda(
        priors[-1] if priors else position_prior_lambda
    )

    report = ba.solve(
        max_iterations=max_iterations,
        verbose=verbose,
        inclusion=inclusion,
        function_tolerance=function_tolerance,
        linear_solver=linear_solver,
    )

    consensus_weights = [
        float(x) for x in consensus_schedule.split(",") if x
    ]
    for j, w in enumerate(consensus_weights):
        log.info(f"Consensus round: weight {w:g}")
        # joint mode: consensus points are variables — initialize them
        # once and let the solver move them; alternated mode recomputes
        # the fixed targets every round
        if not consensus_joint or j == 0:
            ba.update_consensus_targets()
        ba.set_consensus_weight(w)
        report = ba.solve(
            max_iterations=max_iterations,
            verbose=verbose,
            inclusion=inclusion,
            function_tolerance=function_tolerance,
            linear_solver=linear_solver,
        )
    elapsed = time.perf_counter() - t0

    refined = ba.camera_poses
    positions = problem.positions.copy()
    quaternions = problem.quaternions_xyzw.copy()
    for i, (pos, quat) in refined.items():
        positions[i] = pos
        quaternions[i] = quat

    intrinsics = {
        "focal_length": np.asarray(ba.focal_length),
        "principal_point": np.asarray(ba.principal_point),
        "dist_coeffs": np.asarray(ba.distortion_coefficients),
    }
    return positions, quaternions, intrinsics, report, elapsed
