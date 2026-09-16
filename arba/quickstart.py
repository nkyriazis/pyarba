"""Self-contained toy demo of triangulation-free pose refinement.

Synthesises a scene with a known answer, so there is nothing to download:

  1. place N cameras on an arc looking at a cloud of 3D points;
  2. project the points to get exact pixel observations and the
     correspondences between them (this stands in for SIFT + matching,
     which is the only thing images are ever used for);
  3. perturb the camera poses to make a *coarse prior*;
  4. refine the prior with the ray-time bundle adjuster, annealed under
     graduated non-convexity;
  5. compare refined poses against the poses we started from.

No 3D point is ever triangulated and no camera is ever dropped: the
structure is marginalised out per observation as a ray time. The 3D points
exist here only to manufacture the correspondences and the ground truth.

Run (with the built module on PYTHONPATH):

    python -m arba.quickstart

Takes a few seconds and prints prior vs refined pose error. Exits non-zero
if the refinement fails to improve on the prior, so it doubles as a
smoke test.
"""

import argparse
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from arba.refine_core import RefinementProblem, refine
from arba.utils import init_logging, log

WIDTH, HEIGHT = 640, 480
FOCAL = np.array([500.0, 500.0])
PRINCIPAL_POINT = np.array([WIDTH / 2.0, HEIGHT / 2.0])
DIST_COEFFS = np.zeros(4)


def look_at(eye, target, up=np.array([0.0, -1.0, 0.0])):
    """Camera-to-world rotation with OpenCV axes (+x right, +y down, +z
    forward), matching arcore_bundle_adjuster/math.hpp."""
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.stack([right, down, forward], axis=1)


def make_scene(n_cameras: int, n_points: int, seed: int):
    """Cameras on an arc around a cloud of points. Returns ground-truth
    positions (N, 3), quaternions xyzw (N, 4) and points (P, 3)."""
    rng = np.random.default_rng(seed)

    angles = np.linspace(-0.6, 0.6, n_cameras)
    radius = 3.0
    positions = np.stack(
        [
            radius * np.sin(angles),
            0.25 * np.sin(3 * angles),
            -radius * np.cos(angles),
        ],
        axis=1,
    )
    target = np.zeros(3)
    quaternions = np.stack(
        [Rotation.from_matrix(look_at(p, target)).as_quat() for p in positions]
    )

    points = rng.uniform(-0.9, 0.9, size=(n_points, 3))
    points[:, 2] += rng.uniform(-0.4, 0.4, size=n_points)
    return positions, quaternions, points


def project(position, quaternion_xyzw, points):
    """Pinhole projection of world points into one camera. Returns pixels
    (P, 2) and a boolean visibility mask."""
    R = Rotation.from_quat(quaternion_xyzw).as_matrix()
    local = (points - position) @ R  # world -> camera
    in_front = local[:, 2] > 1e-3
    z = np.where(in_front, local[:, 2], 1.0)
    pixels = local[:, :2] / z[:, None] * FOCAL + PRINCIPAL_POINT
    inside = (
        (pixels[:, 0] >= 0)
        & (pixels[:, 0] < WIDTH)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < HEIGHT)
    )
    return pixels, in_front & inside


def synthesise_observations(
    positions, quaternions, points, pixel_noise: float, seed: int
):
    """Exact correspondences: what SIFT + geometric verification would give
    on real images, without the images. Returns (features, matches,
    visibility)."""
    rng = np.random.default_rng(seed + 1)
    features, visible_point_ids = {}, {}
    for i in range(len(positions)):
        pixels, mask = project(positions[i], quaternions[i], points)
        ids = np.flatnonzero(mask)
        obs = pixels[ids]
        if pixel_noise > 0:
            obs = obs + rng.normal(0.0, pixel_noise, size=obs.shape)
        features[i] = obs.astype(np.float64)
        visible_point_ids[i] = ids

    matches = {}
    for i in range(len(positions)):
        index_i = {p: k for k, p in enumerate(visible_point_ids[i])}
        for j in range(i + 1, len(positions)):
            shared = np.intersect1d(
                visible_point_ids[i], visible_point_ids[j], assume_unique=True
            )
            if len(shared) < 15:
                continue
            index_j = {p: k for k, p in enumerate(visible_point_ids[j])}
            matches[(i, j)] = np.array(
                [[index_i[p], index_j[p]] for p in shared], dtype=np.int32
            )
    return features, matches


def perturb(positions, quaternions, rot_deg: float, trans_m: float, seed: int):
    """Turn ground truth into a coarse prior: a random rotation of fixed
    magnitude and a random translation of fixed magnitude per camera."""
    rng = np.random.default_rng(seed + 2)
    n = len(positions)

    axes = rng.normal(size=(n, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    delta = Rotation.from_rotvec(axes * np.deg2rad(rot_deg))

    offsets = rng.normal(size=(n, 3))
    offsets /= np.linalg.norm(offsets, axis=1, keepdims=True)

    return (
        positions + offsets * trans_m,
        (delta * Rotation.from_quat(quaternions)).as_quat(),
    )


def align_sim3(positions, quaternions, gt_positions, gt_quaternions):
    """Umeyama similarity alignment of a pose set onto ground truth.

    The objective is gauge-free: a global similarity of the cameras and
    their rays leaves every cross-projection residual unchanged, and the
    position prior pins the gauge only as firmly as its weight. So the
    honest measure of what refinement recovered is the error after the
    gauge is factored out, which is also the standard evaluation protocol.
    """
    mu_a, mu_b = positions.mean(0), gt_positions.mean(0)
    a, b = positions - mu_a, gt_positions - mu_b
    u, sigma, vt = np.linalg.svd(a.T @ b / len(positions))
    d = np.sign(np.linalg.det(u @ vt))
    rotation = (u @ np.diag([1.0, 1.0, d]) @ vt).T
    scale = (sigma * np.array([1.0, 1.0, d])).sum() * len(positions) / (
        a**2
    ).sum()
    aligned_positions = scale * (rotation @ positions.T).T + (
        mu_b - scale * rotation @ mu_a
    )
    aligned_quaternions = (
        Rotation.from_matrix(rotation) * Rotation.from_quat(quaternions)
    ).as_quat()
    return aligned_positions, aligned_quaternions


def pose_error(positions, quaternions, gt_positions, gt_quaternions):
    """Mean absolute pose error against ground truth, after similarity
    alignment: translation in mm, rotation in degrees."""
    positions, quaternions = align_sim3(
        positions, quaternions, gt_positions, gt_quaternions
    )
    trans_mm = np.linalg.norm(positions - gt_positions, axis=1) * 1000.0
    rel = Rotation.from_quat(quaternions) * Rotation.from_quat(
        gt_quaternions
    ).inv()
    rot_deg = np.rad2deg(np.linalg.norm(rel.as_rotvec(), axis=1))
    return trans_mm.mean(), rot_deg.mean()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cameras", type=int, default=12)
    parser.add_argument("--points", type=int, default=250)
    parser.add_argument(
        "--rot-deg",
        type=float,
        default=3.0,
        help="magnitude of the per-camera orientation error in the prior",
    )
    parser.add_argument(
        "--trans-mm",
        type=float,
        default=60.0,
        help="magnitude of the per-camera position error in the prior",
    )
    parser.add_argument(
        "--pixel-noise",
        type=float,
        default=0.5,
        help="std of the Gaussian noise added to the observations",
    )
    parser.add_argument(
        "--gnc-schedule",
        default="1e4,1e3,1e2",
        help="arctan loss scales, one warm-started solve per stage",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    init_logging()

    gt_positions, gt_quaternions, points = make_scene(
        args.cameras, args.points, args.seed
    )
    features, matches = synthesise_observations(
        gt_positions, gt_quaternions, points, args.pixel_noise, args.seed
    )
    prior_positions, prior_quaternions = perturb(
        gt_positions,
        gt_quaternions,
        args.rot_deg,
        args.trans_mm / 1000.0,
        args.seed,
    )

    log.info(
        f"Toy scene: {args.cameras} cameras, {args.points} points, "
        f"{len(matches)} verified pairs, "
        f"{sum(len(m) for m in matches.values())} correspondences"
    )

    problem = RefinementProblem(
        image_names=[f"{i:04d}.png" for i in range(args.cameras)],
        focal_length=FOCAL.copy(),
        principal_point=PRINCIPAL_POINT.copy(),
        dist_coeffs=DIST_COEFFS.copy(),
        positions=prior_positions.copy(),
        quaternions_xyzw=prior_quaternions.copy(),
        features=features,
        matches=matches,
    )

    positions, quaternions, _, report, elapsed = refine(
        problem,
        gnc_schedule=args.gnc_schedule,
        position_prior_lambda=1e-2,
        optimize="camera_poses",
        verbose=args.verbose,
    )

    prior_mm, prior_deg = pose_error(
        prior_positions, prior_quaternions, gt_positions, gt_quaternions
    )
    refined_mm, refined_deg = pose_error(
        positions, quaternions, gt_positions, gt_quaternions
    )

    print()
    print(f"solved in {elapsed:.2f} s; final Ceres report:")
    print(
        "  "
        + "\n  ".join(
            line
            for line in report.strip().splitlines()
            if line.startswith(("Termination", "Initial", "Final", "Change"))
        )
    )
    print()
    print("mean absolute pose error against the known answer")
    print("(similarity-aligned; the objective is gauge-free)")
    print(f"  prior:    {prior_mm:8.2f} mm   {prior_deg:7.3f} deg")
    print(f"  refined:  {refined_mm:8.2f} mm   {refined_deg:7.3f} deg")
    print(
        f"  improved: {prior_mm / max(refined_mm, 1e-9):8.1f}x  "
        f"{prior_deg / max(refined_deg, 1e-9):7.1f}x"
    )

    # a smoke test, not a benchmark: the defaults recover the ground truth
    # to well under a millimetre, so a 5x margin is a wide one
    if refined_mm * 5 > prior_mm or refined_deg * 5 > prior_deg:
        print("\nFAILED: refinement did not improve on the prior")
        return 1
    print("\nOK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
