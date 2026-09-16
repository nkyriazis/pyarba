"""Pose-accuracy metrics, ported from PoRF (ActiveVisionLab/porf, utils.py)
so that numbers are directly comparable with their published tables.

Poses are 4x4 camera-to-world matrices throughout.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def umeyama_alignment(x: np.ndarray, y: np.ndarray, with_scale: bool = True):
    """Sim(3) least-squares alignment (Umeyama, PAMI 1991).

    x, y: (m, n) — m = dimension, n = number of points. Returns r, t, c
    such that y ≈ c * r @ x + t.
    """
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    m, n = x.shape

    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis]) ** 2)

    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    u, d, v = np.linalg.svd(cov_xy)

    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        s[m - 1, m - 1] = -1

    r = u.dot(s).dot(v)

    c = np.trace(np.diag(d).dot(s)) / sigma_x if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def pose_alignment(poses_pred: np.ndarray, poses_gt: np.ndarray) -> np.ndarray:
    """Sim(3)-align predicted c2w poses to GT (PoRF protocol)."""
    xyz_pred = poses_pred[:, :3, 3].T
    xyz_gt = poses_gt[:, :3, 3].T
    r, t, scale = umeyama_alignment(xyz_pred, xyz_gt, with_scale=True)

    align_transformation = np.eye(4)
    align_transformation[:3, :3] = r
    align_transformation[:3, 3] = t

    aligned = []
    for pose in poses_pred:
        pose = pose.copy()
        pose[:3, 3] *= scale
        aligned.append(align_transformation @ pose)
    return np.stack(aligned)


def rotation_error(pose_error: np.ndarray) -> float:
    """Geodesic rotation angle (radians) of a relative 4x4 pose error."""
    r = Rotation.from_matrix(pose_error[:3, :3]).as_matrix()
    d = 0.5 * (r[0, 0] + r[1, 1] + r[2, 2] - 1.0)
    return float(np.arccos(max(min(d, 1.0), -1.0)))


def translation_error(pose_error: np.ndarray) -> float:
    return float(np.linalg.norm(pose_error[:3, 3]))


def compute_ate(gt: np.ndarray, pred: np.ndarray):
    """Per-frame absolute rotation (rad) / translation errors between
    aligned trajectories."""
    r_errs, t_errs = [], []
    for gt_pose, pred_pose in zip(gt, pred):
        err = np.linalg.inv(gt_pose) @ pred_pose
        r_errs.append(rotation_error(err))
        t_errs.append(translation_error(err))
    return np.array(r_errs), np.array(t_errs)


def compute_rpe(gt: np.ndarray, pred: np.ndarray):
    """Relative pose error between consecutive frames."""
    r_errs, t_errs = [], []
    for i in range(len(gt) - 1):
        gt_rel = np.linalg.inv(gt[i]) @ gt[i + 1]
        pred_rel = np.linalg.inv(pred[i]) @ pred[i + 1]
        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        r_errs.append(rotation_error(rel_err))
        t_errs.append(translation_error(rel_err))
    return np.array(r_errs), np.array(t_errs)


def evaluate_poses(poses_pred: np.ndarray, poses_gt: np.ndarray) -> dict:
    """PoRF-style report: Sim(3) align, then ATE/RPE.

    Returns rotations in degrees and translations in the GT metric unit
    (meters for MobileBrick; multiply by 1e3 for mm).
    """
    aligned = pose_alignment(poses_pred, poses_gt)
    ate_r, ate_t = compute_ate(poses_gt, aligned)
    rpe_r, rpe_t = compute_rpe(poses_gt, aligned)
    return {
        "ate_rot_deg_mean": float(np.rad2deg(ate_r.mean())),
        "ate_rot_deg_max": float(np.rad2deg(ate_r.max())),
        "ate_trans_mean": float(ate_t.mean()),
        "ate_trans_max": float(ate_t.max()),
        "rpe_rot_deg_mean": float(np.rad2deg(rpe_r.mean())),
        "rpe_trans_mean": float(rpe_t.mean()),
        "n_frames": int(len(poses_pred)),
    }
