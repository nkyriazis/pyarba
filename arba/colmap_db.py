import sqlite3
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from tqdm import tqdm


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


def get_data_with_names_from_colmap_db(
    colmap_db_path: Path, table_name: str
) -> List[Dict]:
    with sqlite3.connect(colmap_db_path) as db:
        cursor = db.cursor()

        # fetch two_view_geometries from two_view_geometries table
        cursor.execute(f"PRAGMA table_info({table_name})")
        col_names = cursor.fetchall()
        col_names = [col[1] for col in col_names]

        cursor.execute(f"SELECT * FROM {table_name}")
        data = [
            dict(zip(col_names, row)) for row in tqdm(cursor, f"Reading {table_name}")
        ]

        return data


def read_images(colmap_db_path: Path) -> List[Dict]:
    """
    Read images from colmap database.
    Each entry has image id (image_id), image name (name), and camera id (camera_id).
    """
    images = get_data_with_names_from_colmap_db(colmap_db_path, "images")
    return images


def read_cameras(colmap_db_path: Path) -> List[Dict]:
    """
    Read cameras from colmap database.
    Each entry has camera id (camera_id), camera model (model), and camera parameters (params).
    """
    cameras = get_data_with_names_from_colmap_db(colmap_db_path, "cameras")

    for camera in cameras:
        if camera["params"] is not None:
            camera["params"] = np.frombuffer(camera["params"], dtype=np.float64)

    return cameras


def read_matches(colmap_db_path: Path) -> List[Dict]:
    """
    Read matches from colmap database.
    Each entry has image pair id (pair_id), number of matches (num_matches), and matches (data).
    """
    matches = get_data_with_names_from_colmap_db(colmap_db_path, "matches")

    # decode paid_id
    for match in matches:
        match["pair_id"] = pair_id_to_image_ids(match["pair_id"])

    # decode data as rows * cols * int32
    for match in matches:
        if match["data"] is not None:
            match["data"] = np.frombuffer(match["data"], dtype=np.int32).reshape(
                match["rows"], match["cols"]
            )

    return matches


def read_keypoints(colmap_db_path: Path) -> List[Dict]:
    """
    Read keypoints from colmap database.
    Each entry has image id (image_id), number of keypoints (num_keypoints), and keypoints (data).
    """
    keypoints = get_data_with_names_from_colmap_db(colmap_db_path, "keypoints")

    # decode data as rows * cols * float32
    for keypoint in keypoints:
        if keypoint["data"] is not None:
            keypoint["data"] = np.frombuffer(
                keypoint["data"], dtype=np.float32
            ).reshape(keypoint["rows"], keypoint["cols"])

    return keypoints


def read_two_view_geometries(colmab_db_path: Path) -> List[Dict]:
    """
    Read two_view_geometries from colmap database.
    Each entry has geometrically validated matches (data), along with
    computed Fundamental matrix (F), Essential matrix (E), and Homography matrix (H).
    """

    # fetch two_view_geometries from two_view_geometries table
    two_view_geometries = get_data_with_names_from_colmap_db(
        colmab_db_path, "two_view_geometries"
    )

    for two_view_geometry in two_view_geometries:
        two_view_geometry["pair_id"] = pair_id_to_image_ids(
            two_view_geometry["pair_id"]
        )
        if two_view_geometry["data"] is not None:
            two_view_geometry["data"] = np.frombuffer(
                two_view_geometry["data"], dtype=np.int32
            ).reshape(two_view_geometry["rows"], two_view_geometry["cols"])
        if two_view_geometry["F"] is not None:
            two_view_geometry["F"] = np.frombuffer(
                two_view_geometry["F"], dtype=np.float64
            ).reshape(3, 3)
        if two_view_geometry["E"] is not None:
            two_view_geometry["E"] = np.frombuffer(
                two_view_geometry["E"], dtype=np.float64
            ).reshape(3, 3)
        if two_view_geometry["H"] is not None:
            two_view_geometry["H"] = np.frombuffer(
                two_view_geometry["H"], dtype=np.float64
            ).reshape(3, 3)

    return two_view_geometries
