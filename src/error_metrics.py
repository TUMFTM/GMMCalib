import numpy as np
from scipy.spatial.transform import Rotation as R  # type: ignore
from numpy.typing import NDArray


# given T_gt and T_est
# compute T_error
# see equation (4)
def compute_extrinsic_error(T_gt: NDArray[np.float64], T_est: NDArray[np.float64]):
    """
    Returns T_err
    """
    T_err = np.linalg.inv(T_est) @ T_gt
    return T_err


def rotation_translation_error(T_gt: NDArray[np.float64], T_est: NDArray[np.float64]):

    T_err = compute_extrinsic_error(T_gt, T_est)

    # --- Translation errors ---
    dx, dy, dz = T_err[:3, 3]

    # --- Rotation errors (in roll/pitch/yaw) ---
    R_err = R.from_matrix(T_err[:3, :3])
    # Change 'xyz' to whatever convention you prefer
    roll_deg, pitch_deg, yaw_deg = R_err.as_euler("xyz", degrees=True)

    roll_rad, pitch_rad, yaw_rad = R_err.as_euler("xyz", degrees=False)

    return {
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "roll_deg": roll_deg,
        "pitch_deg": pitch_deg,
        "yaw_deg": yaw_deg,
        "roll_rad": roll_rad,
        "pitch_rad": pitch_rad,
        "yaw_rad": yaw_rad,
    }


def rpy_xyz_to_homogeneous(rpy_xyz):
    """
    rpy_xyz: sequence of 6 floats [roll, pitch, yaw, x, y, z] in radians/meters.
    Returns a 4×4 numpy array T so that
      point_in_ref = T @ [point_in_sensor; 1]
    """
    roll, pitch, yaw, tx, ty, tz = rpy_xyz
    # build rotation (apply roll about X, then pitch about Y, then yaw about Z)
    R_mat = R.from_euler("xyz", [roll, pitch, yaw], degrees=False).as_matrix()
    T = np.eye(4, dtype=float)
    T[:3, :3] = R_mat
    T[:3, 3] = [tx, ty, tz]
    return T


def compute_mean_position_error(T_gt, T_est, V2_list_local):
    """
    Compute mean position error for points already in sensor-2 local frame.

    Parameters
    ----------
    T_gt : np.ndarray (4, 4)
        Ground truth transform (sensor-2 → sensor-1)
    T_est : np.ndarray (4, 4)
        Estimated transform (sensor-2 → sensor-1)
    V2_list_local : List[np.ndarray]
        List of point clouds in sensor-2 LOCAL frame, each (3, N_j)

    Returns
    -------
    delta_X : np.ndarray, shape (3, len(V2_list_local))
        Column j is the mean error vector for observation j.
    """
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3:4]
    R_est, t_est = T_est[:3, :3], T_est[:3, 3:4]

    delta_X = []
    for V2_local in V2_list_local:
        # Apply transforms
        V2_true = R_gt @ V2_local + t_gt  # (3, N_j)
        V2_est = R_est @ V2_local + t_est  # (3, N_j)

        # Compute error
        E_j = V2_true - V2_est  # (3, N_j)

        # Mean over points in this observation
        delta_X.append(E_j.mean(axis=1))  # (3,)

    return np.column_stack(delta_X)  # shape (3, len(V2_list_local))


def distance_based_metric_list(T_gt, T_est, V2_list_local):
    """
    Compute per-point L2 errors for points already in sensor-2 local frame.

    Parameters
    ----------
    T_gt : (4,4) ndarray
        Ground truth transform (sensor-2 → sensor-1)
    T_est : (4,4) ndarray
        Estimated transform (sensor-2 → sensor-1)
    V2_list_local : list of (3, N_j) arrays
        Point clouds in sensor-2 LOCAL frame

    Returns
    -------
    xs_all : (ΣN_j,) array of x-coordinates in sensor-2 frame
    dists_all : (ΣN_j,) array of per-point L2 errors
    m, b : float, float – best-fit line parameters
    sigma : float – standard deviation of residuals
    """
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3:4]
    R_est, t_est = T_est[:3, :3], T_est[:3, 3:4]

    xs_parts, d_parts = [], []

    for V2_local in V2_list_local:
        # Filter forward-facing points (x > 0 in sensor-2 frame)
        mask = V2_local[0, :] > 0
        V2_filtered = V2_local[:, mask]

        # Apply transforms
        P_true = R_gt @ V2_filtered + t_gt
        P_est = R_est @ V2_filtered + t_est

        # Store x-coordinates and errors
        xs_parts.append(V2_filtered[0, :])  # x in sensor-2 frame
        d_parts.append(np.linalg.norm(P_true - P_est, axis=0))

    # Aggregate
    xs_all = np.hstack(xs_parts)
    dists_all = np.hstack(d_parts)

    # Fit linear trend
    m, b = np.polyfit(xs_all, dists_all, 1)

    # Compute residuals
    resid = dists_all - (m * xs_all + b)
    sigma = np.std(resid)

    return xs_all, dists_all, m, b, sigma


# Helper function to transform points from global to local frame
def transform_to_local_frame(points_global_list, T_sensor_to_world):
    """
    Transform list of point clouds from global to sensor local frame.

    Parameters
    ----------
    points_global_list : List[np.ndarray]
        List of point clouds in global frame, each (3, N)
    T_sensor_to_world : np.ndarray (4, 4)
        Sensor to world transform

    Returns
    -------
    points_local_list : List[np.ndarray]
        List of point clouds in sensor local frame
    """
    T_world_to_sensor = np.linalg.inv(T_sensor_to_world)
    R_inv = T_world_to_sensor[:3, :3]
    t_inv = T_world_to_sensor[:3, 3:4]

    points_local_list = []
    for points_global in points_global_list:
        points_local = R_inv @ points_global + t_inv
        points_local_list.append(points_local)

    return points_local_list
