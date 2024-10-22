import numpy as np
from scipy.spatial.transform import Rotation as R 

def compute_global_transform(xyz_cal, rpy_cal):

    rotation_matrix = R.from_euler('xyz', rpy_cal, degrees=False).as_matrix()
    
    calibration_matrix = np.eye(4)
    calibration_matrix[:3, :3] = rotation_matrix
    calibration_matrix[:3, 3] = xyz_cal

    return calibration_matrix

def homogeneous_transform(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def homogeneous_to_euler(T):
    R = T[:3, :3]
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    if np.abs(pitch - np.pi / 2) < 1e-6:
        # Gimbal lock case
        roll = 0
        yaw = np.arctan2(R[0, 1], R[1, 1])
    elif np.abs(pitch + np.pi / 2) < 1e-6:
        # Gimbal lock case
        roll = 0
        yaw = -np.arctan2(R[0, 1], R[1, 1])
    else:
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))

    euler_angles = np.array([roll, pitch, yaw])
    return euler_angles

def euler_to_homogeneous(roll_deg, pitch_deg, yaw_deg, translation):
    roll = np.radians(roll_deg)
    pitch = np.radians(pitch_deg)
    yaw = np.radians(yaw_deg)

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    
    R = np.array([[cos_yaw * cos_pitch, cos_yaw * sin_pitch * sin_roll - sin_yaw * cos_roll, cos_yaw * sin_pitch * cos_roll + sin_yaw * sin_roll],
                  [sin_yaw * cos_pitch, sin_yaw * sin_pitch * sin_roll + cos_yaw * cos_roll, sin_yaw * sin_pitch * cos_roll - cos_yaw * sin_roll],
                  [-sin_pitch, cos_pitch * sin_roll, cos_pitch * cos_roll]])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = translation

    return T

def mean_transform(T):
    
    euler_list = [homogeneous_to_euler(i) for i in T]
    translation_list = [i[:3,3] for i in T]

    euler_mean = np.mean(euler_list, axis=0)
    translation_mean = np.mean(translation_list, axis=0)
    
    return euler_to_homogeneous(np.rad2deg(euler_mean)[0], np.rad2deg(euler_mean)[1], np.rad2deg(euler_mean)[2], translation_mean)
