"""
SMPL utility functions: joint angle computation

Provides functions for extracting joint angles from SMPL parameters
"""

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def axis_angle_to_euler(axis_angle, convention='XYZ'):
    """
    Convert axis-angle representation to Euler angles

    Args:
        axis_angle: (3,) axis-angle vector
        convention: Euler angle order, e.g. 'XYZ', 'ZYX', etc.

    Returns:
        euler_angles: (3,) Euler angles (roll, pitch, yaw) in radians
    """
    # Use scipy's Rotation class
    rotation = Rotation.from_rotvec(axis_angle)
    euler = rotation.as_euler(convention.lower(), degrees=False)
    return euler


def axis_angle_to_degrees(axis_angle):
    """
    Convert axis-angle representation to angle in degrees (preserving axis direction)

    Args:
        axis_angle: (3,) axis-angle vector

    Returns:
        angle_degrees: Rotation angle (degrees)
        axis: (3,) normalized rotation axis
    """
    angle = np.linalg.norm(axis_angle)
    if angle < 1e-8:
        return 0.0, np.array([0, 0, 1])  # Default Z axis

    axis = axis_angle / angle
    angle_degrees = np.degrees(angle)

    return angle_degrees, axis


def extract_lower_body_angles(smpl_params, output_format='axis_angle'):
    """
    Extract lower body joint angles from SMPL parameters

    Args:
        smpl_params: dict with keys 'body_pose', 'global_orient', etc.
                    body_pose: (23, 3) or (69,) array
        output_format: 'axis_angle' | 'euler' | 'degrees'

    Returns:
        dict: {
            'left_hip': (3,),      # Hip joint angles
            'right_hip': (3,),
            'left_knee': (3,),     # Knee joint angles
            'right_knee': (3,),
            'left_ankle': (3,),    # Ankle joint angles
            'right_ankle': (3,),
            'global_orient': (3,)  # Global orientation
        }
    """
    body_pose = smpl_params['body_pose']

    # Ensure body_pose has shape (23, 3)
    if body_pose.ndim == 1:
        body_pose = body_pose.reshape(23, 3)

    # SMPL joint indices (from body_pose)
    # Note: body_pose[0] corresponds to SMPL joint 1 (left hip), excluding the root joint
    joint_indices = {
        'left_hip': 0,       # body_pose[0] = SMPL joint 1
        'right_hip': 1,      # body_pose[1] = SMPL joint 2
        'left_knee': 3,      # body_pose[3] = SMPL joint 4
        'right_knee': 4,     # body_pose[4] = SMPL joint 5
        'left_ankle': 6,     # body_pose[6] = SMPL joint 7
        'right_ankle': 7,    # body_pose[7] = SMPL joint 8
    }

    angles = {}

    for name, idx in joint_indices.items():
        axis_angle = body_pose[idx]  # (3,)

        if output_format == 'axis_angle':
            # Use axis-angle representation directly
            angles[name] = axis_angle

        elif output_format == 'euler':
            # Convert to Euler angles (XYZ order)
            euler = axis_angle_to_euler(axis_angle, convention='XYZ')
            angles[name] = euler

        elif output_format == 'degrees':
            # Convert to angle + rotation axis
            angle_deg, axis = axis_angle_to_degrees(axis_angle)
            angles[name] = {
                'angle': angle_deg,
                'axis': axis
            }

    # Add global orientation
    global_orient = smpl_params['global_orient']
    if output_format == 'axis_angle':
        angles['global_orient'] = global_orient
    elif output_format == 'euler':
        angles['global_orient'] = axis_angle_to_euler(global_orient, convention='XYZ')
    elif output_format == 'degrees':
        angle_deg, axis = axis_angle_to_degrees(global_orient)
        angles['global_orient'] = {
            'angle': angle_deg,
            'axis': axis
        }

    return angles


def compute_joint_angles_from_positions(joints_3d):
    """
    Compute joint angles from 3D joint positions (using vector angles)

    Args:
        joints_3d: (24, 3) or (45, 3) SMPL joint positions

    Returns:
        dict: Joint angles (degrees)
    """
    angles = {}

    # Lower body joint indices
    left_hip_idx = 1
    left_knee_idx = 4
    left_ankle_idx = 7
    left_foot_idx = 10

    right_hip_idx = 2
    right_knee_idx = 5
    right_ankle_idx = 8
    right_foot_idx = 11

    # Left knee angle (thigh-calf angle)
    left_thigh_vec = joints_3d[left_knee_idx] - joints_3d[left_hip_idx]
    left_calf_vec = joints_3d[left_ankle_idx] - joints_3d[left_knee_idx]
    left_knee_angle = compute_angle_between_vectors(left_thigh_vec, left_calf_vec)
    angles['left_knee_flexion'] = left_knee_angle

    # Right knee angle
    right_thigh_vec = joints_3d[right_knee_idx] - joints_3d[right_hip_idx]
    right_calf_vec = joints_3d[right_ankle_idx] - joints_3d[right_knee_idx]
    right_knee_angle = compute_angle_between_vectors(right_thigh_vec, right_calf_vec)
    angles['right_knee_flexion'] = right_knee_angle

    # Left ankle angle (calf-foot angle)
    left_foot_vec = joints_3d[left_foot_idx] - joints_3d[left_ankle_idx]
    left_ankle_angle = compute_angle_between_vectors(left_calf_vec, left_foot_vec)
    angles['left_ankle_flexion'] = left_ankle_angle

    # Right ankle angle
    right_foot_vec = joints_3d[right_foot_idx] - joints_3d[right_ankle_idx]
    right_ankle_angle = compute_angle_between_vectors(right_calf_vec, right_foot_vec)
    angles['right_ankle_flexion'] = right_ankle_angle

    return angles


def compute_angle_between_vectors(v1, v2):
    """
    Compute the angle between two vectors (in degrees)

    Args:
        v1, v2: (3,) vectors

    Returns:
        angle: Angle (degrees)
    """
    v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
    v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)

    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    angle_rad = np.arccos(cos_angle)
    angle_deg = np.degrees(angle_rad)

    return angle_deg


def format_angles_dict(angles_dict, format_type='euler'):
    """
    Format angles dictionary into a human-readable form

    Args:
        angles_dict: Output of extract_lower_body_angles
        format_type: 'euler' | 'axis_angle' | 'degrees'

    Returns:
        formatted_dict: Formatted dictionary
    """
    formatted = {}

    if format_type == 'euler':
        # Euler angles: convert to degrees and label as roll/pitch/yaw
        for joint_name, euler_rad in angles_dict.items():
            if isinstance(euler_rad, dict):  # degrees format
                formatted[joint_name] = euler_rad
            else:
                euler_deg = np.degrees(euler_rad)
                formatted[joint_name] = {
                    'roll': euler_deg[0],
                    'pitch': euler_deg[1],
                    'yaw': euler_deg[2]
                }

    elif format_type == 'axis_angle':
        # Axis-angle: convert to angle values
        for joint_name, axis_angle in angles_dict.items():
            if isinstance(axis_angle, dict):
                formatted[joint_name] = axis_angle
            else:
                angle_deg, axis = axis_angle_to_degrees(axis_angle)
                formatted[joint_name] = {
                    'angle': angle_deg,
                    'axis_x': axis[0],
                    'axis_y': axis[1],
                    'axis_z': axis[2]
                }

    else:
        formatted = angles_dict

    return formatted
