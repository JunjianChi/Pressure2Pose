"""
Evaluation metrics for pose estimation
"""

import torch
import numpy as np
from typing import Union


def compute_mpjpe(pred_joints: Union[torch.Tensor, np.ndarray],
                  gt_joints: Union[torch.Tensor, np.ndarray],
                  mask: Union[torch.Tensor, np.ndarray] = None) -> float:
    """
    Compute Mean Per Joint Position Error (MPJPE)

    Args:
        pred_joints: Predicted joints (N, J, 3) or (J, 3)
        gt_joints: Ground truth joints (N, J, 3) or (J, 3)
        mask: Optional mask for valid joints (N, J) or (J,)

    Returns:
        MPJPE in millimeters
    """
    if torch.is_tensor(pred_joints):
        pred_joints = pred_joints.detach().cpu().numpy()
    if torch.is_tensor(gt_joints):
        gt_joints = gt_joints.detach().cpu().numpy()
    if mask is not None and torch.is_tensor(mask):
        mask = mask.detach().cpu().numpy()

    # Compute L2 distance
    error = np.sqrt(np.sum((pred_joints - gt_joints) ** 2, axis=-1))  # (N, J) or (J,)

    # Apply mask if provided
    if mask is not None:
        error = error[mask > 0]

    # Convert to millimeters and compute mean
    mpjpe = np.mean(error) * 1000.0

    return mpjpe


def compute_pa_mpjpe(pred_joints: Union[torch.Tensor, np.ndarray],
                     gt_joints: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute Procrustes-Aligned Mean Per Joint Position Error (PA-MPJPE)
    Removes global alignment errors (rotation, translation, scale)

    Args:
        pred_joints: Predicted joints (N, J, 3) or (J, 3)
        gt_joints: Ground truth joints (N, J, 3) or (J, 3)

    Returns:
        PA-MPJPE in millimeters
    """
    if torch.is_tensor(pred_joints):
        pred_joints = pred_joints.detach().cpu().numpy()
    if torch.is_tensor(gt_joints):
        gt_joints = gt_joints.detach().cpu().numpy()

    # Handle single sample
    single_sample = False
    if pred_joints.ndim == 2:
        pred_joints = pred_joints[None]
        gt_joints = gt_joints[None]
        single_sample = True

    batch_size = pred_joints.shape[0]
    errors = []

    for i in range(batch_size):
        pred = pred_joints[i]  # (J, 3)
        gt = gt_joints[i]

        # Procrustes alignment
        aligned_pred = procrustes_align(pred, gt)

        # Compute error
        error = np.sqrt(np.sum((aligned_pred - gt) ** 2, axis=-1))  # (J,)
        errors.append(np.mean(error))

    pa_mpjpe = np.mean(errors) * 1000.0  # Convert to mm

    return pa_mpjpe


def procrustes_align(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Align prediction to target using Procrustes analysis

    Args:
        pred: Predicted points (J, 3)
        target: Target points (J, 3)

    Returns:
        Aligned prediction (J, 3)
    """
    # Center
    pred_centered = pred - np.mean(pred, axis=0)
    target_centered = target - np.mean(target, axis=0)

    # Scale
    pred_scale = np.sqrt(np.sum(pred_centered ** 2))
    target_scale = np.sqrt(np.sum(target_centered ** 2))

    pred_normalized = pred_centered / pred_scale
    target_normalized = target_centered / target_scale

    # Rotation (SVD)
    H = pred_normalized.T @ target_normalized
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Apply alignment
    aligned = (target_scale / pred_scale) * pred_centered @ R.T + np.mean(target, axis=0)

    return aligned


def compute_vertex_error(pred_vertices: Union[torch.Tensor, np.ndarray],
                        gt_vertices: Union[torch.Tensor, np.ndarray]) -> float:
    """
    Compute mean vertex position error

    Args:
        pred_vertices: Predicted vertices (N, V, 3) or (V, 3)
        gt_vertices: Ground truth vertices (N, V, 3) or (V, 3)

    Returns:
        Mean vertex error in millimeters
    """
    if torch.is_tensor(pred_vertices):
        pred_vertices = pred_vertices.detach().cpu().numpy()
    if torch.is_tensor(gt_vertices):
        gt_vertices = gt_vertices.detach().cpu().numpy()

    error = np.sqrt(np.sum((pred_vertices - gt_vertices) ** 2, axis=-1))
    return np.mean(error) * 1000.0


def compute_bone_length_error(pred_joints: Union[torch.Tensor, np.ndarray],
                              gt_joints: Union[torch.Tensor, np.ndarray],
                              bone_pairs: list = None) -> float:
    """
    Compute bone length consistency error

    Args:
        pred_joints: Predicted joints (N, J, 3) or (J, 3)
        gt_joints: Ground truth joints (N, J, 3) or (J, 3)
        bone_pairs: List of (joint_i, joint_j) pairs

    Returns:
        Mean bone length error in millimeters
    """
    if bone_pairs is None:
        # Default lower body bones
        bone_pairs = [
            (1, 4),  # Left hip to left knee
            (2, 5),  # Right hip to right knee
            (4, 7),  # Left knee to left ankle
            (5, 8),  # Right knee to right ankle
        ]

    if torch.is_tensor(pred_joints):
        pred_joints = pred_joints.detach().cpu().numpy()
    if torch.is_tensor(gt_joints):
        gt_joints = gt_joints.detach().cpu().numpy()

    errors = []
    for i, j in bone_pairs:
        pred_length = np.linalg.norm(pred_joints[..., i, :] - pred_joints[..., j, :], axis=-1)
        gt_length = np.linalg.norm(gt_joints[..., i, :] - gt_joints[..., j, :], axis=-1)
        errors.append(np.abs(pred_length - gt_length))

    return np.mean(errors) * 1000.0
