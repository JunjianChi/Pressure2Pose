"""
Coordinate system transformation utilities

MediaPipe camera coordinate system -> SMPL standard coordinate system
"""

import numpy as np


def mediapipe_to_smpl_coords(joints_3d):
    """
    Convert MediaPipe camera coordinate system to SMPL standard coordinate system

    MediaPipe coordinate system:
        X: left -> right (right is +)
        Y: up -> down (down is +)  <- opposite to SMPL
        Z: near -> far (far is +)  <- depth distance

    SMPL standard coordinate system:
        X: left -> right (right is +)
        Y: down -> up (up is +)    <- standard up direction
        Z: back -> front (front is +) <- standard forward direction

    Transformation rules:
        X_smpl = X_mediapipe  (unchanged)
        Y_smpl = -Y_mediapipe (flipped, so up is positive)
        Z_smpl = -Z_mediapipe (flipped, so forward is positive)

    Args:
        joints_3d: (N, 3) or (B, N, 3) MediaPipe coordinates

    Returns:
        joints_smpl: Joint positions in SMPL coordinate system
    """
    joints_smpl = joints_3d.copy()

    if joints_3d.ndim == 2:
        # (N, 3)
        joints_smpl[:, 1] = -joints_smpl[:, 1]  # Flip Y
        joints_smpl[:, 2] = -joints_smpl[:, 2]  # Flip Z
    elif joints_3d.ndim == 3:
        # (B, N, 3)
        joints_smpl[:, :, 1] = -joints_smpl[:, :, 1]  # Flip Y
        joints_smpl[:, :, 2] = -joints_smpl[:, :, 2]  # Flip Z

    return joints_smpl


def smpl_to_mediapipe_coords(joints_3d):
    """
    Convert SMPL standard coordinate system back to MediaPipe camera coordinate system
    (used for visual comparison)

    Args:
        joints_3d: (N, 3) or (B, N, 3) SMPL coordinates

    Returns:
        joints_mediapipe: Joint positions in MediaPipe coordinate system
    """
    # The transformation is reversible (flipping twice returns to the original)
    return mediapipe_to_smpl_coords(joints_3d)


def get_standard_views():
    """
    Get standard viewing angles for SMPL human body

    Returns:
        dict: {view_name: (elev, azim, roll)}
    """
    views = {
        'front': (10, -90, 0),      # Front: slightly from above, from the front
        'back': (10, 90, 0),         # Back
        'left': (10, 0, 0),          # Left side
        'right': (10, 180, 0),       # Right side
        'top': (90, -90, 0),         # Top-down view
        'oblique': (20, -60, 0),     # Oblique view (commonly used)
    }
    return views


def rotation_matrix_y(angle_deg):
    """
    Rotation matrix around the Y axis (used to adjust body orientation)

    Args:
        angle_deg: Rotation angle (degrees)

    Returns:
        R: (3, 3) rotation matrix
    """
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)

    R = np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])

    return R


def rotate_joints_y(joints_3d, angle_deg):
    """
    Rotate joints around the Y axis (used to adjust body orientation)

    Args:
        joints_3d: (N, 3) joint coordinates
        angle_deg: Rotation angle

    Returns:
        joints_rotated: Rotated joints
    """
    R = rotation_matrix_y(angle_deg)

    if joints_3d.ndim == 2:
        # (N, 3)
        return joints_3d @ R.T
    elif joints_3d.ndim == 3:
        # (B, N, 3)
        return np.einsum('bnc,dc->bnd', joints_3d, R)


def normalize_pose_orientation(joints_3d, hip_left_idx=5, hip_right_idx=6):
    """
    Normalize pose orientation to face the Z+ direction

    Computes the hip line direction and rotates the body to face Z+

    Args:
        joints_3d: (N, 3) joint coordinates
        hip_left_idx: Left hip joint index
        hip_right_idx: Right hip joint index

    Returns:
        joints_normalized: Normalized joints
        rotation_angle: Applied rotation angle
    """
    # Compute the hip line direction
    hip_left = joints_3d[hip_left_idx]
    hip_right = joints_3d[hip_right_idx]
    hip_vec = hip_right - hip_left  # Left -> right vector

    # Compute current orientation angle (in XZ plane)
    current_angle = np.arctan2(hip_vec[2], hip_vec[0])  # atan2(z, x)

    # Target: hip line parallel to the X axis (i.e., body facing Z+)
    target_angle = 0  # X axis direction

    # Required rotation angle
    rotation_angle = np.degrees(target_angle - current_angle)

    # Apply rotation
    joints_normalized = rotate_joints_y(joints_3d, rotation_angle)

    return joints_normalized, rotation_angle


def visualize_coordinate_systems():
    """
    Visualize the differences between MediaPipe and SMPL coordinate systems
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(12, 5))

    # MediaPipe coordinate system
    ax1 = fig.add_subplot(121, projection='3d')
    origin = np.array([0, 0, 0])

    # Draw coordinate axes
    ax1.quiver(*origin, 1, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=2)
    ax1.text(1.2, 0, 0, 'X (right)', color='r', fontsize=12)

    ax1.quiver(*origin, 0, 1, 0, color='g', arrow_length_ratio=0.2, linewidth=2)
    ax1.text(0, 1.2, 0, 'Y (down)', color='g', fontsize=12)

    ax1.quiver(*origin, 0, 0, 1, color='b', arrow_length_ratio=0.2, linewidth=2)
    ax1.text(0, 0, 1.2, 'Z (away)', color='b', fontsize=12)

    ax1.set_xlim([-0.5, 1.5])
    ax1.set_ylim([-0.5, 1.5])
    ax1.set_zlim([-0.5, 1.5])
    ax1.set_title('MediaPipe Camera Coords', fontsize=14, fontweight='bold')

    # SMPL coordinate system
    ax2 = fig.add_subplot(122, projection='3d')

    ax2.quiver(*origin, 1, 0, 0, color='r', arrow_length_ratio=0.2, linewidth=2)
    ax2.text(1.2, 0, 0, 'X (right)', color='r', fontsize=12)

    ax2.quiver(*origin, 0, 1, 0, color='g', arrow_length_ratio=0.2, linewidth=2)
    ax2.text(0, 1.2, 0, 'Y (up)', color='g', fontsize=12)

    ax2.quiver(*origin, 0, 0, 1, color='b', arrow_length_ratio=0.2, linewidth=2)
    ax2.text(0, 0, 1.2, 'Z (forward)', color='b', fontsize=12)

    ax2.set_xlim([-0.5, 1.5])
    ax2.set_ylim([-0.5, 1.5])
    ax2.set_zlim([-0.5, 1.5])
    ax2.set_title('SMPL Standard Coords', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig('coordinate_systems.png', dpi=150)
    print("Saved coordinate systems visualization")


if __name__ == '__main__':
    # Test coordinate transformation
    print("MediaPipe → SMPL coordinate transformation test")
    print("=" * 60)

    # Test point: a person standing in MediaPipe coordinate system
    # Assume hip at (0.5, 0.3, 5.0) - small Y = above, large Z = away from camera
    hip_mediapipe = np.array([0.5, 0.3, 5.0])
    print(f"MediaPipe coords: {hip_mediapipe}")

    # Convert to SMPL
    hip_smpl = mediapipe_to_smpl_coords(hip_mediapipe.reshape(1, 3))[0]
    print(f"SMPL coords:      {hip_smpl}")
    print(f"  X: {hip_smpl[0]:.3f} (unchanged)")
    print(f"  Y: {hip_smpl[1]:.3f} (flipped: {-0.3:.3f})")
    print(f"  Z: {hip_smpl[2]:.3f} (flipped: {-5.0:.3f})")

    # Visualize coordinate systems
    visualize_coordinate_systems()
