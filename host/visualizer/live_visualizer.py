"""
Live visualization for real-time pressure and pose display

Displays pressure heatmaps and 3D pose side-by-side in real-time
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import argparse


class LiveVisualizer:
    """Real-time visualization of pressure and pose"""

    def __init__(self, pressure_shape=(24, 20)):
        """
        Initialize visualizer

        Args:
            pressure_shape: Shape of pressure matrix (H, W)
        """
        self.pressure_shape = pressure_shape

        # Create figure with subplots
        self.fig = plt.figure(figsize=(15, 5))
        self.ax_left = self.fig.add_subplot(131)  # Left foot pressure
        self.ax_right = self.fig.add_subplot(132)  # Right foot pressure
        self.ax_pose = self.fig.add_subplot(133, projection='3d')  # 3D pose

        # Initialize plots
        self.im_left = None
        self.im_right = None
        self.scatter_joints = None

    def update_pressure(self, matrix_left, matrix_right):
        """
        Update pressure heatmaps

        Args:
            matrix_left: Left foot pressure (H, W)
            matrix_right: Right foot pressure (H, W)
        """
        # Left foot
        self.ax_left.clear()
        self.im_left = self.ax_left.imshow(matrix_left, cmap='hot', interpolation='nearest')
        self.ax_left.set_title('Left Foot Pressure')
        self.ax_left.axis('off')
        plt.colorbar(self.im_left, ax=self.ax_left, fraction=0.046)

        # Right foot
        self.ax_right.clear()
        self.im_right = self.ax_right.imshow(matrix_right, cmap='hot', interpolation='nearest')
        self.ax_right.set_title('Right Foot Pressure')
        self.ax_right.axis('off')
        plt.colorbar(self.im_right, ax=self.ax_right, fraction=0.046)

    def update_pose(self, joints):
        """
        Update 3D pose visualization

        Args:
            joints: Joint positions (24, 3) or (N, 3)
        """
        # Skeleton connections
        connections = [
            (0, 1), (0, 2),  # Pelvis to hips
            (1, 4), (2, 5),  # Hips to knees
            (4, 7), (5, 8),  # Knees to ankles
            (7, 10), (8, 11),  # Ankles to feet
        ]

        self.ax_pose.clear()

        # Plot joints
        self.scatter_joints = self.ax_pose.scatter(
            joints[:, 0], joints[:, 1], joints[:, 2],
            c='red', s=50, marker='o'
        )

        # Plot skeleton
        for i, j in connections:
            if i < len(joints) and j < len(joints):
                self.ax_pose.plot(
                    [joints[i, 0], joints[j, 0]],
                    [joints[i, 1], joints[j, 1]],
                    [joints[i, 2], joints[j, 2]],
                    'r-', linewidth=2
                )

        # Set labels and limits
        self.ax_pose.set_xlabel('X')
        self.ax_pose.set_ylabel('Y')
        self.ax_pose.set_zlabel('Z')
        self.ax_pose.set_title('Predicted 3D Pose')

        # Equal aspect ratio
        max_range = np.array([
            joints[:, 0].max() - joints[:, 0].min(),
            joints[:, 1].max() - joints[:, 1].min(),
            joints[:, 2].max() - joints[:, 2].min()
        ]).max() / 2.0

        mid_x = (joints[:, 0].max() + joints[:, 0].min()) * 0.5
        mid_y = (joints[:, 1].max() + joints[:, 1].min()) * 0.5
        mid_z = (joints[:, 2].max() + joints[:, 2].min()) * 0.5

        self.ax_pose.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax_pose.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax_pose.set_zlim(mid_z - max_range, mid_z + max_range)

    def update(self, matrix_left, matrix_right, joints):
        """
        Update all visualizations

        Args:
            matrix_left: Left foot pressure
            matrix_right: Right foot pressure
            joints: 3D joint positions
        """
        self.update_pressure(matrix_left, matrix_right)
        self.update_pose(joints)
        plt.tight_layout()
        plt.pause(0.001)

    def show(self):
        """Display the visualization"""
        plt.show()


def demo_live_visualization(csv_path, predictions_path):
    """
    Demo function to show live visualization from saved data

    Args:
        csv_path: Path to CSV with pressure data
        predictions_path: Path to saved predictions (.npy)
    """
    import pandas as pd

    # Load data
    df = pd.read_csv(csv_path)
    predictions = np.load(predictions_path, allow_pickle=True)

    # Create visualizer
    visualizer = LiveVisualizer()

    # Animation function
    def animate(frame_idx):
        if frame_idx >= len(predictions):
            return

        # Get pressure data
        row = df.iloc[predictions[frame_idx]['frame']]
        matrix_0 = parse_pressure_matrix(row['Matrix_0']).reshape(24, 20)
        matrix_1 = parse_pressure_matrix(row['Matrix_1']).reshape(24, 20)

        # Get predicted joints
        joints = predictions[frame_idx]['joints']

        # Update visualization
        visualizer.update(matrix_0, matrix_1, joints)

    # Create animation
    anim = FuncAnimation(
        visualizer.fig, animate,
        frames=len(predictions),
        interval=33,  # ~30 FPS
        repeat=True
    )

    plt.show()


def parse_pressure_matrix(matrix_str):
    """Parse comma-separated pressure values"""
    values = np.array([float(x) for x in matrix_str.split(',')])
    return values


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Live visualization')
    parser.add_argument('--csv', type=str, required=True,
                        help='Input CSV file')
    parser.add_argument('--predictions', type=str, required=True,
                        help='Predictions file (.npy)')

    args = parser.parse_args()

    demo_live_visualization(args.csv, args.predictions)
