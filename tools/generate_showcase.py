"""
Generate showcase images: [Left Heatmap] | [Right Heatmap] | [3D SMPL Mesh]

Creates publication-quality figures at 300 DPI for the GitHub project page.

Usage:
    python tools/generate_showcase.py \
        --csv data/dataset/walking1.csv \
        --pkl data/smpl_params/walking1_physics.pkl \
        --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
        --output_dir output/showcase \
        --frames 0 30 60 90 120 150 180 210
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pickle
import torch
import smplx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import argparse


def parse_pressure(matrix_str, H=33, W=15):
    """Parse CSV pressure string to 2D array."""
    vals = np.array([float(x) for x in matrix_str.split(',')])
    target = H * W
    if len(vals) < target:
        vals = np.pad(vals, (0, target - len(vals)))
    elif len(vals) > target:
        vals = vals[:target]
    return vals.reshape(H, W)


def render_smpl_matplotlib(vertices, faces, ax, color=(0.7, 0.8, 0.95)):
    """Render SMPL mesh on a matplotlib 3D axes."""
    mesh = Poly3DCollection(
        vertices[faces],
        alpha=0.7,
        facecolor=color,
        edgecolor=(0.4, 0.4, 0.5),
        linewidths=0.1,
    )
    ax.add_collection3d(mesh)

    # Set axis limits based on mesh bounding box
    v_min = vertices.min(axis=0)
    v_max = vertices.max(axis=0)
    center = (v_min + v_max) / 2
    span = (v_max - v_min).max() * 0.6
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[2] - span, center[2] + span)  # Z -> depth
    ax.set_zlim(center[1] - span, center[1] + span)  # Y -> height

    ax.set_xlabel('X')
    ax.set_ylabel('Z')
    ax.set_zlabel('Y')
    ax.view_init(elev=10, azim=-70)


def generate_showcase(csv_path, pkl_path, smpl_model_path, output_dir,
                      frame_indices=None, gender='neutral', device='cpu',
                      pressure_h=33, pressure_w=15):
    """Generate showcase images for selected frames."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading pressure data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loading SMPL params from {pkl_path}...")
    with open(pkl_path, 'rb') as f:
        smpl_params_list = pickle.load(f)

    # Load SMPL model
    print(f"Loading SMPL model...")
    smpl = smplx.SMPL(model_path=smpl_model_path, gender=gender, batch_size=1).to(device)
    faces = smpl.faces.astype(np.int32)

    if frame_indices is None:
        n = len(smpl_params_list)
        frame_indices = np.linspace(0, n - 1, min(8, n), dtype=int).tolist()

    print(f"Generating {len(frame_indices)} showcase images...")

    for idx in frame_indices:
        if idx >= len(df) or idx >= len(smpl_params_list):
            print(f"  Skipping frame {idx} (out of range)")
            continue

        row = df.iloc[idx]
        sp = smpl_params_list[idx]

        # Parse pressure
        left_pressure = parse_pressure(row['Matrix_0'], pressure_h, pressure_w)
        right_pressure = parse_pressure(row['Matrix_1'], pressure_h, pressure_w)

        # Generate SMPL mesh
        with torch.no_grad():
            output = smpl(
                betas=torch.tensor(sp['betas'], dtype=torch.float32).unsqueeze(0).to(device),
                body_pose=torch.tensor(sp['body_pose'].flatten(), dtype=torch.float32).unsqueeze(0).to(device),
                global_orient=torch.tensor(sp['global_orient'], dtype=torch.float32).unsqueeze(0).to(device),
                transl=torch.tensor(sp['transl'], dtype=torch.float32).unsqueeze(0).to(device),
            )
        verts = output.vertices[0].cpu().numpy()

        # Create figure: [left heatmap] [right heatmap] [3D mesh]
        fig = plt.figure(figsize=(15, 5), dpi=300)

        # Left foot heatmap
        ax1 = fig.add_subplot(1, 3, 1)
        im1 = ax1.imshow(left_pressure, cmap='hot', interpolation='bilinear', aspect='auto')
        ax1.set_title('Left Foot Pressure', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Sensor Column')
        ax1.set_ylabel('Sensor Row')
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        # Right foot heatmap
        ax2 = fig.add_subplot(1, 3, 2)
        im2 = ax2.imshow(right_pressure, cmap='hot', interpolation='bilinear', aspect='auto')
        ax2.set_title('Right Foot Pressure', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Sensor Column')
        ax2.set_ylabel('Sensor Row')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        # 3D SMPL mesh
        ax3 = fig.add_subplot(1, 3, 3, projection='3d')
        render_smpl_matplotlib(verts, faces, ax3)
        ax3.set_title('3D SMPL Mesh', fontsize=12, fontweight='bold')

        fig.suptitle(f'Frame {idx}', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()

        save_path = output_dir / f'showcase_frame_{idx:04d}.png'
        fig.savefig(save_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        plt.close(fig)
        print(f"  Saved: {save_path}")

    print(f"Done! {len(frame_indices)} images saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Generate Pressure2Pose Showcase Images')
    parser.add_argument('--csv', type=str, required=True,
                        help='Pressure CSV file')
    parser.add_argument('--pkl', type=str, required=True,
                        help='SMPL params pickle file')
    parser.add_argument('--smpl_path', type=str, required=True,
                        help='Path to SMPL model directory')
    parser.add_argument('--output_dir', type=str, default='output/showcase',
                        help='Output directory')
    parser.add_argument('--frames', type=int, nargs='+', default=None,
                        help='Frame indices to visualize (default: 8 evenly spaced)')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['male', 'female', 'neutral'])
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--pressure_h', type=int, default=33)
    parser.add_argument('--pressure_w', type=int, default=15)
    args = parser.parse_args()

    generate_showcase(
        csv_path=args.csv,
        pkl_path=args.pkl,
        smpl_model_path=args.smpl_path,
        output_dir=args.output_dir,
        frame_indices=args.frames,
        gender=args.gender,
        device=args.device,
        pressure_h=args.pressure_h,
        pressure_w=args.pressure_w,
    )


if __name__ == '__main__':
    main()
