"""
Data preprocessing script: Relative Normalization method

Uses pelvis-relative normalization to process all 13 joints
This is the best-performing preprocessing method in tests (Loss reduced by 98.9%)

Processing pipeline:
1. Load raw CSV data (13 joints)
2. Handle missing values (0 values -> NaN -> linear interpolation)
3. Remove outliers (Z-score threshold=3)
4. Convert to pelvis-relative coordinates
5. Normalize XYZ to [0,1]
6. Smoothing filter (moving average window=5)
7. Save as *_relative.csv

Usage:
    python host/preprocessing/clean_data_relative.py \
        --input data/walking1.csv \
        --output data/walking1_relative.csv
"""

import sys
# Fix Windows encoding
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import argparse


# All 13 joints
ALL_JOINTS = [
    'head',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_heel', 'right_heel',
    'left_foot_index', 'right_foot_index'
]


def load_data(csv_path):
    """Load CSV data"""
    print(f"\nLoading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")
    return df


def get_joint_columns(df, joint_names):
    """Get joint coordinate column names"""
    cols = []
    for joint in joint_names:
        cols.extend([f'{joint}_x', f'{joint}_y', f'{joint}_z'])
    return cols


def handle_missing_values(df, joint_cols):
    """
    Handle missing values (zero values)

    Method: 0 values -> NaN -> linear interpolation
    """
    print("\nHandling missing values...")
    df_filled = df.copy()

    missing_count = 0
    for col in joint_cols:
        # Mark zero values as NaN
        zero_mask = df_filled[col] == 0
        missing_count += zero_mask.sum()
        df_filled.loc[zero_mask, col] = np.nan

        # Linear interpolation
        df_filled[col] = df_filled[col].interpolate(method='linear', limit_direction='both')

        # If NaN still remains, fill with mean
        df_filled[col] = df_filled[col].fillna(df_filled[col].mean())

    print(f"   Missing values: {missing_count} -> filled")
    return df_filled


def remove_outliers(df, joint_cols, threshold=3):
    """
    Remove outliers using Z-score method

    Args:
        threshold: Z-score threshold, default 3 means 3 standard deviations
    """
    print(f"\nRemoving outliers (Z-score threshold={threshold})...")

    df_clean = df.copy()
    outlier_count = 0

    for col in joint_cols:
        # Compute Z-score
        z_scores = np.abs(stats.zscore(df_clean[col]))

        # Mark outliers as NaN
        outlier_mask = z_scores > threshold
        outlier_count += outlier_mask.sum()
        df_clean.loc[outlier_mask, col] = np.nan

        # Interpolation fill
        df_clean[col] = df_clean[col].interpolate(method='linear', limit_direction='both')
        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())

    print(f"   Outliers: {outlier_count} -> replaced")
    return df_clean


def relative_normalization(df, joint_names):
    """
    Relative Normalization method

    Steps:
    1. Compute pelvis center (midpoint of left and right hip joints)
    2. Convert all joints to pelvis-relative coordinates
    3. Normalize XYZ to [0,1]
    """
    print("\nApplying Relative Normalization...")

    df_norm = df.copy()

    # Extract 3D coordinates for all joints
    joints_3d = {}
    for joint in joint_names:
        joints_3d[joint] = df_norm[[f'{joint}_x', f'{joint}_y', f'{joint}_z']].values

    # Compute pelvis center (midpoint of left and right hip joints)
    left_hip = joints_3d['left_hip']
    right_hip = joints_3d['right_hip']
    pelvis = (left_hip + right_hip) / 2.0  # (N, 3)

    print(f"   Pelvis center range:")
    print(f"     X: [{pelvis[:, 0].min():.3f}, {pelvis[:, 0].max():.3f}]")
    print(f"     Y: [{pelvis[:, 1].min():.3f}, {pelvis[:, 1].max():.3f}]")
    print(f"     Z: [{pelvis[:, 2].min():.3f}, {pelvis[:, 2].max():.3f}]")

    # Convert to pelvis-relative coordinates
    for joint in joint_names:
        joints_3d[joint] = joints_3d[joint] - pelvis

    # Combine all joints for normalization (find global min/max)
    all_coords = np.vstack([joints_3d[j] for j in joint_names])  # (N*13, 3)

    # Normalize each axis
    for axis_idx, axis_name in enumerate(['X', 'Y', 'Z']):
        axis_min = all_coords[:, axis_idx].min()
        axis_max = all_coords[:, axis_idx].max()

        print(f"   {axis_name} axis: [{axis_min:.3f}, {axis_max:.3f}] -> [0, 1]")

        # Normalize this axis for all joints
        for joint in joint_names:
            joints_3d[joint][:, axis_idx] = (joints_3d[joint][:, axis_idx] - axis_min) / (axis_max - axis_min + 1e-8)

    # Write back to DataFrame
    for joint in joint_names:
        df_norm[f'{joint}_x'] = joints_3d[joint][:, 0]
        df_norm[f'{joint}_y'] = joints_3d[joint][:, 1]
        df_norm[f'{joint}_z'] = joints_3d[joint][:, 2]

    print("   Relative normalization complete")

    return df_norm


def smooth_data(df, joint_cols, window=5):
    """
    Smoothing filter (moving average)

    Args:
        window: Window size
    """
    print(f"\nSmoothing data (window={window})...")

    df_smooth = df.copy()

    for col in joint_cols:
        df_smooth[col] = df_smooth[col].rolling(window=window, center=True, min_periods=1).mean()

    print("   Smoothing complete")
    return df_smooth


def main(args):
    """Main function"""
    print("=" * 60)
    print("Relative Normalization Preprocessing")
    print("=" * 60)

    # Load data
    df = load_data(args.input)

    # Get joint columns
    joint_cols = get_joint_columns(df, ALL_JOINTS)

    # Display original data ranges
    print("\nOriginal data ranges:")
    x_cols = [c for c in joint_cols if c.endswith('_x')]
    y_cols = [c for c in joint_cols if c.endswith('_y')]
    z_cols = [c for c in joint_cols if c.endswith('_z')]

    print(f"   X: [{df[x_cols].values.min():.3f}, {df[x_cols].values.max():.3f}]")
    print(f"   Y: [{df[y_cols].values.min():.3f}, {df[y_cols].values.max():.3f}]")
    print(f"   Z: [{df[z_cols].values.min():.3f}, {df[z_cols].values.max():.3f}]")

    # 1. Handle missing values
    df = handle_missing_values(df, joint_cols)

    # 2. Remove outliers
    df = remove_outliers(df, joint_cols, threshold=args.outlier_threshold)

    # 3. Relative Normalization
    df = relative_normalization(df, ALL_JOINTS)

    # 4. Smoothing
    df = smooth_data(df, joint_cols, window=args.smooth_window)

    # Display processed data ranges
    print("\nProcessed data ranges:")
    print(f"   X: [{df[x_cols].values.min():.3f}, {df[x_cols].values.max():.3f}]")
    print(f"   Y: [{df[y_cols].values.min():.3f}, {df[y_cols].values.max():.3f}]")
    print(f"   Z: [{df[z_cols].values.min():.3f}, {df[z_cols].values.max():.3f}]")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Relative Normalization preprocessing')
    parser.add_argument('--input', type=str, required=True,
                       help='Input CSV file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file')
    parser.add_argument('--outlier_threshold', type=float, default=3.0,
                       help='Z-score threshold for outlier detection')
    parser.add_argument('--smooth_window', type=int, default=5,
                       help='Window size for smoothing')

    args = parser.parse_args()
    main(args)
