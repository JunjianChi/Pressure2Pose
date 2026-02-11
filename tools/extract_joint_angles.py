"""
Extract joint angles from SMPL parameters

Usage:
    # Extract angles from SMPL parameter file
    python tools/extract_joint_angles.py \
        --smpl_params data/smpl_params/walking1_smpl.pkl \
        --output output/joint_angles.csv \
        --format euler

    # Supported formats:
    #   - euler: Euler angles (roll, pitch, yaw)
    #   - axis_angle: axis-angle representation
    #   - degrees: rotation angle + rotation axis
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
import pickle
import argparse
from utils.smpl_utils import extract_lower_body_angles, format_angles_dict


def load_smpl_params(pkl_path):
    """Load all SMPL parameters from pickle file"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def extract_angles_sequence(smpl_params_list, output_format='euler'):
    """
    Extract joint angles for the entire sequence

    Args:
        smpl_params_list: List of SMPL parameter dicts
        output_format: 'euler' | 'axis_angle' | 'degrees'

    Returns:
        DataFrame with angles for each frame
    """
    all_angles = []

    for frame_idx, smpl_params in enumerate(smpl_params_list):
        # Extract angles
        angles = extract_lower_body_angles(smpl_params, output_format=output_format)

        # Format
        formatted = format_angles_dict(angles, format_type=output_format)

        # Flatten to single row
        row = {'Frame': smpl_params.get('frame', frame_idx)}

        if output_format == 'euler':
            # Euler angle format: 3 components per joint
            for joint_name, euler_dict in formatted.items():
                if isinstance(euler_dict, dict) and 'roll' in euler_dict:
                    row[f'{joint_name}_roll'] = euler_dict['roll']
                    row[f'{joint_name}_pitch'] = euler_dict['pitch']
                    row[f'{joint_name}_yaw'] = euler_dict['yaw']

        elif output_format == 'axis_angle':
            # Axis-angle format: 4 values per joint (angle + 3 axis components)
            for joint_name, aa_dict in formatted.items():
                if isinstance(aa_dict, dict) and 'angle' in aa_dict:
                    row[f'{joint_name}_angle'] = aa_dict['angle']
                    row[f'{joint_name}_axis_x'] = aa_dict['axis_x']
                    row[f'{joint_name}_axis_y'] = aa_dict['axis_y']
                    row[f'{joint_name}_axis_z'] = aa_dict['axis_z']

        elif output_format == 'degrees':
            # Same as axis_angle
            for joint_name, deg_dict in formatted.items():
                if isinstance(deg_dict, dict) and 'angle' in deg_dict:
                    row[f'{joint_name}_angle'] = deg_dict['angle']
                    row[f'{joint_name}_axis_x'] = deg_dict['axis_x']
                    row[f'{joint_name}_axis_y'] = deg_dict['axis_y']
                    row[f'{joint_name}_axis_z'] = deg_dict['axis_z']

        all_angles.append(row)

    df = pd.DataFrame(all_angles)
    return df


def main(args):
    """Main function"""
    print("=" * 60)
    print("Joint Angle Extraction Tool")
    print("=" * 60)

    # Load SMPL parameters
    print(f"\n📂 Loading SMPL parameters from: {args.smpl_params}")
    smpl_params_list = load_smpl_params(args.smpl_params)
    print(f"   Loaded {len(smpl_params_list)} frames")

    # Extract angles
    print(f"\n🔄 Extracting joint angles (format: {args.format})...")
    df_angles = extract_angles_sequence(smpl_params_list, output_format=args.format)

    print(f"   Extracted angles for {len(df_angles)} frames")
    print(f"   Columns: {len(df_angles.columns)}")

    # Show sample
    print(f"\n📊 Sample (first 3 frames):")
    print(df_angles.head(3).to_string())

    # Save to CSV
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_angles.to_csv(output_path, index=False)
    print(f"\n✅ Saved to: {output_path}")

    # Statistics
    print(f"\n📈 Statistics:")
    print(f"   Total frames: {len(df_angles)}")
    print(f"   Joints: left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle")

    if args.format == 'euler':
        print(f"   Each joint has 3 components: roll, pitch, yaw (degrees)")
        # Sample statistics for one joint
        print(f"\n   Example (left_knee):")
        print(f"     Roll:  [{df_angles['left_knee_roll'].min():.1f}, {df_angles['left_knee_roll'].max():.1f}] deg")
        print(f"     Pitch: [{df_angles['left_knee_pitch'].min():.1f}, {df_angles['left_knee_pitch'].max():.1f}] deg")
        print(f"     Yaw:   [{df_angles['left_knee_yaw'].min():.1f}, {df_angles['left_knee_yaw'].max():.1f}] deg")

    elif args.format in ['axis_angle', 'degrees']:
        print(f"   Each joint has 4 components: angle, axis_x, axis_y, axis_z")
        print(f"\n   Example (left_knee):")
        print(f"     Angle: [{df_angles['left_knee_angle'].min():.1f}, {df_angles['left_knee_angle'].max():.1f}] deg")

    print("\n" + "=" * 60)
    print("✅ Done!")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract joint angles from SMPL parameters')
    parser.add_argument('--smpl_params', type=str, required=True,
                       help='Path to SMPL parameters pickle file')
    parser.add_argument('--output', type=str, required=True,
                       help='Output CSV file path')
    parser.add_argument('--format', type=str, default='euler',
                       choices=['euler', 'axis_angle', 'degrees'],
                       help='Output format')

    args = parser.parse_args()
    main(args)
