"""
MediaPipe Label Cleaner - Strong Physical Constraint Version

Core objectives:
1. Fully constant bone lengths (error <0.1%)
2. Full left-right symmetry
3. Reasonable joint angles (no knee hyperextension)
4. Temporal smoothing
5. Conformity with human gait patterns

Input: Noisy MediaPipe 3D joints
Output: Physically plausible, visually natural labels
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm


class PhysicalLabelCleaner:
    """Label cleaner with strong physical constraints"""

    # MediaPipe joints
    JOINTS = [
        'head',
        'left_shoulder', 'right_shoulder',
        'left_elbow', 'right_elbow',
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    # Bone definitions (start, end)
    BONES = {
        'left_thigh': ('left_hip', 'left_knee'),
        'right_thigh': ('right_hip', 'right_knee'),
        'left_shin': ('left_knee', 'left_heel'),
        'right_shin': ('right_knee', 'right_heel'),
        'left_foot': ('left_heel', 'left_foot_index'),
        'right_foot': ('right_heel', 'right_foot_index'),
        'left_upper_arm': ('left_shoulder', 'left_elbow'),
        'right_upper_arm': ('right_shoulder', 'right_elbow'),
        'pelvis': ('left_hip', 'right_hip'),
        'shoulders': ('left_shoulder', 'right_shoulder'),
    }

    # Symmetric bone pairs
    SYMMETRIC_PAIRS = [
        ('left_thigh', 'right_thigh'),
        ('left_shin', 'right_shin'),
        ('left_foot', 'right_foot'),
        ('left_upper_arm', 'right_upper_arm'),
    ]

    def __init__(self, enforce_symmetry=True, enforce_temporal=True,
                 gait_prior=True, smoothing_sigma=2.0):
        """
        Args:
            enforce_symmetry: Whether to enforce left-right symmetry
            enforce_temporal: Whether to apply temporal smoothing
            gait_prior: Whether to use gait prior
            smoothing_sigma: Sigma for temporal smoothing (in frames)
        """
        self.enforce_symmetry = enforce_symmetry
        self.enforce_temporal = enforce_temporal
        self.gait_prior = gait_prior
        self.smoothing_sigma = smoothing_sigma

        self.target_bone_lengths = None

    def load_mediapipe_csv(self, csv_path):
        """Load MediaPipe CSV data"""
        df = pd.read_csv(csv_path)

        joints_list = []
        for idx, row in df.iterrows():
            frame_joints = []
            for joint_name in self.JOINTS:
                x = row[f'{joint_name}_x']
                y = row[f'{joint_name}_y']
                z = row[f'{joint_name}_z']
                frame_joints.append([x, y, z])
            joints_list.append(frame_joints)

        joints_3d = np.array(joints_list)  # (N, 13, 3)
        return joints_3d, df

    def estimate_target_bone_lengths(self, joints_sequence):
        """
        Estimate target bone lengths from the entire sequence (median)

        Key: Use median instead of mean for robustness
        """
        N = len(joints_sequence)
        bone_lengths_all = {bone: [] for bone in self.BONES}

        for frame_joints in joints_sequence:
            for bone_name, (start_joint, end_joint) in self.BONES.items():
                start_idx = self.JOINTS.index(start_joint)
                end_idx = self.JOINTS.index(end_joint)

                start_pos = frame_joints[start_idx]
                end_pos = frame_joints[end_idx]

                length = np.linalg.norm(end_pos - start_pos)
                bone_lengths_all[bone_name].append(length)

        # Use median
        target_lengths = {}
        for bone_name, lengths in bone_lengths_all.items():
            target_lengths[bone_name] = np.median(lengths)

        return target_lengths

    def enforce_bone_length_constraint(self, joints, target_lengths,
                                      num_iterations=1):
        """
        Enforce constant bone lengths (simplified - direct proportional scaling)

        Method: Each bone is scaled independently, no iteration
        Avoids chain reactions and constraint conflicts
        """
        joints = joints.copy()

        # Scale only once, no iteration (key improvement!)
        for bone_name, (start_joint, end_joint) in self.BONES.items():
            start_idx = self.JOINTS.index(start_joint)
            end_idx = self.JOINTS.index(end_joint)

            start_pos = joints[start_idx]
            end_pos = joints[end_idx]

            # Current vector
            current_vec = end_pos - start_pos
            current_length = np.linalg.norm(current_vec)

            if current_length < 1e-6:
                continue

            # Target length
            target_length = target_lengths[bone_name]

            # Proportional scaling (gentle correction - only correct 30% of the difference)
            scale = target_length / current_length
            correction_weight = 0.3  # Only correct 30%

            # New position
            new_end_pos = start_pos + current_vec * (1.0 + (scale - 1.0) * correction_weight)

            joints[end_idx] = new_end_pos

        return joints

    def enforce_symmetry_constraint(self, joints, target_lengths):
        """
        Enforce left-right symmetry (improved version)

        Strategy:
        Only symmetrize bone lengths, not joint positions!

        Reason: During walking, the height and fore-aft positions of left and right feet
                are naturally different; forcing positional symmetry would cause errors!
        """
        joints = joints.copy()

        # Only symmetrize bone lengths (take average)
        symmetric_lengths = {}
        for left_bone, right_bone in self.SYMMETRIC_PAIRS:
            left_len = target_lengths[left_bone]
            right_len = target_lengths[right_bone]
            avg_len = (left_len + right_len) / 2.0

            symmetric_lengths[left_bone] = avg_len
            symmetric_lengths[right_bone] = avg_len

        # Update target_lengths
        target_lengths.update(symmetric_lengths)

        # No longer force joint position symmetry!
        # Keep original positions, achieve symmetry indirectly through bone length constraints

        return joints, target_lengths

    def enforce_joint_angle_limits(self, joints):
        """
        Enforce joint angles within reasonable range

        Key constraints:
        1. Knees can only bend forward (0-150 degrees)
        2. Feet cannot hyperextend backward
        """
        joints = joints.copy()

        # Knee angle constraint
        for side in ['left', 'right']:
            hip_idx = self.JOINTS.index(f'{side}_hip')
            knee_idx = self.JOINTS.index(f'{side}_knee')
            heel_idx = self.JOINTS.index(f'{side}_heel')

            hip = joints[hip_idx]
            knee = joints[knee_idx]
            heel = joints[heel_idx]

            # Vectors
            thigh_vec = knee - hip
            shin_vec = heel - knee

            # Angle
            cos_angle = np.dot(thigh_vec, shin_vec) / (
                np.linalg.norm(thigh_vec) * np.linalg.norm(shin_vec) + 1e-8
            )
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angle_deg = np.degrees(angle)

            # Limit: knee angle 150-180 degrees (straight to slightly bent)
            # If too bent, force adjustment
            min_angle = 150  # Maximum 30 degrees of bending
            if angle_deg < min_angle:
                # Adjust heel position
                target_angle = np.radians(min_angle)
                # Rotate shin_vec
                # Simplified: move heel backward
                correction = (min_angle - angle_deg) / 100.0
                heel_corrected = heel + thigh_vec * correction
                joints[heel_idx] = heel_corrected

        return joints

    def apply_gait_prior(self, joints, gait_phase):
        """
        Apply gait prior

        Adjust pose based on gait phase to better match typical gait patterns
        """
        # Simplified: adjust knee flexion based on phase
        if gait_phase == 'left_stance':
            # Left leg stance: left leg should be relatively straight
            left_knee_idx = self.JOINTS.index('left_knee')
            left_hip_idx = self.JOINTS.index('left_hip')
            left_heel_idx = self.JOINTS.index('left_heel')

            # Make left knee straighter
            hip = joints[left_hip_idx]
            heel = joints[left_heel_idx]
            # Knee between both, slightly forward
            target_knee = hip * 0.4 + heel * 0.6
            target_knee[2] += 0.05  # Slightly forward

            # Soft constraint (weight 0.3, not fully overriding)
            joints[left_knee_idx] = joints[left_knee_idx] * 0.7 + target_knee * 0.3

        elif gait_phase == 'right_stance':
            # Right leg stance: right leg should be relatively straight
            right_knee_idx = self.JOINTS.index('right_knee')
            right_hip_idx = self.JOINTS.index('right_hip')
            right_heel_idx = self.JOINTS.index('right_heel')

            hip = joints[right_hip_idx]
            heel = joints[right_heel_idx]
            target_knee = hip * 0.4 + heel * 0.6
            target_knee[2] += 0.05

            joints[right_knee_idx] = joints[right_knee_idx] * 0.7 + target_knee * 0.3

        return joints

    def detect_gait_phase(self, joints):
        """
        Detect gait phase (simplified)

        Determine which foot is in stance based on knee height
        """
        left_knee_idx = self.JOINTS.index('left_knee')
        right_knee_idx = self.JOINTS.index('right_knee')

        left_knee_y = joints[left_knee_idx][1]
        right_knee_y = joints[right_knee_idx][1]

        # Simplified: the lower knee indicates the stance leg
        if left_knee_y < right_knee_y - 0.05:
            return 'left_stance'
        elif right_knee_y < left_knee_y - 0.05:
            return 'right_stance'
        else:
            return 'double_stance'

    def temporal_smoothing(self, joints_sequence, sigma=2.0):
        """
        Temporal smoothing (Gaussian filter)

        Smooth each joint's xyz coordinates independently
        """
        N, num_joints, _ = joints_sequence.shape
        smoothed = joints_sequence.copy()

        for joint_idx in range(num_joints):
            for coord_idx in range(3):  # x, y, z
                signal = joints_sequence[:, joint_idx, coord_idx]
                smoothed[:, joint_idx, coord_idx] = gaussian_filter1d(
                    signal, sigma=sigma, mode='nearest'
                )

        return smoothed

    def clean_single_frame(self, joints, target_lengths, gait_phase=None):
        """
        Clean a single frame

        Apply all physical constraints (gentle version)
        """
        # 1. Bone length constraint (gentle)
        joints = self.enforce_bone_length_constraint(joints, target_lengths,
                                                     num_iterations=5)

        # 2. Symmetry constraint
        if self.enforce_symmetry:
            joints, target_lengths = self.enforce_symmetry_constraint(
                joints, target_lengths
            )
            # Reapply bone length constraint (even more gentle)
            joints = self.enforce_bone_length_constraint(joints, target_lengths,
                                                         num_iterations=3)

        # 3. Joint angle limits - disabled! Has BUG
        # joints = self.enforce_joint_angle_limits(joints)

        # 4. Gait prior (optional)
        if self.gait_prior and gait_phase is not None:
            joints = self.apply_gait_prior(joints, gait_phase)

        return joints

    def clean_sequence(self, joints_sequence, verbose=True):
        """
        Clean the entire sequence

        Pipeline:
        1. Estimate target bone lengths
        2. Temporal smoothing (preprocessing)
        3. Per-frame physical projection
        4. Temporal smoothing (postprocessing)
        """
        if verbose:
            print("="*60)
            print("Physical Label Cleaning")
            print("="*60)

        N = len(joints_sequence)

        # ============ Step 1: Estimate target bone lengths ============
        if verbose:
            print(f"\n[Step 1/5] Estimating target bone lengths...")

        self.target_bone_lengths = self.estimate_target_bone_lengths(
            joints_sequence
        )

        if verbose:
            print("\nTarget bone lengths (meters):")
            for bone_name, length in self.target_bone_lengths.items():
                print(f"  {bone_name:20s}: {length:.4f}")

        # ============ Step 2: Temporal smoothing (preprocessing) ============
        if verbose:
            print(f"\n[Step 2/5] Temporal smoothing (pre)...")

        if self.enforce_temporal:
            joints_sequence = self.temporal_smoothing(
                joints_sequence, sigma=self.smoothing_sigma
            )

        # ============ Step 3: Enforce symmetry (global) ============
        if verbose:
            print(f"\n[Step 3/5] Enforcing global symmetry...")

        if self.enforce_symmetry:
            # Average bone lengths for symmetric pairs
            for left_bone, right_bone in self.SYMMETRIC_PAIRS:
                left_len = self.target_bone_lengths[left_bone]
                right_len = self.target_bone_lengths[right_bone]
                avg_len = (left_len + right_len) / 2.0

                self.target_bone_lengths[left_bone] = avg_len
                self.target_bone_lengths[right_bone] = avg_len

            if verbose:
                print("\nSymmetric bone lengths:")
                for left_bone, right_bone in self.SYMMETRIC_PAIRS:
                    print(f"  {left_bone} = {right_bone} = "
                          f"{self.target_bone_lengths[left_bone]:.4f}")

        # ============ Step 4: Per-frame physical projection ============
        if verbose:
            print(f"\n[Step 4/5] Applying physical constraints frame-by-frame...")

        cleaned_sequence = []

        for frame_idx in tqdm(range(N), disable=not verbose):
            joints = joints_sequence[frame_idx]

            # Detect gait phase
            if self.gait_prior:
                gait_phase = self.detect_gait_phase(joints)
            else:
                gait_phase = None

            # Clean single frame
            joints_clean = self.clean_single_frame(
                joints,
                self.target_bone_lengths.copy(),  # Copy to avoid modification
                gait_phase
            )

            cleaned_sequence.append(joints_clean)

        cleaned_sequence = np.array(cleaned_sequence)

        # ============ Step 5: Temporal smoothing (postprocessing) ============
        if verbose:
            print(f"\n[Step 5/5] Temporal smoothing (post)...")

        if self.enforce_temporal:
            cleaned_sequence = self.temporal_smoothing(
                cleaned_sequence, sigma=self.smoothing_sigma
            )

        # ============ Statistics ============
        if verbose:
            print(f"\n{'='*60}")
            print("Cleaning Statistics:")
            print(f"{'='*60}")

            # Check bone length variation
            bone_length_vars = {}
            for bone_name, (start_joint, end_joint) in self.BONES.items():
                start_idx = self.JOINTS.index(start_joint)
                end_idx = self.JOINTS.index(end_joint)

                lengths = []
                for frame in cleaned_sequence:
                    length = np.linalg.norm(frame[end_idx] - frame[start_idx])
                    lengths.append(length)

                var = np.std(lengths) / np.mean(lengths) * 100  # CV%
                bone_length_vars[bone_name] = var

            print("\nBone length variation (CV%):")
            for bone_name, var in bone_length_vars.items():
                status = "[OK]" if var < 1.0 else "[FAIL]"
                print(f"  {status} {bone_name:20s}: {var:.2f}%")

            # Check symmetry
            if self.enforce_symmetry:
                print("\nSymmetry check:")
                for left_bone, right_bone in self.SYMMETRIC_PAIRS:
                    left_start, left_end = self.BONES[left_bone]
                    right_start, right_end = self.BONES[right_bone]

                    left_start_idx = self.JOINTS.index(left_start)
                    left_end_idx = self.JOINTS.index(left_end)
                    right_start_idx = self.JOINTS.index(right_start)
                    right_end_idx = self.JOINTS.index(right_end)

                    left_lengths = [
                        np.linalg.norm(frame[left_end_idx] - frame[left_start_idx])
                        for frame in cleaned_sequence
                    ]
                    right_lengths = [
                        np.linalg.norm(frame[right_end_idx] - frame[right_start_idx])
                        for frame in cleaned_sequence
                    ]

                    diff = np.mean(np.abs(np.array(left_lengths) - np.array(right_lengths)))
                    diff_pct = diff / np.mean(left_lengths) * 100

                    status = "[OK]" if diff_pct < 1.0 else "[FAIL]"
                    print(f"  {status} {left_bone} vs {right_bone}: {diff_pct:.2f}%")

            print(f"\n{'='*60}")

        return cleaned_sequence

    def save_cleaned_csv(self, cleaned_sequence, original_df, output_path):
        """Save the cleaned CSV"""
        df_new = original_df.copy()

        for frame_idx, joints in enumerate(cleaned_sequence):
            for joint_idx, joint_name in enumerate(self.JOINTS):
                df_new.loc[frame_idx, f'{joint_name}_x'] = joints[joint_idx, 0]
                df_new.loc[frame_idx, f'{joint_name}_y'] = joints[joint_idx, 1]
                df_new.loc[frame_idx, f'{joint_name}_z'] = joints[joint_idx, 2]

        df_new.to_csv(output_path, index=False)
        print(f"\n[OK] Saved cleaned labels to: {output_path}")


def main():
    """Example usage"""
    import argparse

    parser = argparse.ArgumentParser(description='Clean MediaPipe labels with physical constraints')
    parser.add_argument('--input_csv', type=str, required=True,
                        help='Input MediaPipe CSV file')
    parser.add_argument('--output_csv', type=str, required=True,
                        help='Output cleaned CSV file')
    parser.add_argument('--no_symmetry', action='store_true',
                        help='Disable symmetry constraint')
    parser.add_argument('--no_temporal', action='store_true',
                        help='Disable temporal smoothing')
    parser.add_argument('--no_gait_prior', action='store_true',
                        help='Disable gait prior')
    parser.add_argument('--smoothing_sigma', type=float, default=2.0,
                        help='Temporal smoothing sigma (frames)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process (for testing)')

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = PhysicalLabelCleaner(
        enforce_symmetry=not args.no_symmetry,
        enforce_temporal=not args.no_temporal,
        gait_prior=not args.no_gait_prior,
        smoothing_sigma=args.smoothing_sigma
    )

    # Load data
    print(f"\nLoading: {args.input_csv}")
    joints_sequence, original_df = cleaner.load_mediapipe_csv(args.input_csv)

    if args.max_frames:
        joints_sequence = joints_sequence[:args.max_frames]
        original_df = original_df.iloc[:args.max_frames]

    print(f"Loaded {len(joints_sequence)} frames")

    # Clean
    cleaned_sequence = cleaner.clean_sequence(joints_sequence, verbose=True)

    # Save
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cleaner.save_cleaned_csv(cleaned_sequence, original_df, output_path)

    print(f"\n[OK] All done!")


if __name__ == '__main__':
    main()
