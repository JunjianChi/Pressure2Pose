"""
Physics-aware SMPL fitting from MediaPipe joints

Core philosophy: SMPL is our physics regularizer.
- Lower loss does not necessarily mean better, because raw data is noisy
- We leverage SMPL's parameterized structure for denoising
- Strong pose prior + joint angle limits > low data loss

Strategy:
  Phase 0: Compute global scale factor between MediaPipe and SMPL
  Phase 1: Estimate a fixed betas (body shape) from bone lengths across all frames
  Phase 2: Fix betas, fit only pose per frame with strong prior and joint limits
  Phase 3: Temporal smoothing of pose parameters
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import smplx


class PhysicsAwareSMPLFitter:
    """
    Physics-constrained SMPL fitting.
    Uses SMPL as a denoising tool, NOT just a shape representation.
    """

    # Only use lower body joints (upper body MediaPipe data is not accurate enough)
    MEDIAPIPE_JOINTS = [
        'left_hip', 'right_hip',
        'left_knee', 'right_knee',
        'left_heel', 'right_heel',
        'left_foot_index', 'right_foot_index'
    ]

    # MediaPipe -> SMPL joint mapping
    JOINT_MAPPING = {
        'left_hip': 1,
        'right_hip': 2,
        'left_knee': 4,
        'right_knee': 5,
        'left_heel': 7,       # heel -> ankle
        'right_heel': 8,
        'left_foot_index': 10,
        'right_foot_index': 11,
    }

    # Confidence weight per joint (hip most reliable, foot least reliable)
    JOINT_WEIGHTS = {
        'left_hip': 1.0,
        'right_hip': 1.0,
        'left_knee': 0.8,
        'right_knee': 0.8,
        'left_heel': 0.5,
        'right_heel': 0.5,
        'left_foot_index': 0.3,
        'right_foot_index': 0.3,
    }

    # SMPL body_pose joint indices (excluding root)
    LOWER_BODY_POSE_INDICES = [
        0,   # left_hip
        1,   # right_hip
        3,   # left_knee
        4,   # right_knee
        6,   # left_ankle
        7,   # right_ankle
        9,   # left_foot
        10,  # right_foot
    ]

    # Joint angle limits (axis-angle, radians)
    JOINT_ANGLE_LIMITS = {
        3: {'x': (0.0, 2.6), 'y': (-0.1, 0.1), 'z': (-0.1, 0.1)},   # left knee
        4: {'x': (0.0, 2.6), 'y': (-0.1, 0.1), 'z': (-0.1, 0.1)},   # right knee
        0: {'x': (-1.5, 1.5), 'y': (-0.8, 0.8), 'z': (-0.5, 0.5)},  # left hip
        1: {'x': (-1.5, 1.5), 'y': (-0.8, 0.8), 'z': (-0.5, 0.5)},  # right hip
        6: {'x': (-0.5, 0.8), 'y': (-0.3, 0.3), 'z': (-0.3, 0.3)},  # left ankle
        7: {'x': (-0.5, 0.8), 'y': (-0.3, 0.3), 'z': (-0.3, 0.3)},  # right ankle
        9:  {'x': (-0.3, 0.5), 'y': (-0.2, 0.2), 'z': (-0.1, 0.1)}, # left foot
        10: {'x': (-0.3, 0.5), 'y': (-0.2, 0.2), 'z': (-0.1, 0.1)}, # right foot
    }

    # SMPL bone connections (used for bone length computation)
    SMPL_BONES = [
        (1, 4),   # left thigh
        (2, 5),   # right thigh
        (4, 7),   # left shin
        (5, 8),   # right shin
    ]

    # MediaPipe bone connections (corresponding to SMPL_BONES order)
    MP_BONES = [
        (0, 2),   # left thigh (left_hip -> left_knee)
        (1, 3),   # right thigh (right_hip -> right_knee)
        (2, 4),   # left shin (left_knee -> left_heel)
        (3, 5),   # right shin (right_knee -> right_heel)
    ]

    def __init__(self, smpl_model_path, gender='neutral', device='cuda'):
        self.device = device
        self.smpl_model = smplx.SMPL(
            model_path=smpl_model_path,
            gender=gender,
            batch_size=1
        ).to(device)

        self.num_betas = 10
        self.num_joints = 23
        self.global_scale = 1.0  # MediaPipe -> SMPL scale

        print(f"[OK] SMPL model loaded ({gender}) on {device}")

    def load_csv_data(self, csv_path):
        """Load lower body joints from CSV"""
        df = pd.read_csv(csv_path)
        joints_list = []
        for idx in range(len(df)):
            row = df.iloc[idx]
            frame_joints = []
            for joint_name in self.MEDIAPIPE_JOINTS:
                x = row[f'{joint_name}_x']
                y = row[f'{joint_name}_y']
                z = row[f'{joint_name}_z']
                frame_joints.append([x, y, z])
            joints_list.append(frame_joints)

        joints_3d = np.array(joints_list)
        timestamps = df['Timestamp'].values
        frames = df['Frame'].values
        return joints_3d, timestamps, frames

    # =========================================================
    # Phase 0: Compute scale factor
    # =========================================================

    def compute_scale_factor(self, joints_3d_all):
        """
        Compute the scale factor between MediaPipe coordinates and SMPL coordinates

        Uses SMPL default pose bone lengths / MediaPipe median bone lengths
        """
        print("\n--- Phase 0: Computing scale factor ---")

        # SMPL default bone lengths (neutral pose, zero betas)
        with torch.no_grad():
            output = self.smpl_model(
                betas=torch.zeros(1, self.num_betas, device=self.device),
                body_pose=torch.zeros(1, self.num_joints * 3, device=self.device),
                global_orient=torch.zeros(1, 3, device=self.device),
                transl=torch.zeros(1, 3, device=self.device)
            )
            smpl_joints = output.joints[0].cpu().numpy()

        smpl_bone_lengths = []
        mp_bone_lengths = []

        for (smpl_s, smpl_e), (mp_s, mp_e) in zip(self.SMPL_BONES, self.MP_BONES):
            # SMPL bone length
            smpl_len = np.linalg.norm(smpl_joints[smpl_e] - smpl_joints[smpl_s])
            smpl_bone_lengths.append(smpl_len)

            # MediaPipe median bone length
            mp_lens = np.linalg.norm(
                joints_3d_all[:, mp_e] - joints_3d_all[:, mp_s], axis=1
            )
            mp_len = np.median(mp_lens)
            mp_bone_lengths.append(mp_len)

            print(f"  SMPL({smpl_s}->{smpl_e}): {smpl_len:.4f}m, "
                  f"MP({mp_s}->{mp_e}): {mp_len:.4f}m, "
                  f"ratio: {smpl_len/mp_len:.3f}")

        # Global scale = SMPL / MediaPipe (average ratio)
        ratios = np.array(smpl_bone_lengths) / np.array(mp_bone_lengths)
        self.global_scale = np.median(ratios)

        print(f"\n  Global scale factor: {self.global_scale:.4f}")
        print(f"  (MediaPipe * {self.global_scale:.3f} = SMPL scale)")

        return self.global_scale

    # =========================================================
    # Phase 1: Estimate body shape (betas) from bone lengths
    # =========================================================

    def estimate_betas_from_bones(self, joints_3d_all, num_iterations=1000):
        """
        Estimate a stable betas from bone lengths across all frames

        Note: MediaPipe bone lengths are already scaled by global_scale
        """
        print("\n--- Phase 1: Estimating body shape (betas) ---")

        # Compute scaled MediaPipe bone lengths
        bone_pairs_full = [
            (0, 2, 1, 4, 'left_thigh'),
            (1, 3, 2, 5, 'right_thigh'),
            (2, 4, 4, 7, 'left_shin'),
            (3, 5, 5, 8, 'right_shin'),
            (4, 6, 7, 10, 'left_foot'),
            (5, 7, 8, 11, 'right_foot'),
        ]

        target_bone_lengths = {}
        for mp_s, mp_e, smpl_s, smpl_e, name in bone_pairs_full:
            mp_lens = np.linalg.norm(
                joints_3d_all[:, mp_e] - joints_3d_all[:, mp_s], axis=1
            )
            # Scale to SMPL dimensions
            target_len = np.median(mp_lens) * self.global_scale
            target_bone_lengths[(smpl_s, smpl_e)] = target_len
            print(f"  {name}: target={target_len:.4f}m (MP median scaled)")

        # Optimize betas
        betas = torch.zeros(1, self.num_betas, requires_grad=True, device=self.device)
        optimizer = torch.optim.Adam([betas], lr=0.05)

        for i in range(num_iterations):
            optimizer.zero_grad()

            output = self.smpl_model(
                betas=betas,
                body_pose=torch.zeros(1, self.num_joints * 3, device=self.device),
                global_orient=torch.zeros(1, 3, device=self.device),
                transl=torch.zeros(1, 3, device=self.device)
            )
            pred_joints = output.joints[0]

            loss_bones = torch.tensor(0.0, device=self.device)
            for (start_idx, end_idx), target_len in target_bone_lengths.items():
                pred_len = torch.norm(pred_joints[end_idx] - pred_joints[start_idx])
                loss_bones = loss_bones + (pred_len - target_len) ** 2

            loss_reg = 0.01 * torch.sum(betas ** 2)
            loss = loss_bones + loss_reg
            loss.backward()
            optimizer.step()

            if i % 200 == 0:
                print(f"  Iter {i}: bone_loss={loss_bones.item():.6f}, "
                      f"betas_norm={torch.norm(betas).item():.3f}")

        # Verify
        with torch.no_grad():
            output = self.smpl_model(
                betas=betas,
                body_pose=torch.zeros(1, self.num_joints * 3, device=self.device),
                global_orient=torch.zeros(1, 3, device=self.device),
                transl=torch.zeros(1, 3, device=self.device)
            )
            pred_joints = output.joints[0]

            print("\n  Bone length verification:")
            for (start_idx, end_idx), target_len in target_bone_lengths.items():
                pred_len = torch.norm(pred_joints[end_idx] - pred_joints[start_idx]).item()
                err_pct = abs(pred_len - target_len) / target_len * 100
                status = "[OK]" if err_pct < 5 else "[!]"
                print(f"    {status} SMPL({start_idx}->{end_idx}): {pred_len:.4f}m "
                      f"vs target {target_len:.4f}m ({err_pct:.1f}%)")

        final_betas = betas.detach().clone()
        print(f"\n  [OK] Betas estimated (norm={torch.norm(final_betas).item():.3f})")
        return final_betas

    # =========================================================
    # Phase 2: Fit pose per frame
    # =========================================================

    def joint_angle_limit_loss(self, body_pose_full):
        """Joint angle limit penalty"""
        loss = torch.tensor(0.0, device=self.device)
        for joint_idx, limits in self.JOINT_ANGLE_LIMITS.items():
            joint_aa = body_pose_full[0, joint_idx*3:(joint_idx+1)*3]
            for axis_idx, axis_name in enumerate(['x', 'y', 'z']):
                if axis_name in limits:
                    min_val, max_val = limits[axis_name]
                    val = joint_aa[axis_idx]
                    if val < min_val:
                        loss = loss + (min_val - val) ** 2
                    elif val > max_val:
                        loss = loss + (val - max_val) ** 2
        return loss

    def fit_single_frame(self, joints_3d, betas_fixed, prev_pose=None,
                         num_iterations=300, lr=0.01):
        """
        Fit SMPL pose for a single frame.

        joints_3d: (8, 3) - raw MediaPipe coordinates (unscaled)
        """
        # Scale MediaPipe data to SMPL dimensions
        joints_3d_scaled = joints_3d * self.global_scale

        # Initialize lower body pose parameters
        lower_body_pose = torch.zeros(
            1, len(self.LOWER_BODY_POSE_INDICES) * 3,
            requires_grad=True, device=self.device
        )

        # Warm start from previous frame
        if prev_pose is not None:
            with torch.no_grad():
                for local_idx, global_idx in enumerate(self.LOWER_BODY_POSE_INDICES):
                    lower_body_pose[0, local_idx*3:(local_idx+1)*3] = \
                        prev_pose[global_idx*3:(global_idx+1)*3]

        global_orient = torch.zeros(1, 3, requires_grad=True, device=self.device)
        transl = torch.zeros(1, 3, requires_grad=True, device=self.device)

        # Initialize translation to pelvis center
        with torch.no_grad():
            pelvis_approx = (joints_3d_scaled[0] + joints_3d_scaled[1]) / 2
            transl[0] = torch.tensor(pelvis_approx, dtype=torch.float32, device=self.device)

        target_joints = torch.tensor(
            joints_3d_scaled, dtype=torch.float32, device=self.device
        )
        target_indices = list(self.JOINT_MAPPING.values())

        joint_weights = torch.tensor(
            [self.JOINT_WEIGHTS[name] for name in self.MEDIAPIPE_JOINTS],
            dtype=torch.float32, device=self.device
        )

        optimizer = torch.optim.Adam(
            [lower_body_pose, global_orient, transl], lr=lr
        )

        upper_body_pose = torch.zeros(1, self.num_joints * 3, device=self.device)

        for i in range(num_iterations):
            optimizer.zero_grad()

            body_pose_full = upper_body_pose.clone()
            for local_idx, global_idx in enumerate(self.LOWER_BODY_POSE_INDICES):
                body_pose_full[0, global_idx*3:(global_idx+1)*3] = \
                    lower_body_pose[0, local_idx*3:(local_idx+1)*3]

            output = self.smpl_model(
                betas=betas_fixed,
                body_pose=body_pose_full,
                global_orient=global_orient,
                transl=transl
            )

            pred_joints = output.joints[:, target_indices, :]

            # Loss 1: Weighted data (Huber-like)
            per_joint_error = torch.sum((pred_joints[0] - target_joints) ** 2, dim=1)
            weighted_error = per_joint_error * joint_weights
            loss_data = torch.mean(weighted_error)

            # Loss 2: Pose prior (STRONG - prevent unnatural poses)
            loss_pose_prior = 0.05 * torch.sum(lower_body_pose ** 2)

            # Loss 3: Joint angle limits
            loss_angle_limits = 10.0 * self.joint_angle_limit_loss(body_pose_full)

            # Loss 4: Temporal smoothness
            loss_temporal = torch.tensor(0.0, device=self.device)
            if prev_pose is not None:
                prev_lower = torch.zeros_like(lower_body_pose)
                for local_idx, global_idx in enumerate(self.LOWER_BODY_POSE_INDICES):
                    prev_lower[0, local_idx*3:(local_idx+1)*3] = \
                        prev_pose[global_idx*3:(global_idx+1)*3]
                loss_temporal = 0.1 * torch.sum((lower_body_pose - prev_lower) ** 2)

            loss = loss_data + loss_pose_prior + loss_angle_limits + loss_temporal

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"  Iter {i}: total={loss.item():.5f} "
                      f"data={loss_data.item():.5f} "
                      f"pose={loss_pose_prior.item():.5f} "
                      f"angle={loss_angle_limits.item():.5f} "
                      f"temp={loss_temporal.item():.5f}")

        # Final pose
        with torch.no_grad():
            body_pose_final = upper_body_pose.clone()
            for local_idx, global_idx in enumerate(self.LOWER_BODY_POSE_INDICES):
                body_pose_final[0, global_idx*3:(global_idx+1)*3] = \
                    lower_body_pose[0, local_idx*3:(local_idx+1)*3]

        return {
            'betas': betas_fixed.detach().cpu().numpy()[0],
            'body_pose': body_pose_final.detach().cpu().numpy()[0].reshape(23, 3),
            'global_orient': global_orient.detach().cpu().numpy()[0],
            'transl': transl.detach().cpu().numpy()[0],
            'loss': loss.item(),
            'loss_data': loss_data.item(),
            'global_scale': self.global_scale,
            '_body_pose_flat': body_pose_final.detach().clone()[0],
        }

    # =========================================================
    # Phase 3: Temporal smoothing
    # =========================================================

    def smooth_pose_sequence(self, results, sigma=1.0):
        """Apply Gaussian smoothing to optimized pose parameters"""
        print(f"\n--- Phase 3: Temporal smoothing (sigma={sigma}) ---")

        all_body_pose = np.array([r['body_pose'] for r in results])
        all_global_orient = np.array([r['global_orient'] for r in results])
        all_transl = np.array([r['transl'] for r in results])

        n_frames = len(results)
        if n_frames < 3:
            print("  [SKIP] Too few frames")
            return results

        for j in range(23):
            for k in range(3):
                all_body_pose[:, j, k] = gaussian_filter1d(
                    all_body_pose[:, j, k], sigma=sigma
                )

        for k in range(3):
            all_global_orient[:, k] = gaussian_filter1d(
                all_global_orient[:, k], sigma=sigma
            )
            all_transl[:, k] = gaussian_filter1d(
                all_transl[:, k], sigma=sigma
            )

        for i in range(n_frames):
            results[i]['body_pose'] = all_body_pose[i]
            results[i]['global_orient'] = all_global_orient[i]
            results[i]['transl'] = all_transl[i]

        print(f"  [OK] Smoothed {n_frames} frames")
        return results

    # =========================================================
    # Main pipeline
    # =========================================================

    def fit_sequence(self, csv_path, output_path,
                     subsample=1, max_frames=None,
                     start_frame=0, end_frame=None,
                     num_iterations=300, smooth_sigma=1.0):
        """
        Complete fitting pipeline:
        0. Compute scale factor
        1. Estimate betas from bone lengths
        2. Fit pose per frame with physics constraints
        3. Temporal smoothing
        """
        print("=" * 60)
        print("Physics-Aware SMPL Fitting")
        print("=" * 60)

        joints_3d, timestamps, frames = self.load_csv_data(csv_path)
        total_csv = len(joints_3d)

        # Slice by start_frame / end_frame (on original CSV indices)
        end_frame = end_frame if end_frame is not None else total_csv
        end_frame = min(end_frame, total_csv)
        joints_3d = joints_3d[start_frame:end_frame]
        timestamps = timestamps[start_frame:end_frame]
        frames = frames[start_frame:end_frame]

        joints_3d = joints_3d[::subsample]
        timestamps = timestamps[::subsample]
        frames = frames[::subsample]

        if max_frames is not None:
            joints_3d = joints_3d[:max_frames]
            timestamps = timestamps[:max_frames]
            frames = frames[:max_frames]

        num_frames = len(joints_3d)
        print(f"CSV total: {total_csv} frames, selected: [{start_frame}:{end_frame}], "
              f"subsample={subsample} -> {num_frames} frames")

        # Phase 0: Scale
        self.compute_scale_factor(joints_3d)

        # Phase 1: Betas
        betas_fixed = self.estimate_betas_from_bones(joints_3d)

        # Phase 2: Pose fitting
        print(f"\n--- Phase 2: Fitting pose ({num_frames} frames) ---")
        results = []
        prev_pose = None

        for i in tqdm(range(num_frames), desc="Fitting"):
            result = self.fit_single_frame(
                joints_3d[i], betas_fixed,
                prev_pose=prev_pose,
                num_iterations=num_iterations
            )
            result['timestamp'] = timestamps[i]
            result['frame'] = frames[i]

            prev_pose = result['_body_pose_flat']
            del result['_body_pose_flat']
            results.append(result)

        # Phase 3: Smoothing
        results = self.smooth_pose_sequence(results, sigma=smooth_sigma)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(results, f)

        losses_data = [r['loss_data'] for r in results]
        print(f"\n{'=' * 60}")
        print(f"Fitting complete!")
        print(f"  Frames: {num_frames}")
        print(f"  Scale factor: {self.global_scale:.4f}")
        print(f"  Mean data loss: {np.mean(losses_data):.6f}")
        print(f"  Output: {output_path}")
        print(f"{'=' * 60}")

        return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Physics-aware SMPL fitting')
    parser.add_argument('--input_csv', type=str, required=True)
    parser.add_argument('--output_pkl', type=str, required=True)
    parser.add_argument('--smpl_model_path', type=str,
                        default='smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models')
    parser.add_argument('--gender', type=str, default='neutral')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subsample', type=int, default=1)
    parser.add_argument('--max_frames', type=int, default=None)
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame index in CSV (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame index in CSV (exclusive, default: last)')
    parser.add_argument('--num_iterations', type=int, default=300)
    parser.add_argument('--smooth_sigma', type=float, default=1.0)

    args = parser.parse_args()

    fitter = PhysicsAwareSMPLFitter(
        smpl_model_path=args.smpl_model_path,
        gender=args.gender,
        device=args.device
    )

    fitter.fit_sequence(
        csv_path=args.input_csv,
        output_path=args.output_pkl,
        subsample=args.subsample,
        max_frames=args.max_frames,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
        num_iterations=args.num_iterations,
        smooth_sigma=args.smooth_sigma
    )


if __name__ == '__main__':
    main()
