"""
Sliding-window temporal dataset for Pressure2Pose
Produces sequences of (T, 2, H, W) pressure frames mapped to the SMPL params of the center frame.

Normalization: Global max normalization — all values divided by the global maximum
across all frames and both feet. Preserves inter-frame and inter-foot magnitude
relationships (important for gait phase / weight distribution).
"""

import numpy as np
import pandas as pd
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class PressureSequenceDataset(Dataset):
    """Sliding-window dataset: (T, 2, H, W) pressure -> SMPL params of center frame."""

    def __init__(self, csv_files, smpl_pkl_files,
                 seq_len=32, stride=1,
                 pressure_shape=(2, 33, 15),
                 normalize=True,
                 global_max=None):
        """
        Args:
            csv_files: List of CSV files with pressure data
            smpl_pkl_files: List of matching SMPL parameter pickle files
            seq_len: Number of frames per window (T)
            stride: Sliding window stride
            pressure_shape: (channels, H, W) for reshaping each foot
            normalize: Whether to apply global max normalization
            global_max: Pre-computed global max (float). If None and normalize=True,
                        it will be computed from all CSV files during loading.
        """
        self.seq_len = seq_len
        self.stride = stride
        self.pressure_shape = pressure_shape
        self.normalize = normalize
        self.global_max = global_max

        # First pass: compute global max if needed
        if self.normalize and self.global_max is None:
            self.global_max = self._compute_global_max(csv_files)
            print(f"  Global pressure max: {self.global_max:.1f}")

        # Load all sequences into memory
        self.sequences = []  # list of (pressures_array, smpl_params_list)
        for csv_file, pkl_file in zip(csv_files, smpl_pkl_files):
            pressures, smpl_params = self._load_sequence(csv_file, pkl_file)
            self.sequences.append((pressures, smpl_params))

        # Build index: (seq_idx, window_start)
        self.windows = []
        for seq_idx, (pressures, smpl_params) in enumerate(self.sequences):
            n_frames = len(smpl_params)
            for start in range(0, n_frames - seq_len + 1, stride):
                self.windows.append((seq_idx, start))

        print(f"PressureSequenceDataset: {len(self.windows)} windows "
              f"(T={seq_len}, stride={stride}) from {len(csv_files)} sequences")

    def _compute_global_max(self, csv_files):
        """Scan all CSV files to find the global pressure maximum."""
        global_max = 0.0
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            for _, row in df.iterrows():
                for col in ('Matrix_0', 'Matrix_1'):
                    vals = [float(x) for x in row[col].split(',')]
                    frame_max = max(vals)
                    if frame_max > global_max:
                        global_max = frame_max
        return max(global_max, 1.0)  # avoid division by zero

    def _parse_pressure_matrix(self, matrix_str):
        """Parse comma-separated pressure values from CSV cell."""
        return np.array([float(x) for x in matrix_str.split(',')])

    def _reshape_pressure(self, pressure_flat):
        """Reshape flat pressure vector to 2D grid, pad/crop as needed."""
        H, W = self.pressure_shape[1], self.pressure_shape[2]
        target_len = H * W
        if len(pressure_flat) < target_len:
            pressure_flat = np.pad(pressure_flat, (0, target_len - len(pressure_flat)))
        elif len(pressure_flat) > target_len:
            pressure_flat = pressure_flat[:target_len]
        return pressure_flat.reshape(H, W)

    def _load_sequence(self, csv_file, pkl_file):
        """Load one walking sequence; returns (pressures_list, smpl_params_list)."""
        df = pd.read_csv(csv_file)
        with open(pkl_file, 'rb') as f:
            smpl_params = pickle.load(f)

        assert len(df) == len(smpl_params), \
            f"Mismatch: CSV {csv_file} has {len(df)} frames, PKL has {len(smpl_params)}"

        pressures = []
        for _, row in df.iterrows():
            left = self._reshape_pressure(self._parse_pressure_matrix(row['Matrix_0']))
            right = self._reshape_pressure(self._parse_pressure_matrix(row['Matrix_1']))

            # Global max normalization: divide by global max, keeps 0 as 0
            if self.normalize and self.global_max is not None:
                left = left / self.global_max
                right = right / self.global_max

            frame = np.stack([left, right], axis=0).astype(np.float32)  # (2, H, W)
            pressures.append(frame)

        return pressures, smpl_params

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        seq_idx, start = self.windows[idx]
        pressures, smpl_params = self.sequences[seq_idx]

        # Center frame index
        center = start + self.seq_len // 2

        # Build pressure window (T, 2, H, W)
        pressure_seq = np.stack(pressures[start:start + self.seq_len], axis=0)

        # Target SMPL params from center frame
        smpl = smpl_params[center]

        return {
            'pressure': torch.from_numpy(pressure_seq),           # (T, 2, H, W)
            'betas': torch.tensor(smpl['betas'], dtype=torch.float32),       # (10,)
            'body_pose': torch.tensor(smpl['body_pose'].flatten(), dtype=torch.float32),  # (69,)
            'global_orient': torch.tensor(smpl['global_orient'], dtype=torch.float32),    # (3,)
            'transl': torch.tensor(smpl['transl'], dtype=torch.float32),                  # (3,)
        }


def create_sequence_dataloaders(data_root, train_sequences, val_sequences,
                                seq_len=32, stride=1, batch_size=16,
                                num_workers=4, pressure_shape=(2, 33, 15)):
    """
    Create train/val DataLoaders for temporal models.

    The global pressure max is computed from the TRAINING set only,
    then applied to both train and val sets (prevents data leakage).
    """
    data_root = Path(data_root)

    def _make_paths(sequences):
        csv = [data_root / f'{s}_cleaned.csv' for s in sequences]
        pkl = [data_root / 'smpl_params' / f'{s}_physics.pkl' for s in sequences]
        return csv, pkl

    train_csv, train_pkl = _make_paths(train_sequences)
    val_csv, val_pkl = _make_paths(val_sequences)

    # Build train dataset (computes global_max internally)
    train_ds = PressureSequenceDataset(
        train_csv, train_pkl,
        seq_len=seq_len, stride=stride,
        pressure_shape=pressure_shape, normalize=True
    )

    # Val dataset reuses the train global_max (no data leakage)
    val_ds = PressureSequenceDataset(
        val_csv, val_pkl,
        seq_len=seq_len, stride=stride,
        pressure_shape=pressure_shape, normalize=True,
        global_max=train_ds.global_max
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
