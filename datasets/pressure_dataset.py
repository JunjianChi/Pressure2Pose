"""
Dataset preparation for Pressure2Pose training
Loads pressure matrices and SMPL parameters

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


class PressureToSMPLDataset(Dataset):
    """Dataset for training Pressure -> SMPL model"""

    def __init__(self, csv_files, smpl_pkl_files,
                 pressure_shape=(2, 33, 15),
                 normalize=True,
                 augment=False,
                 global_max=None):
        """
        Args:
            csv_files: List of CSV files with pressure data
            smpl_pkl_files: List of pickle files with fitted SMPL params
            pressure_shape: Shape to reshape pressure matrix (channels, H, W)
            normalize: Whether to apply global max normalization
            augment: Whether to apply data augmentation
            global_max: Pre-computed global max (float). If None and normalize=True,
                        it will be computed from all CSV files during loading.
        """
        self.pressure_shape = pressure_shape
        self.normalize = normalize
        self.augment = augment
        self.global_max = global_max

        # First pass: compute global max if needed
        if self.normalize and self.global_max is None:
            self.global_max = self._compute_global_max(csv_files)
            print(f"  Global pressure max: {self.global_max:.1f}")

        # Load all data
        self.samples = []
        for csv_file, pkl_file in zip(csv_files, smpl_pkl_files):
            self._load_sequence(csv_file, pkl_file)

        print(f"Loaded {len(self.samples)} samples from {len(csv_files)} sequences")

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

    def _load_sequence(self, csv_file, pkl_file):
        """Load one walking sequence"""
        # Load CSV
        df = pd.read_csv(csv_file)

        # Load SMPL parameters
        with open(pkl_file, 'rb') as f:
            smpl_params = pickle.load(f)

        # Ensure same length
        assert len(df) == len(smpl_params), \
            f"Mismatch: CSV has {len(df)} frames, SMPL has {len(smpl_params)}"

        # Create samples
        for i, (idx, row) in enumerate(df.iterrows()):
            # Parse pressure matrices
            matrix_0 = self._parse_pressure_matrix(row['Matrix_0'])
            matrix_1 = self._parse_pressure_matrix(row['Matrix_1'])

            # Get SMPL params
            smpl = smpl_params[i]

            sample = {
                'pressure_left': matrix_0,
                'pressure_right': matrix_1,
                'betas': smpl['betas'],
                'body_pose': smpl['body_pose'],
                'global_orient': smpl['global_orient'],
                'transl': smpl['transl'],
                'timestamp': row['Timestamp'],
                'frame': row['Frame']
            }

            self.samples.append(sample)

    def _parse_pressure_matrix(self, matrix_str):
        """Parse comma-separated pressure values"""
        values = np.array([float(x) for x in matrix_str.split(',')])
        return values

    def _reshape_pressure(self, pressure_flat):
        """Reshape flat pressure array to 2D grid"""
        H, W = self.pressure_shape[1], self.pressure_shape[2]

        if len(pressure_flat) != H * W:
            # Pad or crop if needed
            if len(pressure_flat) < H * W:
                pressure_flat = np.pad(pressure_flat,
                                      (0, H * W - len(pressure_flat)))
            else:
                pressure_flat = pressure_flat[:H * W]

        return pressure_flat.reshape(H, W)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Reshape pressure matrices
        pressure_left = self._reshape_pressure(sample['pressure_left'])
        pressure_right = self._reshape_pressure(sample['pressure_right'])

        # Global max normalization
        if self.normalize and self.global_max is not None:
            pressure_left = pressure_left / self.global_max
            pressure_right = pressure_right / self.global_max

        # Stack as 2-channel image
        pressure = np.stack([pressure_left, pressure_right], axis=0)  # (2, H, W)

        # Data augmentation
        if self.augment and np.random.rand() > 0.5:
            # Left-right flip
            pressure = np.flip(pressure, axis=0).copy()  # Swap channels
            # TODO: Also flip SMPL pose parameters (mirror left/right joints)

        # Convert to tensors
        pressure = torch.tensor(pressure, dtype=torch.float32)
        betas = torch.tensor(sample['betas'], dtype=torch.float32)
        body_pose = torch.tensor(sample['body_pose'].flatten(), dtype=torch.float32)
        global_orient = torch.tensor(sample['global_orient'], dtype=torch.float32)
        transl = torch.tensor(sample['transl'], dtype=torch.float32)

        return {
            'pressure': pressure,  # (2, H, W)
            'betas': betas,  # (10,)
            'body_pose': body_pose,  # (69,) = 23 joints x 3
            'global_orient': global_orient,  # (3,)
            'transl': transl,  # (3,)
        }


def create_dataloaders(data_root, train_sequences, val_sequences,
                       batch_size=32, num_workers=4,
                       pressure_shape=(2, 33, 15)):
    """
    Create train and validation dataloaders.

    The global pressure max is computed from the TRAINING set only,
    then applied to both train and val sets (prevents data leakage).
    """
    data_root = Path(data_root)

    # Training data
    train_csv_files = [data_root / f'{seq}_cleaned.csv'
                       for seq in train_sequences]
    train_pkl_files = [data_root / 'smpl_params' / f'{seq}_physics.pkl'
                       for seq in train_sequences]

    train_dataset = PressureToSMPLDataset(
        train_csv_files, train_pkl_files,
        pressure_shape=pressure_shape,
        normalize=True, augment=True
    )

    # Validation data — reuse training global_max
    val_csv_files = [data_root / f'{seq}_cleaned.csv'
                     for seq in val_sequences]
    val_pkl_files = [data_root / 'smpl_params' / f'{seq}_physics.pkl'
                     for seq in val_sequences]

    val_dataset = PressureToSMPLDataset(
        val_csv_files, val_pkl_files,
        pressure_shape=pressure_shape,
        normalize=True, augment=False,
        global_max=train_dataset.global_max
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader


def test_dataset():
    """Test data loading"""
    data_root = Path(__file__).parent.parent / 'data'

    train_loader, val_loader = create_dataloaders(
        data_root=data_root,
        train_sequences=['walking1'],
        val_sequences=['walking1'],
        batch_size=8,
        num_workers=0
    )

    batch = next(iter(train_loader))
    print("Batch contents:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape} ({value.dtype})")

    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")


if __name__ == '__main__':
    test_dataset()
