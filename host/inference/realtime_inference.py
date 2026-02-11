"""
Real-time inference from pressure data

Example usage:
    python host/inference/realtime_inference.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/best_model.pth \
        --input data/walking1.csv \
        --visualize
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm

from models import PressureToSMPL
from utils import load_config, SMPLVisualizer


def parse_pressure_matrix(matrix_str):
    """Parse comma-separated pressure values"""
    values = np.array([float(x) for x in matrix_str.split(',')])
    return values


def reshape_and_normalize_pressure(pressure_flat, shape=(24, 20)):
    """Reshape and normalize pressure data"""
    H, W = shape

    # Pad or crop if needed
    if len(pressure_flat) < H * W:
        pressure_flat = np.pad(pressure_flat, (0, H * W - len(pressure_flat)))
    elif len(pressure_flat) > H * W:
        pressure_flat = pressure_flat[:H * W]

    # Reshape
    pressure = pressure_flat.reshape(H, W)

    # Normalize to [0, 1]
    p_min, p_max = pressure.min(), pressure.max()
    if p_max > p_min:
        pressure = (pressure - p_min) / (p_max - p_min)

    return pressure


class RealtimePredictor:
    """Real-time pose prediction from pressure data"""

    def __init__(self, config_path, checkpoint_path, device='cuda'):
        """
        Initialize predictor

        Args:
            config_path: Path to config file
            checkpoint_path: Path to trained model
            device: Device to run on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = load_config(config_path)

        # Load model
        self.model = PressureToSMPL(
            smpl_model_path=self.config['smpl']['model_path'],
            gender=self.config['smpl']['gender'],
            feature_dim=self.config['model']['feature_dim'],
            num_betas=self.config['model']['num_betas'],
            device=self.device
        ).to(self.device)

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"Loaded model from {checkpoint_path} (epoch {checkpoint['epoch']})")
        print(f"Running on {self.device}")

    def predict_frame(self, pressure_matrix):
        """
        Predict pose from single frame pressure data

        Args:
            pressure_matrix: (2, H, W) numpy array or torch tensor

        Returns:
            vertices: (6890, 3) numpy array
            joints: (24, 3) numpy array
        """
        # Convert to tensor if needed
        if isinstance(pressure_matrix, np.ndarray):
            pressure = torch.tensor(pressure_matrix, dtype=torch.float32)
        else:
            pressure = pressure_matrix

        # Add batch dimension and move to device
        pressure = pressure.unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            vertices, joints = self.model.predict_mesh(pressure)

        # Remove batch dimension
        vertices = vertices.cpu().numpy()
        joints = joints.cpu().numpy()

        return vertices, joints

    def predict_sequence(self, csv_path, start_frame=0, num_frames=None):
        """
        Predict poses for a sequence of frames

        Args:
            csv_path: Path to CSV file with pressure data
            start_frame: Starting frame index
            num_frames: Number of frames to process (None = all)

        Returns:
            List of (vertices, joints) tuples
        """
        # Load CSV
        df = pd.read_csv(csv_path)

        # Determine range
        end_frame = len(df) if num_frames is None else start_frame + num_frames
        end_frame = min(end_frame, len(df))

        results = []

        for idx in tqdm(range(start_frame, end_frame), desc='Processing frames'):
            row = df.iloc[idx]

            # Parse pressure matrices
            matrix_0_flat = parse_pressure_matrix(row['Matrix_0'])
            matrix_1_flat = parse_pressure_matrix(row['Matrix_1'])

            # Reshape and normalize
            matrix_0 = reshape_and_normalize_pressure(matrix_0_flat)
            matrix_1 = reshape_and_normalize_pressure(matrix_1_flat)

            # Stack as 2-channel tensor
            pressure = np.stack([matrix_0, matrix_1], axis=0)  # (2, H, W)

            # Predict
            vertices, joints = self.predict_frame(pressure)

            results.append({
                'frame': idx,
                'timestamp': row['Timestamp'],
                'vertices': vertices,
                'joints': joints,
            })

        return results


def main(args):
    """Main inference function"""
    # Initialize predictor
    predictor = RealtimePredictor(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device='cuda' if not args.cpu else 'cpu'
    )

    # Process single frame or sequence
    if args.frame_idx is not None:
        # Single frame
        print(f"Processing single frame {args.frame_idx}...")
        results = predictor.predict_sequence(
            args.input,
            start_frame=args.frame_idx,
            num_frames=1
        )
        result = results[0]

        # Visualize
        if args.visualize:
            visualizer = SMPLVisualizer()
            visualizer.render_mesh_matplotlib(
                result['vertices'],
                joints=result['joints'],
                title=f"Predicted Pose (Frame {args.frame_idx})",
                save_path=args.output
            )

    else:
        # Sequence
        print(f"Processing sequence from {args.input}...")
        results = predictor.predict_sequence(
            args.input,
            start_frame=args.start_frame,
            num_frames=args.num_frames
        )

        print(f"Processed {len(results)} frames")

        # Save results
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(output_path, results)
            print(f"Saved results to {output_path}")

        # Visualize sample frames
        if args.visualize:
            visualizer = SMPLVisualizer()
            sample_indices = [0, len(results) // 2, len(results) - 1]

            for idx in sample_indices:
                result = results[idx]
                output_file = Path(args.output).parent / f'frame_{result["frame"]}.png'
                visualizer.render_mesh_matplotlib(
                    result['vertices'],
                    joints=result['joints'],
                    title=f"Frame {result['frame']}",
                    save_path=output_file
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Real-time pose inference')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input CSV file with pressure data')

    # Processing options
    parser.add_argument('--frame_idx', type=int, default=None,
                        help='Process single frame at this index')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Starting frame for sequence processing')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to process (None = all)')

    # Output options
    parser.add_argument('--output', type=str, default='output/predictions.npy',
                        help='Output file path')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize results')

    # Device
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')

    args = parser.parse_args()
    main(args)
