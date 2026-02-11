"""
Unified inference script for all Pressure2Pose model types.

Handles both single-frame (CNN Baseline) and temporal models (GRU/LSTM/TCN/Transformer).
For temporal models, early frames are zero-padded to fill the sequence window.

Usage:
    python inference.py \
        --config configs/default.yaml \
        --checkpoint checkpoints/cnn_bigru_best.pth \
        --input data/walking1_cleaned.csv \
        --output output/walking1_pred.pkl
"""

import numpy as np
import pandas as pd
import pickle
import torch
import argparse
import time
from pathlib import Path

from models import build_model
from utils import load_config


class InferenceEngine:
    """Unified inference engine for all model architectures."""

    def __init__(self, config, checkpoint_path, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.config = config

        model_type = config['model']['type'].lower()
        self.is_temporal = model_type != 'cnn_baseline'
        self.seq_len = config['model'].get('seq_len', 32)

        ps = config.get('dataset', {}).get('pressure_shape', [2, 33, 15])
        self.pressure_h = ps[1]
        self.pressure_w = ps[2]

        self.global_max = config.get('dataset', {}).get('global_max', None)

        self.model = build_model(config).to(self.device)

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        state = ckpt.get('model_state_dict', ckpt)
        self.model.load_state_dict(state)
        self.model.eval()

        print(f"[InferenceEngine] Loaded {model_type} from {checkpoint_path} "
              f"(temporal={self.is_temporal}, device={self.device})")

    def _parse_pressure_matrix(self, matrix_str):
        values = np.array([float(x) for x in matrix_str.split(',')])
        H, W = self.pressure_h, self.pressure_w
        target = H * W
        if len(values) < target:
            values = np.pad(values, (0, target - len(values)))
        elif len(values) > target:
            values = values[:target]
        return values.reshape(H, W)

    def _normalize(self, arr):
        if self.global_max is not None and self.global_max > 0:
            return arr / self.global_max
        return arr

    def predict_sequence(self, csv_path, output_path=None):
        df = pd.read_csv(csv_path)
        n_frames = len(df)
        print(f"[InferenceEngine] Processing {n_frames} frames from {csv_path}")

        if self.global_max is None:
            gmax = 0.0
            for _, row in df.iterrows():
                for col in ('Matrix_0', 'Matrix_1'):
                    vals = [float(x) for x in row[col].split(',')]
                    gmax = max(gmax, max(vals))
            self.global_max = max(gmax, 1.0)
            print(f"[InferenceEngine] Computed global_max={self.global_max:.1f}")

        frames = []
        for _, row in df.iterrows():
            left = self._normalize(self._parse_pressure_matrix(row['Matrix_0']))
            right = self._normalize(self._parse_pressure_matrix(row['Matrix_1']))
            frame = np.stack([left, right], axis=0).astype(np.float32)
            frames.append(frame)

        results = []
        timings = []

        with torch.no_grad():
            for i in range(n_frames):
                t0 = time.perf_counter()

                if self.is_temporal:
                    half = self.seq_len // 2
                    window = []
                    for j in range(i - half, i - half + self.seq_len):
                        if 0 <= j < n_frames:
                            window.append(frames[j])
                        else:
                            window.append(np.zeros_like(frames[0]))
                    seq = np.stack(window, axis=0)
                    x = torch.from_numpy(seq).unsqueeze(0).to(self.device)
                else:
                    x = torch.from_numpy(frames[i]).unsqueeze(0).to(self.device)

                betas, body_pose, global_orient, transl = self.model(x)

                elapsed = (time.perf_counter() - t0) * 1000
                timings.append(elapsed)

                results.append({
                    'betas': betas[0].cpu().numpy(),
                    'body_pose': body_pose[0].cpu().numpy().reshape(23, 3),
                    'global_orient': global_orient[0].cpu().numpy(),
                    'transl': transl[0].cpu().numpy(),
                    'frame': i,
                })

        avg_time = np.mean(timings)
        print(f"[InferenceEngine] Done. Avg: {avg_time:.1f} ms/frame")

        if output_path:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"[InferenceEngine] Saved {len(results)} frames to {output_path}")

        return results


def main():
    parser = argparse.ArgumentParser(description='Pressure2Pose Inference')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    output = args.output or f"output/{Path(args.input).stem}_pred.pkl"
    device = torch.device(args.device) if args.device else None
    engine = InferenceEngine(config, args.checkpoint, device=device)
    engine.predict_sequence(args.input, output)


if __name__ == '__main__':
    main()
