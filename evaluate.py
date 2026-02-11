"""
Evaluation script for Pressure2Pose

Supports all 5 model architectures via config.

Usage:
    python evaluate.py --config configs/default.yaml --checkpoint checkpoints/cnn_bigru_best.pth
"""

import torch
import numpy as np
import smplx
import argparse
import json
import time
from pathlib import Path
from tqdm import tqdm

from models import build_model
from datasets import PressureSequenceDataset
from utils import (load_config, compute_mpjpe, compute_pa_mpjpe,
                   compute_vertex_error, compute_bone_length_error)


def evaluate(model, dataloader, smpl_layer, device):
    """Evaluate model on dataset, return all metrics."""
    model.eval()
    mpjpe_list, pa_mpjpe_list, ve_list, ble_list, timings = [], [], [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            pressure = batch['pressure'].to(device)
            target_betas = batch['betas'].to(device)
            target_pose = batch['body_pose'].to(device)
            target_orient = batch['global_orient'].to(device)
            target_transl = batch['transl'].to(device)

            t0 = time.perf_counter()
            betas, body_pose, global_orient, transl = model(pressure)
            elapsed = (time.perf_counter() - t0) * 1000 / pressure.shape[0]
            timings.append(elapsed)

            bs = pressure.shape[0]
            smpl_layer.batch_size = bs

            pred_out = smpl_layer(
                betas=betas, body_pose=body_pose,
                global_orient=global_orient, transl=transl)
            tgt_out = smpl_layer(
                betas=target_betas, body_pose=target_pose,
                global_orient=target_orient, transl=target_transl)

            mpjpe_list.append(compute_mpjpe(pred_out.joints, tgt_out.joints))
            pa_mpjpe_list.append(compute_pa_mpjpe(pred_out.joints, tgt_out.joints))
            ve_list.append(compute_vertex_error(pred_out.vertices, tgt_out.vertices))
            ble_list.append(compute_bone_length_error(pred_out.joints, tgt_out.joints))

    return {
        'MPJPE (mm)': float(np.mean(mpjpe_list)),
        'PA-MPJPE (mm)': float(np.mean(pa_mpjpe_list)),
        'Vertex Error (mm)': float(np.mean(ve_list)),
        'Bone Length Error (mm)': float(np.mean(ble_list)),
        'Inference (ms/frame)': float(np.mean(timings)),
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Pressure2Pose')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output', type=str, default=None,
                        help='Save metrics JSON to this path')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataset
    if args.split == 'val':
        sequences = config['dataset']['val_sequences']
    else:
        sequences = config['dataset']['test_sequences']

    data_root = Path(config['dataset']['data_root'])
    ps = config['dataset']['pressure_shape']
    seq_len = config['model'].get('seq_len', 32)

    csv_files = [data_root / f'{s}_cleaned.csv' for s in sequences]
    pkl_files = [data_root / 'smpl_params' / f'{s}_physics.pkl' for s in sequences]

    dataset = PressureSequenceDataset(
        csv_files, pkl_files,
        seq_len=seq_len,
        pressure_shape=tuple(ps),
        normalize=True,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )
    print(f'Loaded {len(dataset)} windows from {len(sequences)} sequences')

    # Model
    ckpt = torch.load(args.checkpoint, map_location=device)
    ckpt_config = ckpt.get('config', config)
    model = build_model(ckpt_config).to(device)
    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state)
    print(f'Loaded {ckpt_config["model"]["type"]} from {args.checkpoint}')

    # SMPL
    smpl_layer = smplx.SMPL(
        model_path=config['smpl']['model_path'],
        gender=config['smpl']['gender'],
        batch_size=1,
        create_transl=False,
    ).to(device)

    # Evaluate
    results = evaluate(model, dataloader, smpl_layer, device)

    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    for metric, value in results.items():
        print(f'  {metric:25s}: {value:.2f}')
    print('=' * 60)

    # Save
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
