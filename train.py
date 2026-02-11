"""
Training script for Pressure2Pose

Supports all 5 model architectures via config.
Uses build_model() registry + external smplx.SMPL for loss computation.

Usage:
    python train.py --config configs/default.yaml
"""

import torch
import torch.optim as optim
import torch.nn as nn
import smplx
import numpy as np
import copy
import argparse
from pathlib import Path
from tqdm import tqdm

from models import build_model, SMPLLoss
from datasets import create_sequence_dataloaders
from utils import load_config, compute_mpjpe


def train_one_epoch(model, train_loader, smpl_layer, criterion, optimizer,
                    device, grad_clip=None):
    model.train()
    total_loss = 0.0

    for batch in train_loader:
        pressure = batch['pressure'].to(device)
        target_betas = batch['betas'].to(device)
        target_pose = batch['body_pose'].to(device)
        target_orient = batch['global_orient'].to(device)
        target_transl = batch['transl'].to(device)

        optimizer.zero_grad()

        betas, body_pose, global_orient, transl = model(pressure)

        bs = pressure.shape[0]
        smpl_layer.batch_size = bs
        pred_out = smpl_layer(
            betas=betas, body_pose=body_pose,
            global_orient=global_orient, transl=transl
        )
        with torch.no_grad():
            tgt_out = smpl_layer(
                betas=target_betas, body_pose=target_pose,
                global_orient=target_orient, transl=target_transl
            )

        target_dict = {'joints': tgt_out.joints, 'vertices': tgt_out.vertices}
        _, loss = criterion(
            pred_out, target_dict,
            pred_params=(betas, body_pose, global_orient, transl)
        )

        loss.backward()
        if grad_clip:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def validate(model, val_loader, smpl_layer, criterion, device):
    model.eval()
    total_loss = 0.0
    mpjpe_list = []

    with torch.no_grad():
        for batch in val_loader:
            pressure = batch['pressure'].to(device)
            target_betas = batch['betas'].to(device)
            target_pose = batch['body_pose'].to(device)
            target_orient = batch['global_orient'].to(device)
            target_transl = batch['transl'].to(device)

            betas, body_pose, global_orient, transl = model(pressure)

            bs = pressure.shape[0]
            smpl_layer.batch_size = bs
            pred_out = smpl_layer(
                betas=betas, body_pose=body_pose,
                global_orient=global_orient, transl=transl
            )
            tgt_out = smpl_layer(
                betas=target_betas, body_pose=target_pose,
                global_orient=target_orient, transl=target_transl
            )

            target_dict = {'joints': tgt_out.joints, 'vertices': tgt_out.vertices}
            _, loss = criterion(
                pred_out, target_dict,
                pred_params=(betas, body_pose, global_orient, transl)
            )
            total_loss += loss.item()
            mpjpe_list.append(compute_mpjpe(pred_out.joints, tgt_out.joints))

    val_loss = total_loss / len(val_loader)
    val_mpjpe = np.mean(mpjpe_list)
    return val_loss, val_mpjpe


def main():
    parser = argparse.ArgumentParser(description='Train Pressure2Pose')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Dataset
    ps = config['dataset']['pressure_shape']
    seq_len = config['model'].get('seq_len', 32)

    train_loader, val_loader = create_sequence_dataloaders(
        data_root=config['dataset']['data_root'],
        train_sequences=config['dataset']['train_sequences'],
        val_sequences=config['dataset']['val_sequences'],
        seq_len=seq_len,
        batch_size=config['training']['batch_size'],
        num_workers=config['training']['num_workers'],
        pressure_shape=tuple(ps),
    )
    print(f'Train: {len(train_loader.dataset)} windows, Val: {len(val_loader.dataset)} windows')

    # Model
    model = build_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Model: {config["model"]["type"]} ({n_params/1e6:.2f}M params)')

    # SMPL layer for loss
    smpl_layer = smplx.SMPL(
        model_path=config['smpl']['model_path'],
        gender=config['smpl']['gender'],
        batch_size=1,
        create_transl=False,
    ).to(device)

    criterion = SMPLLoss(
        lambda_joints=config['training']['loss']['lambda_joints'],
        lambda_betas=config['training']['loss']['lambda_betas'],
        lambda_pose=config['training']['loss']['lambda_pose'],
        lambda_vertices=config['training']['loss']['lambda_vertices'],
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['optimizer']['lr'],
        weight_decay=config['training']['optimizer']['weight_decay'],
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config['training']['scheduler']['step_size'],
        gamma=config['training']['scheduler']['gamma'],
    )

    # Training config
    epochs = config['training']['epochs']
    patience = config['training'].get('patience', 20)
    grad_clip = config['training'].get('grad_clip', None)
    warmup_epochs = config['training'].get('warmup_epochs', 0)
    base_lr = config['training']['optimizer']['lr']

    model_type = config['model']['type'].lower()
    is_rnn = model_type in ('cnn_bigru', 'cnn_bilstm')
    is_transformer = model_type == 'cnn_transformer'
    if not is_rnn:
        grad_clip = None
    if not is_transformer:
        warmup_epochs = 0

    # TensorBoard
    writer = None
    if config['logging'].get('use_tensorboard', False):
        from torch.utils.tensorboard import SummaryWriter
        from datetime import datetime
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_dir = Path(config['logging']['log_dir']) / f'{model_type}_{ts}'
        writer = SummaryWriter(log_dir)
        print(f'TensorBoard: {log_dir}')

    # Checkpoint dir
    ckpt_dir = Path(config['logging']['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Resume
    start_epoch = 1
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f'Resumed from epoch {ckpt["epoch"]}')

    # Training loop
    best_mpjpe = float('inf')
    best_state = None
    wait = 0

    for epoch in range(start_epoch, epochs + 1):
        # LR warmup
        if warmup_epochs > 0 and epoch <= warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = base_lr * epoch / warmup_epochs

        train_loss = train_one_epoch(
            model, train_loader, smpl_layer, criterion, optimizer,
            device, grad_clip=grad_clip
        )

        val_loss, val_mpjpe = validate(
            model, val_loader, smpl_layer, criterion, device
        )

        if epoch > warmup_epochs:
            scheduler.step()

        # Early stopping
        if val_mpjpe < best_mpjpe:
            best_mpjpe = val_mpjpe
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        lr = optimizer.param_groups[0]['lr']
        marker = '*best' if wait == 0 else ''
        if epoch % 5 == 0 or epoch == 1 or wait == 0:
            print(f'Epoch {epoch:3d}  train={train_loss:.4f}  val={val_loss:.4f}  '
                  f'MPJPE={val_mpjpe:.1f}mm  lr={lr:.1e}  {marker}')

        if writer:
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/val', val_loss, epoch)
            writer.add_scalar('MPJPE/val', val_mpjpe, epoch)
            writer.add_scalar('LR', lr, epoch)

        # Save periodic checkpoint
        if epoch % config['logging']['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mpjpe': val_mpjpe,
                'config': config,
            }, ckpt_dir / f'{model_type}_epoch_{epoch}.pth')

        if wait >= patience:
            print(f'Early stopping at epoch {epoch} (patience={patience})')
            break

    # Save best model
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_mpjpe': best_mpjpe,
    }, ckpt_dir / f'{model_type}_best.pth')
    print(f'Best model saved: MPJPE={best_mpjpe:.1f}mm -> {ckpt_dir / f"{model_type}_best.pth"}')

    if writer:
        writer.close()


if __name__ == '__main__':
    main()
