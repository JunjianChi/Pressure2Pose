# Pressure2Pose

Real-time 3D human pose estimation from plantar pressure insoles using SMPL body model.

<p align="center">
  <img src="output/showcase/showcase_frame_0060.png" width="85%">
</p>

This repo contains code for the following projects:

**[ISCAS 2025]** [High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis](https://ieeexplore.ieee.org/abstract/document/11044303)

**[ISCAS 2026 (Accepted)]** Pressure2Pose: Real-Time 3D Human Pose Estimation from Plantar Pressure via Physics-Constrained SMPL Regression

## What's New (ISCAS 2026)

The ISCAS 2025 paper used a CNN-LSTM to predict MediaPipe joint coordinates directly from pressure data. The new work extends this in three ways:

1. **Hardware system** — Custom insole with 33x15 capacitive pressure sensors + ICM-45686 IMU per foot, ESP32-S3 WiFi streaming at 30 Hz
2. **Physics-constrained label generation** — Cleans noisy MediaPipe 3D joints (bone consistency, bilateral symmetry, temporal smoothing) and fits them to the SMPL body model with joint angle limits and pose priors, producing physically plausible ground truth
3. **SMPL parameter regression** — Instead of predicting raw joint positions, the model predicts SMPL parameters (85-dim: betas + pose + orientation + translation), then recovers the full 3D mesh through SMPL forward kinematics. Five architectures compared: CNN baseline, CNN+GRU, CNN+LSTM, CNN+TCN, CNN+Transformer

## Hardware

The insole PCB and sensor design files are in `hardware/`. The MCU firmware (ESP-IDF) is in `firmware/`.

```
hardware/
├── insole_fpc_ver1/      # FPC sensor array v1 (Altium SchDoc + PcbDoc)
├── insole_fpc_ver2/      # FPC sensor array v2 + gerber
├── sensor_laser_cut/     # Laser-cut pressure sensor pattern
└── mcu_pcb/              # ESP32-S3 MCU board (schematic PDF + gerber)
```

**Flash firmware** (requires [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)):

```bash
cd firmware
idf.py set-target esp32s3
idf.py build
idf.py flash monitor
```

**Host-side data collection** — connect to the insole via WiFi UDP:

```bash
# Record pressure + camera simultaneously (for label generation)
python host/data_log/cam_pressure_record.py

# Live pressure heatmap (single insole)
python host/live_pressure_visualize/single_pressure_server.py

# Live pressure heatmap (both insoles)
python host/live_pressure_visualize/double_pressure_server.py

# Live pressure + IMU
python host/live_pressure_visualize/double_pressure_imu_server.py
```

## Installation

```bash
pip install -r requirements.txt
```

Download the SMPL model (requires registration at https://smpl.is.tue.mpg.de/). Extract it so that the model files are at:

```
smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models/
```

## Usage

### 1. Clean MediaPipe labels

Apply physical constraints (bone length consistency, bilateral symmetry, temporal smoothing) to raw MediaPipe 3D joint predictions:

```bash
python tools/clean_mediapipe_labels.py \
    --input_csv data/walking1.csv \
    --output_csv data/walking1_cleaned.csv
```

### 2. Fit SMPL parameters

Physics-aware SMPL fitting: estimates body shape from median bone lengths, then fits lower-body pose per frame with joint angle limits and pose priors:

```bash
python tools/fit_smpl_physics.py \
    --input_csv data/walking1_cleaned.csv \
    --output_pkl data/smpl_params/walking1_physics.pkl \
    --num_iterations 200
```

Use `--max_frames 500` to fit a subset for quick testing.

### 3. Visualize fitted SMPL

Interactive 3D viewer (Open3D):

```bash
# Single frame
python tools/visualize_smpl_open3d.py \
    --smpl_params data/smpl_params/walking1_physics.pkl \
    --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
    --frame_idx 10 --show_joints

# Play sequence
python tools/visualize_smpl_open3d.py \
    --smpl_params data/smpl_params/walking1_physics.pkl \
    --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
    --play_sequence --fps 30
```

### 4. Generate showcase images

Pressure heatmaps alongside 3D SMPL mesh:

```bash
python tools/generate_showcase.py \
    --csv data/walking1_cleaned.csv \
    --pkl data/smpl_params/walking1_physics.pkl \
    --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
    --output_dir output/showcase
```

### 5. Train

Interactive notebook with all 5 model architectures:

```bash
jupyter notebook examples/train_compare.ipynb
```

Or command-line:

```bash
python tools/train.py --config configs/default.yaml
```

### 6. Evaluate

```bash
python tools/evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth
```

### 7. Inference

Run a trained model on new pressure data:

```bash
python tools/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/cnn_bigru_best.pth \
    --input data/walking1_cleaned.csv \
    --output output/walking1_pred.pkl
```

The output PKL can be visualized with `tools/visualize_smpl_open3d.py`.

## Project Structure

```
Pressure2Pose/
├── firmware/                        # ESP32-S3 firmware (ESP-IDF)
├── hardware/                        # PCB + sensor design (Altium)
├── host/                            # Host-side applications
│   ├── data_log/                    #   Pressure + camera recording
│   ├── live_pressure_visualize/     #   Real-time pressure heatmap
│   ├── inference/                   #   Real-time pose inference
│   └── preprocessing/               #   Raw data cleaning
├── tools/                           # Core pipeline scripts
│   ├── clean_mediapipe_labels.py    #   Step 1: Physical cleaning
│   ├── fit_smpl_physics.py          #   Step 2: SMPL fitting
│   ├── visualize_smpl_open3d.py     #   Step 3: 3D visualization
│   ├── train.py                     #   Model training
│   ├── evaluate.py                  #   Model evaluation
│   ├── inference.py                 #   Batch inference
│   └── generate_showcase.py         #   Showcase image generation
├── models/                          # Neural network architectures
│   ├── pressure_to_smpl.py          #   PressureEncoder + SMPLRegressor
│   └── temporal_models.py           #   GRU / LSTM / TCN / Transformer
├── datasets/                        # PyTorch datasets
│   ├── pressure_dataset.py          #   Single-frame dataset
│   └── pressure_sequence_dataset.py #   Sliding-window temporal dataset
├── utils/                           # Metrics, config, logging
├── configs/                         # YAML configs
├── examples/
│   └── train_compare.ipynb          # Multi-model training notebook
├── docs/
│   └── methodology.md               # Technical details (loss, metrics)
├── data/                            # Walking data (CSV + SMPL PKL)
├── smpl_models/                     # SMPL model (download separately)
└── requirements.txt
```

## Acknowledgments

- SMPL body model: Max Planck Institute for Intelligent Systems
- MediaPipe: Google
- ESP-IDF: Espressif Systems
