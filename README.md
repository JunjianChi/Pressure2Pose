# Pressure2Pose

Real-time 3D human pose estimation from plantar pressure insoles using SMPL body model.

<p align="center">
  <img src="output/showcase/showcase_frame_0060.png" width="85%">
</p>

This repo contains code and design files for the following publications and some reconfigurations:

**[ISCAS 2025]** [High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis](https://ieeexplore.ieee.org/abstract/document/11044303)
<br>CNN-LSTM pressure-to-pose pipeline, MediaPipe joint prediction, custom insole hardware design

**[ISCAS 2026]** Multimodal Smart Insole with Crossbar Crosstalk Compensation for Fall-Risk Prediction
<br>Dual-frame readout with dynamic range increment, IMU fusion, fall-risk assessment

**[Reconfiguration]** Physics-constrained Pressure to SMPL Predection
<br>New in this repo: Mediapipe label cleaner, SMPL parameter regression

This repo provides:

1. **Custom pressure insole hardware** — PCB design, MCU firmware, and data collection code.
2. **Physics-constrained label generation** — Cleans noisy MediaPipe 3D joints and fits them to the SMPL body model, producing physically plausible ground truth parameters.
3. **Pressure2Pose training pipeline** — Predicts SMPL parameters from pressure sequences, recovers full 3D body mesh via forward kinematics.

## Hardware

The insole PCB and sensor design files are in `hardware/`. The MCU firmware (ESP-IDF) is in `firmware/`.

```
hardware/
├── insole_fpc_ver1/      # FPC sensor array v1 (Altium SchDoc + PcbDoc)
├── insole_fpc_ver2/      # FPC sensor array v2 + gerber
├── sensor_laser_cut/     # Laser-cut pressure sensor pattern
└── mcu_pcb/              # MCU board (schematic PDF + gerber)
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

## Acknowledgments

- SMPL body model: Max Planck Institute for Intelligent Systems
- MediaPipe: Google
- ESP-IDF: Espressif Systems
