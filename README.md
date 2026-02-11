# Pressure2Pose

Real-time 3D human pose estimation from plantar pressure insoles using SMPL body model.

<p align="center">
  <img src="pics/intro.png" width="85%">
</p>

This repo contains code and design files for the following publications:


**[ISCAS 2025]** [High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis](https://ieeexplore.ieee.org/abstract/document/11044303)
Junjian Chi, Qingyu Zhang, Zibo Zhang, Andreas Demosthenous and Yu Wu, Circuit and System Group, University College London
***CNN-LSTM pressure-to-pose pipeline, MediaPipe joint prediction, custom insole hardware design***

**[ISCAS 2026]** Multimodal Smart Insole with Crossbar Crosstalk Compensation for Fall-Risk Prediction
Junjian Chi, Zibo Zhang, Qingyu Zhang, Andreas Demosthenous and Yu Wu, Circuit and System Group, University College London
***Dual-frame readout with dynamic range increment, IMU fusion, fall-risk assessment***

**[Reconfiguration]** Physics-constrained Pressure to SMPL Prediction
***Mediapipe label cleaner, SMPL parameter regression***


## Hardware

The insole system consists of a flexible printed circuit (FPC) pressure sensor array with concentric circular electrodes, a custom MCU board based on ESP32-S3 (with analog MUX, TIA readout, DAC excitation, and 6-axis IMU ICM-45686), and a host PC for data collection and model inference.

<p align="center">
  <img src="pics/schematic.png" width="85%">
</p>

Design files are organized under `hardware/`:

```
hardware/
├── insole_fpc_ver1/        # FPC sensor array v1 (Altium SchDoc + PcbDoc)
├── insole_fpc_ver2/        # FPC sensor array v2 + production gerber
├── sensor_laser_cut/       # Laser-cut conductive polymer sensor pattern (.dip)
└── mcu_pcb/                # MCU board (schematic PDF + PcbDoc + gerber)
```

### Firmware

The ESP32-S3 firmware is built with [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/) and located in `firmware/`. It handles ADC continuous scanning, DAC excitation, MUX control, IMU SPI communication, WiFi UDP streaming, and sensor configuration.

Flash the firmware step by step:

```bash
cd firmware
```

```bash
idf.py set-target esp32s3
```

```bash
idf.py build
```

```bash
idf.py flash monitor
```

### Data Recording

The host-side scripts in `host/` communicate with the insole hardware over WiFi UDP. Recorded data (pressure CSV files and camera videos) should be saved to the `data/` directory under the project root. You can name the recorded files however you like (e.g., `walking1.csv`, `squat_session2.csv`). The filenames used in the Usage section below (such as `walking1`) are just examples — substitute your own filenames when running the commands.

Each pressure CSV contains columns `Matrix_0` and `Matrix_1` (left and right foot), where each cell is a comma-separated string of 495 pressure values (33 rows x 15 columns).

### Host Tools

- `host/data_log/cam_pressure_record.py` — Records pressure data and camera video simultaneously. The UDP IP, port, and output file paths are hardcoded in the script — edit them before use (default: `192.168.137.1:8999`, sensor grid 33x15).

- `host/live_pressure_visualize/single_pressure_server.py` — Live pressure heatmap visualization for a single foot insole. Receives UDP packets and renders a real-time 33x15 heatmap.

- `host/live_pressure_visualize/double_pressure_server.py` — Live pressure heatmap for dual-foot insoles (left + right side by side).

- `host/live_pressure_visualize/single_pressure_imu_server.py` — Single foot pressure heatmap with IMU data overlay (accelerometer + gyroscope from ICM-45686).

- `host/live_pressure_visualize/double_pressure_imu_server.py` — Dual-foot pressure heatmap with IMU data overlay.

- `host/inference/realtime_inference.py` — Real-time inference using a trained model. Loads a checkpoint and runs predictions on pressure data. Supports single-frame or sequence processing with optional visualization.

- `host/visualizer/live_visualizer.py` — Live 3D body visualization using Open3D, displaying the SMPL mesh in real time.

## Installation

```bash
pip install -r requirements.txt
```

Download the SMPL model (requires registration at https://smpl.is.tue.mpg.de/). Extract it so that the model files are at:

```
smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models/
```

Key dependencies: `torch>=2.0`, `smplx`, `open3d`, `numpy`, `pandas`, `scipy`, `opencv-python`, `tqdm`, `pyyaml`, `tensorboard`, `matplotlib`, `trimesh`

For ESP32-S3 firmware, install [ESP-IDF v5.x](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/).

## Usage

> **Note:** The dataset is not included in this repository. You need to record your own data using the host tools described above, and save the CSV files to the `data/` directory. All filenames below (e.g., `walking1`, `walking1_cleaned`) are examples — replace them with your own filenames.

### 1. Label Generation

**Step 1 — Clean MediaPipe labels** (`preprocessing/clean_mediapipe_labels.py`)

Applies physical constraints to raw MediaPipe 3D joint predictions: bone length consistency, bilateral symmetry, temporal Gaussian smoothing, joint angle limits, and optional gait phase prior.

```bash
python preprocessing/clean_mediapipe_labels.py --input_csv data/walking1.csv --output_csv data/walking1_cleaned.csv
```

| Flag | Required | Description |
|------|----------|-------------|
| `--input_csv` | Yes | Path to the raw MediaPipe CSV file (e.g., `data/walking1.csv`) |
| `--output_csv` | Yes | Path to save the cleaned CSV (e.g., `data/walking1_cleaned.csv`) |
| `--no_symmetry` | No | Disable bilateral symmetry constraint |
| `--no_temporal` | No | Disable temporal Gaussian smoothing |
| `--no_gait_prior` | No | Disable gait phase detection prior |
| `--smoothing_sigma` | No | Temporal smoothing sigma in frames (default: `2.0`) |
| `--max_frames` | No | Maximum number of frames to process (default: all) |

**Step 2 — Fit SMPL parameters** (`preprocessing/fit_smpl_physics.py`)

Estimates body shape (betas) from median bone lengths across all frames, then fits lower-body pose per frame using differentiable optimization with joint angle limits and pose priors. Outputs a PKL file containing per-frame SMPL parameters.

```bash
python preprocessing/fit_smpl_physics.py --input_csv data/walking1_cleaned.csv --output_pkl data/smpl_params/walking1_physics.pkl --num_iterations 200
```

| Flag | Required | Description |
|------|----------|-------------|
| `--input_csv` | Yes | Path to the cleaned MediaPipe CSV from Step 1 |
| `--output_pkl` | Yes | Path to save the fitted SMPL parameters (e.g., `data/smpl_params/walking1_physics.pkl`) |
| `--smpl_model_path` | No | Path to SMPL model directory (default: `smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models`) |
| `--gender` | No | SMPL gender: `neutral`, `male`, or `female` (default: `neutral`) |
| `--device` | No | Compute device: `cuda` or `cpu` (default: `cuda`) |
| `--num_iterations` | No | Number of optimization iterations per frame (default: `300`) |
| `--max_frames` | No | Maximum number of frames to fit (default: all) |
| `--start_frame` | No | Start frame index, inclusive (default: `0`) |
| `--end_frame` | No | End frame index, exclusive (default: last frame) |
| `--subsample` | No | Process every N-th frame (default: `1`, i.e., every frame) |
| `--smooth_sigma` | No | Post-fitting temporal smoothing sigma (default: `1.0`) |

### 2. Visualize SMPL

- `tools/visualize_smpl_open3d.py` — Interactive 3D SMPL body viewer using Open3D. Supports single-frame inspection with optional joint spheres and skeleton overlay, full sequence playback at configurable FPS, video export to MP4, and frame range selection.

View a single frame with joint spheres:

```bash
python tools/visualize_smpl_open3d.py --smpl_params data/smpl_params/walking1_physics.pkl --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models --frame_idx 10 --show_joints
```

Play sequence animation:

```bash
python tools/visualize_smpl_open3d.py --smpl_params data/smpl_params/walking1_physics.pkl --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models --play_sequence --fps 30
```

Save as video file:

```bash
python tools/visualize_smpl_open3d.py --smpl_params data/smpl_params/walking1_physics.pkl --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models --save_video output/walking1.mp4 --fps 30
```

Visualize a specific frame range:

```bash
python tools/visualize_smpl_open3d.py --smpl_params data/smpl_params/walking1_physics.pkl --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models --play_sequence --start_frame 100 --end_frame 500 --fps 30
```

| Flag | Required | Description |
|------|----------|-------------|
| `--smpl_params` | Yes | Path to SMPL parameters PKL file |
| `--smpl_path` | Yes | Path to SMPL model directory |
| `--gender` | No | SMPL gender: `neutral`, `male`, or `female` (default: `neutral`) |
| `--device` | No | Compute device: `cuda` or `cpu` (default: `cuda`) |
| `--frame_idx` | No | Frame index to visualize in single-frame mode (default: `0`) |
| `--show_joints` | No | Show joint spheres on the mesh |
| `--show_skeleton` | No | Show skeleton connections between joints |
| `--play_sequence` | No | Play the entire sequence as animation |
| `--fps` | No | Playback / video FPS (default: `30`) |
| `--start_frame` | No | Start frame index, inclusive (default: `0`) |
| `--end_frame` | No | End frame index, exclusive (default: last frame) |
| `--max_frames` | No | Maximum number of frames to process (default: all) |
| `--save_video` | No | Save sequence as video to this path (e.g., `output/walking1.mp4`) |
| `--width` | No | Video width in pixels (default: `1280`) |
| `--height` | No | Video height in pixels (default: `720`) |

- `tools/generate_showcase.py` — Generates side-by-side showcase images: left foot pressure heatmap, right foot pressure heatmap, and 3D SMPL mesh rendering. Outputs 300 DPI PNG images to the specified directory (8 evenly-spaced frames by default).

```bash
python tools/generate_showcase.py --csv data/walking1_cleaned.csv --pkl data/smpl_params/walking1_physics.pkl --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models --output_dir output/showcase
```

| Flag | Required | Description |
|------|----------|-------------|
| `--csv` | Yes | Path to pressure CSV file |
| `--pkl` | Yes | Path to SMPL parameters PKL file |
| `--smpl_path` | Yes | Path to SMPL model directory |
| `--output_dir` | No | Output directory for showcase images (default: `output/showcase`) |
| `--frames` | No | Specific frame indices to visualize (e.g., `--frames 0 50 100`; default: 8 evenly spaced) |
| `--gender` | No | SMPL gender: `neutral`, `male`, or `female` (default: `neutral`) |
| `--device` | No | Compute device (default: `cpu`) |
| `--pressure_h` | No | Pressure matrix height (default: `33`) |
| `--pressure_w` | No | Pressure matrix width (default: `15`) |

- `tools/extract_joint_angles.py` — Extracts joint angles from SMPL axis-angle parameters and exports to CSV. Useful for biomechanical gait analysis, range of motion assessment, and clinical reporting.

```bash
python tools/extract_joint_angles.py --smpl_params data/smpl_params/walking1_physics.pkl --output output/joint_angles.csv --format euler
```

| Flag | Required | Description |
|------|----------|-------------|
| `--smpl_params` | Yes | Path to SMPL parameters PKL file |
| `--output` | Yes | Output CSV file path |
| `--format` | No | Output angle format: `euler` (roll/pitch/yaw in radians), `axis_angle` (3D rotation vectors), or `degrees` (Euler angles in degrees). Default: `euler` |

### 3. Train

Interactive notebook with all 5 model architectures, training loops, comparison table, and visualization:

```bash
jupyter notebook notebooks/train_compare.ipynb
```

Or command-line training using `configs/default.yaml`:

```bash
python train.py --config configs/default.yaml
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | No | Path to YAML configuration file (default: `configs/default.yaml`) |
| `--resume` | No | Path to a checkpoint file to resume training from (e.g., `checkpoints/cnn_bigru_epoch_40.pth`) |

To change model architecture, edit `model.type` in `configs/default.yaml`. Available architectures:

- `cnn_baseline` — Single-frame CNN baseline. No temporal context; processes each pressure frame independently. Lowest parameter count (~1.09M), serves as the non-temporal reference.

- `cnn_bigru` — CNN + Bidirectional GRU (default). Two-layer BiGRU with hidden size 256 aggregates temporal features from a 32-frame sliding window. Bidirectional design captures both past and future context. ~3.71M parameters.

- `cnn_bilstm` — CNN + Bidirectional LSTM. Same architecture as BiGRU but with LSTM cells, adding cell state for longer-range memory. Slightly more parameters (~4.49M) due to the extra gate.

- `cnn_tcn` — CNN + Temporal Convolutional Network. Four dilated causal Conv1d residual blocks with dilation pattern (1, 2, 4, 8), giving a receptive field of 32 frames matching the window size. Fully convolutional, no recurrence. ~3.72M parameters.

- `cnn_transformer` — CNN + Transformer Encoder. Four-layer TransformerEncoder with 8 attention heads, d_model=512, feedforward dim=1024, and sinusoidal positional encoding. Requires LR warmup (5 epochs) for stable training. Largest model (~9.50M parameters).

Training features: TensorBoard logging (`logs/`), periodic checkpointing (`checkpoints/`, every 10 epochs), early stopping (patience=20 on validation MPJPE), gradient clipping (max norm=1.0 for RNN models), LR warmup (5 epochs for Transformer), StepLR scheduler (step=20, gamma=0.5).

Resume training from a checkpoint:

```bash
python train.py --config configs/default.yaml --resume checkpoints/cnn_bigru_epoch_40.pth
```

### 4. Evaluate

Runs evaluation on the validation or test set. Computes all metrics (MPJPE, PA-MPJPE, per-vertex mesh error, bone length error, per-frame inference time) using differentiable SMPL forward kinematics.

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/cnn_bigru_best.pth
```

Evaluate on test split and save metrics to JSON:

```bash
python evaluate.py --config configs/default.yaml --checkpoint checkpoints/cnn_bigru_best.pth --split test --output output/metrics.json
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | No | Path to YAML configuration file (default: `configs/default.yaml`) |
| `--checkpoint` | Yes | Path to trained model checkpoint |
| `--split` | No | Dataset split to evaluate: `val` or `test` (default: `val`) |
| `--batch_size` | No | Evaluation batch size (default: `16`) |
| `--output` | No | Path to save metrics as JSON (e.g., `output/metrics.json`) |

### 5. Inference

Runs a trained model on new pressure data and saves predicted SMPL parameters as PKL. Handles both single-frame (CNN Baseline) and temporal models (BiGRU/BiLSTM/TCN/Transformer). For temporal models, the sliding window (T=32) is zero-padded at sequence boundaries.

```bash
python inference.py --config configs/default.yaml --checkpoint checkpoints/cnn_bigru_best.pth --input data/walking1_cleaned.csv --output output/walking1_pred.pkl
```

| Flag | Required | Description |
|------|----------|-------------|
| `--config` | Yes | Path to YAML configuration file |
| `--checkpoint` | Yes | Path to trained model checkpoint |
| `--input` | Yes | Path to input pressure CSV file |
| `--output` | No | Path to save predicted SMPL parameters PKL (default: `output/<input_stem>_pred.pkl`) |
| `--device` | No | Compute device: `cuda` or `cpu` (default: auto-detect) |

The output PKL can be visualized directly with `tools/visualize_smpl_open3d.py`.

## Methodology

See [docs/methodology.md](docs/methodology.md) for detailed documentation on:
- Loss function design (joint MSE, vertex MSE, shape/pose regularization, weighting strategy)
- Evaluation metrics (MPJPE, PA-MPJPE, vertex error, bone length error)
- Model architecture details (shared CNN encoder, 4 MLP heads, 5 temporal variants)
- Training configuration and hyperparameters
- Label generation pipeline (MediaPipe cleaning + physics-aware SMPL fitting)

## Acknowledgments

- [SMPL](https://smpl.is.tue.mpg.de/) body model: Max Planck Institute for Intelligent Systems
- [MediaPipe](https://mediapipe.dev/) 3D pose estimation: Google
- [ESP-IDF](https://docs.espressif.com/projects/esp-idf/) firmware framework: Espressif Systems
- [smplx](https://github.com/vchoutas/smplx) PyTorch layer: Vassilis Choutas
- [Open3D](http://www.open3d.org/) 3D visualization: Intel ISL
