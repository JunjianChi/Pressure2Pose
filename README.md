# High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis
[ISCAS 2025] High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis

A custom high-resolution pressure insole system that reconstructs 3D human body pose in real-time using SMPL body model regression. Includes a full pipeline from hardware data collection to multi-architecture deep learning and 3D mesh visualization.

<p align="center">
  <img src="output/showcase/showcase_frame_0060.png" width="90%" alt="Pressure heatmaps and 3D SMPL mesh">
</p>

## Abstract

We present **Pressure2Pose**, an end-to-end system for estimating 3D human body pose from plantar pressure measurements. A custom insole with 33x15 capacitive pressure sensors per foot streams data at 30 Hz via ESP32-S3 WiFi. We compare five neural architectures (CNN, CNN+GRU, CNN+LSTM, CNN+TCN, CNN+Transformer) that predict SMPL body model parameters from pressure sequences, producing physically-constrained 3D meshes. Ground truth labels are generated through a physics-aware pipeline that fits SMPL parameters to cleaned MediaPipe 3D joint detections.

## Pipeline Overview

```
┌──────────────────┐    ┌───────────┐    ┌──────────────────┐    ┌────────────┐
│  Insole Hardware  │──>│ ESP32-S3  │──>│  Pressure CSV     │──>│ CNN / LSTM │──> SMPL 3D Mesh
│  33x15 x2 feet   │   │ WiFi 30Hz │   │  Matrix_0 + _1    │   │  Models    │
└──────────────────┘    └───────────┘    └──────────────────┘    └────────────┘
                                                 │
                        ┌───────────┐    ┌───────┴──────────┐
                        │ MediaPipe │──>│  SMPL Fitting     │──> Ground Truth PKL
                        │ 3D Joints │   │  Physics-aware    │    (training labels)
                        └───────────┘    └──────────────────┘
```

## Results

Five architectures are compared on the walking dataset (80/20 train/val split):

| Model | Params | MPJPE (mm) | PA-MPJPE (mm) | Vertex Err (mm) | Bone Len Err (mm) | Inference (ms) |
|-------|--------|-----------|--------------|----------------|-------------------|---------------|
| A: CNN Baseline | 1.09M | — | — | — | — | — |
| B: CNN + BiGRU | 3.71M | — | — | — | — | — |
| C: CNN + BiLSTM | 4.49M | — | — | — | — | — |
| D: CNN + TCN | 3.72M | — | — | — | — | — |
| E: CNN + Transformer | 9.50M | — | — | — | — | — |

> Run `examples/train_compare.ipynb` to populate this table with actual metrics.

## Installation

```bash
# Clone and install dependencies
git clone https://github.com/your-username/Pressure2Pose.git
cd Pressure2Pose
pip install -r requirements.txt

# Download SMPL model (requires registration at https://smpl.is.tue.mpg.de/)
# Extract to: smpl_models/SMPL_python_v.1.1.0/
```

## Usage

### 1. Data Collection

Flash the ESP32-S3 firmware and record synchronized pressure + camera data:

```bash
# Record pressure data with camera (for label generation)
python host/data\ log/cam_pressure_record.py
```

### 2. Label Generation

Convert noisy MediaPipe 3D joints into clean SMPL parameters:

```bash
# Step 1: Clean MediaPipe labels (bone consistency, symmetry, temporal smoothing)
python tools/clean_mediapipe_labels.py \
    --input data/walking1_processed.csv \
    --output data/walking1_cleaned.csv

# Step 2: Physics-aware SMPL fitting (scale, betas, pose with priors + joint limits)
python tools/fit_smpl_physics.py \
    --input_csv data/walking1_cleaned.csv \
    --output_pkl data/smpl_params/walking1_physics.pkl

# Step 3: Visualize fitted SMPL (interactive 3D viewer)
python tools/visualize_smpl_open3d.py \
    --smpl_params data/smpl_params/walking1_physics.pkl \
    --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
    --play_sequence --fps 30
```

### 3. Training

Train and compare all 5 model architectures interactively:

```bash
# Jupyter notebook (recommended)
jupyter notebook examples/train_compare.ipynb

# Or command-line training
python tools/train.py --config configs/default.yaml
```

### 4. Inference

Run inference on new pressure data:

```bash
python tools/inference.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/cnn_bigru_best.pth \
    --input data/dataset/walking1.csv \
    --output output/walking1_pred.pkl
```

### 5. Visualization

Generate showcase images (pressure heatmaps + 3D mesh):

```bash
python tools/generate_showcase.py \
    --csv data/dataset/walking1.csv \
    --pkl data/smpl_params/walking1_physics.pkl \
    --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
    --output_dir output/showcase
```

## Hardware

| Component | Specification |
|-----------|--------------|
| Sensor Array | 33x15 capacitive pressure grid per foot (495 sensors) |
| IMU | ICM-45686 9-axis motion tracking |
| MCU | ESP32-S3 with WiFi streaming |
| Sampling Rate | ~30 Hz |
| Interface | WiFi UDP streaming to host PC |

See [hardware/](hardware/) for PCB design files and [firmware/](firmware/) for ESP-IDF source code.

## Project Structure

```
Pressure2Pose/
├── models/
│   ├── pressure_to_smpl.py          # PressureEncoder, SMPLRegressor, SMPLLoss
│   └── temporal_models.py           # GRU, LSTM, TCN, Transformer + build_model()
├── datasets/
│   ├── pressure_dataset.py          # Single-frame dataset
│   └── pressure_sequence_dataset.py # Sliding-window temporal dataset
├── tools/
│   ├── clean_mediapipe_labels.py    # Step 1: Physical cleaning
│   ├── fit_smpl_physics.py          # Step 2: SMPL fitting
│   ├── visualize_smpl_open3d.py     # Step 3: Interactive 3D viewer
│   ├── extract_joint_angles.py      # Joint angle extraction
│   ├── train.py                     # CLI training script
│   ├── evaluate.py                  # CLI evaluation script
│   ├── inference.py                 # Unified inference (all model types)
│   └── generate_showcase.py         # Showcase image generation
├── utils/
│   ├── config.py                    # YAML config loading
│   ├── logger.py                    # Logging setup
│   ├── metrics.py                   # MPJPE, PA-MPJPE, vertex error, bone error
│   ├── smpl_utils.py                # Joint angle utilities
│   └── coordinate_transform.py      # MediaPipe <-> SMPL coordinates
├── examples/
│   └── train_compare.ipynb          # Multi-model training + comparison notebook
├── configs/
│   └── default.yaml                 # Default training configuration
├── docs/
│   └── methodology.md               # Loss, metrics, architecture details
├── host/                            # Real-time host application
├── firmware/                        # ESP32-S3 firmware (ESP-IDF)
├── hardware/                        # PCB and sensor design files
├── smpl_models/                     # SMPL model files (download separately)
├── data/                            # Walking data (CSV + SMPL params)
├── checkpoints/                     # Trained model weights
├── output/                          # Visualization outputs
└── requirements.txt
```

## Methodology

See [docs/methodology.md](docs/methodology.md) for detailed documentation on:
- System architecture and end-to-end pipeline
- Label generation pipeline (cleaning + physics-aware SMPL fitting)
- Model architectures (shared encoder/regressor + 5 temporal variants)
- Multi-component loss function (joint MSE, vertex MSE, regularisation)
- Evaluation metrics (MPJPE, PA-MPJPE, vertex error, bone length error)
- Training hyperparameters and schedule

## Related Work

- **ISCAS paper (prior work):** CNN-LSTM architecture predicting MediaPipe joint coordinates directly from pressure. This approach produces raw 3D points without physical constraints, leading to anatomically implausible predictions and no renderable body surface.
- **This work:** Reconfigures the regression target to SMPL body model parameters. The SMPL parameterisation provides a physically-constrained skeleton, smooth mesh output (6890 vertices), and compatibility with the broader pose estimation ecosystem.

## License

MIT License - see [LICENSE](LICENSE).

## Acknowledgments

- **SMPL model:** Max Planck Institute for Intelligent Systems
- **MediaPipe:** Google
- **ESP-IDF:** Espressif Systems
- **smplx:** [https://github.com/vchoutas/smplx](https://github.com/vchoutas/smplx)
