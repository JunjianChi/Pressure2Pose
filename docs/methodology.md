# Methodology

Technical documentation for the Pressure2Pose framework: system overview, label generation pipeline, model architectures, loss functions, evaluation metrics, and training details.

## 1. System Overview

```
  ┌──────────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │  Insole Hardware  │────>│   ESP32-S3   │────>│ Pressure CSV │────>│  CNN + LSTM  │────>│  SMPL Params │
  │  33x15 x2 feet   │     │  WiFi 30 Hz  │     │ Matrix_0 + 1 │     │  (5 models)  │     │  3D Mesh PKL │
  └──────────────────┘     └──────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                       │
                          ┌──────────────┐     ┌───────┴────────┐
                          │  MediaPipe   │────>│  SMPL Fitting  │──> Ground Truth PKL
                          │  3D Joints   │     │ Physics-aware  │    (training labels)
                          └──────────────┘     └────────────────┘
```

### 1.1 Relation to Prior Work

**ISCAS 2025 paper:** Used a CNN-LSTM architecture to predict MediaPipe joint coordinates *directly* from pressure data. This joint regression approach produces 3D points in arbitrary coordinate frames without physical constraints, leading to anatomically implausible bone lengths, inter-penetrating limbs, and no renderable body surface. The original ISCAS pipeline is preserved in `notebooks/iscas2025_cnn_lstm.ipynb`.

**This project (Reconfiguration):** Reconfigures the regression target from raw joint coordinates to **SMPL body model parameters** (betas, body_pose, global_orient, transl). SMPL provides:
- A physically-constrained skeleton with fixed bone topology
- Smooth, watertight mesh output (6890 vertices, 13776 faces) suitable for rendering
- A low-dimensional parameterisation (85 values) that inherently prevents many anatomical violations
- Compatibility with the broader pose estimation ecosystem (SPIN, HMR, PARE, etc.)

### 1.2 SMPL Output Format

The SMPL body model is parameterised by 85 values:

| Parameter | Dim | Description |
|-----------|-----|-------------|
| `betas` | 10 | Body shape coefficients (PCA of body shape space) |
| `body_pose` | 69 | Joint rotations for 23 joints (axis-angle, 3 values each) |
| `global_orient` | 3 | Root orientation (axis-angle) |
| `transl` | 3 | Global translation |

Forward kinematics produces:
- **45 joints** (24 body + 21 extra regressed joints) as 3D positions
- **6890 vertices** defining the full body mesh surface

## 2. Label Generation Pipeline

Raw MediaPipe 3D joints are noisy: 5-10% bone length variation across frames and 8-10% left-right asymmetry. Fitting SMPL directly to these joints produces unnatural poses. The label pipeline therefore has two stages.

### 2.1 Physical Cleaning (`preprocessing/clean_mediapipe_labels.py`)

The `MediaPipeCleaner` class applies a sequence of physical constraints:

1. **Target bone length estimation:** Compute the median bone length for each of the 6 lower-body bone segments (left/right hip-knee, knee-ankle, ankle-foot) across the full sequence. The median is robust to outlier frames.

2. **Bone length enforcement:** For each frame, iteratively project joints so that all bone lengths match the target. Uses a gentle correction strategy (30% per iteration) to avoid oscillation. Converges in 3-5 iterations per frame.

3. **Bilateral symmetry:** Left and right leg bone lengths are averaged and enforced symmetrically. Only bone *lengths* are symmetrised, not joint *positions* — during walking, left and right feet are at different positions, so forcing positional symmetry would be incorrect.

4. **Joint angle limits:** Prevent anatomically impossible poses. Key constraint: knees can only bend forward (150-180 degrees), preventing hyperextension artifacts.

5. **Temporal smoothing:** Gaussian filter (configurable sigma, default 2 frames) on joint positions to remove high-frequency jitter while preserving gait dynamics. Applied both before and after the physical constraint steps.

6. **Gait phase detection:** Optional gait prior that detects stance/swing phases based on vertical foot position and adjusts constraints accordingly.

**Usage:**
```bash
python preprocessing/clean_mediapipe_labels.py --input_csv data/walking1.csv --output_csv data/walking1_cleaned.csv
```

### 2.2 Physics-Aware SMPL Fitting (`preprocessing/fit_smpl_physics.py`)

The `PhysicsSMPLFitter` class fits SMPL parameters to cleaned MediaPipe joints in four phases:

**Phase 0 — Scale estimation:** Compute the ratio between MediaPipe bone lengths and SMPL template bone lengths. Apply a single global scale factor to align coordinate frames. This accounts for the fact that MediaPipe uses arbitrary units while SMPL uses metres.

**Phase 1 — Body shape (betas):** Estimated once from median bone lengths across the full sequence. The optimiser finds the 10-dimensional beta vector whose resulting SMPL skeleton best matches the target bone lengths. This vector is then frozen for all subsequent per-frame fitting.

**Phase 2 — Per-frame pose fitting:** For each frame, optimise `body_pose` (69-dim), `global_orient` (3-dim), and `transl` (3-dim) to minimise weighted joint re-projection error, subject to:
- **Confidence-weighted joints:** Hip joints weighted highest (most reliable from MediaPipe), foot joints weighted lowest
- **Strong pose prior** (weight 0.05): Penalises deviation from neutral pose — more aggressive than typical SMPL fitting to compensate for noisy input
- **Joint angle limit loss:** Prevents hyperextension of knees and ankles using a soft penalty
- **Lower-body only:** Only the 6 lower-body joints (hips, knees, ankles) are used for fitting. Upper body joints remain at neutral pose since plantar pressure provides no upper-body information

**Phase 3 — Temporal smoothing:** Gaussian smoothing on the fitted pose parameters (axis-angle rotations) to remove frame-to-frame jitter.

**Why SMPL as denoiser:** SMPL's forward kinematics enforce a consistent skeleton. Even when the optimisation loss is not fully minimised, the output is always an anatomically valid human body. This makes SMPL fitting an implicit *denoising* step — it projects noisy 3D observations onto the manifold of valid human poses.

**Usage:**
```bash
python preprocessing/fit_smpl_physics.py --input_csv data/walking1_cleaned.csv --output_pkl data/smpl_params/walking1_physics.pkl --num_iterations 200
```

## 3. Model Architectures

All five models share the same **spatial encoder** (`PressureEncoder`) and **parameter regressor** (`SMPLRegressor`), defined in `models/pressure_to_smpl.py`. They differ only in how they aggregate temporal context from a sequence of pressure frames. The temporal models and `build_model()` registry are in `models/temporal_models.py`.

### 3.1 Shared Spatial Encoder: `PressureEncoder`

A 4-layer CNN that processes a single pressure frame (2 channels = left foot + right foot):

| Layer | Output Shape | Details |
|-------|-------------|---------|
| Conv2d + BN + ReLU | (32, H/2, W/2) | 5x5 kernel, stride 2, padding 2 |
| Conv2d + BN + ReLU | (64, H/4, W/4) | 3x3 kernel, stride 2, padding 1 |
| Conv2d + BN + ReLU | (128, H/8, W/8) | 3x3 kernel, stride 2, padding 1 |
| Conv2d + BN + ReLU | (256, H/16, W/16) | 3x3 kernel, stride 2, padding 1 |
| AdaptiveAvgPool2d(1) + Linear | (512,) | Global pooling -> FC projection |

The adaptive average pooling makes the encoder agnostic to input spatial resolution (works for both the 33x15 native grid and any other sensor layout).

### 3.2 Shared Regressor: `SMPLRegressor`

Four independent MLP heads predict SMPL parameter groups from a 512-dim feature vector:

| Head | Architecture | Output |
|------|-------------|--------|
| Shape | Linear(512, 256) -> ReLU -> Dropout(0.2) -> Linear(256, 10) | betas (10) |
| Pose | Linear(512, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 69) | body_pose (69) |
| Orient | Linear(512, 128) -> ReLU -> Linear(128, 3) | global_orient (3) |
| Transl | Linear(512, 128) -> ReLU -> Linear(128, 3) | transl (3) |

Total output: **85 dimensions** = 10 + 69 + 3 + 3.

### 3.3 Five Temporal Variants

For temporal models (B-E), each frame in a sliding window of T=32 is independently encoded by `PressureEncoder`, producing a feature sequence of shape `(B, T, 512)`. The temporal module aggregates this sequence and outputs a single 512-dim vector for the **centre frame** (position T//2). This is then passed to `SMPLRegressor`.

| Model | Config key | Temporal Module | Key Config | ~Params |
|-------|-----------|----------------|-----------|---------|
| **A: CNN Baseline** | `cnn_baseline` | None (centre frame only) | — | ~1.09M |
| **B: CNN + BiGRU** | `cnn_bigru` | Bidirectional GRU -> Linear projection | h=256, 2 layers | ~3.71M |
| **C: CNN + BiLSTM** | `cnn_bilstm` | Bidirectional LSTM -> Linear projection | h=256, 2 layers | ~4.49M |
| **D: CNN + TCN** | `cnn_tcn` | 4 dilated Conv1d residual blocks | d=1,2,4,8, k=3 | ~3.72M |
| **E: CNN + Transformer** | `cnn_transformer` | Sinusoidal PE + TransformerEncoder | d=512, 8 heads, 4 layers | ~9.50M |

**Model selection:** Set `model.type` in `configs/default.yaml` to one of the config keys above. The `build_model(config)` function in `models/temporal_models.py` instantiates the correct model class with the appropriate hyperparameters.

**Centre-frame extraction:** All temporal models extract the representation at position T//2 after temporal processing. This ensures the model has access to both past and future context (bidirectional), which is appropriate for offline analysis of recorded walking data.

**TCN details:** Each TCN block consists of a dilated causal Conv1d followed by BatchNorm, ReLU, and a residual connection. The dilation pattern (1, 2, 4, 8) gives an effective receptive field of 32 frames — matching the window size T=32.

**Transformer details:** Uses sinusoidal positional encoding (not learned) and 4 standard TransformerEncoderLayers with `dim_feedforward=1024`. Requires LR warmup (5 epochs) for training stability.

## 4. Loss Function

The multi-component `SMPLLoss` (defined in `models/pressure_to_smpl.py`) supervises both the reconstructed body and the predicted parameters. An external `smplx.SMPL` layer runs differentiable forward kinematics inside the training loop to compute joint and vertex positions from predicted parameters.

### 4.1 Joint Position Loss

$$L_\text{joints} = \text{MSE}(J_\text{pred}, J_\text{gt})$$

where $J \in \mathbb{R}^{B \times 45 \times 3}$ are the SMPL joint positions obtained by running the differentiable SMPL layer on predicted/ground-truth parameters. This is the **primary supervision signal** (`lambda_joints = 1.0`).

### 4.2 Vertex Position Loss

$$L_\text{verts} = \text{MSE}(V_\text{pred}, V_\text{gt})$$

where $V \in \mathbb{R}^{B \times 6890 \times 3}$ are the SMPL mesh vertices. Supervising vertices ensures the full body surface is accurately reconstructed, not just the skeleton (`lambda_vertices = 0.5`).

### 4.3 Shape Regularisation

$$L_\text{betas} = \frac{1}{N}\sum \beta_i^2$$

Encourages the predicted body shape to stay close to the average human body. Without this term, the model may predict extreme body shapes to compensate for pose errors (`lambda_betas = 0.01`).

### 4.4 Pose Regularisation

$$L_\text{pose} = \frac{1}{N}\sum \theta_i^2$$

Encourages joint rotations to stay near the neutral (T-pose) configuration. Prevents wild rotations in joints with weak supervision signal (`lambda_pose = 0.001`).

### 4.5 Total Loss

$$L = \lambda_j L_\text{joints} + \lambda_v L_\text{verts} + \lambda_b L_\text{betas} + \lambda_p L_\text{pose}$$

| Component | Weight | Purpose |
|-----------|--------|---------|
| Joint MSE | 1.0 | Primary skeleton supervision |
| Vertex MSE | 0.5 | Surface reconstruction |
| Betas L2 | 0.01 | Shape regularisation |
| Pose L2 | 0.001 | Pose regularisation |

## 5. Evaluation Metrics

All metrics are implemented in `utils/metrics.py` and reported in millimetres (mm).

### 5.1 MPJPE — Mean Per Joint Position Error

$$\text{MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \| \hat{p}_j - p_j \|_2$$

The L2 distance between predicted and ground-truth joint positions, averaged over all joints. This is the **standard metric** in 3D human pose estimation (used in Human3.6M, 3DPW, HPS benchmarks). Measures absolute positional accuracy including global alignment.

### 5.2 PA-MPJPE — Procrustes-Aligned MPJPE

$$\text{PA-MPJPE} = \frac{1}{J} \sum_{j=1}^{J} \| R \hat{p}_j + t - p_j \|_2$$

where $R, t$ are obtained by Procrustes analysis (SVD-based alignment) that removes global rotation, translation, and scale. This measures **pose accuracy independent of global positioning** — a model that predicts correct body articulation but wrong global position will have high MPJPE but low PA-MPJPE.

### 5.3 Vertex Error

$$\text{VE} = \frac{1}{V} \sum_{v=1}^{V} \| \hat{v}_i - v_i \|_2$$

Mean L2 error across all 6890 SMPL mesh vertices. Measures body **surface reconstruction quality** — important for applications requiring visual fidelity (e.g., avatar animation, biomechanical surface analysis).

### 5.4 Bone Length Error

$$\text{BLE} = \frac{1}{|B|} \sum_{(i,j) \in B} \left| \| \hat{p}_i - \hat{p}_j \|_2 - \| p_i - p_j \|_2 \right|$$

Difference in bone lengths between predicted and ground-truth skeletons, averaged over lower-body bone pairs (hip-knee, knee-ankle). Measures **anatomical consistency** — a prediction with correct pose but wrong body proportions will have high BLE.

Default bone pairs: left hip-knee (1,4), right hip-knee (2,5), left knee-ankle (4,7), right knee-ankle (5,8).

### 5.5 Inference Time

Per-frame GPU inference latency in milliseconds. Measures **real-time viability** — the insole system streams at 30 Hz (~33 ms/frame budget). Models under this budget are suitable for real-time deployment.

## 6. Training Details

### 6.1 Optimiser and Schedule

| Parameter | Value | Notes |
|-----------|-------|-------|
| Optimiser | Adam | lr=1e-4, weight_decay=1e-4 |
| LR scheduler | StepLR | step=20 epochs, gamma=0.5 (halve LR every 20 epochs) |
| Gradient clipping | 1.0 | RNN models (BiGRU, BiLSTM) only; prevents exploding gradients |
| LR warmup | 5 epochs linear | Transformer model only; stabilises early training |
| Early stopping | Patience 20 | On validation MPJPE; saves best model state |
| Epochs | 80 max | Typically converges in 40-60 epochs |

### 6.2 Data

- **Source:** Walking sequences recorded at 30 Hz with custom pressure insoles + overhead camera
- **Sensor grid:** 33 rows x 15 columns per foot (495 pressure values), 2 feet = shape `(2, 33, 15)` per frame
- **Pressure normalisation:** Global max normalisation across the full sequence (preserves inter-frame and inter-foot magnitude relationships)
- **Label format:** SMPL parameters (PKL) fitted by the physics-aware pipeline (Section 2)
- **Sliding window:** T=32 frames (~1.07 seconds at 30 Hz), stride=1
- **Split:** Train/validation split configured in `configs/default.yaml` (default: walking1-4 train, walking5 val)
- **Batch size:** 16 (for sequence models)

### 6.3 SMPL Configuration

- **Model:** SMPL neutral gender (from `smplx` library)
- **Parameters:** 10 betas + 69 body_pose + 3 global_orient + 3 transl = 85 total
- **Joint output:** 45 joints from SMPL forward kinematics
- **Vertex output:** 6890 vertices, 13776 faces
- **Usage in training loop:** An external `smplx.SMPL` layer runs forward kinematics on both predicted and ground-truth parameters to compute joint/vertex positions for loss computation. The SMPL layer is not part of the model itself — it is used only for loss computation during training and for evaluation.

### 6.4 Configuration File

All hyperparameters are centralised in `configs/default.yaml`:

```yaml
model:
  type: 'cnn_bigru'          # Model architecture
  feature_dim: 512            # Spatial encoder output dimension
  seq_len: 32                 # Temporal window size
  hidden_dim: 256             # RNN hidden size (BiGRU/BiLSTM)
  num_layers: 2               # RNN layers / Transformer layers / TCN blocks

training:
  batch_size: 16
  epochs: 80
  optimizer: { lr: 0.0001, weight_decay: 0.0001 }
  scheduler: { step_size: 20, gamma: 0.5 }
  loss: { lambda_joints: 1.0, lambda_vertices: 0.5, lambda_betas: 0.01, lambda_pose: 0.001 }
  grad_clip: 1.0              # Only applied to RNN models
  patience: 20                # Early stopping patience
  warmup_epochs: 5            # Only applied to Transformer model

dataset:
  pressure_shape: [2, 33, 15] # [channels, height, width]
  train_sequences: ['walking1', 'walking2', 'walking3', 'walking4']
  val_sequences: ['walking5']
```

### 6.5 Reproducibility

- Random seed: 42
- PyTorch deterministic operations where applicable
- All hyperparameters logged in config YAML and saved in checkpoint files
- TensorBoard logging of train/val loss, MPJPE, and learning rate
