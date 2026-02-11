"""
Neural network models for Pressure → SMPL parameter prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import smplx


class PressureEncoder(nn.Module):
    """CNN encoder for pressure matrix features"""

    def __init__(self, in_channels=2, feature_dim=512):
        super().__init__()

        # Simple CNN backbone
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Global average pooling + FC
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Args:
            x: (B, 2, H, W) pressure matrices
        Returns:
            features: (B, feature_dim)
        """
        x = self.conv1(x)  # (B, 32, H/2, W/2)
        x = self.conv2(x)  # (B, 64, H/4, W/4)
        x = self.conv3(x)  # (B, 128, H/8, W/8)
        x = self.conv4(x)  # (B, 256, H/16, W/16)
        x = self.fc(x)     # (B, feature_dim)
        return x


class SMPLRegressor(nn.Module):
    """MLP heads to predict SMPL parameters"""

    def __init__(self, feature_dim=512, num_betas=10, num_joints=23):
        super().__init__()

        self.num_betas = num_betas
        self.num_joints = num_joints

        # Shape head (body shape parameters)
        self.shape_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_betas)
        )

        # Pose head (joint rotations in axis-angle)
        self.pose_head = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_joints * 3)
        )

        # Global orientation head
        self.orient_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        # Translation head
        self.transl_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    def forward(self, features):
        """
        Args:
            features: (B, feature_dim)
        Returns:
            betas: (B, num_betas)
            body_pose: (B, num_joints * 3)
            global_orient: (B, 3)
            transl: (B, 3)
        """
        betas = self.shape_head(features)
        body_pose = self.pose_head(features)
        global_orient = self.orient_head(features)
        transl = self.transl_head(features)

        return betas, body_pose, global_orient, transl


class PressureToSMPL(nn.Module):
    """Complete end-to-end model: Pressure → SMPL"""

    def __init__(self, smpl_model_path, gender='neutral',
                 feature_dim=512, num_betas=10, device='cuda'):
        super().__init__()

        self.device = device

        # Encoder
        self.encoder = PressureEncoder(in_channels=2, feature_dim=feature_dim)

        # Regressor
        self.regressor = SMPLRegressor(feature_dim, num_betas)

        # SMPL layer (for differentiable forward pass)
        self.smpl_layer = smplx.SMPL(
            model_path=smpl_model_path,
            gender=gender,
            batch_size=1,
            create_transl=False
        ).to(device)

    def forward(self, pressure, return_smpl_output=False):
        """
        Args:
            pressure: (B, 2, H, W) pressure matrices
            return_smpl_output: If True, run SMPL forward pass

        Returns:
            If return_smpl_output=False:
                betas, body_pose, global_orient, transl
            If return_smpl_output=True:
                smpl_output (with vertices and joints)
        """
        # Extract features
        features = self.encoder(pressure)

        # Predict SMPL parameters
        betas, body_pose, global_orient, transl = self.regressor(features)

        if not return_smpl_output:
            return betas, body_pose, global_orient, transl

        # Run SMPL forward pass
        batch_size = pressure.shape[0]
        self.smpl_layer.batch_size = batch_size

        smpl_output = self.smpl_layer(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl
        )

        return smpl_output

    def predict_mesh(self, pressure):
        """
        Convenient method for inference

        Args:
            pressure: (B, 2, H, W) or (2, H, W)

        Returns:
            vertices: (B, 6890, 3) or (6890, 3)
            joints: (B, 24, 3) or (24, 3)
        """
        single_sample = False
        if pressure.dim() == 3:
            pressure = pressure.unsqueeze(0)
            single_sample = True

        with torch.no_grad():
            smpl_output = self.forward(pressure, return_smpl_output=True)

        vertices = smpl_output.vertices
        joints = smpl_output.joints

        if single_sample:
            vertices = vertices[0]
            joints = joints[0]

        return vertices, joints


class SMPLLoss(nn.Module):
    """Multi-component loss for SMPL training"""

    def __init__(self, lambda_joints=1.0, lambda_betas=0.01,
                 lambda_pose=0.001, lambda_vertices=0.5):
        super().__init__()

        self.lambda_joints = lambda_joints
        self.lambda_betas = lambda_betas
        self.lambda_pose = lambda_pose
        self.lambda_vertices = lambda_vertices

    def forward(self, pred_smpl_output, target_smpl_params,
                pred_params=None):
        """
        Args:
            pred_smpl_output: SMPL output from model
            target_smpl_params: Ground truth SMPL parameters dict
            pred_params: Predicted SMPL parameters (for regularization)

        Returns:
            loss_dict: Dictionary of individual losses
            total_loss: Weighted sum of all losses
        """
        losses = {}

        # Joint loss (primary supervision)
        if 'joints' in target_smpl_params:
            pred_joints = pred_smpl_output.joints
            target_joints = target_smpl_params['joints']
            losses['joints'] = F.mse_loss(pred_joints, target_joints)

        # Vertex loss (if available)
        if 'vertices' in target_smpl_params:
            pred_vertices = pred_smpl_output.vertices
            target_vertices = target_smpl_params['vertices']
            losses['vertices'] = F.mse_loss(pred_vertices, target_vertices)

        # Parameter regularization
        if pred_params is not None:
            betas, body_pose, global_orient, transl = pred_params

            # Shape regularization (encourage average body shape)
            losses['betas'] = torch.mean(betas ** 2)

            # Pose regularization (encourage neutral pose)
            losses['pose'] = torch.mean(body_pose ** 2)

        # Compute total loss
        total_loss = 0.0
        if 'joints' in losses:
            total_loss += self.lambda_joints * losses['joints']
        if 'vertices' in losses:
            total_loss += self.lambda_vertices * losses['vertices']
        if 'betas' in losses:
            total_loss += self.lambda_betas * losses['betas']
        if 'pose' in losses:
            total_loss += self.lambda_pose * losses['pose']

        losses['total'] = total_loss

        return losses, total_loss


def test_model():
    """Test model architecture"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Dummy SMPL path (replace with actual path)
    smpl_path = 'path/to/smpl/models'

    # Create model
    model = PressureToSMPL(
        smpl_model_path=smpl_path,
        gender='neutral',
        device=device
    ).to(device)

    # Test input
    batch_size = 4
    pressure = torch.randn(batch_size, 2, 24, 20).to(device)

    # Forward pass (parameters only)
    betas, body_pose, global_orient, transl = model(pressure)
    print(f"Betas: {betas.shape}")
    print(f"Body pose: {body_pose.shape}")
    print(f"Global orient: {global_orient.shape}")
    print(f"Translation: {transl.shape}")

    # Forward pass (with SMPL)
    smpl_output = model(pressure, return_smpl_output=True)
    print(f"\nVertices: {smpl_output.vertices.shape}")
    print(f"Joints: {smpl_output.joints.shape}")

    # Test loss
    criterion = SMPLLoss()
    target_params = {
        'joints': torch.randn_like(smpl_output.joints)
    }
    losses, total_loss = criterion(
        smpl_output, target_params,
        pred_params=(betas, body_pose, global_orient, transl)
    )
    print(f"\nLosses: {losses}")


if __name__ == '__main__':
    test_model()
