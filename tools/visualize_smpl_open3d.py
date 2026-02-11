"""
Interactive SMPL model visualization using Open3D

Features:
- Real-time interactive 3D viewing (rotate, zoom)
- Clear human body mesh rendering
- View single frame or play entire sequence
- Support for exporting images and videos

Usage:
    # View single frame
    python tools/visualize_smpl_open3d.py \
        --smpl_params data/smpl_params/walking1_real_scale.pkl \
        --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
        --frame_idx 5

    # Play sequence animation
    python tools/visualize_smpl_open3d.py \
        --smpl_params data/smpl_params/walking1_real_scale.pkl \
        --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
        --play_sequence \
        --fps 30

    # Save video
    python tools/visualize_smpl_open3d.py \
        --smpl_params data/smpl_params/walking1_real_scale.pkl \
        --smpl_path smpl_models/SMPL_python_v.1.1.0/SMPL_python_v.1.1.0/smpl/models \
        --save_video output/smpl_animation.mp4 \
        --fps 30
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pickle
import torch
import smplx
import open3d as o3d
import argparse
from tqdm import tqdm
import time


class SMPLOpen3DVisualizer:
    """Visualize SMPL model using Open3D"""

    def __init__(self, smpl_model_path, gender='neutral', device='cuda'):
        """
        Initialize the visualizer

        Args:
            smpl_model_path: Path to SMPL model
            gender: 'male', 'female', or 'neutral'
            device: 'cuda' or 'cpu'
        """
        self.device = device

        # Load SMPL model
        print(f"[*] Loading SMPL model ({gender})...")
        self.smpl_model = smplx.SMPL(
            model_path=smpl_model_path,
            gender=gender,
            batch_size=1
        ).to(device)

        print(f"   [OK] SMPL loaded: {self.smpl_model.faces.shape[0]} faces, "
              f"{len(self.smpl_model.v_template)} vertices")

    def smpl_to_mesh(self, betas, body_pose, global_orient, transl, color='skin'):
        """
        Convert SMPL parameters to Open3D mesh

        Args:
            betas: (10,) shape parameters
            body_pose: (23, 3) pose parameters
            global_orient: (3,) global orientation
            transl: (3,) global translation
            color: 'skin', 'blue', 'red', or RGB tuple

        Returns:
            o3d.geometry.TriangleMesh
        """
        # Convert to tensor
        betas_t = torch.tensor(betas, dtype=torch.float32).unsqueeze(0).to(self.device)
        body_pose_t = torch.tensor(body_pose.flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
        global_orient_t = torch.tensor(global_orient, dtype=torch.float32).unsqueeze(0).to(self.device)
        transl_t = torch.tensor(transl, dtype=torch.float32).unsqueeze(0).to(self.device)

        # SMPL forward
        with torch.no_grad():
            output = self.smpl_model(
                betas=betas_t,
                body_pose=body_pose_t,
                global_orient=global_orient_t,
                transl=transl_t
            )

        vertices = output.vertices[0].cpu().numpy()  # (6890, 3)
        faces = self.smpl_model.faces.astype(np.int32)  # (F, 3)

        # Create Open3D mesh
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)

        # Compute normals (for lighting effects)
        mesh.compute_vertex_normals()

        # Set color
        if color == 'skin':
            mesh_color = [0.9, 0.7, 0.6]  # skin tone
        elif color == 'blue':
            mesh_color = [0.3, 0.6, 0.9]
        elif color == 'red':
            mesh_color = [0.9, 0.3, 0.3]
        elif isinstance(color, (list, tuple)):
            mesh_color = color
        else:
            mesh_color = [0.8, 0.8, 0.8]  # gray

        mesh.paint_uniform_color(mesh_color)

        return mesh

    def visualize_single_frame(self, smpl_params, show_joints=True, show_skeleton=False):
        """
        Visualize a single SMPL frame (interactive)

        Args:
            smpl_params: Dict containing betas, body_pose, global_orient, transl
            show_joints: Whether to display joint points
            show_skeleton: Whether to display skeleton connections
        """
        print(f"\n[*] Creating 3D visualization...")

        # Create mesh
        mesh = self.smpl_to_mesh(
            smpl_params['betas'],
            smpl_params['body_pose'],
            smpl_params['global_orient'],
            smpl_params['transl'],
            color='skin'
        )

        # Create list of visualization objects
        geometries = [mesh]

        # Add coordinate frame (helps understand orientation)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        geometries.append(coord_frame)

        # Add joint points (optional)
        if show_joints or show_skeleton:
            # Get joint positions
            betas_t = torch.tensor(smpl_params['betas'], dtype=torch.float32).unsqueeze(0).to(self.device)
            body_pose_t = torch.tensor(smpl_params['body_pose'].flatten(), dtype=torch.float32).unsqueeze(0).to(self.device)
            global_orient_t = torch.tensor(smpl_params['global_orient'], dtype=torch.float32).unsqueeze(0).to(self.device)
            transl_t = torch.tensor(smpl_params['transl'], dtype=torch.float32).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.smpl_model(
                    betas=betas_t,
                    body_pose=body_pose_t,
                    global_orient=global_orient_t,
                    transl=transl_t
                )
            joints = output.joints[0].cpu().numpy()  # (45, 3)

            if show_joints:
                # Display joints as small spheres
                for joint in joints[:24]:  # Only show first 24 joints
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
                    sphere.translate(joint)
                    sphere.paint_uniform_color([1.0, 0.0, 0.0])  # red
                    geometries.append(sphere)

            if show_skeleton:
                # Display skeleton connections
                # SMPL skeleton connection definitions
                skeleton_connections = [
                    (0, 1), (0, 2), (0, 3),  # pelvis -> hips, spine
                    (1, 4), (2, 5),  # hips -> knees
                    (4, 7), (5, 8),  # knees -> ankles
                    (7, 10), (8, 11),  # ankles -> feet
                    (3, 6), (6, 9),  # spine -> chest -> head
                    (9, 12), (9, 13), (9, 14),  # chest -> neck, shoulders
                    (12, 15),  # neck -> head
                    (13, 16), (14, 17),  # shoulders -> elbows
                    (16, 18), (17, 19),  # elbows -> wrists
                    (18, 20), (19, 21),  # wrists -> hands
                    (20, 22), (21, 23),  # hands -> fingers
                ]

                for i, j in skeleton_connections:
                    if i < len(joints) and j < len(joints):
                        points = np.array([joints[i], joints[j]])
                        lines = [[0, 1]]
                        line_set = o3d.geometry.LineSet()
                        line_set.points = o3d.utility.Vector3dVector(points)
                        line_set.lines = o3d.utility.Vector2iVector(lines)
                        line_set.paint_uniform_color([0.0, 0.0, 1.0])  # blue
                        geometries.append(line_set)

        # Create visualization window
        print(f"\n[*] Opening interactive viewer...")
        print(f"   Controls:")
        print(f"     - Left mouse: rotate")
        print(f"     - Right mouse: translate")
        print(f"     - Scroll: zoom")
        print(f"     - Q: quit")

        o3d.visualization.draw_geometries(
            geometries,
            window_name='SMPL Model - Open3D Interactive Viewer',
            width=1280,
            height=720,
            left=50,
            top=50,
            mesh_show_back_face=True
        )

    def visualize_sequence(self, smpl_params_list, fps=30, max_frames=None):
        """
        Play SMPL sequence animation (interactive)

        Args:
            smpl_params_list: List of SMPL parameters
            fps: Playback frame rate
            max_frames: Maximum number of frames (for testing)
        """
        if max_frames is not None:
            smpl_params_list = smpl_params_list[:max_frames]

        print(f"\n[*] Playing SMPL sequence ({len(smpl_params_list)} frames @ {fps} fps)...")

        # Create visualization window
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='SMPL Sequence - Open3D Player',
            width=1280,
            height=720
        )

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)

        # Initialize first frame
        current_mesh = self.smpl_to_mesh(
            smpl_params_list[0]['betas'],
            smpl_params_list[0]['body_pose'],
            smpl_params_list[0]['global_orient'],
            smpl_params_list[0]['transl'],
            color='skin'
        )
        vis.add_geometry(current_mesh)

        # Set render options
        render_option = vis.get_render_option()
        render_option.mesh_show_back_face = True
        render_option.background_color = np.array([0.1, 0.1, 0.1])  # dark gray background

        print(f"\n   Press Q to quit, Space to pause/resume")
        print(f"   Frame: 0/{len(smpl_params_list)}")

        frame_delay = 1.0 / fps
        frame_idx = 0
        paused = False

        # Playback loop
        while True:
            if not paused:
                # Update mesh
                new_mesh = self.smpl_to_mesh(
                    smpl_params_list[frame_idx]['betas'],
                    smpl_params_list[frame_idx]['body_pose'],
                    smpl_params_list[frame_idx]['global_orient'],
                    smpl_params_list[frame_idx]['transl'],
                    color='skin'
                )

                # Update geometry
                current_mesh.vertices = new_mesh.vertices
                current_mesh.triangles = new_mesh.triangles
                current_mesh.vertex_normals = new_mesh.vertex_normals
                current_mesh.vertex_colors = new_mesh.vertex_colors

                vis.update_geometry(current_mesh)

                # Update window title
                vis.get_render_option()
                print(f"\r   Frame: {frame_idx+1}/{len(smpl_params_list)}", end='', flush=True)

                frame_idx = (frame_idx + 1) % len(smpl_params_list)

            # Update display
            if not vis.poll_events():
                break
            vis.update_renderer()

            # Control frame rate
            time.sleep(frame_delay)

        vis.destroy_window()
        print(f"\n\n[OK] Playback finished!")

    def save_video(self, smpl_params_list, output_path, fps=30, max_frames=None,
                   width=1280, height=720):
        """
        Save SMPL sequence as video

        Args:
            smpl_params_list: List of SMPL parameters
            output_path: Output video path (.mp4)
            fps: Frame rate
            max_frames: Maximum number of frames
            width, height: Video resolution
        """
        if max_frames is not None:
            smpl_params_list = smpl_params_list[:max_frames]

        print(f"\n[*] Saving video to {output_path}...")
        print(f"   Frames: {len(smpl_params_list)}, FPS: {fps}, Resolution: {width}x{height}")

        # Create temporary image folder
        temp_dir = Path('temp_frames')
        temp_dir.mkdir(exist_ok=True)

        # Create visualization window (offscreen rendering)
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=width, height=height, visible=False)

        # Add coordinate frame
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        vis.add_geometry(coord_frame)

        # Initialize mesh
        current_mesh = self.smpl_to_mesh(
            smpl_params_list[0]['betas'],
            smpl_params_list[0]['body_pose'],
            smpl_params_list[0]['global_orient'],
            smpl_params_list[0]['transl'],
            color='skin'
        )
        vis.add_geometry(current_mesh)

        # Set camera viewpoint
        ctr = vis.get_view_control()
        ctr.set_zoom(0.7)
        ctr.set_front([0, 0, -1])
        ctr.set_up([0, 1, 0])

        # Render each frame
        frame_paths = []
        for i, params in enumerate(tqdm(smpl_params_list, desc="Rendering frames")):
            # Update mesh
            new_mesh = self.smpl_to_mesh(
                params['betas'],
                params['body_pose'],
                params['global_orient'],
                params['transl'],
                color='skin'
            )

            current_mesh.vertices = new_mesh.vertices
            current_mesh.triangles = new_mesh.triangles
            current_mesh.vertex_normals = new_mesh.vertex_normals
            current_mesh.vertex_colors = new_mesh.vertex_colors

            vis.update_geometry(current_mesh)
            vis.poll_events()
            vis.update_renderer()

            # Save image
            frame_path = temp_dir / f"frame_{i:05d}.png"
            vis.capture_screen_image(str(frame_path), do_render=True)
            frame_paths.append(frame_path)

        vis.destroy_window()

        # Encode video using OpenCV
        print(f"\n   Encoding video...")
        import cv2

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        h, w = first_frame.shape[:2]

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        # Write all frames
        for frame_path in tqdm(frame_paths, desc="Encoding"):
            frame = cv2.imread(str(frame_path))
            out.write(frame)

        out.release()

        # Clean up temporary files
        for frame_path in frame_paths:
            frame_path.unlink()
        temp_dir.rmdir()

        print(f"\n[OK] Video saved to: {output_path}")


def load_smpl_params(pkl_path):
    """Load SMPL parameters"""
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    return results


def main(args):
    """Main function"""
    print("=" * 70)
    print("SMPL Open3D Interactive Visualizer")
    print("=" * 70)

    # Initialize visualizer
    visualizer = SMPLOpen3DVisualizer(
        smpl_model_path=args.smpl_path,
        gender=args.gender,
        device=args.device
    )

    # Load SMPL parameters
    print(f"\n[*] Loading SMPL params from: {args.smpl_params}")
    smpl_params_list = load_smpl_params(args.smpl_params)
    print(f"   [OK] Loaded {len(smpl_params_list)} frames")

    # Slice start_frame / end_frame range
    start = args.start_frame
    end = args.end_frame if args.end_frame is not None else len(smpl_params_list)
    end = min(end, len(smpl_params_list))
    if start > 0 or end < len(smpl_params_list):
        smpl_params_list = smpl_params_list[start:end]
        print(f"   [OK] Using frames {start}–{end-1} ({len(smpl_params_list)} frames)")

    if args.save_video:
        # Save video mode
        visualizer.save_video(
            smpl_params_list,
            output_path=args.save_video,
            fps=args.fps,
            max_frames=args.max_frames,
            width=args.width,
            height=args.height
        )

    elif args.play_sequence:
        # Play sequence mode
        visualizer.visualize_sequence(
            smpl_params_list,
            fps=args.fps,
            max_frames=args.max_frames
        )

    else:
        # Single frame mode
        if args.frame_idx >= len(smpl_params_list):
            print(f"[ERROR] Error: frame_idx {args.frame_idx} >= {len(smpl_params_list)}")
            return

        smpl_params = smpl_params_list[args.frame_idx]
        print(f"\n[*] Frame {args.frame_idx} info:")
        print(f"   Loss: {smpl_params.get('loss', 'N/A')}")
        if 'mean_error_mm' in smpl_params:
            print(f"   Error: {smpl_params['mean_error_mm']:.1f} mm")

        visualizer.visualize_single_frame(
            smpl_params,
            show_joints=args.show_joints,
            show_skeleton=args.show_skeleton
        )

    print("\n" + "=" * 70)
    print("[OK] Done!")
    print("=" * 70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SMPL Open3D Visualization')

    # Input parameters
    parser.add_argument('--smpl_params', type=str, required=True,
                        help='Path to SMPL params pickle file')
    parser.add_argument('--smpl_path', type=str, required=True,
                        help='Path to SMPL model folder')
    parser.add_argument('--gender', type=str, default='neutral',
                        choices=['male', 'female', 'neutral'])
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])

    # Single frame mode parameters
    parser.add_argument('--frame_idx', type=int, default=0,
                        help='Frame index to visualize (single frame mode)')
    parser.add_argument('--show_joints', action='store_true',
                        help='Show joint spheres')
    parser.add_argument('--show_skeleton', action='store_true',
                        help='Show skeleton connections')

    # Sequence mode parameters
    parser.add_argument('--play_sequence', action='store_true',
                        help='Play the entire sequence')
    parser.add_argument('--fps', type=int, default=30,
                        help='Playback FPS')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='Maximum frames to process')
    parser.add_argument('--start_frame', type=int, default=0,
                        help='Start frame index (inclusive)')
    parser.add_argument('--end_frame', type=int, default=None,
                        help='End frame index (exclusive, default: last frame)')

    # Video saving parameters
    parser.add_argument('--save_video', type=str, default=None,
                        help='Save sequence as video (e.g., output.mp4)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Video width')
    parser.add_argument('--height', type=int, default=720,
                        help='Video height')

    args = parser.parse_args()

    main(args)
