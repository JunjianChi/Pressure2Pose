import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread, Lock
import csv
import time
import pyrealsense2 as rs
import cv2
import mediapipe as mp
import os

# UDP Configuration for pressure data
UDP_IP = "192.168.137.1"
UDP_PORT = 8999
BUFFER_SIZE = 508

MATRIX_ROW = 33
MATRIX_COLUMN = 15

# Mutex for synchronizing access to pressure data
data_lock = Lock()

# Create lookup table and image pattern for pressure mapping

image_pattern = np.array([
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
])

lookup_table = [
    16, 32, 41, 59, 70, 81, 91, 101, 111, 121, 130, 139, 140, 147, 154, 161, 162, 168, 174, 180,
    192, 199, 200, 207, 214, 221, 228, 235, 236, 242, 248, 249, 4, 24, 33, 50, 60, 71, 82, 92, 102,
    112, 122, 131, 132, 141, 148, 155, 156, 163, 169, 175, 186, 193, 194, 201, 208, 215, 222, 229,
    230, 237, 243, 244, 10, 17, 25, 42, 51, 61, 72, 83, 93, 103, 113, 123, 124, 133, 142, 149, 150,
    157, 164, 170, 181, 187, 188, 195, 202, 209, 216, 223, 224, 231, 238, 250, 5, 11, 18, 34, 43,
    52, 62, 73, 84, 94, 104, 114, 115, 125, 134, 143, 144, 151, 158, 165, 176, 182, 183, 189, 196,
    203, 210, 217, 218, 225, 232, 251, 0, 6, 12, 26, 35, 44, 53, 63, 74, 85, 95, 105, 106, 116, 126,
    135, 136, 145, 152, 159, 171, 177, 178, 184, 190, 197, 204, 211, 212, 219, 226, 245, 1, 2, 7,
    19, 27, 36, 45, 54, 64, 75, 86, 96, 97, 107, 117, 127, 128, 137, 146, 153, 166, 172, 173, 179,
    185, 191, 198, 205, 206, 213, 220, 239, 3, 13, 20, 28, 37, 46, 55, 65, 76, 87, 88, 98, 108, 118,
    119, 129, 138, 160, 167, 227, 233, 8, 14, 21, 29, 38, 47, 56, 66, 77, 78, 89, 99, 109, 110, 120,
    234, 240, 9, 15, 22, 30, 39, 48, 57, 67, 68, 79, 90, 100, 241, 246, 23, 31, 40, 49, 58, 69, 80,
    247, 252
]


# Camera and MediaPipe configuration
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Filters for depth processing
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()

# Align depth frame to color frame
align_to = rs.stream.color
align = rs.align(align_to)

# Initialize MediaPipe Pose module
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Keypoint dictionary for pose detection
keypoints = {
    'head': mp_pose.PoseLandmark.NOSE,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER,
    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW,
    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE,
    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL,
    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL,
    'left_foot_index': mp_pose.PoseLandmark.LEFT_FOOT_INDEX,
    'right_foot_index': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX
}

# Initialize heatmaps for pressure data
data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)

# Plotting setup
fig, (ax0, ax1) = plt.subplots(1, 2)
heatmap_0 = ax0.imshow(data_matrix_0, cmap='viridis', interpolation='nearest', vmin=0, vmax=2000)
heatmap_1 = ax1.imshow(data_matrix_1, cmap='viridis', interpolation='nearest', vmin=0, vmax=2000)
plt.colorbar(heatmap_0, ax=ax0)
plt.colorbar(heatmap_1, ax=ax1)

# Folder path for storing CSV
folder_path = "dataset/standing"
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# CSV file path
csv_file_path = os.path.join(folder_path, 'standing_hanson_3.csv')
csv_file = open(csv_file_path, mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(
    ['Timestamp', 'Frame'] + [f'{kp}_{axis}' for kp in keypoints for axis in ['x', 'y', 'z']] + ['Matrix_0', 'Matrix_1'])

frame_number = 0
timestamps_and_matrices = []


def get_depth_intrinsics(depth_frame):
    """ Get depth frame intrinsics for 3D point calculation """
    return depth_frame.profile.as_video_stream_profile().intrinsics


def convert_to_3d(depth_frame, intrinsics, x, y):
    """ Convert 2D keypoint coordinates to 3D using depth data """
    depth = depth_frame.get_distance(x, y)
    if depth == 0:
        return None
    return rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)


def receive_data():
    """ Receive pressure data in a separate thread """
    global data_matrix_0, data_matrix_1, timestamps_and_matrices
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            timestamp = time.time()

            decimal_array = np.frombuffer(data, dtype=np.uint16).reshape(254)
            client = decimal_array[-1]
            decimal_array = decimal_array[:-1]

            sorted_one_d_array = np.zeros(len(decimal_array), dtype=int)
            for i, index in enumerate(lookup_table):
                sorted_one_d_array[index] = decimal_array[i]

            indices = np.argwhere(image_pattern == 1)

            with data_lock:
                if client == 0:
                    for (i, j), value in zip(indices, sorted_one_d_array):
                        data_matrix_0[i, j] = value if value > 500 else 0
                elif client == 1:
                    for (i, j), value in zip(indices, sorted_one_d_array):
                        data_matrix_1[i, 14 - j] = value if value > 500 else 0
                # Save the current matrices with a timestamp for synchronization
                timestamps_and_matrices.append((timestamp, np.copy(data_matrix_0), np.copy(data_matrix_1)))

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()


def update_heatmap_and_pose(frame):
    """ Update heatmaps and pose data in a separate thread """
    global frame_number, timestamps_and_matrices

    timestamp = time.time()

    # RealSense frame handling
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    if not aligned_depth_frame or not color_frame:
        return

    color_image = np.asanyarray(color_frame.get_data())
    rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb_image)
    row = [timestamp, frame_number]

    if results.pose_landmarks:
        depth_intrinsics = get_depth_intrinsics(aligned_depth_frame)

        for keypoint_name, landmark_index in keypoints.items():
            landmark = results.pose_landmarks.landmark[landmark_index]
            x, y = int(landmark.x * color_image.shape[1]), int(landmark.y * color_image.shape[0])

            if 0 <= x < color_image.shape[1] and 0 <= y < color_image.shape[0]:
                point_3d = convert_to_3d(aligned_depth_frame, depth_intrinsics, x, y)
                if point_3d:
                    row.extend(point_3d)
                else:
                    row.extend([0, 0, 0])
            else:
                row.extend([0, 0, 0])

            cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
    else:
        for keypoint_name in keypoints:
            row.extend([0, 0, 0])

    # Find the closest matching pressure data based on the timestamp
    with data_lock:
        closest_matrix = min(timestamps_and_matrices, key=lambda x: abs(x[0] - timestamp))
        timestamps_and_matrices.remove(closest_matrix)

    # Update heatmaps for pressure data
    data_matrix_0, data_matrix_1 = closest_matrix[1], closest_matrix[2]
    heatmap_0.set_data(data_matrix_0)
    heatmap_1.set_data(data_matrix_1)

    # Add heatmap data to CSV row
    matrix_0_flat = data_matrix_0.flatten()
    matrix_1_flat = data_matrix_1.flatten()
    row.append(','.join(map(str, matrix_0_flat)))
    row.append(','.join(map(str, matrix_1_flat)))

    # Write to CSV
    csv_writer.writerow(row)
    frame_number += 1

    # Show color image and heatmap
    cv2.imshow('Pose Detection with Depth', color_image)
    return heatmap_0, heatmap_1


# Start receiving pressure data
data_thread = Thread(target=receive_data)
data_thread.daemon = True
data_thread.start()

# Real-time animation and pose detection loop
ani = animation.FuncAnimation(fig, update_heatmap_and_pose, interval=20, blit=True, cache_frame_data=False)

# Display plots and video
plt.show()

# Close resources after execution
csv_file.close()
pipeline.stop()
cv2.destroyAllWindows()
