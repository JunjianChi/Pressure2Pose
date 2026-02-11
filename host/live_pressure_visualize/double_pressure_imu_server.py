import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# ==============================
# Network configuration
# ==============================
UDP_IP = "0.0.0.0"
UDP_PORT = 12345

PAYLOAD_SIZE = 513          # 253 + 253 + 6 + 1
BUFFER_SIZE = PAYLOAD_SIZE * 2  # uint16 -> 2 bytes

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

image_pattern = [[0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,1,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,0,0,0,0,0],
                [0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
                [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
                ]
image_pattern = np.array(image_pattern)

lookup_table = [16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
                4,24,33,50,60,71,82,92,102,112,122,131,132,141,148,155,156,163,169,175,186,193,194,201,208,215,222,229,230,237,243,244,
                10,17,25,42,51,61,72,83,93,103,113,123,124,133,142,149,150,157,164,170,181,187,188,195,202,209,216,223,224,231,238,250,
                5,11,18,34,43,52,62,73,84,94,104,114,115,125,134,143,144,151,158,165,176,182,183,189,196,203,210,217,218,225,232,251,
                0,6,12,26,35,44,53,63,74,85,95,105,106,116,126,135,136,145,152,159,171,177,178,184,190,197,204,211,212,219,226,245,
                1,2,7,19,27,36,45,54,64,75,86,96,97,107,117,127,128,137,146,153,166,172,173,179,185,191,198,205,206,213,220,239,
                3,13,20,28,37,46,55,65,76,87,88,98,108,118,119,129,138,160,167,227,233,
                8,14,21,29,38,47,56,66,77,78,89,99,109,110,120,234,240,
                9,15,22,30,39,48,57,67,68,79,90,100,241,246,
                23,31,40,49,58,69,80,247,252
                ]

fig, (ax0, ax1) = plt.subplots(1, 2)
heatmap_0 = ax0.imshow(np.zeros((33, 15)), cmap='inferno', interpolation='nearest', vmin=0, vmax=3500)
heatmap_1 = ax1.imshow(np.zeros((33, 15)), cmap='inferno', interpolation='nearest', vmin=0, vmax=3500)
plt.colorbar(heatmap_0, ax=ax0)
plt.colorbar(heatmap_1, ax=ax1)
ax0.set_title("Left Insole")
ax1.set_title("Right Insole")

data_matrix_0 = np.zeros((33, 15), dtype=int)
data_matrix_1 = np.zeros((33, 15), dtype=int)

# ==============================
# Main update loop
# ==============================
def update(frame):
    data, addr = sock.recvfrom(BUFFER_SIZE)

    # Decode payload
    payload = np.frombuffer(data, dtype=np.uint16)

    if payload.size != PAYLOAD_SIZE:
        print("Invalid packet size:", payload.size)
        return heatmap_0, heatmap_1

    frame1 = payload[0:253]
    frame2 = payload[253:506]

    # Extract IMU values (6 values)
    imu = payload[506:512]
    ax_ = int(imu[0])
    ay_ = int(imu[1])
    az_ = int(imu[2])
    gx_ = int(imu[3])
    gy_ = int(imu[4])
    gz_ = int(imu[5])

    # Extract client ID
    client = int(payload[512])
    
    print(f"[Client {client}] IMU: AX={ax_} AY={ay_} AZ={az_}  GX={gx_} GY={gy_} GZ={gz_}")

    # Map sensor data via lookup table for left (frame1)
    sorted_array1 = np.zeros(len(frame1), dtype=int)
    for i, index in enumerate(lookup_table):
        sorted_array1[index] = int(frame1[i])

    # Map sensor data via lookup table for right (frame2)
    sorted_array2 = np.zeros(len(frame2), dtype=int)
    for i, index in enumerate(lookup_table):
        sorted_array2[index] = int(frame2[i])

    # Combine for left insole
    combined_left = sorted_array1
    indices = np.argwhere(image_pattern == 1)
    for (i, j), value in zip(indices, combined_left):
        if value > 0:
            data_matrix_0[i, j] = value
        else:
            data_matrix_0[i, j] = 0

    # Combine for right insole (mirror)
    combined_right = sorted_array2
    for (i, j), value in zip(indices, combined_right):
        if value > 0:
            data_matrix_1[i, 14-j] = value
        else:
            data_matrix_1[i, 14-j] = 0

    heatmap_0.set_data(data_matrix_0)
    heatmap_1.set_data(data_matrix_1)

    return heatmap_0, heatmap_1

ani = animation.FuncAnimation(fig, update, interval=1)
plt.show()
