import socket
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from threading import Thread
import math

# UDP Configuration
UDP_IP = "192.168.137.1"  # IP address to bind to
UDP_PORT = 8999           # Port to listen on

# Payload: 254 sensor readings + 6 IMU values = 260 uint16 values
PAYLOAD_SIZE = 254 + 6
BUFFER_SIZE = PAYLOAD_SIZE * 2  # bytes

# Matrix dimensions
MATRIX_ROW = 33
MATRIX_COLUMN = 15

# Lookup pattern and table (as before)
image_pattern = np.array([
    [0,0,0,0,0,0,1,1,1,1,0,0,0,0,0],
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
    [0,0,0,0,0,1,1,1,1,1,1,0,0,0,0],
    [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0],
])
lookup_table = [
    16,32,41,59,70,81,91,101,111,121,130,139,140,147,154,161,162,168,174,180,192,199,200,207,214,221,228,235,236,242,248,249,
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

# IMU indices in the payload
IDX_FOOT = 253
IDX_AX   = 254
IDX_AY   = 255
IDX_AZ   = 256
IDX_GX   = 257
IDX_GY   = 258
IDX_GZ   = 259

# Initialize data matrices
data_matrix_0 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)
data_matrix_1 = np.zeros((MATRIX_ROW, MATRIX_COLUMN), dtype=int)

# Setup plot
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))
heatmap_0 = ax0.imshow(data_matrix_0, cmap='viridis', interpolation='nearest', vmin=0, vmax=4000)
heatmap_1 = ax1.imshow(data_matrix_1, cmap='viridis', interpolation='nearest', vmin=0, vmax=1500)
fig.colorbar(heatmap_0, ax=ax0)
fig.colorbar(heatmap_1, ax=ax1)
ax0.set_title("Left Insole")
ax1.set_title("Right Insole")

def update_heatmap(frame):
    heatmap_0.set_data(data_matrix_0)
    heatmap_1.set_data(data_matrix_1)
    return heatmap_0, heatmap_1

def receive_data():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    print(f"Listening for UDP packets on {UDP_IP}:{UDP_PORT}")

    try:
        while True:
            data, addr = sock.recvfrom(BUFFER_SIZE)
            payload = np.frombuffer(data, dtype=np.uint16)

            # Extract foot side and sensor readings
            client  = int(payload[IDX_FOOT])
            sensors = payload[:IDX_FOOT]

            # Extract IMU values
            ax = int(payload[IDX_AX])
            ay = int(payload[IDX_AY])
            az = int(payload[IDX_AZ])
            gx = int(payload[IDX_GX])
            gy = int(payload[IDX_GY])
            gz = int(payload[IDX_GZ])
            print(f"[Client {client}] IMU: AX={ax} AY={ay} AZ={az}  GX={gx} GY={gy} GZ={gz}")

            # Map sensor data via lookup table
            sorted_sensors = np.zeros_like(sensors, dtype=int)
            for i, idx in enumerate(lookup_table):
                sorted_sensors[idx] = int(sensors[i])

            # Fill matrix according to pattern
            indices = np.argwhere(image_pattern == 1)
            if client == 0:  # left
                for (i, j), v in zip(indices, sorted_sensors):
                    data_matrix_0[i, j] = v if v > 100 else 0
            else:           # right
                for (i, j), v in zip(indices, sorted_sensors):
                    data_matrix_1[i, MATRIX_COLUMN-1-j] = v if v > 200 else 0

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        sock.close()

if __name__ == "__main__":
    # Start UDP listener thread
    t = Thread(target=receive_data, daemon=True)
    t.start()

    # Start animation
    ani = animation.FuncAnimation(fig, update_heatmap, interval=20, blit=True, cache_frame_data=False)
    plt.show()
