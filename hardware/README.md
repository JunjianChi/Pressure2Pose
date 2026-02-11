# Hardware Design Files

This directory contains PCB design files (Altium Designer), production Gerbers, and laser-cut patterns for the plantar pressure insole system.

For detailed fabrication procedures, refer to our ISCAS 2025 paper: [High-Resolution Plantar Pressure Insole System for Enhanced Lower Body Biomechanical Analysis](https://ieeexplore.ieee.org/abstract/document/11044303).

## Directory Structure

### [`insole_fpc_ver1`](insole_fpc_ver1/)

First-generation FPC (flexible printed circuit) sensor array. This version implements a 32x10 electrode-only crossbar connection without isolation diodes.

| File | Description |
|------|-------------|
| `insole_ver1.SchDoc` | Altium schematic |
| `insole_ver1.PcbDoc` | Altium PCB layout |

### [`insole_fpc_ver2`](insole_fpc_ver2/)

Second-generation FPC sensor array. This version adds isolation diodes to each sensing node to eliminate crossbar crosstalk, improving measurement accuracy.

| File | Description |
|------|-------------|
| `insole_ver2.SchDoc` | Altium schematic |
| `insole_ver2.PcbDoc` | Altium PCB layout |
| `insole_ver2_gerber.zip` | Production Gerber files for manufacturing |

### [`sensor_laser_cut`](sensor_laser_cut/)

Laser-cut pattern for the piezoresistive sensor layer. The material used is Velostat (conductive polymer film). The pattern can be imported into a laser cutter to produce circular sensing dots that match the insole electrode array geometry.

| File | Description |
|------|-------------|
| `insole_42.dip` | Laser-cut pattern file |

### [`mcu_pcb`](mcu_pcb/)

MCU main board based on ESP32-S3, integrating analog MUX, TIA readout circuit, DAC excitation, and 6-axis IMU (ICM-45686).

| File | Description |
|------|-------------|
| `insole_mcu.SchDoc` | Altium schematic |
| `insole_mcu.PcbDoc` | Altium PCB layout |
| `insole_mcu_gerber.zip` | Production Gerber files for manufacturing |
| `insole_mcu_schematic.pdf` | Schematic PDF export for quick reference |
