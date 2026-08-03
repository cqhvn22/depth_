# IMU

## Overview

This folder contains the driver and demo application for the **ICM-20948** 9-DOF IMU used alongside the IMX219-83 stereo camera on the Jetson Orin Nano. It provides orientation (roll/pitch/yaw), accelerometer, gyroscope, and magnetometer readings over I2C, printed to the terminal. This is a standalone C program with no ROS or camera dependency — it's used to verify the IMU wiring/readout independently, and the same driver source is reused in the ROS 2 `stereo_camera` package's `icm20948_node`.

**This code is not original to this repository.** It is the ICM-20948 demo provided by Waveshare for their 9-DOF IMU module, included here unmodified so it can be built and run directly on the Jetson Orin Nano. Source:

```bash
wget https://files.waveshare.com/upload/e/eb/D219-9dof.tar.gz
tar zxvf D219-9dof.tar.gz
cd D219-9dof/07-icm20948-demo
make
./ICM20948-Demo
```

## Structure

```text
IMU/
├── ICM20948.c          # Waveshare ICM-20948 driver (I2C register access, sensor fusion)
├── ICM20948.h          # Driver header (types, function declarations)
├── main.c              # Demo entry point: reads and prints angles/accel/gyro/mag in a loop
├── Makefile            # Builds ICM20948_Demo from main.c + ICM20948.c
├── ICM20948_Demo        # Pre-built binary
├── ICM20948.o / main.o  # Pre-built object files
```

## Installation / Dependencies

Runs on the Jetson Orin Nano with the ICM-20948 wired over I2C — see `../Pinout.png` for the connection reference.

```bash
sudo apt update
sudo apt install build-essential
# I2C must be enabled/accessible (e.g. the user has access to /dev/i2c-*)
```

No external libraries beyond the standard C library and `libm` are required (see `Makefile`: `gcc ... -lm -std=gnu99`).

## Running

### Build

```bash
cd IMU
make
```

This produces `ICM20948_Demo` from `main.c` and `ICM20948.c`.

### Run

```bash
./ICM20948_Demo
```

Continuously prints (every 100 ms):

- Angles (Roll / Pitch / Yaw, degrees)
- Acceleration (X/Y/Z, g)
- Gyroscope (X/Y/Z, dps)
- Magnetic field (X/Y/Z, µT)

On startup it also prints whether an ICM-20948 was detected.

### Clean

```bash
make clean
```

