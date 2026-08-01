```
python3 stereo_calibration_capture.py \
    --left-sensor 0 \
    --right-sensor 1 \
    --board-size 9x6 \
    --target-pairs 35
```

```
python3 verify_stereo_calibration.py \
    --calibration stereo_calibration.npz \
    --left-dir calibration_images/left \
    --right-dir calibration_images/right \
    --board-size 9x6 \
    --expected-baseline 60
```