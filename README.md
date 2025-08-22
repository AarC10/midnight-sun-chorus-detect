## 
# Generating Numpy Files for Detection Script
The first argument is the audio file you want detected
The second argument is the output numpy file to create
```
python3 src/chorus_bin_gen.py SAN_Chorus.wav out.npy
```

## Live Detection
Run live detection (no UART broadcast):
```
python3 src/detect_chorus.py out.npy
```

With UART control using ttyUSB0 as an example:
```
python3 src/detect_chorus.py out.npy /dev/ttyUSB0
```

## File Analysis
Analyze a recorded file for detection:
```
python3 src/detect_chorus.py SAN.wav out.npy
```

### List Audio Devices
```
python3 src/detect_chorus.py --list-devices
```

## Configuration

Control detection behavior with environment variables:
You can specify these before the python3 command to quickly iterate

### Detection Settings
```
CHORUS_THRESHOLD=0.55           # Detection sensitivity (0.0-1.0)
CHORUS_INPUT_DEVICE=5           # Audio input device ID
CHORUS_METHOD=dtw               # Detection method: 'dtw' or 'hyst'
```

### Timing Controls
```
CHORUS_LIVE_COOLDOWN_SEC=5.0    # Seconds between detections
CHORUS_MIN_RUN_SEC=1.0          # Minimum detection duration
CHORUS_MAX_GAP_SEC=0.25         # Max gap in detection
```

### Signal Processing
```
CHORUS_SMOOTH_SEC=2.0           # Rolling average window (0 disables)
CHORUS_TEMPLATE_SR=22050        # Template sample rate
CHORUS_ALIGN_TOL_SEC=0.23       # Frame alignment tolerance
CHORUS_START_THRESHOLD=0.55     # Initial detection threshold
CHORUS_CONT_THRESHOLD=0.47      # Continuation threshold
CHORUS_SEGMENT_SIGNAL=raw       # Segmentation signal: 'raw' or 'smooth'
```

You will most likely only care about the detection settings and timing controls.
I have mostly just needed to mess with the threshold setting and have been running this command, changing the threshold

```
CHORUS_THRESHOLD=0.45 python3 src/detect_chorus.py out.npy
```
