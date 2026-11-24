# Standalone webcam gaze tracker (legacy)

> **Superseded by `gaze_engine/`** (see the main README, "Eye tracking: the gaze engine").
> This tracker maps iris-offset features to pixels with no head pose, so head movement after
> calibration shows up as gaze error (about 400 px median measured). It is kept for reference and
> as the browser's offline "legacy" fallback; do not use it for study data.

This is an experimental prototype, independent of the poster app, FastAPI,
React, YOLO and poster AOIs. Camera frames are processed on-device; MediaPipe's
published statement says input images/video are not sent to Google, but the
Tasks package does send performance and utilization metrics. This app does not
send gaze CSVs or predictions. It cannot promise zero outbound traffic because
of MediaPipe's metrics behavior. Landmark inference uses the MediaPipe Tasks
Face Landmarker API; it does not use legacy `mp.solutions.face_mesh`.

## Install on macOS

MediaPipe's published Python classifiers currently list Python 3.9–3.12. The
project machine's Python 3.14 is outside that declared range. Python 3.11 is
available on this Mac and works with the tracker requirements. Use it in a
virtual environment; do not install into the system interpreter.

```sh
cd custom_eye_tracker
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Download the official [Face Landmarker model asset](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)
while in this folder:

```sh
mkdir -p models
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" -o models/face_landmarker.task
cd ..
```

The model asset is present in this working copy at `models/face_landmarker.task`.
If it is missing from another checkout, run the download command above; do not
start the tracker until the file exists at that path.

From the repository root, first inspect the live face/eye status and frame rate:

```sh
python -m custom_eye_tracker.main --diagnostics
```

Press `q` to close diagnostics. Then start a fresh calibration run:

```sh
python -m custom_eye_tracker.main
```

Select a camera with `--camera 1`; change the output folder with `--output records`.
On first launch, macOS asks Terminal (or the terminal app you use) for camera permission.
If permission was denied, enable it in **System Settings → Privacy & Security →
Camera**, quit and reopen the terminal, then retry. A probe from this tool-run
process was denied by macOS AVFoundation, so webcam access from your usual
Terminal remains unverified. Camera index 0 produced a permission error in the
probe; check Terminal's permission or try `--camera 1` for another device.

## Use

The app opens a full-screen camera preview. For each target, keep your head in a
comfortable, fixed position and look at the dot. It waits 0.7 seconds, then
collects valid frames for 2 seconds (at least 8 valid frames). Space skips a
target; `q` quits. After fitting and validation, it runs the final held-out
targets. Press `r` to start/stop CSV recording in the live view; press `q` to
quit. Invalid/blink frames are not predicted and are marked invalid in CSV.

The video preview and landmark calculations are local. The tracker uses the
macOS primary/default display selected at launch; choose another primary
display in macOS Display settings before starting if needed. Restart calibration
after changing display, resolution or scale. Targets and gaze outputs use
pixels of the OpenCV full-screen drawable area, recorded in model metadata.
The view is resampled to that drawable size; the coordinates are not CSS pixels
or poster-image coordinates. OpenCV/macOS window backends can expose logical or
backing pixels differently on Retina/external displays. Do not resize or
reconfigure the display during a run. Check the saved
`screen.width_px`/`height_px` against the display and confirm the dot is at the
reported target coordinate before relying on pixel errors. Changing display,
resolution, camera or feature setup requires a fresh calibration; this program
always makes a new calibration and never silently loads a previous one.

## Calibration and accuracy protocol

Each fresh run presents 25 target locations split by location: 15 training, 5
model-selection validation, and 5 final-test positions. The new coordinate grid
is offset from the initial pilot run so that its four already-seen final-test
locations are not reused for this next evaluation. Training includes both
horizontal screen edges; validation covers all quadrants plus the upper center;
the final test uses four corners and screen center. Frames collected at one
position are never divided between sets. Per-target feature summaries use the
median of the valid repeated frames; repeated frames do not add independent
locations. Calibration order is randomized within each phase. A regularized
linear ridge model is the baseline. Its regularization value is chosen using
validation median location error only. Smoothing time constant is also selected
from validation locations. The final test starts after model fitting and
settings have been saved and printed. Do not tune using the final test.

Each run gets a unique `records/run_<timestamp>/` folder, preserving earlier
results. `calibration_samples.json` contains location IDs, target pixel
coordinates, sample counts and raw per-frame feature vectors for diagnosis.
`calibration_model.json` stores the model and setup/version metadata.
`evaluation.json` reports validation and final-test raw and filtered
errors, mean/median/P90/worst error, error by quadrant, threshold proportions,
coverage, ridge-regression call latency and observed camera frame rate. The
regression latency does not include camera capture or landmark inference. The
test estimate has only five independent locations and is descriptive, not a
scientifically validated performance claim. Pixel error changes with display
size and setup. No accuracy numbers are claimed before a real run.

The current feature vector is normalized iris-landmark position relative to eye
corners plus eye openness for both eyes. Iris landmarks are coarse image
geometry, not a precise pupil sensor. The eye-openness blink threshold is a
heuristic. Head pose is not included: the optional transformation matrix needs
coordinate convention and reliability validation before treating it as a gaze
feature. Quality score is visible-eye coverage, a heuristic rather than a
statistically calibrated confidence. The final filtered per-target estimate
uses a configurable exponential smoother; raw coordinates remain separately
reported. The fixation detector uses a 70 drawable-pixel dispersion radius and
100 ms minimum elapsed duration. IDs are provisional while active and finalized
on invalid input or shutdown; invalid input ends fixation and clears its window.

## Recording fields

CSV output has monotonic timestamp, actual elapsed time since prior written
sample, raw and filtered drawable-pixel gaze, validity/quality/reason, fixation
ID and elapsed duration, and normalized left/right iris and eye-openness values.
Out-of-bounds raw predictions are not clipped in CSV; the live dot is visually
clipped to the drawable to keep it visible. Head fields are blank because head
pose is not currently used.

## Tests and troubleshooting

From repository root, with the environment activated:

```sh
python -m pytest -q custom_eye_tracker/tests
```

`BLOCKER: MediaPipe is missing` means the active environment is not the
documented environment. `Face Landmarker model not found` means the `.task`
asset is missing or the `--model` path is wrong. `Cannot open camera` means the
device index is unavailable or macOS permission is off (`--camera 1` can select
another device). `No face`/eye status in the preview means the face is outside
the frame, lighting is insufficient, or eyes are occluded. Skipped or incomplete
locations prevent the corresponding accuracy report; rerun calibration in
steady lighting and keep both eyes visible.
