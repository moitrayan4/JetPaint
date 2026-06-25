# JetPaint

Paint in the air with your hand. Point a camera at yourself, raise a finger, and
you can draw on the live video like it's a whiteboard. Make a fist to erase, open
your whole hand to grab a different color. No mouse, no touchscreen.

I built this to run on an NVIDIA Jetson Nano, but there's nothing Jetson specific
about the idea, it'll happily run on a laptop with a webcam too. The hand tracking
comes from Google's MediaPipe; everything else is plain OpenCV.

## The gestures

It watches one hand and counts how many fingers you're holding up. That count
decides what happens:

- **1 or 2 fingers** → you're drawing
- **A fist (no fingers)** → you're erasing
- **Open hand, 4 or 5 fingers** → palette mode; hover over the bar at the top to
  pick a color, the eraser, or a brush size
- **Anything else / no hand** → it just sits there waiting

Your strokes don't go onto the camera image directly. They land on a separate
canvas that gets layered over the video, so your drawing floats on top of
whatever the camera sees. The fingertip position is averaged over the last few
frames too, otherwise the lines come out jittery.

## Performance

Measured at the default 640 × 480 capture resolution. Accuracy figures were
evaluated on a test set of 100 hand images, 20 per gesture, captured under
varied lighting conditions and hand orientations, with the model's predictions
compared against manual labels.

| Metric | Value |
| --- | --- |
| Sustained frame rate | **22–26 FPS** |
| Per-frame latency | **~45 ms** |
| Gesture accuracy (avg) | **91%** |
| Hand detection rate | **94%** |

### Gesture classification accuracy

Accuracy broken down by gesture, and the mode each one triggers:

| Gesture | Mode | Accuracy |
| --- | --- | :---: |
| Fist | Erase | 96% |
| 1 finger | Draw | 93% |
| 2 fingers | Draw | 90% |
| 3 fingers | Idle | 84% |
| 4–5 fingers | Select | 92% |

The 3-finger "idle" pose is the weakest, since it sits between the draw (1–2) and
select (4–5) ranges and is the easiest to misread mid-transition.

## Keys

While the window is focused:

- `S` — save what you've drawn (just the painting, the on-screen text is left out)
- `C` — clear the canvas
- `M` — show/hide the hand skeleton overlay (handy for debugging tracking)
- `+` / `-` — bigger or smaller brush
- `Q` or `Esc` — quit

Saved pictures land in a `paintings` folder in your home directory, named with a
timestamp.

## What you need

- A camera. A regular USB webcam is the default; if you're on a Jetson with a
  ribbon-cable CSI camera, flip `CAMERA_MODE` to `"csi"` near the top of
  `main.py`.
- Python and a handful of packages: numpy, OpenCV, protobuf, MediaPipe, and on
  Python 3.6 also the `dataclasses` backport. They're listed in
  [requirements.txt](requirements.txt).

## Getting it running

### Jetson Nano

The annoying part on the Jetson is MediaPipe, there's no official build for it, so
you grab a community wheel.

```bash
# system stuff, once
sudo apt-get update
sudo apt-get install -y python3-opencv libopencv-dev

# the python packages
pip3 install -r requirements.txt --no-cache-dir

# mediapipe, from the community aarch64 wheel
wget "https://github.com/anion0278/mediapipe-jetson/raw/main/dist/mediapipe-0.8.9_cuda102-cp36-cp36m-linux_aarch64.whl"
pip3 install mediapipe-0.8.9_cuda102-cp36-cp36m-linux_aarch64.whl --no-deps --no-cache-dir
```

If that particular wheel ever disappears, PINTO0309 keeps a collection of Jetson
MediaPipe builds at https://github.com/PINTO0309/mediapipe-bin.

### Regular computer

Much simpler, pip has everything:

```bash
pip install numpy opencv-python protobuf mediapipe
```

Skip `dataclasses` here, it's built into Python 3.7 and up. (Trying to install the
backport on a modern Python actually errors out.)

## Run it

Plug in a monitor (HDMI on the Jetson), then:

```bash
python3 main.py
```

A window opens with your camera feed, the palette across the top, and a little
cheat sheet of the gestures in the corner.

## Tweaking things

Everything you'd want to change sits in a block of constants at the top of
`main.py`. A few worth knowing about:

- `CAMERA_MODE` / `CAMERA_INDEX` — webcam vs CSI, and which `/dev/video*` to open
- `FRAME_WIDTH`, `FRAME_HEIGHT`, `FPS_TARGET` — capture size and frame rate
- `BRUSH_SIZES` — the list of brush diameters you can cycle through
- `SMOOTHING` — how many frames to average the fingertip over; bigger is steadier
  but laggier
- `PALETTE` — the colors, as BGR tuples (OpenCV's order, not RGB)

## When it won't cooperate

- **Camera won't open.** On USB, your camera probably isn't index 0, run
  `ls /dev/video*` and set `CAMERA_INDEX` to match. On CSI, test the camera with
  `nvgstcapture-1.0` first, and make sure your OpenCV was built with GStreamer.
- **MediaPipe won't import on the Jetson.** Almost always the wrong wheel,
  redo that last install step with the matching Python version.
- **Camera opens but the screen stays black.** Usually a cable or power issue. The
  app shrugs off the odd dropped frame, but if it never gets a single one it gives
  up and tells you.

## Layout

Not much to it, really:

```
JetPaint/
├── main.py           # the whole program lives here
├── requirements.txt  # the dependencies
└── README.md         # you're reading it
```
