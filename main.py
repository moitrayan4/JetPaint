import cv2
import numpy as np
import time
import os
import collections
import sys
from datetime import datetime
import mediapipe as mp



CAMERA_MODE    = "usb"
CAMERA_INDEX   = 0          
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
FPS_TARGET     = 30
BRUSH_SIZES    = [4, 8, 16, 24, 40]
DEFAULT_BRUSH  = 1
SMOOTHING      = 7      
MIN_DRAW_DIST  = 4      

PALETTE = {
    "Red":    (0,   0,   220),
    "Orange": (0,   140, 255),
    "Yellow": (0,   220, 220),
    "Green":  (50,  200,  50),
    "Cyan":   (220, 200,  50),
    "Blue":   (220,  80,  20),
    "Purple": (200,  40, 180),
    "White":  (255, 255, 255),
    "Black":  (20,   20,  20),
}

ERASER_SIZE = 40
_FINGER_TIPS = [8, 12, 16, 20]      
_FINGER_PIPS = [6, 10, 14, 18]      
_THUMB_TIP   = 4
_THUMB_IP    = 3



class HandAnalyzer:

    def __init__(self):
        self._mp_hands = mp.solutions.hands
        self._mp_draw  = mp.solutions.drawing_utils
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            # min_detection_confidence — lower = faster but more false positives
            min_detection_confidence=0.7,
            min_tracking_confidence=0.6,
        )
        self._tip_buf = collections.deque(maxlen=SMOOTHING)


    def _lm_px(self, lm_list, idx, w, h):
        lm = lm_list[idx]
        return int(lm.x * w), int(lm.y * h)

    def _count_fingers(self, lm_list, w, h):
        count = 0

        tx, _  = self._lm_px(lm_list, _THUMB_TIP, w, h)
        ipx, _ = self._lm_px(lm_list, _THUMB_IP,  w, h)
        if tx < ipx:   
            count += 1

        for tip_id, pip_id in zip(_FINGER_TIPS, _FINGER_PIPS):
            _, ty = self._lm_px(lm_list, tip_id, w, h)
            _, py = self._lm_px(lm_list, pip_id, w, h)
            if ty < py:
                count += 1

        return count



    def process(self, frame):
        empty = {"detected": False, "fingers": 0,
                 "tip": None, "centroid": None, "lm_frame": None}

        if frame is None or frame.size == 0:
            return empty

        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            self._tip_buf.clear()
            return empty

        hand_lm = results.multi_hand_landmarks[0]  # first detected hand
        lm_list = hand_lm.landmark

        # Draw landmarks onto a copy for optional debug display
        lm_frame = frame.copy()
        self._mp_draw.draw_landmarks(
            lm_frame, hand_lm,
            self._mp_hands.HAND_CONNECTIONS,
            self._mp_draw.DrawingSpec(color=(0, 220, 100), thickness=2, circle_radius=3),
            self._mp_draw.DrawingSpec(color=(255, 255, 255), thickness=1),
        )

        fingers  = self._count_fingers(lm_list, w, h)

        raw_tip  = self._lm_px(lm_list, 8,  w, h)
        centroid = self._lm_px(lm_list, 0,  w, h)   

        # Smooth the tip position
        self._tip_buf.append(raw_tip)
        xs  = [p[0] for p in self._tip_buf]
        ys  = [p[1] for p in self._tip_buf]
        tip = (int(np.mean(xs)), int(np.mean(ys)))

        # Clamp to frame bounds
        tip = (max(0, min(w - 1, tip[0])),
               max(0, min(h - 1, tip[1])))

        return {"detected": True, "fingers": fingers,
                "tip": tip, "centroid": centroid,
                "lm_frame": lm_frame}

    def close(self):
        self._hands.close()



class Mode:
    IDLE   = "IDLE"
    DRAW   = "DRAW"
    ERASE  = "ERASE"
    SELECT = "SELECT"

def classify(info):
    if not info["detected"]:
        return Mode.IDLE
    f = info["fingers"]
    if f == 0:
        return Mode.ERASE
    if f <= 2:
        return Mode.DRAW
    if f >= 4:
        return Mode.SELECT
    return Mode.IDLE


class PaletteUI:
    SW = 52
    SH = 42
    MG = 5

    def __init__(self):
        self.colors = list(PALETTE.items())
        self._build()

    def _build(self):
        x = self.MG
        self.color_rects = []
        for name, bgr in self.colors:
            self.color_rects.append({
                "name": name, "color": bgr,
                "r": (x, self.MG, x + self.SW, self.MG + self.SH),
            })
            x += self.SW + self.MG
        self.color_rects.append({
            "name": "Eraser", "color": (200, 200, 200),
            "r": (x, self.MG, x + self.SW, self.MG + self.SH),
        })
        x += self.SW + self.MG * 3
        self.brush_rects = []
        for i, sz in enumerate(BRUSH_SIZES):
            self.brush_rects.append({
                "size": sz,
                "r": (x, self.MG, x + self.SH, self.MG + self.SH),
            })
            x += self.SH + self.MG

    @property
    def bar_h(self):
        return self.SH + self.MG * 2 + 4

    def draw(self, frame, active_name, active_brush):
        cv2.rectangle(frame, (0, 0), (frame.shape[1], self.bar_h), (30, 30, 30), -1)
        for item in self.color_rects:
            x1, y1, x2, y2 = item["r"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), item["color"], -1)
            if item["name"] == active_name:
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (255, 255, 255), 2)
            cv2.putText(frame, item["name"][:3], (x1+3, y2-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        for i, item in enumerate(self.brush_rects):
            x1, y1, x2, y2 = item["r"]
            cx, cy = (x1+x2)//2, (y1+y2)//2
            r = item["size"]//2 + 2
            cv2.circle(frame, (cx, cy), r, (180, 180, 180), -1)
            if i == active_brush:
                cv2.circle(frame, (cx, cy), r+2, (255, 255, 255), 2)

    def hit_color(self, px, py):
        for item in self.color_rects:
            x1, y1, x2, y2 = item["r"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return item["name"], item["color"]
        return None, None

    def hit_brush(self, px, py):
        for i, item in enumerate(self.brush_rects):
            x1, y1, x2, y2 = item["r"]
            if x1 <= px <= x2 and y1 <= py <= y2:
                return i
        return None


class VirtualPainter:

    def __init__(self):
        self.cap        = self._init_cam()
        self.analyzer   = HandAnalyzer()
        self.ui         = PaletteUI()
        self.canvas     = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.uint8)
        self.mode       = Mode.IDLE
        self.color_name = "Red"
        self.color_bgr  = PALETTE["Red"]
        self.brush_idx  = DEFAULT_BRUSH
        self.erasing    = False
        self.prev_pt    = None
        self._fps_buf   = collections.deque(maxlen=30)
        self.show_mask  = False
        self.save_dir   = os.path.expanduser("~/paintings")
        os.makedirs(self.save_dir, exist_ok=True)


    def _gstreamer_csi_pipeline(self):
        return (
            "nvarguscamerasrc ! "
            "video/x-raw(memory:NVMM), width=(int){w}, height=(int){h}, "
            "format=(string)NV12, framerate=(fraction){fps}/1 ! "
            "nvvidconv flip-method=0 ! "
            "video/x-raw, width=(int){w}, height=(int){h}, format=(string)BGRx ! "
            "videoconvert ! "
            "video/x-raw, format=(string)BGR ! appsink"
        ).format(w=FRAME_WIDTH, h=FRAME_HEIGHT, fps=FPS_TARGET)

    def _init_cam(self):
        if CAMERA_MODE == "csi":
            pipeline = self._gstreamer_csi_pipeline()
            print("[Camera] Opening CSI camera via GStreamer...")
            print("[Camera] Pipeline:", pipeline)
            cap = cv2.VideoCapture(pipeline)
            if not cap.isOpened():
                raise RuntimeError(
                    "Cannot open CSI camera via GStreamer.\n"
                    "Check that:\n"
                    "  1. Camera is connected to the CSI port.\n"
                    "  2. JetPack / nvarguscamerasrc is installed.\n"
                    "  3. Try:  nvgstcapture-1.0  to verify camera works.\n"
                    "  4. To use a USB webcam instead, set CAMERA_MODE = 'usb'."
                )
        else:
            print("[Camera] Opening USB camera at index {} ...".format(CAMERA_INDEX))
            cap = cv2.VideoCapture(CAMERA_INDEX)
            if not cap.isOpened():
                raise RuntimeError(
                    "Cannot open USB camera at index {}.\n"
                    "Run:  ls /dev/video*  to list cameras,\n"
                    "then change CAMERA_INDEX at the top of the script.\n"
                    "For a CSI camera, set CAMERA_MODE = 'csi'.".format(CAMERA_INDEX)
                )
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)

        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                print("[Camera] Ready. Resolution: {}x{}".format(
                    int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
                return cap
        raise RuntimeError(
            "Camera opened but returned no frames.\n"
            "Check power/cable and try again."
        )


    def _stroke(self, pt):
        bsize = ERASER_SIZE if self.erasing else BRUSH_SIZES[self.brush_idx]
        color = (0, 0, 0)  if self.erasing else self.color_bgr
        thick = max(1, bsize * 2)   # thickness must be >= 1
        if (self.prev_pt is not None and
                np.hypot(pt[0] - self.prev_pt[0],
                         pt[1] - self.prev_pt[1]) >= MIN_DRAW_DIST):
            cv2.line(self.canvas, self.prev_pt, pt, color, thick)
        cv2.circle(self.canvas, pt, thick // 2, color, -1)
        self.prev_pt = pt


    def _composite(self, frame):
        if self.canvas.shape[:2] != frame.shape[:2]:
            self.canvas = cv2.resize(self.canvas, (frame.shape[1], frame.shape[0]))
        gray = cv2.cvtColor(self.canvas, cv2.COLOR_BGR2GRAY)
        _, m = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        mi   = cv2.bitwise_not(m)
        bg   = cv2.bitwise_and(frame,        frame,        mask=mi)
        fg   = cv2.bitwise_and(self.canvas,  self.canvas,  mask=m)
        return cv2.add(bg, fg)


    def _hud(self, frame, fps, fingers):
        h, w = frame.shape[:2]
        mc = {
            Mode.DRAW:   (50,  220, 50),
            Mode.ERASE:  (50,  80,  220),
            Mode.SELECT: (220, 200, 50),
            Mode.IDLE:   (120, 120, 120),
        }
        cv2.putText(frame, "Mode: {}  Fingers: {}".format(self.mode, fingers),
                    (10, h-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mc[self.mode], 2)
        label = "ERASER" if self.erasing else self.color_name
        cv2.circle(frame, (165, h-58), 10, self.color_bgr, -1)
        cv2.putText(frame, label, (182, h-52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
        cv2.putText(frame, "Brush: {}px".format(BRUSH_SIZES[self.brush_idx]),
                    (10, h-38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
        cv2.putText(frame, "FPS: {:.1f}".format(fps),
                    (w-110, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 220, 100), 1)
        guide = [
            "1-2 fingers = Draw",
            "Fist (0)    = Erase",
            "4-5 fingers = Palette",
            "S=Save  C=Clear",
            "M=Landmarks  Q=Quit",
            "+/-=Brush Size",
        ]
        for i, g in enumerate(guide):
            cv2.putText(frame, g, (w-200, h-140+i*22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


    def _save(self, frame):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.save_dir, "painting_{}.png".format(ts))
        # save a clean composite, not the frame with the HUD drawn on top
        clean = self._composite(frame)
        cv2.imwrite(path, clean)
        print("[Saved] {}".format(path))


    def run(self):
        print("=" * 56)
        print("  Virtual Painter  |  Jetson Nano  |  MediaPipe")
        print("=" * 56)
        print("  Gestures:")
        print("    1-2 fingers up  → Draw")
        print("    Fist (0 fingers)→ Erase")
        print("    4-5 fingers up  → Select palette / brush")
        print("  Keys: S=Save  C=Clear  M=Landmarks  +/-=Brush  Q=Quit")
        print("=" * 56)

        bar_h = self.ui.bar_h

        while True:
            ret, frame = self.cap.read()

            # don't crash on occasional dropped frames — retry next iteration
            if not ret or frame is None or frame.size == 0:
                print("[Warning] Dropped frame, retrying...")
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            # if the camera delivers a different resolution than expected,
            # resize so the canvas always matches
            fh, fw = frame.shape[:2]
            if fw != FRAME_WIDTH or fh != FRAME_HEIGHT:
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

            info    = self.analyzer.process(frame)
            self.mode = classify(info)
            tip     = info["tip"]
            fingers = info["fingers"]

            if tip is not None:
                tx, ty = tip

                if self.mode == Mode.SELECT:
                    cname, cbgr = self.ui.hit_color(tx, ty)
                    if cname == "Eraser":
                        self.erasing = True
                    elif cname is not None:
                        self.color_name = cname
                        self.color_bgr  = cbgr
                        self.erasing    = False
                    bidx = self.ui.hit_brush(tx, ty)
                    if bidx is not None:
                        self.brush_idx = bidx
                    self.prev_pt = None

                elif self.mode == Mode.DRAW and ty > bar_h:
                    self.erasing = False
                    self._stroke((tx, ty))

                elif self.mode == Mode.ERASE and ty > bar_h:
                    self.erasing = True
                    self._stroke((tx, ty))

                else:
                    self.prev_pt = None

                # Fingertip cursor ring
                cur_col = (50, 50, 200) if self.erasing else self.color_bgr
                sz = ERASER_SIZE if self.erasing else BRUSH_SIZES[self.brush_idx]
                sz = max(1, sz)   # radius must be >= 1
                cv2.circle(frame, (tx, ty), sz, cur_col, 2)

            else:
                self.prev_pt = None

            # Wrist centroid dot on main frame
            if info["centroid"] is not None:
                cv2.circle(frame, info["centroid"], 5, (0, 100, 255), -1)

            # FPS
            self._fps_buf.append(time.time())
            elapsed = self._fps_buf[-1] - self._fps_buf[0]
            fps = len(self._fps_buf) / (elapsed if elapsed > 0 else 1e-6)

            # Composite, palette bar, HUD
            frame = self._composite(frame)
            self.ui.draw(frame, self.color_name if not self.erasing else "Eraser",
                         self.brush_idx)
            self._hud(frame, fps, fingers)

            # Optional hand-landmark debug window
            if self.show_mask and info["lm_frame"] is not None:
                cv2.imshow("Landmarks", info["lm_frame"])

            cv2.imshow("Virtual Painter - Jetson Nano", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('s'):
                self._save(frame)
            elif key == ord('c'):
                self.canvas[:] = 0
                self.prev_pt   = None
                print("[Cleared]")
            elif key == ord('m'):
                self.show_mask = not self.show_mask
                print("[Landmarks] {}".format("ON" if self.show_mask else "OFF"))
            elif key in (ord('+'), ord('=')):
                self.brush_idx = min(self.brush_idx + 1, len(BRUSH_SIZES) - 1)
            elif key == ord('-'):
                self.brush_idx = max(self.brush_idx - 1, 0)

        self.analyzer.close()
        self.cap.release()
        cv2.destroyAllWindows()
        print("Done.")


def _check_dependencies():
    """Verify required packages are present before starting."""
    missing = []
    try:
        import cv2
        build_info = cv2.getBuildInformation()
        if CAMERA_MODE == "csi" and "GStreamer" not in build_info:
            print("[Warning] OpenCV built without GStreamer — CSI camera mode won't work.")
            print("          Install Jetson-optimized OpenCV: sudo apt-get install python3-opencv")
    except ImportError:
        missing.append("opencv  →  sudo apt-get install python3-opencv")

    try:
        import numpy
    except ImportError:
        missing.append("numpy   →  pip3 install numpy")

    try:
        import mediapipe
    except ImportError:
        missing.append(
            "mediapipe (aarch64)  →  see install instructions at the top of this file\n"
            "              or: https://github.com/PINTO0309/mediapipe-bin"
        )

    if missing:
        print("=" * 60)
        print("  Missing dependencies — please install:")
        for m in missing:
            print("    " + m)
        print("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    _check_dependencies()
    painter = VirtualPainter()
    painter.run()