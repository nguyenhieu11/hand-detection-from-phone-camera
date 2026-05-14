# YOLOv8 Hand Detection — C++ / ONNX Runtime

Real-time hand detection using a YOLOv8n model exported to ONNX, running on CPU with ONNX Runtime. Supports a local webcam or an RTSP stream (e.g. from an iPhone), and serves annotated video over HTTP as an MJPEG stream.

---

## Architecture

```
Video source  (webcam  or  iPhone → Larix → MediaMTX RTSP)
       │
       │  cv::VideoCapture (V4L2 or RTSP/FFMPEG)
       ▼
C++ inference app  (main.cpp)
  ├── Grab thread   — drains decoder buffer, always keeps latest frame
  ├── Inference     — YOLOv8 ONNX via ONNX Runtime (CPU)
  ├── Draw          — bounding boxes + confidence labels
  └── MJPEG server  — serves annotated frames over HTTP (:8080)
              │
              ▼
     Browser  →  http://<server-ip>:8080
```

---

## Requirements

| | Version |
|---|---|
| Ubuntu | 24.04 LTS |
| GCC / G++ | ≥ 11 |
| CMake | ≥ 3.18 |
| OpenCV 4 | `sudo apt install libopencv-dev` |
| ONNX Runtime | ≥ 1.20 — downloaded automatically by `build.sh` |

---

## Quick Start

### Webcam

```bash
chmod +x build.sh
./build.sh                         # installs deps, builds
./build/yolov8_hand best.onnx      # webcam on /dev/video0, HTTP on :8080
```

Open `http://localhost:8080` in a browser to view the annotated stream.

### iPhone camera over RTSP

**Step 1 — Start MediaMTX** (RTSP relay, included in repo)

```bash
./mediamtx
```

**Step 2 — Configure Larix Broadcaster on iPhone** (free, App Store)

- Connections → ＋ → URL: `rtsp://<server-ip>:8554/live`
- Settings → Audio/Video → enable **Background streaming**
- Tap **Start**

**Step 3 — Start inference**

```bash
./build/yolov8_hand best.onnx rtsp://127.0.0.1:8554/live 8080
```

Open `http://<server-ip>:8080` in Safari on the iPhone.

### CLI

```
./build/yolov8_hand  [model.onnx]  [rtsp://... | 0]  [http-port]
```

| Argument | Default | Description |
|---|---|---|
| `model.onnx` | `best.onnx` | YOLOv8 ONNX model |
| `rtsp-url \| 0` | `0` | RTSP URL, or `0` for local webcam |
| `http-port` | `8080` | MJPEG HTTP server port |

---

## Pipeline

### Preprocessing → `[1, 3, 320, 320]` float32 tensor

1. **Letterbox resize** — scale the frame so the longer side fits 320 px, pad the rest with grey (114). Records `scale`, `padLeft`, `padTop` for later inverse mapping.
2. **BGR → RGB** — OpenCV is BGR; the model was trained on RGB.
3. **HWC → CHW + normalise ÷ 255** — done in one pass via `cv::dnn::blobFromImage`.

### Inference

```cpp
sessionOpts.SetIntraOpNumThreads(4);
sessionOpts.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
```

Input/output names and shapes are queried at runtime — no hardcoded tensor names.

### Postprocessing

Output tensor shape: `[1, 5, 2100]` — `(cx, cy, w, h, conf)` × 2100 anchors.  
YOLOv8 has **no separate objectness score**; the class score is the confidence.

1. Filter anchors below `CONF_THRESHOLD` (0.60).
2. Map boxes from letterboxed space back to the original frame:
   $$x = (x_{lb} - padLeft) / scale$$
3. NMS via `cv::dnn::NMSBoxes` (IoU threshold 0.45).
4. Draw green boxes + `hand XX%` labels; overlay inference time.

### Grab thread (RTSP latency control)

A background thread calls `cap.read()` in a tight loop and always overwrites `g_latestFrame`. This drains the RTSP decoder buffer so inference never processes stale frames.

---

## Tunable Parameters

| Constant | Default | Effect |
|---|---|---|
| `CONF_THRESHOLD` | `0.60` | Raise → fewer false positives |
| `NMS_THRESHOLD` | `0.45` | Raise → allow more overlapping boxes |
| `INPUT_W / INPUT_H` | `320` | Must match the exported model |

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ERROR: cannot open source` | Start Larix *before* the inference app |
| High latency (5–10 s) | Grab thread already handles this; check network/Wi-Fi |
| `bind() failed` | Port in use — pass a different port as 3rd argument |
