"""Gate 2: Log yaw angles on the angled session to confirm foreshortening hypothesis.

Prints per-frame yaw (degrees) for the first 300 frames of j2b3jd.
If yaw is consistently 20-40°+, proceed with cos(yaw) MAR correction.
"""

import math
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, str(Path(__file__).parent.parent))
from chewing.engines.ours import MODEL_PATH, _ensure_model

SESSION = "20260517T142102_j2b3jd"
SESSIONS_DIR = Path(__file__).parent.parent / "sessions"
video_path = next((SESSIONS_DIR / SESSION).glob("video_*.mp4"))

MAX_FRAMES = 300
SAMPLE_EVERY = 5  # print every 5th frame to keep output manageable


def extract_yaw_deg(matrix_4x4) -> float:
    """Extract yaw (rotation about Y-axis) from 4x4 rigid transformation matrix.

    MediaPipe's facial_transformation_matrixes uses OpenGL convention:
      R = matrix[:3, :3]
      yaw = atan2(R[0,2], R[2,2])  (rotation about Y / left-right head turn)
    """
    data = matrix_4x4.data  # flat list, row-major 4x4
    R = np.array(data, dtype=float).reshape(4, 4)[:3, :3]
    yaw_rad = math.atan2(R[0, 2], R[2, 2])
    return math.degrees(yaw_rad)


model_path = _ensure_model()
base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
options = mp_vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=True,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_faces=1,
)

cap = cv2.VideoCapture(str(video_path))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

yaws = []
frame_idx = 0

with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
    while frame_idx < MAX_FRAMES:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        t_ms = int(frame_idx / fps * 1000)
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        detection = landmarker.detect_for_video(mp_image, t_ms)

        if detection.facial_transformation_matrixes:
            yaw = extract_yaw_deg(detection.facial_transformation_matrixes[0])
            yaws.append(yaw)
            if frame_idx % SAMPLE_EVERY == 0:
                print(f"  frame {frame_idx:4d}  t={frame_idx/fps:.2f}s  yaw={yaw:+.1f}°")
        else:
            if frame_idx % SAMPLE_EVERY == 0:
                print(f"  frame {frame_idx:4d}  t={frame_idx/fps:.2f}s  [no face]")

        frame_idx += 1

cap.release()

if yaws:
    arr = np.array(yaws)
    abs_arr = np.abs(arr)
    print(f"\n--- Yaw summary ({len(yaws)} frames with face) ---")
    print(f"  mean abs yaw : {abs_arr.mean():.1f}°")
    print(f"  median abs   : {np.median(abs_arr):.1f}°")
    print(f"  max abs      : {abs_arr.max():.1f}°")
    print(f"  pct > 20°    : {(abs_arr > 20).mean()*100:.0f}%")
    print(f"  pct > 30°    : {(abs_arr > 30).mean()*100:.0f}%")
    print(f"  pct > 45°    : {(abs_arr > 45).mean()*100:.0f}%")
    print(f"\nConclusion: ", end="")
    if np.median(abs_arr) >= 20:
        print("HYPOTHESIS CONFIRMED — proceed with cos(yaw) MAR correction")
    else:
        print("Yaw is small — foreshortening hypothesis may be wrong")
else:
    print("No frames with face detected.")
