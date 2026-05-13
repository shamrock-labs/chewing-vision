"""
chewing_demo.py — Day 1 데모: 영상 한 편 → chewing count + 신호 그래프

흐름:
  1) 로컬 영상 파일 읽기 (mp4/mov)
  2) MediaPipe Face Landmarker로 매 프레임 478 landmark + blendshape 추출
  3) MAR (Mouth Aspect Ratio, 직접 계산) + jawOpen (모델이 주는 값) 두 신호 시계열화
  4) scipy.signal.find_peaks 로 봉우리 찾기 (씹기 1회 = 봉우리 1개로 가정)
  5) 결과: 콘솔에 chew 수/속도, PNG로 시계열+peak 그래프 저장

사용법:
    python chewing_demo.py path/to/video.mp4
    python chewing_demo.py path/to/video.mp4 --start 10 --end 70   # 10~70초 구간만
    python chewing_demo.py path/to/video.mp4 --out my_plot.png

읽기 전 알아둘 점:
  * MediaPipe FaceMesh는 정면 얼굴 가정. 얼굴이 화면에 안 잡힌 프레임은
    `face detection rate` 로 보고됨. 70% 미만이면 영상 자체가 부적합.
  * "chew count" 는 신호 봉우리 개수에 대한 휴리스틱 추정. 실제 정답 라벨이
    아니라 첫 번째 sanity check 수치.
  * MAR과 jawOpen은 서로 다른 정의이므로 count 가 다르게 나오는 게 정상.
    어느 쪽이 영상의 실제 chewing과 더 잘 맞는지 본인 눈으로 비교가 목적.
"""

import argparse
import os
import sys
import urllib.request
from dataclasses import dataclass

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = "face_landmarker.task"

# 478-landmark FaceMesh 기준 입 주요 인덱스
# 윗입술 안쪽 가운데 / 아랫입술 안쪽 가운데 / 왼·오른쪽 입꼬리 안쪽
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_CORNER_IDX = 78
RIGHT_CORNER_IDX = 308


def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[setup] face_landmarker 모델을 다운로드합니다 → {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[setup] 완료.")


def compute_mar(landmarks, w, h):
    """Mouth Aspect Ratio = (입 위·아래 거리) / (입 좌우 거리)."""
    p_up = np.array([landmarks[UPPER_LIP_IDX].x * w, landmarks[UPPER_LIP_IDX].y * h])
    p_dn = np.array([landmarks[LOWER_LIP_IDX].x * w, landmarks[LOWER_LIP_IDX].y * h])
    p_l = np.array([landmarks[LEFT_CORNER_IDX].x * w, landmarks[LEFT_CORNER_IDX].y * h])
    p_r = np.array([landmarks[RIGHT_CORNER_IDX].x * w, landmarks[RIGHT_CORNER_IDX].y * h])
    vertical = np.linalg.norm(p_up - p_dn)
    horizontal = np.linalg.norm(p_l - p_r)
    return float(vertical / horizontal) if horizontal > 0 else 0.0


def get_blendshape(blendshapes, name):
    """blendshape 결과 리스트에서 이름으로 0~1 값 가져오기."""
    for b in blendshapes:
        if b.category_name == name:
            return float(b.score)
    return 0.0


@dataclass
class FrameSignal:
    t: float
    mar: float
    jaw_open: float
    face_found: bool


def process_video(video_path, start=None, end=None):
    ensure_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base_options,
        output_face_blendshapes=True,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"영상을 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = int(start * fps) if start is not None else 0
    end_frame = int(end * fps) if end is not None else n_frames
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    print(f"[run] fps={fps:.1f}, 처리 프레임 {start_frame} ~ {end_frame}")

    signals = []
    with mp_vision.FaceLandmarker.create_from_options(options) as landmarker:
        frame_idx = start_frame
        last_log = 0
        while frame_idx < end_frame:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            t = frame_idx / fps
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            h, w = frame_rgb.shape[:2]
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            result = landmarker.detect_for_video(mp_image, int(t * 1000))

            if result.face_landmarks:
                mar = compute_mar(result.face_landmarks[0], w, h)
                jaw_open = (
                    get_blendshape(result.face_blendshapes[0], "jawOpen")
                    if result.face_blendshapes
                    else 0.0
                )
                signals.append(FrameSignal(t, mar, jaw_open, True))
            else:
                signals.append(FrameSignal(t, np.nan, np.nan, False))

            frame_idx += 1
            if t - last_log >= 5.0:
                print(f"[run] {t:.1f}s 처리 ...")
                last_log = t

    cap.release()
    return signals, fps


def detect_chews(values, fps, min_freq=0.8, max_freq=2.5, prominence_ratio=0.05):
    """씹기 1회 = 봉우리 1개. NaN 보간 → 중앙값 제거 → find_peaks."""
    arr = np.array(values, dtype=float)
    valid = ~np.isnan(arr)
    if valid.sum() < 10:
        return np.array([], dtype=int), arr

    arr_interp = np.interp(np.arange(len(arr)), np.where(valid)[0], arr[valid])
    arr_norm = arr_interp - np.nanmedian(arr_interp)

    # 봉우리 간 최소 간격: 1초에 max_freq 회보다 빨리는 못 씹는다고 가정
    min_distance = max(1, int(fps / max_freq))
    # 봉우리 prominence(돋보임 정도) 최소값
    rng = float(np.nanmax(arr_interp) - np.nanmin(arr_interp))
    prominence = prominence_ratio * rng if rng > 0 else 0.0

    peaks, _ = find_peaks(arr_norm, distance=min_distance, prominence=prominence)
    return peaks, arr_interp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", help="로컬 영상 파일 경로")
    ap.add_argument("--start", type=float, default=None, help="시작 시각(초)")
    ap.add_argument("--end", type=float, default=None, help="끝 시각(초)")
    ap.add_argument("--out", default="chewing_signal.png", help="결과 PNG 경로")
    args = ap.parse_args()

    print(f"[run] 영상 처리: {args.video}")
    signals, fps = process_video(args.video, start=args.start, end=args.end)
    if not signals:
        sys.exit("프레임을 하나도 읽지 못했습니다.")

    times = np.array([s.t for s in signals])
    mar_series = np.array([s.mar for s in signals])
    jaw_series = np.array([s.jaw_open for s in signals])

    face_rate = float(np.mean([s.face_found for s in signals]))
    print(f"[stats] face detection rate: {face_rate*100:.1f}%")
    if face_rate < 0.7:
        print("[stats] ⚠ 얼굴 인식률이 낮습니다(<70%). 다른 영상/구간 권장.")

    mar_peaks, mar_interp = detect_chews(mar_series, fps)
    jaw_peaks, jaw_interp = detect_chews(jaw_series, fps)

    duration = float(times[-1] - times[0]) if len(times) > 1 else 1.0
    print(f"[result] duration       : {duration:.1f} s")
    print(
        f"[result] MAR     chews  : {len(mar_peaks):4d}   "
        f"rate {len(mar_peaks)/duration*60:.1f} chews/min"
    )
    print(
        f"[result] jawOpen chews  : {len(jaw_peaks):4d}   "
        f"rate {len(jaw_peaks)/duration*60:.1f} chews/min"
    )

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    axes[0].plot(times, mar_interp, label="MAR (직접 계산)")
    axes[0].plot(
        times[mar_peaks],
        mar_interp[mar_peaks],
        "ro",
        markersize=4,
        label=f"peaks ({len(mar_peaks)})",
    )
    axes[0].set_ylabel("MAR")
    axes[0].set_title(f"Chewing signal — {os.path.basename(args.video)}")
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.3)

    axes[1].plot(times, jaw_interp, color="C1", label="jawOpen (blendshape)")
    axes[1].plot(
        times[jaw_peaks],
        jaw_interp[jaw_peaks],
        "ro",
        markersize=4,
        label=f"peaks ({len(jaw_peaks)})",
    )
    axes[1].set_ylabel("jawOpen")
    axes[1].set_xlabel("time (sec)")
    axes[1].legend(loc="upper right")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"[run] 그래프 저장: {args.out}")


if __name__ == "__main__":
    main()
