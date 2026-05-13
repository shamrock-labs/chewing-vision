"""IMU 로드, timestamp align, sliding window 생성 유틸리티."""

import json
import numpy as np
import pandas as pd
from pathlib import Path


IMU_AXES = [
    "rotation_x", "rotation_y", "rotation_z",
    "user_accel_x", "user_accel_y", "user_accel_z",
]


def load_session(imu_path: str, session_json_path: str) -> tuple[pd.DataFrame, float]:
    """IMU CSV와 session.json을 읽고 (imu_df, video_start_sec) 반환.

    video_start_sec: baseline_mach 기준으로 변환한 비디오 시작 시각(초).
    imu.t_rel_sec + video_start_sec ≈ vision.t_sec (비디오 기준 절대 시각)
    """
    imu = pd.read_csv(imu_path)

    with open(session_json_path) as f:
        meta = json.load(f)

    numer = meta["clock"]["mach_timebase_numer"]
    denom = meta["clock"]["mach_timebase_denom"]
    baseline = meta["clock"]["baseline_mach"]
    video_start_mach = meta["video"]["start_mach"]

    video_start_sec = (video_start_mach - baseline) * numer / denom / 1e9

    # vision 기준 시각으로 정렬된 컬럼 추가
    imu["t_vision"] = imu["t_rel_sec"] + video_start_sec
    return imu, video_start_sec


def load_labels(labels_path: str) -> pd.DataFrame:
    """chewing-vision labels_ours.csv 로드."""
    return pd.read_csv(labels_path)


def assign_label(t_start: float, t_end: float, labels: pd.DataFrame,
                 chewing_threshold: float = 0.5) -> int:
    """윈도우 [t_start, t_end) 구간에서 chewing 비율 >= threshold 이면 1."""
    overlap = labels[
        (labels["t_end"] > t_start) & (labels["t_start"] < t_end)
    ]
    if overlap.empty:
        return 0
    chewing_count = (overlap["label"] == "chewing").sum()
    return 1 if chewing_count / len(overlap) >= chewing_threshold else 0


def make_windows(imu: pd.DataFrame, labels: pd.DataFrame,
                 window_sec: float = 2.0, stride_sec: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """슬라이딩 윈도우로 feature matrix X와 label vector y 생성."""
    X, y, _ = make_windows_with_times(imu, labels, window_sec, stride_sec)
    return X, y


def make_windows_with_times(imu: pd.DataFrame, labels: pd.DataFrame,
                             window_sec: float = 2.0, stride_sec: float = 0.5
                             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """슬라이딩 윈도우로 feature matrix X, label vector y, t_start 배열 생성."""
    t_min = imu["t_vision"].min()
    t_max = imu["t_vision"].max() - window_sec

    X, y, t_starts = [], [], []
    t = t_min
    while t <= t_max:
        window = imu[(imu["t_vision"] >= t) & (imu["t_vision"] < t + window_sec)]
        if len(window) < 5:
            t += stride_sec
            continue

        features = extract_features(window)
        label = assign_label(t, t + window_sec, labels)
        X.append(features)
        y.append(label)
        t_starts.append(t)
        t += stride_sec

    return np.array(X), np.array(y), np.array(t_starts)


def extract_features(window: pd.DataFrame) -> list[float]:
    """12-dim feature vector: RMS + std for 6 IMU axes."""
    features = []
    for axis in IMU_AXES:
        vals = window[axis].values
        features.append(float(np.sqrt(np.mean(vals ** 2))))
        features.append(float(np.std(vals)))
    return features


FEATURE_NAMES = [f"{axis}_{stat}" for axis in IMU_AXES for stat in ("rms", "std")]
