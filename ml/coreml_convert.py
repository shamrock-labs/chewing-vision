"""sklearn 1.5.1 환경에서 실행 — Random Forest 재학습 → Core ML 변환.

사용법 (별도 venv):
    <cml_env>/bin/python ml/coreml_convert.py
"""

import sys
from pathlib import Path

# ml/ 디렉터리를 import 경로에 추가
sys.path.insert(0, str(Path(__file__).parent))

import joblib
import coremltools as ct
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import FEATURE_NAMES, load_labels, load_session, make_windows

DATA_DIR    = Path("/Users/bohyeong/Desktop/공부/project/soma/chewing_collector_data")
LABELS_PATH = Path("/tmp/chewing_a61a4c/labels_ours.csv")
SESSION_ID  = "20260513T145413_a61a4c"
OUT_DIR     = Path(__file__).parent

IMU_PATH     = DATA_DIR / f"imu_{SESSION_ID}.csv"
SESSION_PATH = DATA_DIR / f"session_{SESSION_ID}.json"


def main() -> None:
    print(f"scikit-learn: {__import__('sklearn').__version__}")
    print(f"coremltools:  {ct.__version__}")

    imu, video_start_sec = load_session(str(IMU_PATH), str(SESSION_PATH))
    labels = load_labels(str(LABELS_PATH))
    X, y = make_windows(imu, labels, window_sec=2.0, stride_sec=0.5)

    split = int(len(X) * 0.8)
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X[:split], y[:split])

    from sklearn.metrics import classification_report
    y_pred = clf.predict(X[split:])
    print(classification_report(y[split:], y_pred, target_names=["rest", "chewing"]))

    # sklearn 1.5.x 호환 모델도 joblib 저장
    joblib_path = OUT_DIR / "model_rf_cml.joblib"
    joblib.dump(clf, joblib_path)
    print(f"joblib 저장: {joblib_path}")

    # Core ML 변환
    coreml = ct.converters.sklearn.convert(clf, FEATURE_NAMES, "chewing_label")

    # 메타데이터
    coreml.short_description = "AirPods IMU 씹기 감지 (Random Forest)"
    coreml.input_description["rotation_x_rms"]    = "rotation X RMS (2s window)"
    coreml.output_description["chewing_label"]    = "0=rest, 1=chewing"

    coreml_path = str(OUT_DIR / "ChewingClassifier.mlmodel")
    coreml.save(coreml_path)
    print(f"Core ML 저장: {coreml_path}")


if __name__ == "__main__":
    main()
