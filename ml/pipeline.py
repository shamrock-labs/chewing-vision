"""AirPods IMU 씹기 감지 Random Forest — 전 세션 학습 + CoreML 내보내기.

평가는 ml/compare_sessions.py의 LOSO CV가 담당.
이 스크립트는 4세션 전체로 최종 모델을 학습한다.

사용법:
    cd /Users/bohyeong/Desktop/공부/project/soma/chewing-vision
    .venv/bin/python ml/pipeline.py
"""

from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from utils import FEATURE_NAMES, load_labels, load_session, make_windows_with_times

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/Users/bohyeong/Desktop/공부/project/soma/chewing_collector_data")
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SESSIONS = [
    {"id": "20260513T145413_a61a4c", "labels_path": "/tmp/chewing_a61a4c/labels_ours.csv"},
    {"id": "20260513T145034_838cje",  "labels_path": "/tmp/chewing_838cje/labels_ours.csv"},
    {"id": "20260513T161848_uj2e92",  "labels_path": "/tmp/chewing_uj2e92/labels_ours.csv"},
    {"id": "20260514T115953_n1xetu",  "labels_path": "/tmp/chewing_n1xetu/labels_ours.csv"},
]


def main() -> None:
    # 1. 전 세션 데이터 로드 + concat
    print("▶ 데이터 로드 중...")
    all_X, all_y = [], []
    for sess in SESSIONS:
        imu_path     = DATA_DIR / "sessions" / sess["id"] / "imu.csv"
        session_path = DATA_DIR / "sessions" / sess["id"] / "session.json"
        imu, _  = load_session(str(imu_path), str(session_path))
        labels  = load_labels(sess["labels_path"])
        X, y, _ = make_windows_with_times(imu, labels)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            print(f"  {sess['id']}: {len(X)} windows (chewing={y.sum()}, rest={(y==0).sum()})")

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"  총 윈도우: {len(X)}  (chewing={y.sum()}, rest={(y==0).sum()})")

    # 2. 전체 학습 (holdout 없음 — 평가는 LOSO compare_sessions.py 가 담당)
    print("▶ Random Forest 학습 중 (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    print(f"  Trained on {len(y)} windows, chewing ratio: {y.mean():.1%}")

    # 3. Feature importance
    importances = clf.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("\nTop-5 feature importance:")
    for i in top5:
        print(f"  {FEATURE_NAMES[i]}: {importances[i]:.3f}")

    # 4. 모델 저장 (joblib)
    model_path = OUT_DIR / "model_rf.joblib"
    joblib.dump(clf, model_path)
    print(f"\n▶ 모델 저장: {model_path}")

    # 5. Core ML 변환 (sklearn 1.6+ 버전 게이트 우회 패치)
    try:
        import coremltools as ct
        import coremltools._deps as _ct_deps
        import coremltools.converters.sklearn._tree_ensemble as _te
        from sklearn.tree import _tree as _sk_tree

        _ct_deps._HAS_SKLEARN = True
        _te._HAS_SKLEARN = True
        _te._tree = _sk_tree

        coreml_spec = _te.convert_tree_ensemble(
            clf,
            input_features=FEATURE_NAMES,
            output_features="chewing_label",
            mode="classifier",
            class_labels=[0, 1],
        )
        from coremltools.models import MLModel
        coreml = MLModel(coreml_spec)
        coreml.short_description = "AirPods IMU 씹기 감지 (Random Forest, 4-session LOSO)"
        coreml_path = str(OUT_DIR / "ChewingClassifier.mlmodel")
        coreml.save(coreml_path)
        print(f"▶ Core ML 저장: {coreml_path}")
    except Exception as e:
        print(f"▶ Core ML 변환 실패: {e}")


if __name__ == "__main__":
    main()
