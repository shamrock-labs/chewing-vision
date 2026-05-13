"""AirPods IMU 씹기 감지 Random Forest baseline.

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
from sklearn.metrics import classification_report, confusion_matrix

from utils import FEATURE_NAMES, load_labels, load_session, make_windows

# ── 경로 ──────────────────────────────────────────────────────────────────────
DATA_DIR    = Path("/Users/bohyeong/Desktop/공부/project/soma/chewing_collector_data")
LABELS_PATH = Path("/tmp/chewing_a61a4c/labels_ours.csv")
SESSION_ID  = "20260513T145413_a61a4c"
OUT_DIR     = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

IMU_PATH     = DATA_DIR / "sessions" / SESSION_ID / "imu.csv"
SESSION_PATH = DATA_DIR / "sessions" / SESSION_ID / "session.json"


def main() -> None:
    # 1. 데이터 로드 + 타임스탬프 정렬
    print("▶ 데이터 로드 중...")
    imu, video_start_sec = load_session(str(IMU_PATH), str(SESSION_PATH))
    labels = load_labels(str(LABELS_PATH))
    print(f"  IMU 샘플: {len(imu)}, GT 라벨 windows: {len(labels)}")
    print(f"  video_start_sec offset: {video_start_sec:.4f}s")

    # 2. 슬라이딩 윈도우 생성
    print("▶ 윈도우 생성 중 (2s / 0.5s stride)...")
    X, y = make_windows(imu, labels, window_sec=2.0, stride_sec=0.5)
    print(f"  총 윈도우: {len(X)}  (chewing={y.sum()}, rest={(y==0).sum()})")

    # 3. Time-based split (앞 80% train)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"  train: {len(X_train)}, test: {len(X_test)}")

    # 4. Random Forest 학습
    print("▶ Random Forest 학습 중...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    # 5. 평가
    y_pred = clf.predict(X_test)
    print("\n── Classification Report ──")
    print(classification_report(y_test, y_pred, target_names=["rest", "chewing"]))

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix (rest / chewing):")
    print(cm)

    # 6. Feature importance
    importances = clf.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("\nTop-5 feature importance:")
    for i in top5:
        print(f"  {FEATURE_NAMES[i]}: {importances[i]:.3f}")

    # 7. 예측 vs GT 시각화
    _plot_predictions(y_test, y_pred, OUT_DIR / "prediction_vs_gt.png")

    # 8. 모델 저장
    model_path = OUT_DIR / "model_rf.joblib"
    joblib.dump(clf, model_path)
    print(f"\n▶ 모델 저장: {model_path}")

    # 9. Core ML 변환 (sklearn 1.6+ 버전 게이트 우회 패치)
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
        coreml.short_description = "AirPods IMU 씹기 감지 (Random Forest, F1=0.79)"
        coreml_path = str(OUT_DIR / "ChewingClassifier.mlmodel")
        coreml.save(coreml_path)
        print(f"▶ Core ML 저장: {coreml_path}")
    except Exception as e:
        print(f"▶ Core ML 변환 실패: {e}")


def _plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 4), sharex=True)
    axes[0].step(range(len(y_true)), y_true, where="post", color="steelblue")
    axes[0].set_title("GT label"); axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["rest", "chewing"])
    axes[1].step(range(len(y_pred)), y_pred, where="post", color="tomato")
    axes[1].set_title("Predicted"); axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["rest", "chewing"])
    axes[1].set_xlabel("Window index (test set, 0.5s stride)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"▶ 시각화 저장: {out_path}")


if __name__ == "__main__":
    main()
