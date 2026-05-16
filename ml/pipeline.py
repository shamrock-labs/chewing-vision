"""AirPods IMU 씹기 감지 Random Forest baseline.

sessions/ 디렉토리를 자동 탐색해 IMU + labels_ours.csv가 모두 있는
세션을 전부 학습에 사용한다. 평가는 ml/compare_sessions.py의 LOSO CV가 담당.

사용법:
    cd chewing-vision
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

WORKTREE_ROOT = Path(__file__).resolve().parents[1]
MAIN_REPO_ROOT = Path(__file__).resolve().parents[4]
SESSIONS_DIR = MAIN_REPO_ROOT / "sessions"
OUT_DIR      = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def _discover_sessions() -> list[dict]:
    sessions = []
    for sdir in sorted(SESSIONS_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        imu_files     = sorted(sdir.glob("imu_*.csv"))
        session_files = sorted(sdir.glob("session_*.json"))
        labels_path   = sdir / "labels_ours.csv"
        if imu_files and session_files and labels_path.exists():
            sessions.append({
                "id":           sdir.name,
                "imu_path":     str(imu_files[0]),
                "session_path": str(session_files[0]),
                "labels_path":  str(labels_path),
            })
    return sessions


def main() -> None:
    sessions = _discover_sessions()
    if not sessions:
        print(f"[pipeline] No valid sessions found in {SESSIONS_DIR}")
        return
    print(f"[pipeline] Found {len(sessions)} sessions: {[s['id'] for s in sessions]}")

    # 1. 전 세션 데이터 로드 + concat
    print("▶ 데이터 로드 중...")
    all_X, all_y = [], []
    for sess in sessions:
        imu, _ = load_session(sess["imu_path"], sess["session_path"])
        labels = load_labels(sess["labels_path"])
        X, y, _ = make_windows_with_times(imu, labels)
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            print(f"  {sess['id']}: {len(X)} windows (chewing={y.sum()}, rest={(y==0).sum()})")

    if not all_X:
        print("[pipeline] No windows produced.")
        return

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    print(f"  총 윈도우: {len(X)}  (chewing={y.sum()}, rest={(y==0).sum()})")

    # 2. 전체 데이터로 학습 (holdout 없음 — 평가는 LOSO compare_sessions.py 가 담당)
    print("▶ Random Forest 학습 중 (n_estimators=100)...")
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X, y)
    print(f"  Trained on {len(X)} windows, chewing ratio: {y.mean()*100:.1f}%")

    # 3. Feature importance
    importances = clf.feature_importances_
    top5 = np.argsort(importances)[::-1][:5]
    print("\nTop-5 feature importance:")
    for i in top5:
        print(f"  {FEATURE_NAMES[i]}: {importances[i]:.3f}")

    # 4. 모델 저장
    model_path = OUT_DIR / "model_rf.joblib"
    joblib.dump(clf, model_path)
    print(f"\n▶ 모델 저장: {model_path}")

    # 5. Core ML 변환
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
        coreml.short_description = "AirPods IMU 씹기 감지 (Random Forest, composite w=0.3)"
        coreml_path = str(OUT_DIR / "ChewingClassifier.mlmodel")
        coreml.save(coreml_path)
        print(f"▶ Core ML 저장: {coreml_path}")
    except Exception as e:
        print(f"▶ Core ML 변환 실패: {e}")


if __name__ == "__main__":
    main()
