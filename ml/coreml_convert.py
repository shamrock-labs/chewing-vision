"""Train on ALL sessions + export CoreML .mlpackage.

Requires sklearn ≤ 1.5.1 (coremltools limit). Run in a separate venv:
    python -m venv cml_env
    cml_env/bin/pip install scikit-learn==1.5.1 coremltools numpy
    cml_env/bin/python ml/coreml_convert.py [--notes "설명"]

Output: ml/models/chewing_v{timestamp}.mlpackage
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

MAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_ROOT))

from ml.utils import FEATURE_NAMES, load_labels, load_session, make_windows_with_times
from ml.save_loso_results import _gt_from_overlap, fetch_windows_from_db

SESSIONS_DIR = MAIN_ROOT / "sessions"
MODELS_DIR   = MAIN_ROOT / "ml" / "models"


def _discover_sessions(labels_suffix: str = "") -> list[dict]:
    sessions = []
    for sdir in sorted(SESSIONS_DIR.iterdir()):
        if not sdir.is_dir():
            continue
        imu_files     = sorted(sdir.glob("imu_*.csv"))
        session_files = sorted(sdir.glob("session_*.json"))
        labels_path   = sdir / f"labels_ours{labels_suffix}.csv"
        if imu_files and session_files and labels_path.exists():
            sessions.append({
                "id":           sdir.name,
                "imu_path":     str(imu_files[0]),
                "session_path": str(session_files[0]),
                "labels_path":  str(labels_path),
            })
    return sessions


def build_full_dataset(sessions: list[dict]):
    """Apply same overlap-voting GT resolution as save_loso_results."""
    X_all, y_all = [], []
    for s in sessions:
        imu, _ = load_session(s["imu_path"], s["session_path"])
        labels  = load_labels(s["labels_path"])
        X, y, t_starts = make_windows_with_times(imu, labels)

        win_map = fetch_windows_from_db(s["id"])
        overridden = 0
        for i, t in enumerate(t_starts):
            if not win_map:
                continue
            gt = _gt_from_overlap(t, win_map)
            if gt and gt["source"].startswith("human"):
                y[i] = 1 if gt["label"] == "chewing" else 0
                overridden += 1
        print(f"  [{s['id']}] {len(y)} windows, human_label override: {overridden}")
        if len(X) == 0:
            continue  # skip sessions with no usable windows (e.g. lhqzzv after boundary filter)
        X_all.append(X)
        y_all.append(y)

    return np.vstack(X_all), np.concatenate(y_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="", help="Optional description for metadata")
    ap.add_argument("--labels-suffix", default="_chin",
                    help="Label file suffix (default: _chin → labels_ours_chin.csv)")
    args = ap.parse_args()

    import coremltools as ct
    print(f"coremltools: {ct.__version__}")

    sessions = _discover_sessions(labels_suffix=args.labels_suffix)
    if not sessions:
        print("No sessions found.")
        sys.exit(1)

    print(f"Building dataset from {len(sessions)} sessions...")
    X, y = build_full_dataset(sessions)
    print(f"Total: {len(X)} windows, chew={y.sum()}, rest={(y==0).sum()}")

    print("Training RandomForest on full dataset...")
    clf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                 class_weight="balanced")
    clf.fit(X, y)

    y_pred_train = clf.predict(X)
    report = classification_report(y, y_pred_train, target_names=["rest", "chewing"],
                                   output_dict=True, zero_division=0)
    f1_chew = report["chewing"]["f1-score"]
    f1_rest = report["rest"]["f1-score"]
    print(f"Train (full): F1-chew={f1_chew:.3f} F1-rest={f1_rest:.3f}")

    # CoreML export
    coreml = ct.converters.sklearn.convert(clf, FEATURE_NAMES, "chewing_label")
    coreml.short_description = "AirPods IMU 씹기 감지 (Random Forest)"
    coreml.author = "세잎클로버 / chewing-vision"
    notes_str = f" | {args.notes}" if args.notes else ""
    coreml.description = (
        f"sessions={len(sessions)}, F1-chew={f1_chew:.3f}, F1-rest={f1_rest:.3f}{notes_str}"
    )
    for fn in FEATURE_NAMES:
        coreml.input_description[fn] = fn
    coreml.output_description["chewing_label"] = "0=rest, 1=chewing"

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_path = MODELS_DIR / f"chewing_v{ts}.mlmodel"
    coreml.save(str(out_path))
    print(f"\nSaved: {out_path}")
    print(f"Sessions: {[s['id'] for s in sessions]}")


if __name__ == "__main__":
    main()
