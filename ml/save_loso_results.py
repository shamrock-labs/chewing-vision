"""Run LOSO CV and save results to InsForge DB.

Usage:
    .venv/bin/python ml/save_loso_results.py [--notes "some note"]

GT priority:
  1. human_label (agree with jaw_open_label) → gt_confidence='high'
  2. human_label (disagree with jaw_open_label) → gt_confidence='low'  (human wins)
  3. composite_label fallback → gt_confidence='machine'
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

MAIN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MAIN_ROOT))

from ml.utils import load_labels, load_session, make_windows_with_times
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

SESSIONS_DIR = MAIN_ROOT / "sessions"


def _discover_sessions():
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


def _query(sql, as_json=False):
    cmd = ["npx", "@insforge/cli", "db", "query", sql]
    if as_json:
        cmd.append("--json")
    result = subprocess.run(cmd, cwd=str(MAIN_ROOT), capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def _import_sql(sql: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False) as f:
        f.write(sql)
        tmp = f.name
    result = subprocess.run(
        ["npx", "@insforge/cli", "db", "import", tmp],
        cwd=str(MAIN_ROOT), capture_output=True, text=True
    )
    Path(tmp).unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    return result.stdout


def _bulk_update_gt(updates: list[tuple[int, str, str]]):
    """Write gt_label / gt_confidence back to DB windows in chunks."""
    for chunk_start in range(0, len(updates), 200):
        chunk = updates[chunk_start:chunk_start + 200]
        cases_label = " ".join(f"WHEN {wid} THEN '{lbl}'" for wid, lbl, _ in chunk)
        cases_conf  = " ".join(f"WHEN {wid} THEN '{conf}'" for wid, _, conf in chunk)
        ids = ", ".join(str(wid) for wid, _, _ in chunk)
        sql = (
            f"UPDATE windows SET "
            f"gt_label = CASE id {cases_label} END, "
            f"gt_confidence = CASE id {cases_conf} END "
            f"WHERE id IN ({ids});"
        )
        _import_sql(sql)


def fetch_windows_from_db(session_id: str) -> dict:
    """Return {t_start_db: {'id', 't_end', 'human_label', 'jaw_open_label', 'composite_label'}}."""
    sql = (f"SELECT id, t_start, t_end, human_label, jaw_open_label, composite_label "
           f"FROM windows WHERE session_id = '{session_id}' "
           f"ORDER BY t_start, labeled_at DESC NULLS LAST;")
    out = _query(sql, as_json=True)
    try:
        data = json.loads(out)
        rows = data.get("rows", data) if isinstance(data, dict) else data
        if not isinstance(rows, list):
            return {}
    except (json.JSONDecodeError, ValueError):
        return {}
    result = {}
    for row in rows:
        try:
            t = float(row["t_start"])
            new_row = {
                "id":               int(row["id"]),
                "t_end":            float(row["t_end"]),
                "human_label":      row.get("human_label"),
                "jaw_open_label":   row.get("jaw_open_label"),
                "composite_label":  row.get("composite_label"),
            }
            existing = result.get(t)
            if existing and existing["human_label"] is not None and new_row["human_label"] is None:
                continue  # keep the labeled row when duplicates share a t_start
            result[t] = new_row
        except (KeyError, TypeError, ValueError):
            pass
    return result


LOSO_WINDOW_SEC = 2.0       # must match make_windows_with_times default
TIE_WARN_THRESHOLD = 0.05   # flag sessions where >5% of overrides were ties


def _gt_from_overlap(t_loso, win_map, window_sec=LOSO_WINDOW_SEC):
    """Resolve GT for LOSO window [t, t+window_sec) via duration-weighted overlap voting.

    Returns dict {label, source, best, is_tie} or None when no DB window overlaps.
    Tie policy: chew_w == rest_w → 'rest' (conservative on positive class).
    """
    loso_end = t_loso + window_sec
    hits = []
    for tb, e in win_map.items():
        dur = max(0.0, min(loso_end, e["t_end"]) - max(t_loso, tb))
        if dur > 0:
            hits.append((dur, e))
    if not hits:
        return None

    chew_w = sum(d for d, e in hits if e.get("human_label") == "chewing")
    rest_w = sum(d for d, e in hits if e.get("human_label") == "rest")
    if chew_w > 0 or rest_w > 0:
        if chew_w > rest_w:
            label, is_tie = "chewing", False
        elif rest_w > chew_w:
            label, is_tie = "rest", False
        else:
            label, is_tie = "rest", True
        matched = [(d, e) for d, e in hits if e.get("human_label") == label]
        best = (max(matched, key=lambda x: x[0]) if matched
                else max(hits, key=lambda x: x[0]))[1]
        return {"label": label, "source": "human-tie" if is_tie else "human",
                "best": best, "is_tie": is_tie}

    best = max(hits, key=lambda x: x[0])[1]
    comp = best.get("composite_label")
    if comp in ("chewing", "rest"):
        return {"label": comp, "source": "machine", "best": best, "is_tie": False}
    return None


def run_loso(sessions):
    cache = {}
    for s in sessions:
        imu, _ = load_session(s["imu_path"], s["session_path"])
        labels  = load_labels(s["labels_path"])
        X, y, t_starts = make_windows_with_times(imu, labels)

        win_map = fetch_windows_from_db(s["id"])
        window_ids = [None] * len(t_starts)
        overridden = 0
        tie_count = 0

        gt_updates: list[tuple[int, str, str]] = []  # (window_id, gt_label, gt_confidence)

        for i, t in enumerate(t_starts):
            if not win_map:
                continue
            gt = _gt_from_overlap(t, win_map)
            if gt is None:
                continue
            wid = gt["best"]["id"]
            window_ids[i] = wid
            if gt["source"].startswith("human"):
                y[i] = 1 if gt["label"] == "chewing" else 0
                overridden += 1
                if gt["is_tie"]:
                    tie_count += 1
                confidence = "high" if gt["label"] == gt["best"].get("jaw_open_label") else "low"
                gt_updates.append((wid, gt["label"], confidence))
            else:
                gt_updates.append((wid, gt["label"], "machine"))

        if gt_updates:
            _bulk_update_gt(gt_updates)

        if win_map:
            tie_ratio = tie_count / overridden if overridden else 0.0
            tie_warn = " ⚠ >5%" if tie_ratio > TIE_WARN_THRESHOLD else ""
            tie_part = f", ties: {tie_count} ({tie_ratio:.1%}){tie_warn}" if overridden else ""
            msg = f"human_label override: {overridden}/{len(y)}{tie_part}"
        else:
            msg = "no DB windows"
        print(f"    [{s['id']}] {msg}")
        cache[s["id"]] = {"X": X, "y": y, "t_starts": t_starts, "window_ids": window_ids}

    fold_results = []
    all_y_true, all_y_pred = [], []
    all_predictions = []

    for test_id in cache:
        X_test       = cache[test_id]["X"]
        y_test       = cache[test_id]["y"]
        t_starts_test = cache[test_id]["t_starts"]
        wids_test     = cache[test_id]["window_ids"]

        if len(X_test) == 0:
            continue
        X_train = np.vstack([v["X"] for k, v in cache.items() if k != test_id and len(v["X"]) > 0])
        y_train = np.concatenate([v["y"] for k, v in cache.items() if k != test_id and len(v["y"]) > 0])
        if len(X_train) == 0:
            continue

        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        report = classification_report(y_test, y_pred, labels=[0, 1],
                                       target_names=["rest", "chewing"],
                                       output_dict=True, zero_division=0)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        for i in range(len(y_test)):
            all_predictions.append({
                "session_id": test_id,
                "window_id":  wids_test[i],
                "t_start":    float(t_starts_test[i]),
                "y_true":     int(y_test[i]),
                "y_pred":     int(y_pred[i]),
            })

        fold_results.append({
            "session_id":       test_id,
            "accuracy":         report["accuracy"],
            "f1_chewing":       report["chewing"]["f1-score"],
            "f1_rest":          report["rest"]["f1-score"],
            "n_train":          int(len(X_train)),
            "n_test":           int(len(X_test)),
            "train_chew_ratio": float(y_train.mean()),
            "test_chew_ratio":  float(y_test.mean()),
            "estimated_chews":  int(y_pred.sum()),
        })
        print(f"  [{test_id}] acc={report['accuracy']:.3f} "
              f"F1-chew={report['chewing']['f1-score']:.3f} "
              f"F1-rest={report['rest']['f1-score']:.3f}")

    pooled = classification_report(all_y_true, all_y_pred, labels=[0, 1],
                                   target_names=["rest", "chewing"],
                                   output_dict=True, zero_division=0)
    return fold_results, pooled, all_predictions


def save_to_db(fold_results, pooled, notes, predictions):
    n    = len(fold_results)
    pacc = pooled["accuracy"]
    pf1c = pooled["chewing"]["f1-score"]
    pf1r = pooled["rest"]["f1-score"]

    gt_note = "[GT: human+machine fallback]"
    full_notes = f"{gt_note} {notes}".strip() if notes else gt_note

    out = _query(
        f"INSERT INTO loso_runs (n_sessions, pooled_accuracy, pooled_f1_chewing, pooled_f1_rest, notes) "
        f"VALUES ({n}, {pacc}, {pf1c}, {pf1r}, '{full_notes}') RETURNING id;"
    )
    match = re.search(r'\b(\d+)\b', out)
    if not match:
        raise RuntimeError(f"Could not parse run_id from: {out}")
    run_id = int(match.group(1))
    print(f"  loso_runs id={run_id}")

    for r in fold_results:
        _query(
            f"INSERT INTO loso_results "
            f"(run_id, session_id, accuracy, f1_chewing, f1_rest, "
            f"n_train, n_test, train_chew_ratio, test_chew_ratio, estimated_chews) "
            f"VALUES ({run_id}, '{r['session_id']}', {r['accuracy']}, {r['f1_chewing']}, {r['f1_rest']}, "
            f"{r['n_train']}, {r['n_test']}, {r['train_chew_ratio']}, {r['test_chew_ratio']}, {r['estimated_chews']});"
        )
    print(f"  {n} fold results saved.")

    if predictions:
        rows = []
        for p in predictions:
            wid = str(p["window_id"]) if p["window_id"] is not None else "NULL"
            rows.append(
                f"({run_id}, {wid}, '{p['session_id']}', {p['t_start']}, {p['y_true']}, {p['y_pred']})"
            )
        for chunk_start in range(0, len(rows), 500):
            chunk = rows[chunk_start:chunk_start + 500]
            sql = (
                "INSERT INTO loso_predictions "
                "(run_id, window_id, session_id, t_start, y_true, y_pred) VALUES\n"
                + ",\n".join(chunk) + ";"
            )
            _import_sql(sql)
        print(f"  {len(predictions)} predictions saved.")
    return run_id


def _trigger_coreml_export(run_id: int, notes: str) -> None:
    """Invoke coreml_convert.py in cml_env subprocess. Skips with hint if env missing."""
    cml_python = MAIN_ROOT / "cml_env" / "bin" / "python"
    if not cml_python.exists():
        print(f"⚠ cml_env not found at {cml_python}. Run: bash ml/setup_cml_env.sh")
        return
    print(f"\n[save_loso_results] Triggering CoreML export (run_id={run_id})...")
    export_notes = notes or f"auto from loso_runs id={run_id}"
    result = subprocess.run(
        [str(cml_python), str(MAIN_ROOT / "ml" / "coreml_convert.py"),
         "--notes", export_notes],
        cwd=str(MAIN_ROOT),
    )
    if result.returncode != 0:
        print(f"⚠ CoreML export failed (exit {result.returncode})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notes", default="", help="Optional run description")
    ap.add_argument("--export-coreml", action="store_true",
                    help="Train on full dataset and export .mlmodel via cml_env after LOSO")
    args = ap.parse_args()

    sessions = _discover_sessions()
    if len(sessions) < 2:
        print(f"[save_loso_results] Need at least 2 sessions, found {len(sessions)}")
        return

    print(f"[save_loso_results] {len(sessions)} sessions -> LOSO CV")
    fold_results, pooled, predictions = run_loso(sessions)
    print(f"\n  Pooled acc={pooled['accuracy']:.3f} "
          f"F1-chew={pooled['chewing']['f1-score']:.3f} "
          f"F1-rest={pooled['rest']['f1-score']:.3f}")

    print("\n[save_loso_results] Saving to InsForge DB...")
    run_id = save_to_db(fold_results, pooled, args.notes, predictions)
    print("[save_loso_results] Done.")

    if args.export_coreml:
        _trigger_coreml_export(run_id, args.notes)


if __name__ == "__main__":
    main()
