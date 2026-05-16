"""4세션 LOSO(Leave-One-Session-Out) ML 결과 비교 + HTML 시각화.

사용법:
    cd /Users/bohyeong/Desktop/공부/project/soma/chewing-vision
    .venv/bin/python ml/compare_sessions.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import LeaveOneGroupOut

from utils import FEATURE_NAMES, load_labels, load_session, make_windows_with_times


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_bouts(t_starts: np.ndarray, y_pred: np.ndarray,
                  stride_sec: float = 0.5, max_gap: float = 1.0) -> list[tuple]:
    """각 bout의 (start, end, chew_count) 리스트 반환."""
    if len(t_starts) == 0:
        return []
    bouts, bout_start, prev_end = [], None, None
    for t, lbl in zip(t_starts, y_pred):
        if lbl == 1:
            w_end = t + stride_sec
            if prev_end is not None and t - prev_end > max_gap:
                count = max(1, int((prev_end - bout_start) * 1.2 + 0.5))
                bouts.append((bout_start, prev_end, count))
                bout_start = t
            elif bout_start is None:
                bout_start = t
            prev_end = w_end
    if bout_start is not None:
        count = max(1, int((prev_end - bout_start) * 1.2 + 0.5))
        bouts.append((bout_start, prev_end, count))
    return bouts


# ── IMU 신호 서브플롯 ──────────────────────────────────────────────────────────

def _plot_imu_signal(ax, imu, labels, t_starts, y_pred, session_label, f1_val):
    """rotation_y signal + GT chewing regions (background) + ML prediction strip (bottom 10%)."""
    t   = imu["t_vision"].values
    sig = imu["rotation_y"].values
    t_min, t_max = t.min(), t.max()

    # GT chewing regions as background shading (above prediction strip)
    for _, row in labels.iterrows():
        if row["label"] == "chewing":
            ax.axvspan(row["t_start"], row["t_end"],
                       ymin=0.12, ymax=1.0,
                       alpha=0.22, color="#27ae60", linewidth=0)

    # IMU rotation_y signal
    ax.plot(t, sig, color="#2980b9", linewidth=0.5, alpha=0.85, rasterized=True)

    # ML prediction strip at bottom 10% of axes (colored spans in axes coords)
    for t_s, pred in zip(t_starts, y_pred):
        color = "#27ae60" if pred == 1 else "#bdc3c7"
        ax.axvspan(t_s, t_s + 0.5, ymin=0, ymax=0.10,
                   alpha=0.85, color=color, linewidth=0)

    ax.set_xlim(t_min, t_max)
    ax.set_title(f"{session_label}  rotation_y  (F1={f1_val:.2f})", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("rad/s", fontsize=7)
    ax.tick_params(labelsize=7)

    ax.legend(handles=[
        mpatches.Patch(color="#27ae60", alpha=0.22, label="GT chewing"),
        mpatches.Patch(color="#2980b9", alpha=0.85, label="rotation_y"),
        mpatches.Patch(color="#27ae60", alpha=0.85, label="ML: chewing"),
        mpatches.Patch(color="#bdc3c7", alpha=0.85, label="ML: rest"),
    ], fontsize=6, loc="upper right")


# ── 시각화 (5행 × (n+1)열 GridSpec) ─────────────────────────────────────────

def plot_comparison(fold_results: list[dict], pooled_report: dict,
                    pooled_cm: np.ndarray, session_cache: dict,
                    out_dir: Path | None = None) -> None:
    n = len(fold_results)
    ncols = n + 1  # sessions + pooled column
    fig = plt.figure(figsize=(5.5 * ncols, 25))
    fig.suptitle("AirPods IMU Chewing Detection — LOSO Session Comparison",
                 fontsize=15, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(5, ncols, figure=fig, hspace=0.60, wspace=0.40)

    labels_x = [r["label"] for r in fold_results] + ["Pooled"]

    # ── Row 0: F1 bar chart (all columns) ─────────────────────────────────────
    ax_f1 = fig.add_subplot(gs[0, 0:ncols])
    f1_chew = [r["report"]["chewing"]["f1-score"] for r in fold_results] + [pooled_report["chewing"]["f1-score"]]
    f1_rest = [r["report"]["rest"]["f1-score"]    for r in fold_results] + [pooled_report["rest"]["f1-score"]]
    x = np.arange(len(labels_x)); w = 0.35
    ax_f1.bar(x - w/2, f1_chew, w, label="F1 (chewing)", color="#2ecc71", alpha=0.85)
    ax_f1.bar(x + w/2, f1_rest,  w, label="F1 (rest)",    color="#95a5a6", alpha=0.85)
    ax_f1.axhline(0.70, color="red", linestyle="--", linewidth=1, label="baseline 0.70")
    ax_f1.set_xticks(x); ax_f1.set_xticklabels(labels_x, fontsize=9)
    ax_f1.set_ylim(0, 1.15); ax_f1.set_ylabel("F1 Score")
    ax_f1.set_title("F1 Score by Fold (LOSO) + Pooled")
    ax_f1.legend(fontsize=8); ax_f1.grid(axis="y", alpha=0.3)
    for xi, v in zip(x - w/2, f1_chew):
        ax_f1.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # ── Row 1: IMU signal + GT labels per session ──────────────────────────────
    for i, r in enumerate(fold_results):
        ax = fig.add_subplot(gs[1, i])
        _plot_imu_signal(ax,
                         r["imu"], r["labels"],
                         r["t_starts"], r["y_pred"],
                         r["label"], r["report"]["chewing"]["f1-score"])

    # Pooled column Row 1: summary text
    ax_pooled_r1 = fig.add_subplot(gs[1, n])
    ax_pooled_r1.axis("off")
    ax_pooled_r1.text(0.5, 0.5, "Pooled\n(4 sessions)\n\nSee metrics table\nbelow",
                      ha="center", va="center", fontsize=11, transform=ax_pooled_r1.transAxes)
    ax_pooled_r1.set_title("Pooled (4 sessions)", fontsize=9)

    # ── Row 2: GT vs Pred prediction timeline ─────────────────────────────────
    colors_map = {0: "#bdc3c7", 1: "#27ae60"}
    for i, r in enumerate(fold_results):
        ax = fig.add_subplot(gs[2, i])
        y_t = r["y"]; y_p = r["y_pred"]
        for j in range(len(y_t)):
            ax.barh(0.7, 1, left=j, height=0.4,
                    color=colors_map[y_t[j]], alpha=0.7, linewidth=0)
            ax.barh(0.2, 1, left=j, height=0.4,
                    color=colors_map[y_p[j]], alpha=0.7, linewidth=0)
        ax.set_xlim(0, len(y_t))
        ax.set_yticks([0.2, 0.7])
        ax.set_yticklabels(["Pred", "GT"], fontsize=8)
        f1_val = r["report"]["chewing"]["f1-score"]
        ax.set_title(f"{r['label']}  (F1={f1_val:.2f})", fontsize=10)
        ax.set_xlabel("Window index (test session)", fontsize=8)
        ax.legend(handles=[
            mpatches.Patch(color="#27ae60", label="chewing", alpha=0.7),
            mpatches.Patch(color="#bdc3c7", label="rest",    alpha=0.7),
        ], fontsize=7, loc="upper right")

    # Pooled column Row 2: summary text
    ax_pooled_r2 = fig.add_subplot(gs[2, n])
    ax_pooled_r2.axis("off")
    pacc = pooled_report["accuracy"]
    pf1c = pooled_report["chewing"]["f1-score"]
    pf1r = pooled_report["rest"]["f1-score"]
    ax_pooled_r2.text(0.5, 0.5,
                      f"Pooled\n(4 sessions)\n\nAcc: {pacc:.3f}\nF1-chew: {pf1c:.3f}\nF1-rest: {pf1r:.3f}",
                      ha="center", va="center", fontsize=11, transform=ax_pooled_r2.transAxes)
    ax_pooled_r2.set_title("Pooled summary", fontsize=9)

    # ── Row 3: Confusion matrices (n individual + 1 pooled) ───────────────────
    all_cms = [(r["label"], r["cm"]) for r in fold_results] + [("Pooled", pooled_cm)]
    for i, (lbl, cm) in enumerate(all_cms):
        ax = fig.add_subplot(gs[3, i])
        ax.imshow(cm, cmap="Greens", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["rest", "chewing"], fontsize=8)
        ax.set_yticklabels(["rest", "chewing"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(f"Confusion — {lbl}", fontsize=8)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm[row, col] > cm.max() * 0.5 else "black")

    # ── Row 4: Bout timeline ───────────────────────────────────────────────────
    for i, r in enumerate(fold_results):
        ax = fig.add_subplot(gs[4, i])
        t_min = r["imu"]["t_vision"].min()
        t_max = r["imu"]["t_vision"].max()
        bouts = r["bouts"]

        ax.set_xlim(t_min, t_max)
        ax.set_ylim(0, 1)
        ax.set_yticks([])

        # rest background
        ax.axhspan(0, 1, color="#ecf0f1", alpha=0.5)

        if bouts:
            cmap  = plt.cm.YlOrRd
            max_c = max(b[2] for b in bouts)
            for b_start, b_end, count in bouts:
                width = max(b_end - b_start, 0.5)
                color = cmap(count / max_c)
                ax.axvspan(b_start, b_start + width, ymin=0.1, ymax=0.9,
                           alpha=0.85, color=color)
                mid = b_start + width / 2
                ax.text(mid, 0.5, str(count),
                        ha="center", va="center", fontsize=9, fontweight="bold",
                        color="white" if count / max_c > 0.5 else "#2c3e50")
        else:
            ax.text(0.5, 0.5, "no chewing detected", ha="center", va="center",
                    transform=ax.transAxes, fontsize=9, color="gray")

        ax.set_xlabel("Time (s)", fontsize=8)
        ax.set_title(f"{r['label']}  predicted bouts + chew count  (total={r['estimated_chews']})",
                     fontsize=9)

    # Pooled column Row 4: summary text
    ax_pooled_r4 = fig.add_subplot(gs[4, n])
    ax_pooled_r4.axis("off")
    total_chews = sum(r["estimated_chews"] for r in fold_results)
    ax_pooled_r4.text(0.5, 0.5,
                      f"Pooled\nTotal estimated\nchews: {total_chews}",
                      ha="center", va="center", fontsize=11, transform=ax_pooled_r4.transAxes)
    ax_pooled_r4.set_title("Pooled bouts", fontsize=9)

    _out = out_dir or (Path(__file__).parent / "outputs")
    _out.mkdir(parents=True, exist_ok=True)
    out_path = _out / "session_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  saved: {out_path}")


# ── HTML 요약 ──────────────────────────────────────────────────────────────────

def save_html_table(fold_results: list[dict], pooled_report: dict,
                    session_cache: dict,
                    out_dir: Path | None = None) -> None:
    def badge(v, threshold=0.70):
        color = "#27ae60" if v >= threshold else "#e74c3c"
        return f'<span style="color:{color};font-weight:bold">{v:.3f}</span>'

    # Dataset overview rows
    overview_rows = ""
    for label, info in session_cache.items():
        n_win = len(info["y"])
        chew_pct = f"{info['chew_ratio']*100:.1f}%"
        rest_pct = f"{(1-info['chew_ratio'])*100:.1f}%"
        overview_rows += f"""
        <tr>
          <td>{label}</td>
          <td>{info['duration']:.1f}</td>
          <td>{n_win}</td>
          <td>{chew_pct}</td>
          <td>{rest_pct}</td>
          <td>{info['n_gt_bouts']}</td>
        </tr>"""

    # LOSO fold rows
    fold_rows = ""
    for r in fold_results:
        acc  = r["report"]["accuracy"]
        f1c  = r["report"]["chewing"]["f1-score"]
        f1r  = r["report"]["rest"]["f1-score"]
        fold_rows += f"""
        <tr>
          <td>{r['label']}</td>
          <td>{r['n_train']}</td>
          <td>{r['train_chew_ratio']*100:.1f}%</td>
          <td>{r['n_test']}</td>
          <td>{r['test_chew_ratio']*100:.1f}%</td>
          <td>{acc*100:.1f}%</td>
          <td>{badge(f1c)}</td>
          <td>{badge(f1r)}</td>
        </tr>"""

    # Pooled row
    pacc = pooled_report["accuracy"]
    pf1c = pooled_report["chewing"]["f1-score"]
    pf1r = pooled_report["rest"]["f1-score"]
    n_total = sum(r["n_test"] for r in fold_results)
    fold_rows += f"""
        <tr style="background:#f0f8f0;font-weight:bold">
          <td>Pooled</td>
          <td>—</td>
          <td>—</td>
          <td>{n_total}</td>
          <td>—</td>
          <td>{pacc*100:.1f}%</td>
          <td>{badge(pf1c)}</td>
          <td>{badge(pf1r)}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>씹기 감지 세션 비교 (LOSO)</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1100px; margin: 40px auto; padding: 0 24px; color: #222; }}
  h1   {{ font-size: 1.6rem; margin-bottom: 4px; }}
  .subtitle {{ color: #666; margin-bottom: 32px; font-size: 0.95rem; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.9rem; }}
  th, td {{ border: 1px solid #ddd; padding: 10px 14px; text-align: center; }}
  th {{ background: #f5f5f5; font-weight: 600; }}
  tr:hover {{ background: #fafafa; }}
  .section {{ margin-top: 40px; }}
  img {{ width: 100%; border-radius: 8px; border: 1px solid #eee; margin-top: 12px; }}
  .note {{ background: #fffbe6; border-left: 4px solid #f1c40f; padding: 12px 16px;
           border-radius: 4px; font-size: 0.88rem; color: #666; margin-top: 24px; }}
  .made-with {{ margin-top: 40px; padding: 16px; background: #f8f8f8;
                border-radius: 8px; font-size: 0.85rem; color: #555; }}
  code {{ background: #f0f0f0; padding: 2px 5px; border-radius: 3px; font-size: 0.85em; }}
</style>
</head>
<body>

<h1>AirPods IMU 씹기 감지 — LOSO 세션 비교</h1>
<p class="subtitle">Leave-One-Session-Out CV / Random Forest n=20 / window 2s stride 0.5s</p>

<div class="section">
  <h2>Dataset Overview</h2>
  <table>
    <thead>
      <tr>
        <th>Session</th><th>Duration (s)</th><th>Total Windows</th>
        <th>Chewing %</th><th>Rest %</th><th>GT Chewing Bouts</th>
      </tr>
    </thead>
    <tbody>{overview_rows}</tbody>
  </table>
</div>

<div class="section">
  <h2>LOSO 성능 비교 테이블</h2>
  <table>
    <thead>
      <tr>
        <th>Fold (held out)</th><th>Train Windows</th><th>Train Chew%</th>
        <th>Test Windows</th><th>Test Chew%</th>
        <th>Accuracy</th><th>F1 (chewing)</th><th>F1 (rest)</th>
      </tr>
    </thead>
    <tbody>{fold_rows}</tbody>
  </table>
  <div class="note">
    기준치: F1 (chewing) >= 0.70 | 빨간색 = 미달 / 초록색 = 통과<br>
    LOSO: 각 fold에서 해당 세션을 test로, 나머지 3세션을 train으로 사용.<br>
    Pooled: 전체 fold 예측을 합쳐 계산한 통합 메트릭.
  </div>
</div>

<div class="section">
  <h2>시각화</h2>
  <p style="font-size:0.85rem;color:#888">
    Row 1: F1 비교 바차트 (LOSO fold + Pooled)<br>
    Row 2: IMU rotation_y 신호 + GT 라벨(녹색 배경) + ML 예측 스트립(하단 10%)<br>
    Row 3: GT vs 예측 타임라인 (각 fold의 test 세션 전체)<br>
    Row 4: Confusion Matrix (fold별 + Pooled)<br>
    Row 5: Bout 타임라인 + 씹기 수 추정
  </p>
  <img src="session_comparison.png" alt="LOSO 세션 비교 시각화">
</div>

<div class="made-with">
  <strong>어떻게 만들어졌나요?</strong><br><br>
  이 파일은 <code>ml/compare_sessions.py</code>로 자동 생성되었습니다.<br><br>
  <strong>데이터 흐름:</strong><br>
  1. <strong>chewing-vision CLI</strong> — MediaPipe FaceLandmarker로 영상에서 MAR(입 열림 비율) 추출 → 1초 window GT 라벨 생성<br>
  2. <strong>IMU 전처리</strong> (<code>ml/utils.py · make_windows_with_times</code>) — AirPods 50Hz 신호를 2초/0.5s stride 슬라이딩 윈도우로 분할, 15차원 피처 추출<br>
  3. <strong>LOSO CV</strong> (<code>sklearn LeaveOneGroupOut</code>, RF n=20) — 4세션 × "3개 train / 1개 test"<br>
  4. <strong>씹기 카운팅</strong> — ML 예측 결과에서 연속 chewing 윈도우를 bout 단위로 묶어 duration × 1.2 Hz로 씹기 수 추정<br>
  5. <strong>시각화</strong> (<code>matplotlib</code>) — F1 바차트, IMU 신호+GT라벨, 예측 타임라인, Confusion Matrix, Bout 타임라인<br>
  6. <strong>이 HTML</strong> — Python f-string으로 생성 (<code>ml/compare_sessions.py</code>)<br><br>
  재실행: <code>.venv/bin/python ml/compare_sessions.py</code>
</div>

</body>
</html>"""

    _out = out_dir or (Path(__file__).parent / "outputs")
    _out.mkdir(parents=True, exist_ok=True)
    out_path = _out / "session_comparison.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main(sessions_dir: Path | None = None, output_dir: Path | None = None) -> None:
    import argparse as _ap
    if sessions_dir is None:
        _p = _ap.ArgumentParser(prog="compare_sessions",
                                description="LOSO CV across downloaded sessions.")
        _p.add_argument("--sessions-dir", default="./sessions", metavar="DIR",
                        help="Root dir produced by `chewing-vision fetch --all` (default: ./sessions)")
        _p.add_argument("-o", "--output", default=str(Path(__file__).parent / "outputs"),
                        metavar="DIR", help="Where to write PNG + HTML (default: ml/outputs)")
        _a = _p.parse_args()
        sessions_dir = Path(_a.sessions_dir)
        output_dir   = Path(_a.output)

    from datetime import datetime as _dt
    run_ts  = _dt.now().strftime("%Y%m%dT%H%M%S")
    out_dir = output_dir / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # update 'latest' symlink
    latest_link = output_dir / "latest"
    if latest_link.is_symlink() or latest_link.exists():
        latest_link.unlink()
    latest_link.symlink_to(run_ts)

    # ── 0. Auto-discover sessions ─────────────────────────────────────────────
    SESSIONS = []
    if sessions_dir.exists():
        for d in sorted(sessions_dir.iterdir()):
            if not d.is_dir():
                continue
            imu_files     = sorted(d.glob("imu*.csv"))
            session_files = sorted(d.glob("session*.json"))
            labels_file   = d / "labels_ours.csv"
            if imu_files and session_files and labels_file.exists():
                short = d.name.split("_")[-1] if "_" in d.name else d.name
                SESSIONS.append({
                    "id":           d.name,
                    "label":        short,
                    "dir":          d,
                    "imu_file":     imu_files[0],
                    "session_file": session_files[0],
                })

    if len(SESSIONS) < 2:
        print(f"[ERROR] Need ≥2 sessions with imu.csv + session.json + labels_ours.csv in {sessions_dir}")
        print(f"  Run: chewing-vision fetch --all -o {sessions_dir}")
        print(f"  Then: chewing-vision analyze <session_dir>/video.mp4 -o <session_dir>/")
        return

    print(f"Found {len(SESSIONS)} sessions in {sessions_dir}")

    # ── 1. 전 세션 데이터 로드 ──────────────────────────────────────────────────
    all_X, all_y, all_groups = [], [], []
    session_cache = {}

    for sess in SESSIONS:
        imu_path     = sess.get("imu_file",     sess["dir"] / "imu.csv")
        session_path = sess.get("session_file", sess["dir"] / "session.json")
        imu, _       = load_session(str(imu_path), str(session_path))
        labels       = load_labels(str(sess["dir"] / "labels_ours.csv"))
        X, y, t_starts = make_windows_with_times(imu, labels)
        if len(X) == 0:
            print(f"[WARN] {sess['label']}: no windows, skipping")
            continue

        all_X.append(X)
        all_y.append(y)
        all_groups.append(np.full(len(y), sess["label"]))

        # GT 씹기 bout 수 계산 (labels DataFrame 기준)
        chew_labels = labels[labels["label"] == "chewing"]
        n_gt_bouts = 0
        if not chew_labels.empty:
            prev_end = None
            for _, row in chew_labels.iterrows():
                if prev_end is None or row["t_start"] - prev_end > 1.0:
                    n_gt_bouts += 1
                prev_end = row["t_end"]

        session_cache[sess["label"]] = {
            "imu":         imu,
            "labels":      labels,
            "X":           X,
            "y":           y,
            "t_starts":    t_starts,
            "duration":    float(imu["t_vision"].max() - imu["t_vision"].min()),
            "chew_ratio":  float(y.mean()),
            "n_gt_bouts":  n_gt_bouts,
        }

    X_all  = np.vstack(all_X)
    y_all  = np.concatenate(all_y)
    groups = np.concatenate(all_groups)

    # ── 2. LOSO 학습 루프 ────────────────────────────────────────────────────────
    logo = LeaveOneGroupOut()
    fold_results = []

    for train_idx, test_idx in logo.split(X_all, y_all, groups):
        held_out = groups[test_idx[0]]
        clf = RandomForestClassifier(n_estimators=20, class_weight='balanced', random_state=42)
        clf.fit(X_all[train_idx], y_all[train_idx])
        y_pred = clf.predict(X_all[test_idx])

        report = classification_report(
            y_all[test_idx], y_pred,
            labels=[0, 1], target_names=["rest", "chewing"],
            output_dict=True, zero_division=0,
        )
        cm    = confusion_matrix(y_all[test_idx], y_pred, labels=[0, 1])
        bouts = compute_bouts(session_cache[held_out]["t_starts"], y_pred)

        fold_results.append({
            "label":              held_out,
            "n_train":            len(train_idx),
            "train_chew_ratio":   float(y_all[train_idx].mean()),
            "n_test":             len(test_idx),
            "test_chew_ratio":    float(y_all[test_idx].mean()),
            "report":             report,
            "cm":                 cm,
            "bouts":              bouts,
            "estimated_chews":    sum(b[2] for b in bouts),
            "feature_importances": clf.feature_importances_,
            "imu":                session_cache[held_out]["imu"],
            "labels":             session_cache[held_out]["labels"],
            "y":                  y_all[test_idx],
            "y_pred":             y_pred,
            "t_starts":           session_cache[held_out]["t_starts"],
        })

    # ── 3. Pooled 메트릭 ────────────────────────────────────────────────────────
    y_true_pooled = np.concatenate([r["y"]      for r in fold_results])
    y_pred_pooled = np.concatenate([r["y_pred"] for r in fold_results])
    pooled_report = classification_report(
        y_true_pooled, y_pred_pooled,
        labels=[0, 1], target_names=["rest", "chewing"],
        output_dict=True, zero_division=0,
    )
    pooled_cm = confusion_matrix(y_true_pooled, y_pred_pooled, labels=[0, 1])

    # ── 4. 콘솔 출력 ────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"{'Fold (held out)':<14} {'Train n':>8} {'Train chew%':>12} "
          f"{'Test n':>8} {'Test chew%':>11} {'Acc':>6} {'F1-chew':>8} {'F1-rest':>8}")
    print(f"{'-'*70}")
    for r in fold_results:
        acc = r["report"]["accuracy"]
        f1c = r["report"]["chewing"]["f1-score"]
        f1r = r["report"]["rest"]["f1-score"]
        print(f"{r['label']:<14} {r['n_train']:>8} {r['train_chew_ratio']:>11.1%} "
              f"{r['n_test']:>8} {r['test_chew_ratio']:>10.1%} {acc:>6.3f} {f1c:>8.3f} {f1r:>8.3f}")
    print(f"{'-'*70}")
    pacc = pooled_report["accuracy"]
    pf1c = pooled_report["chewing"]["f1-score"]
    pf1r = pooled_report["rest"]["f1-score"]
    n_total = sum(r["n_test"] for r in fold_results)
    print(f"{'Pooled':<14} {'—':>8} {'—':>12} {n_total:>8} {'—':>11} "
          f"{pacc:>6.3f} {pf1c:>8.3f} {pf1r:>8.3f}")
    print(f"{'='*70}\n")

    # ── 5. 시각화 + HTML ─────────────────────────────────────────────────────────
    plot_comparison(fold_results, pooled_report, pooled_cm, session_cache, out_dir=out_dir)
    save_html_table(fold_results, pooled_report, session_cache, out_dir=out_dir)
    print(f"Report saved to {out_dir / 'session_comparison.html'}")


if __name__ == "__main__":
    main()
