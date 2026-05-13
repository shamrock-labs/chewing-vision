"""3개 세션 ML 결과 비교 + HTML 시각화.

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

from utils import FEATURE_NAMES, load_labels, load_session, make_windows_with_times

DATA_DIR = Path("/Users/bohyeong/Desktop/공부/project/soma/chewing_collector_data")
OUT_DIR  = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

SESSIONS = [
    {
        "id":          "20260513T145413_a61a4c",
        "label":       "a61a4c",
        "labels_path": "/tmp/chewing_a61a4c/labels_ours.csv",
    },
    {
        "id":          "20260513T145034_838cje",
        "label":       "838cje",
        "labels_path": "/tmp/chewing_838cje/labels_ours.csv",
    },
    {
        "id":          "20260513T161848_uj2e92",
        "label":       "uj2e92",
        "labels_path": "/tmp/chewing_uj2e92/labels_ours.csv",
    },
]


# ── helpers ───────────────────────────────────────────────────────────────────

def compute_estimated_chews(t_starts: np.ndarray, y_pred: np.ndarray,
                             stride_sec: float = 0.5, max_gap: float = 1.0) -> int:
    """Bout-based chew estimation: group consecutive chewing windows, duration × 1.2 Hz."""
    if len(t_starts) == 0:
        return 0
    bouts, bout_start, prev_end = [], None, None
    for t, lbl in zip(t_starts, y_pred):
        if lbl == 1:
            w_end = t + stride_sec
            if prev_end is not None and t - prev_end > max_gap:
                bouts.append((bout_start, prev_end))
                bout_start = t
            elif bout_start is None:
                bout_start = t
            prev_end = w_end
    if bout_start is not None:
        bouts.append((bout_start, prev_end))
    return sum(max(1, int((end - start) * 1.2 + 0.5)) for start, end in bouts)


# ── 1. 세션별 독립 학습 + 평가 ────────────────────────────────────────────────

def run_session(s: dict) -> dict:
    imu, _  = load_session(
        str(DATA_DIR / "sessions" / s['id'] / "imu.csv"),
        str(DATA_DIR / "sessions" / s['id'] / "session.json"),
    )
    labels       = load_labels(s["labels_path"])
    X, y, t_starts = make_windows_with_times(imu, labels, window_sec=2.0, stride_sec=0.5)
    split        = int(len(X) * 0.8)

    clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    clf.fit(X[:split], y[:split])
    y_pred_test = clf.predict(X[split:])
    y_all       = clf.predict(X)          # full-session predictions for signal plot

    report = classification_report(y[split:], y_pred_test,
                                   target_names=["rest", "chewing"], output_dict=True)
    cm     = confusion_matrix(y[split:], y_pred_test)

    duration      = imu["t_vision"].max() - imu["t_vision"].min()
    chewing_ratio = y.mean()
    top_feat      = sorted(zip(FEATURE_NAMES, clf.feature_importances_),
                           key=lambda x: -x[1])[:3]

    return {
        "label":           s["label"],
        "duration_sec":    round(duration, 1),
        "n_windows":       len(X),
        "chewing_ratio":   round(chewing_ratio * 100, 1),
        "accuracy":        round(report["accuracy"] * 100, 1),
        "f1_chewing":      round(report["chewing"]["f1-score"], 3),
        "f1_rest":         round(report["rest"]["f1-score"], 3),
        "precision_chew":  round(report["chewing"]["precision"], 3),
        "recall_chew":     round(report["chewing"]["recall"], 3),
        "confusion":       cm,
        "top_features":    top_feat,
        "y_test":          y[split:],
        "y_pred":          y_pred_test,
        "t_starts":        t_starts,       # all windows
        "t_starts_test":   t_starts[split:],
        "y_all":           y_all,
        "imu":             imu,
        "labels":          labels,
        "estimated_chews": compute_estimated_chews(t_starts, y_all),
    }


# ── 2. 합산 학습 (3세션) ──────────────────────────────────────────────────────

def run_combined() -> dict:
    all_X, all_y = [], []
    for s in SESSIONS:
        imu, _     = load_session(str(DATA_DIR / "sessions" / s['id'] / "imu.csv"),
                                  str(DATA_DIR / "sessions" / s['id'] / "session.json"))
        labels     = load_labels(s["labels_path"])
        X, y, _    = make_windows_with_times(imu, labels, window_sec=2.0, stride_sec=0.5)
        all_X.append(X); all_y.append(y)

    X     = np.vstack(all_X)
    y     = np.concatenate(all_y)
    split = int(len(X) * 0.8)

    clf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    clf.fit(X[:split], y[:split])
    y_pred = clf.predict(X[split:])

    report = classification_report(y[split:], y_pred,
                                   target_names=["rest", "chewing"], output_dict=True)
    return {
        "label":      "Combined (3 sessions)",
        "n_windows":  len(X),
        "accuracy":   round(report["accuracy"] * 100, 1),
        "f1_chewing": round(report["chewing"]["f1-score"], 3),
        "f1_rest":    round(report["rest"]["f1-score"], 3),
        "confusion":  confusion_matrix(y[split:], y_pred),
        "y_test":     y[split:],
        "y_pred":     y_pred,
    }


# ── 3. IMU 신호 서브플롯 ──────────────────────────────────────────────────────

def _plot_imu_signal(ax, imu, labels, t_starts_all, y_all, session_label, f1_val):
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
    for t_s, pred in zip(t_starts_all, y_all):
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


# ── 4. 시각화 (4행 × 4열 GridSpec) ───────────────────────────────────────────

def plot_comparison(results: list[dict], combined: dict, out_path: Path) -> None:
    fig = plt.figure(figsize=(22, 20))
    fig.suptitle("AirPods IMU Chewing Detection — Session Comparison",
                 fontsize=15, fontweight="bold", y=0.995)

    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.55, wspace=0.40)

    labels_x = [r["label"] for r in results] + [combined["label"]]

    # ── Row 0-A: F1 bar chart (cols 0-1) ──────────────────────────────────────
    ax_f1 = fig.add_subplot(gs[0, 0:2])
    f1_chew = [r["f1_chewing"] for r in results] + [combined["f1_chewing"]]
    f1_rest = [r["f1_rest"]    for r in results] + [combined["f1_rest"]]
    x = np.arange(len(labels_x)); w = 0.35
    ax_f1.bar(x - w/2, f1_chew, w, label="F1 (chewing)", color="#2ecc71", alpha=0.85)
    ax_f1.bar(x + w/2, f1_rest,  w, label="F1 (rest)",    color="#95a5a6", alpha=0.85)
    ax_f1.axhline(0.70, color="red", linestyle="--", linewidth=1, label="baseline 0.70")
    ax_f1.set_xticks(x); ax_f1.set_xticklabels(labels_x, fontsize=9)
    ax_f1.set_ylim(0, 1.15); ax_f1.set_ylabel("F1 Score")
    ax_f1.set_title("F1 Score by Session")
    ax_f1.legend(fontsize=8); ax_f1.grid(axis="y", alpha=0.3)
    for xi, v in zip(x - w/2, f1_chew):
        ax_f1.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

    # ── Row 0-B: Chewing ratio (col 2) ────────────────────────────────────────
    ax_rat = fig.add_subplot(gs[0, 2])
    ratios = [r["chewing_ratio"] for r in results]
    colors = ["#27ae60" if r > 40 else "#e67e22" for r in ratios]
    bars = ax_rat.bar([r["label"] for r in results], ratios, color=colors, alpha=0.85)
    ax_rat.set_ylabel("Chewing Ratio (%)")
    ax_rat.set_title("Chewing Ratio")
    ax_rat.set_ylim(0, 100); ax_rat.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, ratios):
        ax_rat.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{v}%", ha="center", va="bottom", fontsize=9)

    # ── Row 0-C: Estimated chew count (col 3) ─────────────────────────────────
    ax_cnt = fig.add_subplot(gs[0, 3])
    chew_counts = [r["estimated_chews"] for r in results]
    bars3 = ax_cnt.bar([r["label"] for r in results], chew_counts,
                       color=["#8e44ad", "#9b59b6", "#a569bd"], alpha=0.85)
    ax_cnt.set_ylabel("Estimated Chews")
    ax_cnt.set_title("Chew Count (ML, 1.2 Hz)")
    ax_cnt.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars3, chew_counts):
        ax_cnt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # ── Row 1: IMU signal + GT labels per session ──────────────────────────────
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[1, i])
        _plot_imu_signal(ax, r["imu"], r["labels"],
                         r["t_starts"], r["y_all"],
                         r["label"], r["f1_chewing"])

    # ── Row 2: GT vs Pred prediction timeline ─────────────────────────────────
    colors_map = {0: "#bdc3c7", 1: "#27ae60"}
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[2, i])
        y_t = r["y_test"]; y_p = r["y_pred"]
        for j in range(len(y_t)):
            ax.barh(0.7, 1, left=j, height=0.4,
                    color=colors_map[y_t[j]], alpha=0.7, linewidth=0)
            ax.barh(0.2, 1, left=j, height=0.4,
                    color=colors_map[y_p[j]], alpha=0.7, linewidth=0)
        ax.set_xlim(0, len(y_t))
        ax.set_yticks([0.2, 0.7])
        ax.set_yticklabels(["Pred", "GT"], fontsize=8)
        ax.set_title(f"{r['label']}  (F1={r['f1_chewing']:.2f})", fontsize=10)
        ax.set_xlabel("Window index (test set)", fontsize=8)
        ax.legend(handles=[
            mpatches.Patch(color="#27ae60", label="chewing", alpha=0.7),
            mpatches.Patch(color="#bdc3c7", label="rest",    alpha=0.7),
        ], fontsize=7, loc="upper right")

    # ── Row 3: Confusion matrices (3 individual + 1 combined) ─────────────────
    for i, r in enumerate(results + [combined]):
        ax = fig.add_subplot(gs[3, i])
        cm = r["confusion"]
        ax.imshow(cm, cmap="Greens", aspect="auto")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["rest", "chewing"], fontsize=8)
        ax.set_yticklabels(["rest", "chewing"], fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8); ax.set_ylabel("Actual", fontsize=8)
        ax.set_title(f"Confusion — {r['label']}", fontsize=8)
        for row in range(2):
            for col in range(2):
                ax.text(col, row, str(cm[row, col]),
                        ha="center", va="center", fontsize=12, fontweight="bold",
                        color="white" if cm[row, col] > cm.max() * 0.5 else "black")

    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"  saved: {out_path}")


# ── 5. HTML 요약 ──────────────────────────────────────────────────────────────

def save_html_table(results: list[dict], combined: dict, out_path: Path) -> None:
    def badge(v, threshold=0.70):
        color = "#27ae60" if v >= threshold else "#e74c3c"
        return f'<span style="color:{color};font-weight:bold">{v:.3f}</span>'

    rows_html = ""
    for r in results + [combined]:
        is_combined = "Combined" in r["label"]
        style = 'style="background:#f0f8f0;font-weight:bold"' if is_combined else ""
        rows_html += f"""
        <tr {style}>
          <td>{r['label']}</td>
          <td>{r.get('duration_sec', '—')}</td>
          <td>{r.get('n_windows', '—')}</td>
          <td>{r.get('chewing_ratio', '—')}%</td>
          <td>{r.get('estimated_chews', '—')}</td>
          <td>{r.get('accuracy', '—')}%</td>
          <td>{badge(r['f1_chewing'])}</td>
          <td>{badge(r['f1_rest'])}</td>
          <td>{r.get('precision_chew', '—')}</td>
          <td>{r.get('recall_chew', '—')}</td>
        </tr>"""

    top_feat_html = ""
    for r in results:
        top_feat_html += f"<h4>{r['label']}</h4><ol>"
        for name, imp in r["top_features"]:
            top_feat_html += f"<li><code>{name}</code>: {imp:.3f}</li>"
        top_feat_html += "</ol>"

    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>씹기 감지 세션 비교</title>
<style>
  body {{ font-family: -apple-system, sans-serif; max-width: 1000px; margin: 40px auto; padding: 0 24px; color: #222; }}
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

<h1>AirPods IMU 씹기 감지 — 세션 비교</h1>
<p class="subtitle">3개 세션 독립 학습 + 합산 학습 / Random Forest n=20 / window 2s stride 0.5s</p>

<div class="section">
  <h2>성능 비교 테이블</h2>
  <table>
    <thead>
      <tr>
        <th>세션</th><th>길이(초)</th><th>윈도우</th><th>씹기 비율</th>
        <th>추정 씹기 수</th><th>Accuracy</th>
        <th>F1 (chewing)</th><th>F1 (rest)</th>
        <th>Precision↑</th><th>Recall↑</th>
      </tr>
    </thead>
    <tbody>{rows_html}</tbody>
  </table>
  <div class="note">
    🎯 기준치: F1 (chewing) ≥ 0.70 | 빨간색 = 미달 / 초록색 = 통과<br>
    추정 씹기 수 = ML 예측 기반 씹기 구간 × 1.2 Hz (평균 씹기 주파수)<br>
    합산 행은 3세션 데이터를 80/20 time-based split으로 학습한 결과입니다.
  </div>
</div>

<div class="section">
  <h2>시각화</h2>
  <p style="font-size:0.85rem;color:#888">
    Row 1: F1 비교 바차트 / 씹기 비율 / 추정 씹기 수<br>
    Row 2: IMU rotation_y 신호 + GT 라벨(녹색 배경) + ML 예측 스트립(하단 10%)<br>
    Row 3: GT vs 예측 타임라인 (test set)<br>
    Row 4: Confusion Matrix (세션별 + 합산)
  </p>
  <img src="session_comparison.png" alt="세션 비교 시각화">
</div>

<div class="section">
  <h2>Top-3 중요 피처 (세션별)</h2>
  {top_feat_html}
</div>

<div class="made-with">
  <strong>🛠 어떻게 만들어졌나요?</strong><br><br>
  이 파일은 <code>ml/compare_sessions.py</code>로 자동 생성되었습니다.<br><br>
  <strong>데이터 흐름:</strong><br>
  1. <strong>chewing-vision CLI</strong> — MediaPipe FaceLandmarker로 영상에서 MAR(입 열림 비율) 추출 → 1초 window GT 라벨 생성<br>
  2. <strong>IMU 전처리</strong> (<code>ml/utils.py · make_windows_with_times</code>) — AirPods 50Hz 신호를 2초/0.5s stride 슬라이딩 윈도우로 분할, 6축×(RMS+Std) = 12차원 피처 추출<br>
  3. <strong>Random Forest</strong> (<code>sklearn</code>, n=20) — time-based 80/20 split으로 학습·평가<br>
  4. <strong>씹기 카운팅</strong> — ML 예측 결과에서 연속 chewing 윈도우를 bout 단위로 묶어 duration × 1.2 Hz로 씹기 수 추정<br>
  5. <strong>시각화</strong> (<code>matplotlib</code>) — F1 바차트, IMU 신호+GT라벨, 예측 타임라인, Confusion Matrix<br>
  6. <strong>이 HTML</strong> — Python f-string으로 생성 (<code>ml/compare_sessions.py</code>)<br><br>
  재실행: <code>.venv/bin/python ml/compare_sessions.py</code>
</div>

</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    print(f"  saved: {out_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print("▶ 세션별 분석 중...")
    results = []
    for s in SESSIONS:
        print(f"  -> {s['label']}")
        results.append(run_session(s))

    print("▶ 합산 학습 중...")
    combined = run_combined()

    print("\n── 결과 ──")
    header = f"{'세션':>14}  {'길이':>6}  {'윈도우':>6}  {'씹기%':>6}  {'추정씹기':>8}  {'Acc':>6}  {'F1-chew':>8}  {'F1-rest':>8}"
    print(header)
    print("─" * len(header))
    for r in results + [combined]:
        print(f"{r['label']:>14}  "
              f"{r.get('duration_sec','—'):>6}  "
              f"{r.get('n_windows','—'):>6}  "
              f"{r.get('chewing_ratio','—'):>6}  "
              f"{str(r.get('estimated_chews','—')):>8}  "
              f"{r.get('accuracy','—'):>6}  "
              f"{r['f1_chewing']:>8.3f}  "
              f"{r['f1_rest']:>8.3f}")

    print("\n▶ 시각화 생성 중...")
    plot_comparison(results, combined, OUT_DIR / "session_comparison.png")
    save_html_table(results, combined, OUT_DIR / "session_comparison.html")
    print("\n완료. 브라우저에서 열기:")
    print(f"  open {OUT_DIR / 'session_comparison.html'}")


if __name__ == "__main__":
    main()
