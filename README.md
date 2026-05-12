# chewing-vision

## Overview

`chewing-vision` is a local CLI and Python package that turns a video of eating into chewing signals, event labels, window labels, bout summaries, plots, and a demo overlay MP4. It is designed as a weak-label generator for AirPods IMU ground-truth workflows, not as a medical or nutrition product.

## Installation

Use the project virtualenv and avoid system Python:

```bash
cd /Users/bohyeong/Desktop/공부/project/soma/chewing-vision
.venv/bin/pip install -e ".[dev]"
```

The package requires Python 3.10+, MediaPipe, OpenCV, SciPy, Matplotlib, pandas, Pillow, and `orofacIAnalysis==0.1.2`.

## Quickstart

```bash
.venv/bin/chewing demo tests/fixtures/sample_chewing_1.mp4 -o /tmp/cv_demo_out/
```

Expected console shape:

```text
engine  duration_s  n_chews  chews/min  face_rate  warnings
------  ----------  -------  ---------  ---------  --------
ours    91.03       85       56.0       1.00       0
orofac  91.07       95       62.6       N/A        2
demo outputs written to /tmp/cv_demo_out/
```

## CLI Reference

```bash
.venv/bin/chewing analyze VIDEO.mp4 --engine ours|orofac|both -o out/
```

Writes frame signals, window labels, events, bouts, and `summary.json`.

```bash
.venv/bin/chewing plot VIDEO.mp4 --engine ours -o signals.png
```

Writes a two-panel MAR/jawOpen PNG with peak markers and chewing windows.

```bash
.venv/bin/chewing overlay VIDEO.mp4 --engine ours -o demo.mp4
```

Writes a 1600x840 demo MP4 with video, sidebar, rolling signal trace, and peak markers.

```bash
.venv/bin/chewing eval --auto AUTO_LABELS.csv --human HUMAN_LABELS.csv
```

Compares window labels and prints JSON metrics. Optional event inputs:

```bash
.venv/bin/chewing eval --auto AUTO.csv --human HUMAN.csv --auto-events AUTO_EVENTS.csv --human-events HUMAN_EVENTS.csv
```

```bash
.venv/bin/chewing compare --a LABELS_A.csv --b LABELS_B.csv
```

Compares two label CSVs and prints cross-engine agreement metrics.

```bash
.venv/bin/chewing demo VIDEO.mp4 -o out/
```

Runs analyze with both engines, writes `signals.png`, renders `demo.mp4`, and prints a summary table.

## Output Schemas

`frame_signals_{engine}.csv`:

```text
t_sec,frame_index,face_found,mar,jaw_open,chin_y,head_motion,quality
```

`labels_{engine}.csv`:

```text
t_start,t_end,label,confidence,quality,n_events,engine,mar_mean,jaw_open_mean
```

`events_{engine}.csv`:

```text
t_sec,frame_index,signal,value,confidence,engine,side
```

`bouts_{engine}.csv`:

```text
t_start,t_end,n_events,chews_per_min,confidence,engine
```

`summary.json` contains the analyzed video metadata, per-engine counts, quality/warnings, and agreement metrics. See `SPEC.md` §7 for the canonical schema.

## License + Attribution

This project is MIT licensed. See `LICENSE`.

Uses orofacIAnalysis (MIT) by Cameron Maloney — see ATTRIBUTION.md.
