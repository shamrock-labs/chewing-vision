"""Import existing sessions CSV data into InsForge DB.

Reads sessions/ directory, loads labels_ours.csv + labels_ours_jaw_open.csv,
generates a SQL file and imports via: npx @insforge/cli db import

Usage:
    .venv/bin/python ml/import_to_db.py [--mode disagree|all]

    --mode disagree  (default) only windows where engines disagree
    --mode all       all windows
"""
import argparse
import csv
import subprocess
import tempfile
from pathlib import Path

MAIN_ROOT    = Path(__file__).resolve().parents[1]
SESSIONS_DIR = MAIN_ROOT / "sessions"


def _esc(v) -> str:
    if v is None or v == "":
        return "NULL"
    if isinstance(v, float):
        return str(v)
    return "'" + str(v).replace("'", "''") + "'"


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def build_sql(mode: str) -> str:
    lines: list[str] = []

    dirs = sorted(d for d in SESSIONS_DIR.iterdir() if d.is_dir())
    print(f"[import_to_db] {len(dirs)} session dirs, mode={mode}")

    for sdir in dirs:
        comp_path = sdir / "labels_ours.csv"
        jaw_path  = sdir / "labels_ours_jaw_open.csv"
        if not comp_path.exists() or not jaw_path.exists():
            print(f"  [skip] {sdir.name}: missing label CSVs")
            continue

        comp_rows = _read_csv(comp_path)
        jaw_rows  = _read_csv(jaw_path)
        jaw_by_t  = {float(r["t_start"]): r for r in jaw_rows}

        windows: list[dict] = []
        for crow in comp_rows:
            t_start = float(crow["t_start"])
            jrow    = jaw_by_t.get(t_start)
            if jrow is None:
                continue
            if mode == "disagree" and crow["label"] == jrow["label"]:
                continue
            windows.append({
                "session_id":      sdir.name,
                "t_start":         t_start,
                "t_end":           float(crow["t_end"]),
                "composite_label": crow["label"],
                "jaw_open_label":  jrow["label"],
                "jaw_open_mean":   float(crow["jaw_open_mean"]) if crow.get("jaw_open_mean") else None,
                "mar_mean":        float(crow["mar_mean"])       if crow.get("mar_mean")       else None,
            })

        if not windows:
            print(f"  [skip] {sdir.name}: no windows for mode={mode}")
            continue

        # Session upsert
        lines.append(
            f"INSERT INTO sessions (id) VALUES ({_esc(sdir.name)}) "
            f"ON CONFLICT (id) DO NOTHING;"
        )
        # Window UPSERT: refresh machine labels but preserve human_label/labeled_at
        for w in windows:
            lines.append(
                f"INSERT INTO windows "
                f"(session_id, t_start, t_end, composite_label, jaw_open_label, jaw_open_mean, mar_mean) "
                f"VALUES ({_esc(w['session_id'])}, {w['t_start']}, {w['t_end']}, "
                f"{_esc(w['composite_label'])}, {_esc(w['jaw_open_label'])}, "
                f"{_esc(w['jaw_open_mean'])}, {_esc(w['mar_mean'])}) "
                f"ON CONFLICT (session_id, t_start) DO UPDATE SET "
                f"t_end = EXCLUDED.t_end, "
                f"composite_label = EXCLUDED.composite_label, "
                f"jaw_open_label = EXCLUDED.jaw_open_label, "
                f"jaw_open_mean = EXCLUDED.jaw_open_mean, "
                f"mar_mean = EXCLUDED.mar_mean;"
            )
        print(f"  {sdir.name}: {len(windows)} windows queued")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["disagree", "all"], default="disagree")
    args = ap.parse_args()

    sql = build_sql(args.mode)
    if not sql.strip():
        print("[import_to_db] Nothing to import.")
        return

    with tempfile.NamedTemporaryFile(suffix=".sql", mode="w", delete=False) as f:
        f.write(sql)
        sql_path = f.name

    print(f"[import_to_db] Running: npx @insforge/cli db import {sql_path}")
    result = subprocess.run(
        ["npx", "@insforge/cli", "db", "import", sql_path],
        cwd=str(MAIN_ROOT),
    )
    if result.returncode != 0:
        print("[import_to_db] Import failed.")
    else:
        print("[import_to_db] Done.")


if __name__ == "__main__":
    main()
