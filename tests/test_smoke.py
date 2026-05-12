import subprocess
import time
from pathlib import Path

import pytest

from chewing.engines.orofac import OrofacEngine
from chewing.engines.ours import OursEngine


FIXTURE = Path("tests/fixtures/sample_chewing_1.mp4")


pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(),
    reason="sample chewing fixture is not present in this checkout",
)


def test_fixture_exists_and_min_size():
    assert FIXTURE.exists()
    assert FIXTURE.stat().st_size > 100_000


@pytest.mark.slow
def test_ours_engine_smoke():
    start = time.monotonic()
    result = OursEngine().analyze(str(FIXTURE))
    elapsed = time.monotonic() - start

    assert elapsed < 300
    assert result.n_chews > 0
    assert result.face_detection_rate >= 0.7
    assert len(result.frames) > 0


@pytest.mark.slow
def test_orofac_engine_smoke():
    start = time.monotonic()
    result = OrofacEngine().analyze(str(FIXTURE))
    elapsed = time.monotonic() - start

    assert elapsed < 300
    assert result.n_chews > 0


@pytest.mark.slow
def test_demo_command_produces_files(tmp_path):
    completed = subprocess.run(
        [".venv/bin/chewing", "demo", str(FIXTURE), "-o", str(tmp_path)],
        check=False,
        timeout=600,
    )

    assert completed.returncode == 0
    expected = [
        "frame_signals_ours.csv",
        "frame_signals_orofac.csv",
        "labels_ours.csv",
        "labels_orofac.csv",
        "events_ours.csv",
        "events_orofac.csv",
        "bouts_ours.csv",
        "bouts_orofac.csv",
        "summary.json",
        "signals.png",
        "demo.mp4",
    ]
    for name in expected:
        assert (tmp_path / name).exists(), name
