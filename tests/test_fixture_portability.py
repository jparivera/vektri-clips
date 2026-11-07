"""Fixture setup must not depend on the optional GNU time executable."""
import importlib.util
from pathlib import Path
import subprocess
import sys
import time

spec = importlib.util.spec_from_file_location("fixture_generator", Path(__file__).parents[1] / "scripts/generate_fixtures.py")
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_encoder_usage_preserves_exit_status():
    for code in (0, 7):
        started = time.perf_counter()
        proc = subprocess.Popen([sys.executable, "-c", f"raise SystemExit({code})"])
        usage = generator.wait_with_usage(proc, started)
        assert proc.returncode == code
        assert usage["elapsed_seconds"] >= 0
        assert usage["peak_rss_kib"] is None or usage["peak_rss_kib"] > 0
        assert proc.wait() == code


def test_unavailable_usage_is_not_fabricated(monkeypatch):
    monkeypatch.delattr(generator.os, "wait4", raising=False)
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    usage = generator.wait_with_usage(proc, time.perf_counter())
    assert proc.returncode == 0
    assert usage["peak_rss_kib"] is None
    assert usage["peak_rss_method"] == "unavailable"
