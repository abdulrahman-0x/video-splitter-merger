"""Tests for the WP1-WP8 changes: tail integrity, dual-mono, temp dir, settings, messages."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest

from src.core import settings as app_settings
from src.core.denoise import (
    DF_DELAY_SAMPLES,
    DenoisePartStatus,
    DenoiseReport,
    build_padded_input_filter,
    choose_temp_base,
    cleanup_temp_base,
    estimate_ram_mb,
    guard_admits,
    pcm16_rms_from_wav,
    tail_is_damaged,
    wav_file_sample_count,
    write_report_file,
)
from src.core.ffmpeg_manager import probe_media

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FX_ROOT = PROJECT_ROOT / "tmp" / "dfn_fix_tests"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fx_dir() -> "object":
    if FX_ROOT.exists():
        shutil.rmtree(FX_ROOT, ignore_errors=True)
    FX_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(PROJECT_ROOT)).free
    if free < 300 * 1024 * 1024:
        pytest.skip(f"المساحة الحرّة غير كافية: {free // (1024 * 1024)} MB")
    yield FX_ROOT
    shutil.rmtree(FX_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def ffmpeg_tools() -> Tuple[Path, Path]:
    from src.core.ffmpeg_manager import find_tool

    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe غير متوفرة")
    return ffmpeg, ffprobe


@pytest.fixture(scope="session")
def df_exe() -> Path:
    from src.core.denoise import find_deep_filter

    exe = find_deep_filter()
    if exe is None:
        pytest.skip("deep-filter.exe غير موجود؛ تُخطّى الاختبارات التي تحتاجه")
    return exe


def _sub(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_wav(path: Path) -> "object":
    np = pytest.importorskip("numpy")
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return frames.reshape(-1, nch).astype("float64") / 32768.0


def _trailing_zeros(x: "object", limit: int) -> int:
    np = pytest.importorskip("numpy")
    tail = x[-limit:]
    nz = np.nonzero(tail)[0]
    return int(len(tail)) if len(nz) == 0 else int(len(tail) - 1 - nz[-1])


def _make_wav(path: Path, samples: List[int], channels: int = 1) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(48000)
        wf.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return path


# ---------------------------------------------------------------------------
# WP1: tail integrity primitives
# ---------------------------------------------------------------------------

class TestTailPrimitives:
    def test_pcm16_rms_detects_silent_tail(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "tailrms")
        # 1 s total so the 0.1 s window really is only the tail
        loud = _make_wav(d / "loud.wav", [3000] * 48000)
        silent_tail = _make_wav(d / "silent.wav", [3000] * 43200 + [0] * 4800)
        assert pcm16_rms_from_wav(loud) > 0.05
        assert pcm16_rms_from_wav(silent_tail) == pytest.approx(0.0, abs=1e-12)
        # the first-100 ms variant sees the loud part
        assert pcm16_rms_from_wav(silent_tail, from_end=False) > 0.05

    def test_pcm16_rms_rejects_non_wav(self, fx_dir: Path) -> None:
        p = _sub(fx_dir, "tailbad") / "x.bin"
        p.write_bytes(b"not a wav")
        assert pcm16_rms_from_wav(p) is None
        assert pcm16_rms_from_wav(fx_dir / "missing.wav") is None

    def test_tail_is_damaged_matrix(self) -> None:
        assert tail_is_damaged(0.05, 0.0) is True       # audible source, silent output
        assert tail_is_damaged(0.05, 0.04) is False     # both audible
        assert tail_is_damaged(0.0, 0.0) is False       # source itself silent
        assert tail_is_damaged(0.0001, 0.0) is False    # source effectively silent
        assert tail_is_damaged(None, 0.0) is False
        assert tail_is_damaged(0.05, None) is False


# ---------------------------------------------------------------------------
# WP2: RAM formula and guard
# ---------------------------------------------------------------------------

class TestRamModel:
    def test_formula_matches_measured_table(self) -> None:
        measured = {
            (120.0, 1): 96.1, (600.0, 1): 259.7, (1800.0, 1): 699.3,
            (120.0, 2): 174.5, (600.0, 2): 701.7, (1200.1, 2): 1360.7,
            (1800.0, 2): 2020.2,
        }
        for (dur, ch), actual in measured.items():
            assert abs(estimate_ram_mb(dur, ch) - actual) / actual <= 0.05

    def test_dual_mono_uses_mono_cost(self) -> None:
        # The shortcut must be priced as 1 channel.
        assert estimate_ram_mb(600, 1) < estimate_ram_mb(600, 2) / 2.5

    def test_guard_decision_table(self) -> None:
        cases = [
            # (duration, channels, avail, admitted)
            (600, 1, 2700, True),
            (600, 2, 2700, True),
            (1800, 1, 2700, True),
            (1800, 2, 2700, False),
            (1200, 2, 2700, True),
            (600, 2, 1200, False),   # limit 432 MB < 804 MB needed
            (60, 1, 900, True),
            (600, 1, 800, False),    # limit 32 MB
        ]
        for dur, ch, avail, expected in cases:
            ok, est, limit = guard_admits(dur, ch, avail)
            assert ok is expected, (dur, ch, avail, ok, est, limit)


# ---------------------------------------------------------------------------
# WP3: dual-mono detection
# ---------------------------------------------------------------------------

class TestDualMono:
    def test_detects_identical_channels(
        self, fx_dir: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        from src.core.denoise import _CancelState, is_dual_mono

        ffmpeg, _ = ffmpeg_tools
        d = _sub(fx_dir, "dm_yes")
        src = d / "dm.wav"
        subprocess.run([
            str(ffmpeg), "-nostdin", "-y", "-f", "lavfi",
            "-i", "sine=frequency=440:duration=12:sample_rate=48000",
            "-af", "pan=stereo|c0=c0|c1=c0", "-c:a", "pcm_s16le", str(src),
        ], check=True, capture_output=True)
        dual, residual = is_dual_mono(ffmpeg, src, 2, 12.0, _CancelState(None))
        assert dual is True
        assert residual is not None and residual > 60.0

    def test_rejects_different_channels(
        self, fx_dir: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        from src.core.denoise import _CancelState, is_dual_mono

        ffmpeg, _ = ffmpeg_tools
        d = _sub(fx_dir, "dm_no")
        src = d / "st.wav"
        subprocess.run([
            str(ffmpeg), "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12:sample_rate=48000",
            "-f", "lavfi", "-i", "sine=frequency=660:duration=12:sample_rate=48000",
            "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(src),
        ], check=True, capture_output=True)
        dual, residual = is_dual_mono(ffmpeg, src, 2, 12.0, _CancelState(None))
        assert dual is False
        assert residual is not None and residual < 10.0

    def test_rejects_near_identical_at_minus_40(
        self, fx_dir: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        """A -40 dB residual must NOT be treated as dual-mono (conservative)."""
        from src.core.denoise import _CancelState, is_dual_mono

        ffmpeg, _ = ffmpeg_tools
        d = _sub(fx_dir, "dm_40")
        src = d / "near.wav"
        # L = tone, R = tone + very quiet independent noise (-40 dB)
        subprocess.run([
            str(ffmpeg), "-nostdin", "-y",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=12:sample_rate=48000",
            "-f", "lavfi", "-i",
            "anoisesrc=color=white:amplitude=0.01:duration=12:sample_rate=48000:seed=3",
            "-filter_complex",
            "[0:a]asplit=2[l][r];[r][1:a]amix=inputs=2:duration=first:normalize=0[rr];"
            "[l][rr]amerge=inputs=2[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(src),
        ], check=True, capture_output=True)
        dual, residual = is_dual_mono(ffmpeg, src, 2, 12.0, _CancelState(None))
        assert dual is False
        assert residual is not None and residual < 60.0

    def test_single_channel_never_shortcuts(self, fx_dir: Path) -> None:
        from src.core.denoise import _CancelState, is_dual_mono

        dual, residual = is_dual_mono(Path("ffmpeg"), fx_dir / "nope.wav", 1, 10.0,
                                      _CancelState(None))
        assert dual is False and residual is None


# ---------------------------------------------------------------------------
# WP4: temp location
# ---------------------------------------------------------------------------

class TestTempLocation:
    def test_temp_inside_parts_folder(self, fx_dir: Path) -> None:
        parts = _sub(fx_dir, "tempin")
        path, created = choose_temp_base(parts)
        try:
            assert created is True
            assert path.parent == parts
            assert path.name.startswith(".dfn_tmp_")
            assert path.is_dir()
        finally:
            cleanup_temp_base(path, created)
        assert not path.exists()

    def test_explicit_request_is_used_verbatim(self, fx_dir: Path) -> None:
        explicit = _sub(fx_dir, "tempexp")
        path, created = choose_temp_base(fx_dir / "parts", str(explicit))
        assert path == explicit and created is False
        cleanup_temp_base(path, created)
        assert path.exists()  # user-supplied dirs are not deleted

    def test_fallback_when_parts_folder_unwritable(self, fx_dir: Path, monkeypatch) -> None:
        import tempfile

        parts = _sub(fx_dir, "tempfb")
        real_mkdir = Path.mkdir

        def boom(self, *a, **k):
            if ".dfn_tmp_" in str(self):
                raise PermissionError("denied")
            return real_mkdir(self, *a, **k)

        monkeypatch.setattr(Path, "mkdir", boom)
        path, created = choose_temp_base(parts)
        assert path == Path(tempfile.gettempdir())
        assert created is False

    def test_cleanup_ignores_foreign_dir(self, fx_dir: Path) -> None:
        keep = _sub(fx_dir, "tempkeep")
        cleanup_temp_base(keep, created_by_us=False)
        assert keep.exists()


# ---------------------------------------------------------------------------
# WP5: settings persistence
# ---------------------------------------------------------------------------

class TestSettings:
    def test_round_trip(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        data = {
            "enabled": True, "strength_db": 20.0, "auto_convert": False,
            "restore_original_rate": False, "denoise_dual_mono_shortcut": False,
            "deep_filter_path": r"F:\DeepFilterNet\deep-filter.exe",
            "rtf_history": {"1": [0.2, 0.21], "2": [0.3]},
        }
        assert app_settings.save(data, p) is True
        loaded = app_settings.load(p)
        for key in ("enabled", "strength_db", "auto_convert", "restore_original_rate",
                    "denoise_dual_mono_shortcut", "deep_filter_path"):
            assert loaded[key] == data[key], key
        assert loaded["rtf_history"]["1"] == [0.2, 0.21]
        assert json.loads(p.read_text(encoding="utf-8"))["version"] == 1

    def test_missing_file_gives_defaults(self, tmp_path: Path) -> None:
        loaded = app_settings.load(tmp_path / "nope.json")
        assert loaded == app_settings.sanitize({})
        assert loaded["strength_db"] == 12.0
        assert loaded["denoise_dual_mono_shortcut"] is True

    def test_corrupt_file_gives_defaults(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text("{ this is not json", encoding="utf-8")
        assert app_settings.load(p) == app_settings.sanitize({})
        p.write_text("[]", encoding="utf-8")
        assert app_settings.load(p) == app_settings.sanitize({})

    def test_unknown_keys_ignored(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"version": 1, "strength_db": 30, "mystery": {"a": 1}}),
                     encoding="utf-8")
        loaded = app_settings.load(p)
        assert loaded["strength_db"] == 30.0
        assert "mystery" not in loaded

    def test_invalid_values_fall_back(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({
            "strength_db": "abc", "enabled": "maybe", "rtf_history": {"2": [0.3, "x", 99, -1]},
        }), encoding="utf-8")
        loaded = app_settings.load(p)
        assert loaded["strength_db"] == 12.0
        assert loaded["enabled"] is False
        assert loaded["rtf_history"]["2"] == [0.3]

    def test_rtf_history_capped(self, tmp_path: Path) -> None:
        p = tmp_path / "settings.json"
        p.write_text(json.dumps({"rtf_history": {"1": [0.1 * i for i in range(1, 21)]}}),
                     encoding="utf-8")
        assert len(app_settings.load(p)["rtf_history"]["1"]) == app_settings.RTF_HISTORY_CAP

    def test_atomic_write_keeps_old_file_on_failure(self, tmp_path: Path, monkeypatch) -> None:
        p = tmp_path / "settings.json"
        assert app_settings.save({"strength_db": 6.0}, p) is True
        before = p.read_text(encoding="utf-8")
        real_replace = app_settings.os.replace

        def boom(src, dst):
            raise OSError("disk full")

        monkeypatch.setattr(app_settings.os, "replace", boom)
        assert app_settings.save({"strength_db": 100.0}, p) is False
        monkeypatch.setattr(app_settings.os, "replace", real_replace)
        assert p.read_text(encoding="utf-8") == before
        assert app_settings.load(p)["strength_db"] == 6.0
        # no stray temp files left behind
        assert list(tmp_path.glob("settings_*.tmp")) == []

    def test_strength_is_clamped(self) -> None:
        assert app_settings.sanitize({"strength_db": 0})["strength_db"] == 1.0
        assert app_settings.sanitize({"strength_db": 500})["strength_db"] == 100.0
        assert app_settings.sanitize({"strength_db": float("nan")})["strength_db"] == 12.0


# ---------------------------------------------------------------------------
# WP6: counters, message semantics, report file
# ---------------------------------------------------------------------------

class TestReportSemantics:
    def test_counters_separate_skip_from_failure(self) -> None:
        report = DenoiseReport(statuses=[
            DenoisePartStatus(part=Path("a.wav"), ok=True, channels=1),
            DenoisePartStatus(part=Path("b.wav"), ok=False, skipped=True, reason="ليس WAV"),
            DenoisePartStatus(part=Path("c.wav"), ok=False, reason="نفدت الذاكرة"),
        ])
        counts = report.counts()
        assert counts == {"denoised": 1, "skipped": 1, "failed": 1, "total": 3}
        assert [s.outcome for s in report.statuses] == ["denoised", "skipped", "failed"]
        assert len(report.skips) == 1
        assert len(report.failures) == 1

    def test_zero_denoised_never_says_success(self) -> None:
        report = DenoiseReport(statuses=[
            DenoisePartStatus(part=Path("a.wav"), ok=False, reason="الذاكرة غير كافية"),
        ])
        lines = report.summary_lines()
        assert lines
        assert "بنجاح" not in lines[0]
        assert "لم يتم تقليل الضوضاء لأي جزء" in lines[0]
        assert "الذاكرة غير كافية" in lines[0]

    def test_skip_is_reported_as_warning_not_failure(self) -> None:
        report = DenoiseReport(statuses=[
            DenoisePartStatus(part=Path("a.wav"), ok=True),
            DenoisePartStatus(part=Path("b.wav"), ok=False, skipped=True, reason="تحويل مغلق"),
        ])
        lines = report.summary_lines()
        assert "1 من 2" in lines[0]
        assert "تُخطّي 1" in lines[0]
        assert any("b.wav" in line and "تحويل مغلق" in line for line in lines)

    def test_report_file_contents(self, tmp_path: Path) -> None:
        report = DenoiseReport(statuses=[
            DenoisePartStatus(part=Path("p1.wav"), ok=True, rtf=0.31, clip_events=4,
                              wall_s=12.5, channels=1, dual_mono=True, tail_restored=True),
            DenoisePartStatus(part=Path("p2.wav"), ok=False, skipped=True,
                              reason="غير مطابق", wall_s=0.1),
        ])
        report.elapsed_s = 13.0
        path = write_report_file(tmp_path, report)
        assert path is not None and path.is_file()
        text = path.read_text(encoding="utf-8")
        assert "denoised" in text and "skipped" in text
        assert "p1.wav" in text and "p2.wav" in text
        assert "yes" in text          # dual_mono / tail_restored columns
        assert "\t0.3100\t" in text   # rtf column

    def test_report_file_failure_is_not_fatal(self, tmp_path: Path) -> None:
        missing = tmp_path / "no_such_dir" / "x"
        # Path.write_text raises OSError (missing parent); must return None
        assert write_report_file(missing, DenoiseReport()) is None


# ---------------------------------------------------------------------------
# WP1 integration: the real tool must not zero the tail
# ---------------------------------------------------------------------------

@pytest.mark.deepfilter_exe
class TestTailWithRealTool:
    def test_s1_landmark_tail_survives(
        self, fx_dir: Path, df_exe: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        """440 Hz + a noise burst ending at EOF: the burst must survive."""
        np = pytest.importorskip("numpy")
        from src.core.denoise import denoise_wav

        ffmpeg, ffprobe = ffmpeg_tools
        d = _sub(fx_dir, "wp1_s1")
        src = d / "s1.wav"
        n = 48000 * 4
        t = np.arange(n) / 48000.0
        x = 0.3 * np.sin(2 * np.pi * 440 * t)
        rng = np.random.default_rng(11)
        x[n - 9600:] += rng.standard_normal(9600) * 0.25
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
        with wave.open(str(src), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(pcm.tobytes())

        dst = d / "s1_out.wav"
        info = denoise_wav(src, dst, strength_db=12, exe=df_exe, ffmpeg=ffmpeg,
                           ffprobe=ffprobe, temp_base=fx_dir)
        out = _read_wav(dst)[:, 0]
        assert wav_file_sample_count(dst) == wav_file_sample_count(src)
        assert _trailing_zeros(out, 4800) <= 2
        # the burst is still there: the last 100 ms is well above silence
        tail_rms = float(np.sqrt(np.mean(out[-4800:] ** 2)))
        assert tail_rms > 0.01
        assert info["tail_restored"] is False

    def test_real_speech_tail_survives(
        self, fx_dir: Path, df_exe: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        np = pytest.importorskip("numpy")
        from src.core.denoise import denoise_wav

        ffmpeg, ffprobe = ffmpeg_tools
        d = _sub(fx_dir, "wp1_speech")
        # 20 s of speech-like material cut mid-signal (no fade at the end)
        src = d / "speech.wav"
        subprocess.run([
            str(ffmpeg), "-nostdin", "-y", "-f", "lavfi",
            "-i", "sine=frequency=220:duration=20:sample_rate=48000",
            "-f", "lavfi", "-i",
            "anoisesrc=color=pink:amplitude=0.08:duration=20:sample_rate=48000:seed=5",
            "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
            "-map", "[a]", "-c:a", "pcm_s16le", str(src),
        ], check=True, capture_output=True)
        dst = d / "speech_out.wav"
        denoise_wav(src, dst, strength_db=12, exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe,
                    temp_base=fx_dir)
        out = _read_wav(dst)[:, 0]
        assert _trailing_zeros(out, 4800) <= 2
        assert float(np.sqrt(np.mean(out[-4800:] ** 2))) > 0.001

    def test_ten_parts_have_no_silent_join(
        self, fx_dir: Path, df_exe: Path, ffmpeg_tools: Tuple[Path, Path]
    ) -> None:
        """Ten 3 s parts denoised separately: no all-zero tail at any join."""
        np = pytest.importorskip("numpy")
        from src.core.denoise import denoise_wav

        ffmpeg, ffprobe = ffmpeg_tools
        d = _sub(fx_dir, "wp1_join")
        part_s = 3.0
        parts: List[Path] = []
        for i in range(10):
            p = d / f"p{i:02d}.wav"
            subprocess.run([
                str(ffmpeg), "-nostdin", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={300 + 20 * i}:duration={part_s}:sample_rate=48000",
                "-f", "lavfi", "-i",
                f"anoisesrc=color=pink:amplitude=0.05:duration={part_s}"
                f":sample_rate=48000:seed={i + 1}",
                "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=first:normalize=0[a]",
                "-map", "[a]", "-c:a", "pcm_s16le", str(p),
            ], check=True, capture_output=True)
            out = d / f"p{i:02d}_out.wav"
            denoise_wav(p, out, strength_db=12, exe=df_exe, ffmpeg=ffmpeg,
                        ffprobe=ffprobe, temp_base=fx_dir)
            parts.append(out)

        n = int(part_s * 48000)
        worst_step = 0.0
        worst_p99 = 1e-9
        for p in parts:
            x = _read_wav(p)[:, 0]
            assert len(x) == n
            assert _trailing_zeros(x, 4800) <= 2, p.name
            last200 = x[-9600:]
            worst_step = max(worst_step, float(np.abs(np.diff(last200)).max()))
            worst_p99 = max(worst_p99, float(np.percentile(np.abs(np.diff(x[:-9600])), 99)))
        assert worst_step <= 3.0 * worst_p99, (worst_step, worst_p99)

        # and the concatenation has no zero run at any boundary
        cat = np.concatenate([_read_wav(p)[:, 0] for p in parts])
        for k in range(1, 10):
            assert _trailing_zeros(cat[: k * n], 1440) <= 2, f"join {k}"
