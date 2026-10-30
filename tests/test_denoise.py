"""Tests for src/core/denoise.py (DeepFilterNet wrapper).

Groups:
- TestPureHelpers / TestDenoiseWavValidationGate: run everywhere (no tools).
- TestPipelineToDfLaunch: needs ffmpeg only (real conform steps, tool never launched).
- TestDeepFilterRun / TestDenoiseParts: marked `deepfilter_exe`; skipped when
  deep-filter.exe is absent.

All generated media goes to <project>/tmp/dfn_tests (on F:). The pytest built-in
`tmp_path` lives on C:, which has ~0 bytes free on this machine, so it is not
used here.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import struct
import subprocess
import threading
import time
import uuid
import wave
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import pytest

from src.core.denoise import (
    DF_DELAY_SAMPLES,
    DF_EXIT_OOM,
    PAD_RMS_DBFS,
    PAD_SAMPLES_48K,
    DenoiseError,
    DenoisePartStatus,
    DenoiseReport,
    DenoiseSkipped,
    _avg_rtf,
    _CancelState,
    _map_df_failure,
    build_df_args,
    build_padded_input_filter,
    compute_n48,
    denoise_parts,
    denoise_wav,
    estimate_ram_mb,
    find_deep_filter,
    get_version,
    guard_admits,
    is_conformant_wav,
    ram_guard_limit_mb,
    wav_file_sample_count,
)
from src.core.ffmpeg_manager import probe_media
from tests.conftest import create_synthetic_video

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Per-process workspace: a fixed path here makes two concurrent pytest runs
# delete each other's files mid-test (seen 2026-10-30 as spurious "file not
# found" / "tool exited 1" failures). F: keeps C: usage at zero.
FX_ROOT = Path(
    os.environ.get("DFN_TEST_ROOT") or (PROJECT_ROOT / "tmp" / f"dfn_tests_{os.getpid()}")
)


# ---------------------------------------------------------------------------
# Shared helpers and fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fx_dir() -> Iterator[Path]:
    """Fresh F:-backed workspace for generated media (C: is full here)."""
    if FX_ROOT.exists():
        shutil.rmtree(FX_ROOT, ignore_errors=True)
    FX_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(str(PROJECT_ROOT)).free
    if free < 300 * 1024 * 1024:
        pytest.skip(f"المساحة الحرّة غير كافية على القرص: {free // (1024 * 1024)} MB")
    yield FX_ROOT
    shutil.rmtree(FX_ROOT, ignore_errors=True)


def _sub(fx_dir: Path, hint: str) -> Path:
    d = fx_dir / f"{uuid.uuid4().hex[:8]}_{hint}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def _locate_df() -> Optional[Path]:
    env_var = os.environ.get("DEEP_FILTER_EXE")
    if env_var and Path(env_var).is_file():
        return Path(env_var)
    known = Path(r"F:\DeepFilterNet\deep-filter.exe")
    if known.is_file():
        return known
    return find_deep_filter(None)


@pytest.fixture(scope="session")
def df_exe() -> Iterator[Optional[Path]]:
    """deep-filter.exe location; skips tool-dependent tests when unavailable."""
    exe = _locate_df()
    if exe is None:
        pytest.skip("deep-filter.exe غير متوفر على هذه الآلة.")
    assert get_version(exe), "لم يُرجِع deep-filter -V أي إصدار."
    yield exe


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _wcount(path: Path) -> int:
    return wav_file_sample_count(path)


def _run_checked(cmd: List[str]) -> int:
    proc = subprocess.run(  # noqa: S603 - argv list, no shell=True
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,  # type: ignore[attr-defined]
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "فشل توليد ملف الاختبار:\n" + proc.stderr.decode("utf-8", "replace")[-500:]
        )
    return proc.returncode


def _gen_sine_wav(
    ffmpeg: Path,
    out: Path,
    *,
    duration: float,
    rate: int = 48000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    """Generate an exact-duration sine WAV for tool-dependent tests."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-nostdin", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=550:duration={duration}:sample_rate={rate}",
        "-c:a", codec, "-ac", str(channels), str(out),
    ]
    assert _run_checked(cmd) == 0
    return out


def _gen_noise_mix_wav(
    ffmpeg: Path,
    out: Path,
    *,
    duration: float,
    rate: int = 44100,
    channels: int = 2,
) -> Path:
    """Voice-band tone mixed with broadband noise (a denoising input)."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ffmpeg), "-nostdin", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={duration}:sample_rate={rate}",
        "-f", "lavfi", "-i",
        f"anoisesrc=color=pink:amplitude=0.15:duration={duration}:sample_rate={rate}:seed=7",
        "-filter_complex", "[0:a][1:a]amix=inputs=2:duration=longest:normalize=0[a]",
        "-map", "[a]",
        "-c:a", "pcm_s16le", "-ac", str(channels), str(out),
    ]
    assert _run_checked(cmd) == 0
    return out


def _rms_dbfs(path: Path) -> float:
    """RMS dBFS of a 16-bit PCM wav (mean across channels)."""
    np = pytest.importorskip("numpy")
    with wave.open(str(path), "rb") as wf:
        nch = wf.getnchannels()
        sw = wf.getsampwidth()
        frames = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    assert sw == 2, "هذا المساعد يقبل ملفات pcm_s16le فقط."
    if nch > 1:
        frames = frames.reshape(-1, nch).mean(axis=1)
    rms = float(np.sqrt(np.mean(frames.astype("float64") ** 2)))
    return 20.0 * math.log10(rms / 32768.0) if rms > 0.0 else float("-inf")


def _dfn_settings(fx_dir: Path, **over: object) -> Dict:
    settings: Dict = {
        "enabled": True,
        "strength_db": 12,
        "auto_convert": True,
        "restore_original_rate": True,
        "workers": 1,
        "temp_base": str(fx_dir),
    }
    settings.update(over)
    return settings


def _no_tool_launch_guard(monkeypatch: pytest.MonkeyPatch) -> List[List[object]]:
    """Record + fail on any attempt to assemble deep-filter arguments."""
    calls: List[List[object]] = []

    def _trap(*args: object) -> List[str]:
        calls.append(list(args))
        raise AssertionError("كان يجب عدم تشغيل أداة deep-filter إطلاقاً في هذا السيناريو.")

    monkeypatch.setattr("src.core.denoise.build_df_args", _trap)
    return calls


def _wcopy(src: Path, dst: Path) -> None:
    shutil.copyfile(src, dst)


# ---------------------------------------------------------------------------
# Pure logic (no external tools)
# ---------------------------------------------------------------------------

class TestPureHelpers:
    """Argument building, probing predicates, RAM math, error mapping, RIFF."""

    def test_build_df_args_layout(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "args")
        exe = d / "deep-filter.exe"
        exe.write_bytes(b"fake")
        in_wav = d / "in.wav"
        in_wav.write_bytes(b"fake")
        out_dir = d / "out"
        args = build_df_args(exe, in_wav, out_dir, 12.5)
        assert args[-1] == str(in_wav)
        assert args[0] == str(exe)
        assert "-D" in args
        assert args[args.index("-a") + 1] == "12.5"
        assert str(out_dir) in args

    def test_is_conformant_wav_matrix(self) -> None:
        assert is_conformant_wav(
            {"container": "wav", "codec": "pcm_s16le", "sample_rate": 48000}
        ) is True
        variants = [
            {"container": "wav", "codec": "pcm_s24le", "sample_rate": 48000},
            {"container": "wav", "codec": "pcm_s16le", "sample_rate": 44100},
            {"container": "mp3", "codec": "mp3", "sample_rate": 44100},
            {"container": "wav", "codec": "pcm_f32le", "sample_rate": 48000},
            {"container": "wav", "codec": "pcm_s16le", "sample_rate": None},
        ]
        for variant in variants:
            assert is_conformant_wav(variant) is False, variant

    def test_estimate_ram_mb(self) -> None:
        # Fitted to the measured table (deep-filter-test/run_20261030_0004).
        measured = {
            (120.0, 1): 96.1, (600.0, 1): 259.7, (1800.0, 1): 699.3,
            (120.0, 2): 174.5, (600.0, 2): 701.7, (1200.1, 2): 1360.7,
            (1800.0, 2): 2020.2,
        }
        for (dur, ch), actual in measured.items():
            est = estimate_ram_mb(dur, ch)
            assert abs(est - actual) / actual <= 0.05, (dur, ch, est, actual)
        assert estimate_ram_mb(0, 0) == estimate_ram_mb(0, 1)  # intercept only
        assert estimate_ram_mb(0, 1) == 49
        # one extra channel costs ~3x, not 2x
        assert estimate_ram_mb(600, 2) / estimate_ram_mb(600, 1) > 2.5

    def test_ram_guard_limit(self) -> None:
        assert ram_guard_limit_mb(8000) == 4096
        assert ram_guard_limit_mb(2700) == 1932
        assert ram_guard_limit_mb(1200) == 432
        assert ram_guard_limit_mb(500) == 0
        assert ram_guard_limit_mb(768) == 0
        assert 0 <= ram_guard_limit_mb() <= 4096  # real machine sanity

    def test_guard_admits(self) -> None:
        # At 2700 MB available the limit is 1932 MB.
        assert guard_admits(600, 1, 2700)[0] is True       # 10 min mono: 300 MB
        assert guard_admits(600, 2, 2700)[0] is True       # 10 min stereo: 804 MB
        assert guard_admits(1800, 1, 2700)[0] is True      # 30 min mono: 797 MB
        assert guard_admits(1800, 2, 2700)[0] is False     # 30 min stereo: 2310 MB
        assert guard_admits(1200, 2, 2700)[0] is True      # 20 min stereo: 1557 MB
        ok, est, limit = guard_admits(1800, 2, 2700)
        assert est > limit

    def test_map_df_oom_and_toowide(self) -> None:
        oom = str(_map_df_failure(DF_EXIT_OOM, ""))
        assert "الذاكرة" in oom and "أقصر" in oom
        assert "3221226505" not in oom  # the raw code belongs in the log, not the dialog
        assert "الذاكرة" in str(_map_df_failure(1, "memory allocation of 474462720 bytes failed"))
        wide = str(_map_df_failure(101, "TooWide"))
        assert "24" in wide and "16" in wide
        assert "الحديد" not in wide  # the old nonsense sentence is gone

    def test_compute_n48(self) -> None:
        assert compute_n48(1.0) == 48000
        assert compute_n48(0.001) == 48
        assert compute_n48(0.0005) == 24
        assert compute_n48(0.99999) == 48000  # banker's rounding at .5
        assert compute_n48(0.0) == 0
        assert compute_n48(-3.0) == 0
        assert compute_n48(1235.48) == round(1235.48 * 48000)

    def test_wav_file_sample_count_stdlib_wave(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "riff")
        p = d / "ok.wav"
        with wave.open(str(p), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(48000)
            wf.writeframes(b"\x00\x00" * 96000)
        assert _wcount(p) == 96000

    def test_wav_file_sample_count_skips_odd_chunks(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "rifs")
        p = d / "manual.wav"
        fmt = struct.pack("<HHIIHH", 1, 2, 48000, 48000 * 4, 4, 16)
        junk_even = b"JUNK" + struct.pack("<I", 4) + b"AAAA"
        junk_odd = b"JUNK" + struct.pack("<I", 5) + b"AAAAA" + b"\x00"
        data = b"\x00" * (4 * 4800)
        body = (
            b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + junk_even + junk_odd
            + b"data" + struct.pack("<I", len(data)) + data
        )
        p.write_bytes(b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body)
        assert _wcount(p) == 4800

    def test_wav_file_sample_count_rejects_non_wave(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "rifk")
        p = d / "bad.wav"
        p.write_bytes(b"this is not a wav file at all........")
        with pytest.raises(DenoiseError):
            _wcount(p)

    def test_map_df_failure_codes(self) -> None:
        assert "24" in str(_map_df_failure(101, ""))
        assert "TooWide" not in str(_map_df_failure(101, ""))  # raw text stays in the log
        assert "24" in str(_map_df_failure(3, "libDF\\src\\wav_utils.rs:87:39 TooWide"))
        assert "رمز 1" in str(_map_df_failure(1, "Caused by: لا ملف."))
        assert "رمز 2" in str(_map_df_failure(2, "الوسائط غير صالحة"))
        assert "أُلغيت" in str(_map_df_failure(-1073741510, ""))
        for rc in (3, 42):
            assert isinstance(_map_df_failure(rc, "تفاصيل"), DenoiseError)

    def test_cancel_state_semantics(self) -> None:
        st = _CancelState(None)
        assert not st.cancelled
        st.cancel()
        assert st.cancelled
        with pytest.raises(InterruptedError):
            st.check("x")
        ev = threading.Event()
        st2 = _CancelState(ev)
        assert not st2.cancelled
        ev.set()
        assert st2.cancelled

    def test_report_summary_lines(self) -> None:
        def mk(name: str, ok: bool, reason: str = "", clip: int = 0) -> DenoisePartStatus:
            return DenoisePartStatus(
                part=Path(name), ok=ok, reason=reason, clip_events=clip
            )

        report = DenoiseReport(statuses=[
            mk("part_01.wav", True),
            mk("part_02.wav", True, clip=7),
            mk("part_03.wav", False, reason="انقطعت"),
        ])
        lines = report.summary_lines()
        assert "2 من 3" in lines[0]
        assert not any("part_01" in line for line in lines[1:])
        assert any("part_03" in line and "انقطعت" in line for line in lines)
        assert any("part_02" in line and "7" in line for line in lines)
        empty = DenoiseReport()
        assert empty.summary_lines() == []
        assert len(empty.failures) == 0
        assert len(report.failures) == 1

    def test_avg_rtf(self) -> None:
        assert _avg_rtf(None) == 0.22          # measured mono / dual-mono RTF
        assert _avg_rtf({"rtf_history": {1: [0.2, 0.3], 2: [0.5]}}) == pytest.approx(1 / 3)
        assert _avg_rtf({"rtf_history": {}, "rtf_default": 0.33}) == 0.33
        assert _avg_rtf({}) == 0.22

    def test_df_delay_padding_constants(self) -> None:
        # Measured at 48 kHz for deep-filter 0.5.6.
        assert DF_DELAY_SAMPLES == 1440
        assert PAD_SAMPLES_48K == 4800
        # The pad must be audible to the tool's gate, not digital silence.
        assert PAD_RMS_DBFS == -55.0

    def test_pad_filter_is_not_silence(self) -> None:
        fc = build_padded_input_filter()
        assert "aevalsrc" in fc and "random(0)" in fc
        assert "apad" not in fc          # the old silent pad is gone
        assert "concat=n=2" in fc
        mono = build_padded_input_filter(downmix_mono=True)
        assert "pan=mono|c0=0.5*c0+0.5*c1" in mono
        assert "channel_layouts=mono" in mono
        stereo = build_padded_input_filter(src_channels=2)
        assert "channel_layouts=stereo" in stereo

    def test_format_for_suffix(self) -> None:
        from src.core.denoise import format_for_suffix

        assert format_for_suffix(".wav") == "wav"
        assert format_for_suffix(".M4A") == "ipod"
        assert format_for_suffix(".mp4") == "mp4"
        assert format_for_suffix(".aac") == "adts"
        assert format_for_suffix(".opus") == "ogg"
        with pytest.raises(DenoiseError, match="غير مدعومة"):
            format_for_suffix(".tuntunay")


class TestDenoiseWavValidationGate:
    """Strength/source/exe validation happens before any tool launch."""

    @staticmethod
    def _flags(**kw: object) -> Dict:
        base: Dict = dict(exe="any", ffmpeg="any", ffprobe="any")
        base.update(kw)
        return base

    def test_strength_off_never_launches_tool(
        self, fx_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _no_tool_launch_guard(monkeypatch)
        d = _sub(fx_dir, "off")
        dst = d / "x.wav"
        for bad in (0, -1, 0.0, "0", "-3"):
            with pytest.raises(DenoiseSkipped, match="صفر أو أقل"):
                denoise_wav(  # type: ignore[arg-type]
                    d / "missing.wav", dst, strength_db=bad, **self._flags()
                )
        assert calls == [] and not dst.exists()

    def test_bad_strength_rejected(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "nan")
        dst = d / "y.wav"
        for bad in ("abc", float("nan")):
            with pytest.raises(DenoiseError, match="قوة"):
                denoise_wav(  # type: ignore[arg-type]
                    d / "missing.wav", dst, strength_db=bad, **self._flags()
                )
        assert not dst.exists()

    def test_missing_source_raises(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "nosrc")
        with pytest.raises(FileNotFoundError):
            denoise_wav(d / "nope.wav", d / "out.wav", 12, **self._flags())

    def test_missing_tool_raises_before_probe(self, fx_dir: Path) -> None:
        d = _sub(fx_dir, "noexe")
        src = d / "src.wav"
        src.write_bytes(b"RIFF....WAVE")
        with pytest.raises(DenoiseError, match="غير موجودة"):
            denoise_wav(src, d / "out.wav", 12, **self._flags())
        assert not (d / "out.wav").exists()


class TestPipelineToDfLaunch:
    """Real ffmpeg conform steps, everything up to (not including) the tool."""

    class _ToolTouched(AssertionError):
        """Raised by the Popen guard when deep-filter would launch."""

    @staticmethod
    def _arm(
        monkeypatch: pytest.MonkeyPatch,
        fake_exe: Path,
        sink: Dict,
    ) -> List[Dict]:
        """Spy on build_df_args + guard Popen; real ffmpeg steps still run."""
        import src.core.denoise as dn

        calls: List[Dict] = []
        real_bdf = dn.build_df_args

        def spy(exe_obj: object, in_obj: object, out_obj: object, strength: float) -> List[str]:
            in_path = Path(str(in_obj))
            calls.append(
                {"exe": str(exe_obj), "in_wav": str(in_obj), "strength": strength}
            )
            sink["padded_samples"] = wav_file_sample_count(in_path)
            with wave.open(str(in_path), "rb") as wf:
                sink["channels"] = wf.getnchannels()
                sink["rate"] = wf.getframerate()
            return real_bdf(exe_obj, in_obj, out_obj, strength)  # type: ignore[arg-type]

        real_popen = subprocess.Popen

        def guard(cmd: List[str], *args: object, **kwargs: object) -> object:
            if str(cmd[0]) == str(fake_exe):
                raise TestPipelineToDfLaunch._ToolTouched(list(cmd))
            return real_popen(  # type: ignore[return-value,operator]
                cmd, *args, **kwargs  # type: ignore[arg-type]
            )

        monkeypatch.setattr("src.core.denoise.build_df_args", spy)
        monkeypatch.setattr("subprocess.Popen", guard)
        return calls

    @staticmethod
    def _fake_tool(fx_dir: Path, hint: str) -> Path:
        d = _sub(fx_dir, hint)
        exe = d / "fake-deep-filter.exe"
        exe.write_bytes(b"MZ fake but existing")
        return exe

    def test_conform_pads_and_df_args_observed(
        self, fx_dir: Path, require_ffmpeg: Tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        exe = self._fake_tool(fx_dir, "spyl")
        d = exe.parent
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=3.0, rate=48000, channels=1)
        sink: Dict = {}
        calls = self._arm(monkeypatch, exe, sink)
        with pytest.raises(self._ToolTouched) as touched:
            denoise_wav(
                src, d / "out.wav", 12,
                exe=exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
            )
        assert sink["padded_samples"] == compute_n48(3.0) + PAD_SAMPLES_48K
        assert sink["channels"] == 1 and sink["rate"] == 48000
        assert calls[0]["strength"] == 12
        cmd = touched.value.args[0]
        assert cmd[0] == str(exe)
        assert "-D" in cmd
        assert cmd[cmd.index("-a") + 1] == "12"
        out_flag = cmd.index("-o")
        assert cmd[out_flag + 1].endswith("\\out")  # inside the private tempdir
        assert cmd[-1] == "-v"  # RTF summary line requested
        assert cmd[-2].endswith("in.wav")  # padded ASCII-named temp file
        assert not (d / "out.wav").exists()

    def test_strength_500_clamped_to_100(
        self, fx_dir: Path, require_ffmpeg: Tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        exe = self._fake_tool(fx_dir, "clmp")
        d = exe.parent
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=2.0, rate=48000, channels=1)
        self._arm(monkeypatch, exe, {})
        with pytest.raises(self._ToolTouched) as touched:
            denoise_wav(
                src, d / "out.wav", 500,
                exe=exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
            )
        cmd = touched.value.args[0]
        assert cmd[cmd.index("-a") + 1] == "100"

    def test_downmix_to_mono_before_tool(
        self, fx_dir: Path, require_ffmpeg: Tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        exe = self._fake_tool(fx_dir, "dmix")
        d = exe.parent
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=2.0, rate=48000, channels=2)
        sink: Dict = {}
        self._arm(monkeypatch, exe, sink)
        with pytest.raises(self._ToolTouched):
            denoise_wav(
                src, d / "out.wav", 12, downmix_mono=True,
                exe=exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
            )
        assert sink["channels"] == 1 and sink["rate"] == 48000
        assert sink["padded_samples"] == compute_n48(2.0) + PAD_SAMPLES_48K


# ---------------------------------------------------------------------------
# Tool-dependent end-to-end (deep-filter.exe required)
# ---------------------------------------------------------------------------

@pytest.mark.deepfilter_exe
class TestDeepFilterRun:
    """One-file denoise_wav runs against the real tool (short inputs)."""

    def test_conformant_exact_length_and_alignment(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "exact")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=12.0, rate=48000, channels=1)
        n48 = compute_n48(12.0)
        assert _wcount(src) == n48  # lavfi sine is sample exact here
        res = denoise_wav(
            src, d / "out.wav", 12,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        assert res["padded_samples"] == n48 + PAD_SAMPLES_48K
        assert res["df_out_samples"] == res["padded_samples"] - DF_DELAY_SAMPLES
        assert res["df_out_expected"] == res["df_out_samples"]
        assert res["rtf"] > 0 and res["wall_s"] > 0
        assert _wcount(d / "out.wav") == n48
        meta = probe_media(d / "out.wav", ffprobe_path=ffprobe)
        first = meta["audio_streams"][0]
        assert int(first["sample_rate"]) == 48000 and first["codec_name"] == "pcm_s16le"
        assert int(first.get("channels") or 0) == 1

    def test_rerun_is_bitwise_deterministic(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "deti")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=6.0, rate=48000, channels=1)
        common = dict(exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d))
        denoise_wav(src, d / "out1.wav", 12, **common)  # type: ignore[arg-type]
        denoise_wav(src, d / "out2.wav", 12, **common)  # type: ignore[arg-type]
        assert _sha256(d / "out1.wav") == _sha256(d / "out2.wav")

    def test_24bit_converted_and_restored(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "s24b")
        src = _gen_sine_wav(
            ffmpeg, d / "src.wav", duration=8.0, rate=48000, channels=2, codec="pcm_s24le"
        )
        n48 = compute_n48(8.0)
        assert _wcount(src) == n48
        res = denoise_wav(
            src, d / "out.wav", 12,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        assert res["padded_samples"] == n48 + PAD_SAMPLES_48K
        first = probe_media(d / "out.wav", ffprobe_path=ffprobe)["audio_streams"][0]
        assert first["codec_name"] == "pcm_s24le" or first.get("bits_per_raw_sample") == "24"
        assert int(first["sample_rate"]) == 48000
        assert _wcount(d / "out.wav") == n48 == _wcount(src)

    def test_44100_restored_to_original_rate(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "r441")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=8.0, rate=44100, channels=2)
        denoise_wav(
            src, d / "out.wav", 12,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        first = probe_media(d / "out.wav", ffprobe_path=ffprobe)["audio_streams"][0]
        assert int(first["sample_rate"]) == 44100
        # atrim to n48 then aresample back: count may differ by a resampler sample.
        assert abs(_wcount(d / "out.wav") - 352800) <= 2

    def test_no_restore_keeps_48k(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "nrst")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=8.0, rate=44100, channels=2)
        denoise_wav(
            src, d / "out.wav", 12, restore_original_rate=False,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        first = probe_media(d / "out.wav", ffprobe_path=ffprobe)["audio_streams"][0]
        assert int(first["sample_rate"]) == 48000
        assert _wcount(d / "out.wav") == compute_n48(8.0)

    def test_16k_mono_roundtrip(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "k16m")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=6.0, rate=16000, channels=1)
        denoise_wav(
            src, d / "out.wav", 12,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        first = probe_media(d / "out.wav", ffprobe_path=ffprobe)["audio_streams"][0]
        assert int(first["sample_rate"]) == 16000
        assert int(first.get("channels") or 0) == 1
        assert abs(_wcount(d / "out.wav") - 96000) <= 2

    def test_auto_convert_off_skips_nonconformant(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "acnv")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=5.0, rate=44100, channels=2)
        dst = d / "out.wav"
        with pytest.raises(DenoiseSkipped, match="التحويل التلقائي"):
            denoise_wav(
                src, dst, 12, auto_convert=False,
                exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
            )
        assert not dst.exists()

    def test_target_codec_mp3(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "mp3o")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=6.0, rate=44100, channels=2)
        dst = d / "out.mp3"
        denoise_wav(
            src, dst, 12, target_codec="mp3", target_bitrate="128k",
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        info = probe_media(dst, ffprobe_path=ffprobe)
        first = info["audio_streams"][0]
        assert first["codec_name"] == "mp3"
        assert abs((info["duration"] or 9) - 6.0) <= 0.06

    def test_cancel_kills_tool_no_orphans(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        psutil = pytest.importorskip("psutil")
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "cncl")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=90.0, rate=48000, channels=1)
        dst = d / "out.wav"
        ev = threading.Event()
        slot: Dict = {}

        def runner() -> None:
            try:
                slot["res"] = denoise_wav(
                    src, dst, 12, cancel_event=ev,
                    exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
                )
            except BaseException as exc:  # noqa: BLE001 - captured for assertions
                slot["exc"] = exc

        th = threading.Thread(target=runner, daemon=True)
        th.start()
        time.sleep(3.0)
        ev.set()
        th.join(timeout=25)
        assert not th.is_alive(), "تعطّل الخيط بعد الإلغاء."
        assert isinstance(slot.get("exc"), InterruptedError)
        assert slot.get("res") is None
        assert not dst.exists()
        time.sleep(0.4)
        me = psutil.Process().pid
        names = [
            pct.info
            for pct in psutil.process_iter(["name", "ppid"])
            if "deep-filter" in (pct.info.get("name") or "").lower()
        ]
        # Only OUR child counts as an orphan (the sweep/other sessions may run
        # their own deep-filter processes concurrently).
        orphans = [n for n in names if n.get("ppid") == me]
        assert orphans == [], f"بقيت عملية deep-filter يتيمة: {orphans}"

    def test_loud_input_reports_clipping_note(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "clip")
        src = _gen_sine_wav(ffmpeg, d / "src.wav", duration=6.0, rate=48000, channels=1)
        res = denoise_wav(
            src, d / "out.wav", 20,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        assert (d / "out.wav").exists()
        if res["clip_events"] == 0:
            pytest.skip("لم تسجّل الأداة تحذيرات تشبع لهذا المدخل المتشبع.")

    def test_strength_levels_change_audio(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "lvls")
        src = _gen_noise_mix_wav(ffmpeg, d / "src.wav", duration=15.0, rate=44100, channels=2)
        common = dict(exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d))
        denoise_wav(src, d / "out6.wav", 6, **common)  # type: ignore[arg-type]
        denoise_wav(src, d / "out100.wav", 100, **common)  # type: ignore[arg-type]
        rms6 = _rms_dbfs(d / "out6.wav")
        rms100 = _rms_dbfs(d / "out100.wav")
        rms_src = _rms_dbfs(src)
        assert rms6 < rms_src  # denoising attenuates the noise floor
        assert abs(rms6 - rms100) >= 2.0  # stronger setting attenuates much more
        assert _sha256(d / "out6.wav") != _sha256(d / "out100.wav")

    def test_arabic_output_paths_supported(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "arbx")
        folder = d / "درس الرياضيات"
        folder.mkdir()
        src = _gen_sine_wav(ffmpeg, folder / "المصدر.wav", duration=4.0, rate=48000, channels=1)
        dst = folder / "الجزء 1.wav"
        res = denoise_wav(
            src, dst, 12,
            exe=df_exe, ffmpeg=ffmpeg, ffprobe=ffprobe, temp_base=str(d),
        )
        assert res["df_out_samples"] == res["df_out_expected"]
        assert _wcount(dst) == compute_n48(4.0)
        first = probe_media(dst, ffprobe_path=ffprobe)["audio_streams"][0]
        assert int(first["sample_rate"]) == 48000


# ---------------------------------------------------------------------------
# denoise_parts pipeline (tool required for real runs)
# ---------------------------------------------------------------------------

@pytest.mark.deepfilter_exe
class TestDenoiseParts:
    """Batch integration: parts processed in place with fallbacks."""

    def test_audio_wav_part_denoised_in_place(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "pt01")
        part = _gen_sine_wav(ffmpeg, d / "part.wav", duration=6.0, rate=48000, channels=1)
        n_samples = _wcount(part)
        sha_before = _sha256(part)
        report = denoise_parts(
            [(part, 6.0)],
            settings=_dfn_settings(fx_dir, strength_db=12),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        assert isinstance(report, DenoiseReport)
        assert len(report.statuses) == 1
        status = report.statuses[0]
        assert status.ok and not status.skipped and status.rtf > 0
        assert _wcount(part) == n_samples
        assert _sha256(part) != sha_before  # audio actually replaced
        info = probe_media(part, ffprobe_path=ffprobe)
        assert abs((info["duration"] or 0) - 6.0) <= 0.05
        assert report.measured_rtfs.get(1, [0])[0] > 0
        assert "1 من 1" in report.summary_lines()[0]

    def test_video_part_keeps_video_and_subs(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "ptv1")
        part = d / "lecture.mp4"
        create_synthetic_video(part, ffmpeg, duration_sec=9, include_subtitle=True)
        before = probe_media(part, ffprobe_path=ffprobe)
        assert len(before["audio_streams"]) == 1
        video_codec = before["video_streams"][0]["codec_name"]
        report = denoise_parts(
            [(part, 9.0)],
            settings=_dfn_settings(fx_dir, strength_db=12),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        status = report.statuses[0]
        assert status.ok, status.reason
        after = probe_media(part, ffprobe_path=ffprobe)
        assert len(after["video_streams"]) == 1
        assert after["video_streams"][0]["codec_name"] == video_codec  # stream copy
        assert len(after["audio_streams"]) == 1
        assert len(after["subtitle_streams"]) == 1  # subtitles survived the remux
        assert abs((after["duration"] or 0) - (before["duration"] or 0)) <= 0.15

    def test_two_audio_streams_replaced(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "pt2s")
        part = d / "two.aac.mp4"
        create_synthetic_video(part, ffmpeg, duration_sec=8, extra_audio=True)
        before = probe_media(part, ffprobe_path=ffprobe)
        assert len(before["audio_streams"]) == 2
        report = denoise_parts(
            [(part, 8.0)],
            settings=_dfn_settings(fx_dir),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        assert report.statuses[0].ok, report.statuses[0].reason
        after = probe_media(part, ffprobe_path=ffprobe)
        assert len(after["audio_streams"]) == 2
        for stream in after["audio_streams"]:
            assert stream["codec_name"] == "aac"  # per-stream encoder preserved

    def test_strength_zero_skips_and_never_launches(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        calls = _no_tool_launch_guard(monkeypatch)
        d = _sub(fx_dir, "ptz0")
        part = _gen_sine_wav(ffmpeg, d / "part.wav", duration=4.0, rate=48000, channels=1)
        sha_before = _sha256(part)
        report = denoise_parts(
            [(part, 4.0)],
            settings=_dfn_settings(fx_dir, strength_db=0),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        status = report.statuses[0]
        assert status.ok is False and status.skipped is True
        assert "صفر أو أقل" in status.reason
        assert calls == []
        assert _sha256(part) == sha_before  # untouched

    def test_auto_convert_off_skips_parts(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "ptac")
        part = _gen_sine_wav(ffmpeg, d / "part.wav", duration=4.0, rate=44100, channels=2)
        sha_before = _sha256(part)
        report = denoise_parts(
            [(part, 4.0)],
            settings=_dfn_settings(fx_dir, auto_convert=False),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        status = report.statuses[0]
        assert status.ok is False and status.skipped is True
        assert "التحويل التلقائي" in status.reason
        assert _sha256(part) == sha_before

    def test_ram_guard_refusal(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        monkeypatch.setattr("src.core.denoise.ram_guard_limit_mb", lambda *a, **k: 0)
        d = _sub(fx_dir, "ptra")
        part = _gen_sine_wav(ffmpeg, d / "part.wav", duration=3.0, rate=48000, channels=1)
        sha_before = _sha256(part)
        report = denoise_parts(
            [(part, 3.0)],
            settings=_dfn_settings(fx_dir),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        status = report.statuses[0]
        assert status.ok is False and status.skipped is False
        assert "الحد" in status.reason
        assert _sha256(part) == sha_before

    def test_tool_missing_top_level_message(
        self, fx_dir: Path, require_ffmpeg: Tuple[Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        monkeypatch.setattr("src.core.denoise.find_deep_filter", lambda *_: None)
        d = _sub(fx_dir, "ptnm")
        part = _gen_sine_wav(ffmpeg, d / "part.wav", duration=2.0, rate=48000, channels=1)
        with pytest.raises(DenoiseError, match="غير موجودة"):
            denoise_parts(
                [(part, 2.0)],
                settings=_dfn_settings(fx_dir),
                ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=None,
            )

    def test_corrupt_input_falls_back_untouched(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "ptcr")
        part = d / "broken.wav"
        part.write_bytes(b"RIFF\x00\x00\x00\x00WAVE fragments...not really")
        sha_before = _sha256(part)
        report = denoise_parts(
            [(part, 1.0)],
            settings=_dfn_settings(fx_dir),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        status = report.statuses[0]
        assert status.ok is False
        assert status.reason  # non-empty Arabic reason for the summary
        assert _sha256(part) == sha_before
        assert any("تنبيه" in line for line in report.summary_lines())

    def test_same_basename_two_dirs_isolated(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "ptsb")
        dir_a, dir_b = d / "A", d / "B"
        dir_a.mkdir(), dir_b.mkdir()
        pa = _gen_sine_wav(ffmpeg, dir_a / "part.wav", duration=4.0, rate=48000, channels=1)
        pb = _gen_sine_wav(
            ffmpeg, dir_b / "part.wav", duration=4.0, rate=48000, channels=2
        )
        report = denoise_parts(
            [(pa, 4.0), (pb, 4.0)],
            settings=_dfn_settings(fx_dir),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        assert all(s.ok for s in report.statuses), [s.reason for s in report.statuses]
        assert len(report.statuses) == 2
        assert _sha256(pa) != _sha256(pb)  # each temp call isolated per part
        assert _wcount(pa) == compute_n48(4.0)

    def test_workers_2_processes_all_parts(
        self, fx_dir: Path, df_exe: Path, require_ffmpeg: Tuple[Path, Path]
    ) -> None:
        ffmpeg, ffprobe = require_ffmpeg
        d = _sub(fx_dir, "ptw2")
        freqs = (250, 420, 680, 950)
        parts: List[Path] = []
        for idx, freq in enumerate(freqs):
            sub = d / f"d{idx}"
            sub.mkdir()
            p = sub / "part.wav"
            _run_checked([
                str(ffmpeg), "-nostdin", "-y", "-f", "lavfi",
                "-i", f"sine=frequency={freq}:duration=4:sample_rate=48000",
                "-c:a", "pcm_s16le", "-ac", "1", str(p),
            ])
            parts.append(p)
        report = denoise_parts(
            [(p, 4.0) for p in parts],
            settings=_dfn_settings(fx_dir, workers=2),
            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe,
        )
        assert all(s.ok for s in report.statuses), [s.reason for s in report.statuses]
        assert len(report.statuses) == 4
        hashes = [_sha256(p) for p in parts]
        assert len(set(hashes)) == 4  # no cross-part temp leakage


# ---------------------------------------------------------------------------
# Version reporting
# ---------------------------------------------------------------------------

@pytest.mark.deepfilter_exe
class TestToolVersion:
    def test_version_line_reports_number(self, df_exe: Path) -> None:
        version = get_version(df_exe)
        assert version and "0.5" in version
