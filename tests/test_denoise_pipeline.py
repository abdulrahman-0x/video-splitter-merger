"""Integration tests for WP3 (dual-mono end-to-end) and WP7 (stream order/tags)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Tuple

import pytest

from src.core.denoise import denoise_parts, wav_file_sample_count
from src.core.ffmpeg_manager import find_tool

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FX_ROOT = PROJECT_ROOT / "tmp" / "dfn_int_tests"


@pytest.fixture(scope="session")
def fx_dir() -> "object":
    if FX_ROOT.exists():
        shutil.rmtree(FX_ROOT, ignore_errors=True)
    FX_ROOT.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(str(PROJECT_ROOT)).free < 300 * 1024 * 1024:
        pytest.skip("المساحة الحرّة غير كافية")
    yield FX_ROOT
    shutil.rmtree(FX_ROOT, ignore_errors=True)


@pytest.fixture(scope="session")
def tools() -> Tuple[Path, Path]:
    ffmpeg, ffprobe = find_tool("ffmpeg"), find_tool("ffprobe")
    if not ffmpeg or not ffprobe:
        pytest.skip("ffmpeg/ffprobe غير متوفرة")
    return ffmpeg, ffprobe


@pytest.fixture(scope="session")
def df_exe() -> Path:
    from src.core.denoise import find_deep_filter

    exe = find_deep_filter()
    if exe is None:
        pytest.skip("deep-filter.exe غير موجود")
    return exe


def _probe(ffprobe: Path, p: Path) -> dict:
    r = subprocess.run([str(ffprobe), "-v", "error", "-show_streams", "-show_format",
                        "-of", "json", str(p)], capture_output=True, text=True,
                       encoding="utf-8", check=True)
    return json.loads(r.stdout)


def _pcm(ffmpeg: Path, p: Path) -> "object":
    np = pytest.importorskip("numpy")
    raw = subprocess.run([str(ffmpeg), "-v", "error", "-i", str(p), "-map", "0:a:0",
                          "-f", "f32le", "-acodec", "pcm_f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4")


def _make_dual_mono(ffmpeg: Path, dst: Path, seconds: float = 20.0) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ffmpeg), "-nostdin", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}:sample_rate=48000",
        "-f", "lavfi", "-i",
        f"anoisesrc=color=pink:amplitude=0.06:duration={seconds}:sample_rate=48000:seed=9",
        "-filter_complex",
        "[0:a][1:a]amix=inputs=2:duration=first:normalize=0,"
        "pan=stereo|c0=c0|c1=c0[a]",
        "-map", "[a]", "-c:a", "pcm_s24le", "-ar", "48000", str(dst),
    ], check=True, capture_output=True)
    return dst


def _make_true_stereo(ffmpeg: Path, dst: Path, seconds: float = 20.0) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        str(ffmpeg), "-nostdin", "-y",
        "-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}:sample_rate=48000",
        "-f", "lavfi", "-i", f"sine=frequency=520:duration={seconds}:sample_rate=48000",
        "-filter_complex", "[0:a][1:a]amerge=inputs=2[a]",
        "-map", "[a]", "-c:a", "pcm_s24le", "-ar", "48000", str(dst),
    ], check=True, capture_output=True)
    return dst


@pytest.mark.deepfilter_exe
class TestDualMonoEndToEnd:
    def test_shortcut_used_and_layout_restored(
        self, fx_dir: Path, tools: Tuple[Path, Path], df_exe: Path
    ) -> None:
        np = pytest.importorskip("numpy")
        ffmpeg, ffprobe = tools
        d = fx_dir / "dm_on"
        src = _make_dual_mono(ffmpeg, d / "dm.wav")
        before = _probe(ffprobe, src)["streams"][0]
        rep = denoise_parts([(src, 20.0)],
                            settings={"enabled": True, "strength_db": 12, "auto_convert": True,
                                      "restore_original_rate": True,
                                      "denoise_dual_mono_shortcut": True, "workers": 1,
                                      "parts_dir": str(d)},
                            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe)
        st = rep.statuses[0]
        assert st.ok and st.dual_mono is True
        assert st.channels == 1                      # priced/processed as mono
        after = _probe(ffprobe, src)["streams"][0]
        assert after["channels"] == before["channels"] == 2
        assert after["codec_name"] == before["codec_name"]
        x = _pcm(ffmpeg, src).reshape(-1, 2)
        assert np.array_equal(x[:, 0], x[:, 1])      # layout restored by duplication
        assert wav_file_sample_count(src) > 0

    def test_true_stereo_is_not_shortcut(
        self, fx_dir: Path, tools: Tuple[Path, Path], df_exe: Path
    ) -> None:
        ffmpeg, ffprobe = tools
        d = fx_dir / "dm_off"
        src = _make_true_stereo(ffmpeg, d / "st.wav")
        rep = denoise_parts([(src, 20.0)],
                            settings={"enabled": True, "strength_db": 12, "auto_convert": True,
                                      "restore_original_rate": True,
                                      "denoise_dual_mono_shortcut": True, "workers": 1,
                                      "parts_dir": str(d)},
                            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe)
        st = rep.statuses[0]
        assert st.ok and st.dual_mono is False
        assert st.channels == 2
        assert _probe(ffprobe, src)["streams"][0]["channels"] == 2

    def test_shortcut_disabled_keeps_stereo_path(
        self, fx_dir: Path, tools: Tuple[Path, Path], df_exe: Path
    ) -> None:
        ffmpeg, ffprobe = tools
        d = fx_dir / "dm_disabled"
        src = _make_dual_mono(ffmpeg, d / "dm.wav")
        rep = denoise_parts([(src, 20.0)],
                            settings={"enabled": True, "strength_db": 12, "auto_convert": True,
                                      "restore_original_rate": True,
                                      "denoise_dual_mono_shortcut": False, "workers": 1,
                                      "parts_dir": str(d)},
                            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe)
        st = rep.statuses[0]
        assert st.ok and st.dual_mono is False and st.channels == 2

    def test_report_file_written_next_to_parts(
        self, fx_dir: Path, tools: Tuple[Path, Path], df_exe: Path
    ) -> None:
        ffmpeg, ffprobe = tools
        d = fx_dir / "dm_report"
        src = _make_dual_mono(ffmpeg, d / "dm.wav", seconds=8.0)
        rep = denoise_parts([(src, 8.0)],
                            settings={"enabled": True, "strength_db": 12, "auto_convert": True,
                                      "restore_original_rate": True,
                                      "denoise_dual_mono_shortcut": True, "workers": 1,
                                      "parts_dir": str(d)},
                            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe)
        assert rep.report_path is not None and rep.report_path.is_file()
        text = rep.report_path.read_text(encoding="utf-8")
        assert "dm.wav" in text and "denoised" in text
        assert "yes" in text  # dual_mono column


@pytest.mark.deepfilter_exe
class TestStreamPreservation:
    """WP7: order, tags and chapters survive the remux."""

    def _make_video(self, ffmpeg: Path, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        srt = d / "s.srt"
        srt.write_text("1\n00:00:00,000 --> 00:00:04,000\nمرحبا\n", encoding="utf-8")
        ch = d / "ch.txt"
        ch.write_text(
            ";FFMETADATA1\ntitle=T\n\n[CHAPTER]\nTIMEBASE=1/1000\nSTART=0\nEND=4000\ntitle=A\n",
            encoding="utf-8")
        out = d / "v.mkv"
        subprocess.run([
            str(ffmpeg), "-nostdin", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=6",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=6:sample_rate=48000",
            "-i", str(srt), "-i", str(ch),
            "-t", "6",
            "-map", "0:v", "-map", "1:a", "-map", "2:s", "-map_metadata", "3",
            "-map_chapters", "3",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-g", "25",
            "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-c:s", "srt",
            "-metadata:s:a:0", "language=ara", "-metadata:s:a:0", "title=صوت",
            "-metadata:s:s:0", "language=ara",
            str(out),
        ], check=True, capture_output=True)
        return out

    def test_order_tags_chapters_preserved(
        self, fx_dir: Path, tools: Tuple[Path, Path], df_exe: Path
    ) -> None:
        ffmpeg, ffprobe = tools
        d = fx_dir / "wp7"
        src = self._make_video(ffmpeg, d)
        before = _probe(ffprobe, src)
        rep = denoise_parts([(src, 6.0)],
                            settings={"enabled": True, "strength_db": 12, "auto_convert": True,
                                      "restore_original_rate": True,
                                      "denoise_dual_mono_shortcut": True, "workers": 1,
                                      "parts_dir": str(d)},
                            ffmpeg_exe=ffmpeg, ffprobe_exe=ffprobe, deep_filter_exe=df_exe)
        assert rep.statuses[0].ok, rep.statuses[0].reason
        after = _probe(ffprobe, src)
        assert ([s["codec_type"] for s in before["streams"]]
                == [s["codec_type"] for s in after["streams"]]), "stream order changed"
        ba = before["streams"][1]
        aa = after["streams"][1]
        assert ba["codec_type"] == "audio"
        assert (ba.get("tags") or {}).get("language") == (aa.get("tags") or {}).get("language")
        assert (ba.get("tags") or {}).get("title") == (aa.get("tags") or {}).get("title")
        assert len(before.get("chapters") or []) == len(after.get("chapters") or [])
