"""Pytest fixtures and test video generators using FFmpeg lavfi."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Generator

import pytest

from src.utils.helpers import get_unique_path


def find_test_ffmpeg() -> tuple[Path | None, Path | None]:
    """Find ffmpeg and ffprobe either in bin/ or in system PATH."""
    bin_dir = Path("bin").resolve()
    ffmpeg_local = bin_dir / "ffmpeg.exe"
    ffprobe_local = bin_dir / "ffprobe.exe"

    if ffmpeg_local.is_file() and ffprobe_local.is_file():
        return ffmpeg_local, ffprobe_local

    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")

    if ffmpeg_path and ffprobe_path:
        return Path(ffmpeg_path), Path(ffprobe_path)

    return None, None


@pytest.fixture(scope="session")
def ffmpeg_binaries() -> tuple[Path | None, Path | None]:
    """Session fixture returning (ffmpeg_path, ffprobe_path)."""
    return find_test_ffmpeg()


@pytest.fixture(scope="session")
def require_ffmpeg(ffmpeg_binaries: tuple[Path | None, Path | None]) -> tuple[Path, Path]:
    """Fixture that skips test if ffmpeg or ffprobe are not available."""
    ffmpeg, ffprobe = ffmpeg_binaries
    if not ffmpeg or not ffprobe:
        pytest.skip("FFmpeg أو FFprobe غير متوفرين لتشغيل هذا الاختبار.")
    return ffmpeg, ffprobe


@pytest.fixture
def temp_workspace(tmp_path: Path) -> Generator[Path, None, None]:
    """Provide a temporary directory for test outputs."""
    yield tmp_path


def create_synthetic_video(
    output_path: Path,
    ffmpeg_path: Path,
    duration_sec: int = 30,
    gop_size: int = 25,
    fps: int = 25,
    extra_audio: bool = False,
    include_subtitle: bool = False,
) -> Path:
    """Generate a synthetic test video using FFmpeg lavfi testsrc and sine audio."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Assemble all inputs first
    inputs = [
        "-f", "lavfi",
        "-i", f"testsrc=duration={duration_sec}:size=320x240:rate={fps}",
        "-f", "lavfi",
        "-i", f"sine=frequency=1000:duration={duration_sec}",
    ]

    maps = [
        "-map", "0:v:0",
        "-map", "1:a:0",
    ]

    if extra_audio:
        inputs += [
            "-f", "lavfi",
            "-i", f"sine=frequency=440:duration={duration_sec}",
        ]
        maps += ["-map", "2:a:0"]

    sub_encoding: list[str] = []
    if include_subtitle:
        srt_file = output_path.with_suffix(".srt")
        srt_content = (
            "1\n00:00:01,000 --> 00:00:05,000\nترجمة اختبارية تجريبية\n\n"
            "2\n00:00:06,000 --> 00:00:15,000\nالجزء الثاني من الترجمة\n\n"
        )
        srt_file.write_text(srt_content, encoding="utf-8")
        sub_idx = 3 if extra_audio else 2
        inputs += ["-i", str(srt_file)]
        maps += ["-map", f"{sub_idx}:s:0"]
        sub_encoding = ["-c:s", "srt"]

    # Assemble complete command
    cmd = [
        str(ffmpeg_path),
        "-y",
        *inputs,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-g", str(gop_size),
        "-keyint_min", str(gop_size),
        "-sc_threshold", "0",
        "-c:a", "aac",
        "-b:a", "128k",
        *sub_encoding,
        *maps,
        str(output_path),
    ]

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"فشل توليد فيديو الاختبار بـ FFmpeg:\n{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    return output_path


def create_synthetic_audio(
    output_path: Path,
    ffmpeg_path: Path,
    duration_sec: int = 30,
    frequency: int = 1000,
    sample_rate: int = 44100,
    channels: int = 2,
    with_cover_art: bool = False,
) -> Path:
    """Generate a synthetic audio test file (wav, mp3, flac, m4a, ogg)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ext = output_path.suffix.lower()

    cmd = [
        str(ffmpeg_path),
        "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency={frequency}:duration={duration_sec}:sample_rate={sample_rate}",
    ]

    if with_cover_art:
        # Create a tiny 1-frame dummy image as album cover
        cmd += [
            "-f", "lavfi",
            "-i", "color=c=blue:s=120x120:d=1",
        ]

    # Select codec based on extension
    if ext == ".wav":
        codec_args = ["-c:a", "pcm_s16le"]
    elif ext == ".mp3":
        codec_args = ["-c:a", "libmp3lame", "-b:a", "192k"]
    elif ext == ".flac":
        codec_args = ["-c:a", "flac"]
    elif ext in (".m4a", ".aac"):
        codec_args = ["-c:a", "aac", "-b:a", "192k"]
    elif ext in (".ogg", ".opus"):
        codec_args = ["-c:a", "libopus", "-b:a", "128k"]
    else:
        codec_args = ["-c:a", "pcm_s16le"]

    cmd.extend(codec_args)
    cmd.extend(["-ac", str(channels)])

    if with_cover_art:
        cmd.extend([
            "-map", "0:a",
            "-map", "1:v",
            "-c:v", "mjpeg",
            "-disposition:v:0", "attached_pic",
        ])

    cmd.append(str(output_path))

    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"فشل توليد ملف الصوت التجريبي بـ FFmpeg:\n{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    return output_path
