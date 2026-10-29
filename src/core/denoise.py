"""DeepFilterNet noise-reduction wrapper for the splitter/merger app.

Wraps the external `deep-filter` command-line tool (DeepFilterNet3, tested with
0.5.6 on Windows) and exposes one function per call (one input file per call:

1. The tool only reads 8/16/22.05/32/44.1/48 kHz u8/s16/f32 WAV input safely and
   panics (exit 101, `TooWide`) on 24-bit and 32-bit PCM, which is the app's
   normal source format. Every input is therefore re-encoded by ffmpeg to
   48 kHz `pcm_s16le` inside a private temp dir (ASCII names only).
2. The tool's enhanced output is delayed by exactly 1440 samples (30 ms at
   48 kHz) and, with `-D`, exactly 1440 samples shorter than the input: the
   last 1440 samples of real content are **dropped**, not merely shifted.
   Measured 2026-10-30 (deep-filter 0.5.6): the tool zeroes any output frame
   whose *input* frame is digital silence, and because of the 1440-sample
   lookahead the final 1440 samples of content always fall into the region
   where a silent pad has already begun. Padding with digital silence of any
   length (0.03 s to 0.5 s) does not help. Padding with **low-level noise at
   -55 dBFS RMS** does: the tail survives with 0 trailing zeros, lag 0, and a
   bit-identical body (measured: 1059511 of 2880000 samples change by at most
   3.4e-4, all of them inside the last 1440). That is what `PAD_RMS_DBFS`
   below implements.
3. The pipeline pads with that noise, runs with `-D`, then trims back to the
   exact original sample count (`atrim=end_sample=N48,apad=whole_len=N48`).
   A content check after the tool run catches a regression (see
   `_tail_is_damaged`): if the source tail is audible and the output tail is
   digital silence, the last 1440 samples are restored from the source with a
   5 ms crossfade instead of shipping a hole.
4. `-a 0` is a bit-exact passthrough that mis-aligns when combined with `-D`:
   strength <= 0 is treated as "feature off" and the tool is never launched.
5. The tool emits no progress; the caller estimates ETA with
   ETA = remaining audio duration * RTF (measured: mono 0.19-0.22,
   stereo 0.30-0.36, dual-mono downmix 0.22).

All measured behavior above (2026-10-29/30, deep-filter 0.5.6, ffmpeg 9.0.2) is
reproducible with the scripts kept in
`deep-filter-test/run_20261030_0004/scripts/`.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import shutil
import subprocess
import threading
import time
import tempfile
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from src.core.ffmpeg_manager import CREATE_NO_WINDOW, find_tool, probe_media

logger = logging.getLogger(__name__)

# The tool's enhanced output is delayed 1440 samples without `-D` and, with
# `-D`, the last 1440 samples of (padded) input are dropped. Measured with
# deep-filter 0.5.6 (DeepFilterNet3, lookahead 2 frames) at 48 kHz on this
# machine: output length == (padded input length - 1440) within 0 samples.
DF_DELAY_SAMPLES = 1440

# The pad appended before denoising must NOT be digital silence: the tool zeroes
# output frames whose input frame is silent, so a silent pad makes the last 30 ms
# of real content vanish. Measured threshold on real speech: a -70 dBFS pad still
# loses the tail, a -68 dBFS pad keeps it. -55 dBFS gives ~13 dB of margin while
# staying ~35 dB below typical speech (-20 dBFS), and it is inaudible in practice.
PAD_SECONDS = 0.1
PAD_SAMPLES_48K = int(PAD_SECONDS * 48000)
PAD_RMS_DBFS = -55.0
# Uniform white noise of RMS R has peak R + 4.77 dB; aevalsrc takes the peak.
PAD_PEAK = 10 ** ((PAD_RMS_DBFS + 4.7712) / 20.0)

# Crossfade used when the tail safety net restores source samples.
TAIL_RESTORE_CROSSFADE_SAMPLES = int(0.005 * 48000)  # 5 ms

# Convergence-independent defaults when no RTF has been measured yet in this
# session. Measured 2026-10-30: mono 0.19-0.22, true stereo 0.30-0.36,
# stereo downmixed to mono 0.22 (dual-mono shortcut uses the mono figure).
DEFAULT_RTF_BY_CHANNELS = {1: 0.22, 2: 0.32}

# Peak working set of deep-filter.exe, fitted 2026-10-30 over 2/10/20/30-minute
# units (see deep-filter-test/run_20261030_0004/results/q3_ram_all.json):
#   mono   peak_MB = 48.8 + 0.3609 * d
#   stereo peak_MB = 42.6 + 1.0986 * d
# so one extra channel costs ~3.0x, not 2x. The generic form below reproduces
# both fits and is an extrapolation for channel counts other than 1 and 2.
RAM_INTERCEPT_MB = 45.0
RAM_MB_PER_SECOND_PER_CHANNEL_POWER = 0.36
RAM_CHANNEL_EXPONENT = 1.6
RAM_FIT_HEADROOM = 1.15

# Upper bound for a single denoise unit and the RAM the rest of the machine keeps.
RAM_GUARD_MAX_MB = 4096
RAM_GUARD_RESERVE_MB = 768
# Kept for backwards compatibility with callers that still read the old name.
RAM_MB_PER_SECOND_PER_CHANNEL = 0.45

# Exit code the tool uses when it cannot allocate its working set (0xC0000409).
DF_EXIT_OOM = 3221226505

# Regex for the `-v` summary line: `Enhanced audio file <name> in <s> (RTF: 0.17)`
_RTF_LINE_RE = re.compile(r"\(RTF:\s*([0-9]*\.?[0-9]+)\)", re.IGNORECASE)
# Count of clipping WARN lines C:\DF\libdf noise detector can spam on hot source material.
CLIPPING_WARN_TEXT = "Possible clipping detected"

# ffprobe codec name -> ffmpeg encoder name. Anything else raises DenoiseError.
CODEC_TO_ENCODER = {
    "pcm_s16le": "pcm_s16le",
    "pcm_s24le": "pcm_s24le",
    "pcm_s32le": "pcm_s32le",
    "pcm_u8": "pcm_u8",
    "pcm_f32le": "pcm_f32le",
    "flac": "flac",
    "mp3": "libmp3lame",
    "aac": "aac",
    "opus": "libopus",
    "vorbis": "libvorbis",
}

# File suffix -> ffmpeg muxer. Needed because temp outputs end with ".tmp" and
# ffmpeg cannot infer the container from that extension.
FORMAT_BY_SUFFIX = {
    ".wav": "wav",
    ".flac": "flac",
    ".mp3": "mp3",
    ".m4a": "ipod",
    ".mp4": "mp4",
    ".m4b": "ipod",
    ".mov": "mov",
    ".avi": "avi",
    ".mkv": "matroska",
    ".webm": "webm",
    ".aac": "adts",
    ".ogg": "ogg",
    ".opus": "ogg",
    ".wma": "asf",
    ".alac": "ipod",
    ".aiff": "aiff",
}


def format_for_suffix(suffix: Optional[str]) -> str:
    """ffmpeg muxer name for a media suffix (DenoiaseError on unknown)."""
    fmt = FORMAT_BY_SUFFIX.get((suffix or "").lower())
    if not fmt:
        raise DenoiseError(f"صيغة المخرجات غير مدعومة للكتابة: {suffix or 'بدون امتداد'}")
    return fmt


class DenoiseError(RuntimeError):
    """Raised when the noise reduction could not be performed for an item."""


class DenoiseSkipped(Exception):
    """Raised when the item is intentionally not denoised (e.g. auto-convert off)."""


# ---------------------------------------------------------------------------
# Pure helpers (unit tested without any external tool)
# ---------------------------------------------------------------------------

def find_deep_filter(configured_path: Optional[str | Path] = None) -> Optional[Path]:
    """Locate the deep-filter executable.

    Order (plan section 6):
    1. Explicit path from settings if it exists.
    2. `deep-filter` on PATH.
    3. Versioned alias `deep-filter-0.5.6-x86_64-pc-windows-msvc` on PATH.

    Note: a process only sees PATH changes that existed at its start, so the
    explicit settings path remains important; the resolved path is logged.
    """
    try:
        if configured_path:
            candidate = Path(configured_path)
            if candidate.is_file():
                return candidate
            # settings may point at a directory housing the exe
            if candidate.is_dir() and (candidate / "deep-filter.exe").is_file():
                return candidate / "deep-filter.exe"
            logger.warning("لم يتم العثور على deep-filter في المسار المحدد: %s", candidate)
    except OSError as exc:
        logger.warning("فشل فحص مسار deep-filter المحدد: %s", exc)

    for name in ("deep-filter", "deep-filter-0.5.6-x86_64-pc-windows-msvc"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def get_version(exe: str | Path) -> Optional[str]:
    """Return the tool version string (e.g. 'deep_filter 0.5.6') or None."""
    try:
        proc = subprocess.run(
            [str(exe), "-V"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        logger.debug("تعذر الحصول على إصدار deep-filter: %s", exc)
        return None
    text = proc.stdout.decode("utf-8", errors="replace").strip()
    if not text:
        text = proc.stderr.decode("utf-8", errors="replace").strip()
    return text.splitlines()[0].strip() if text else None


def build_df_args(exe: str | Path, in_wav: Path, out_dir: Path, strength_db: float) -> List[str]:
    """Build the deep-filter argument list (plan section 6, step 5)."""
    return [
        str(exe),
        "-D",                                   # always align output to input
        "-a", f"{strength_db:g}",               # attenuation limit in dB
        "-o", str(out_dir),
        str(in_wav),
    ]


def is_conformant_wav(meta: Dict) -> bool:
    """Plan section 6, step 3: WAV container + pcm_s16le + 48000 Hz."""
    return bool(
        meta.get("container", "").startswith("wav")
        and meta.get("codec") == "pcm_s16le"
        and meta.get("sample_rate") == 48000
    )


def estimate_ram_mb(duration_s: float, channels: int) -> int:
    """Peak working set of deep-filter.exe for a unit of this length.

    Measured 2026-10-30 on this machine (2/10/20/30-minute units):
      mono   `48.8 + 0.3609 * d`
      stereo `42.6 + 1.0986 * d`
    One extra channel costs ~3.0x, not 2x. For channel counts other than 1 and 2
    the generic form `45 + 0.36 * ch**1.6 * d` is used; that is an
    **extrapolation** and is not backed by measurements.
    """
    ch = max(1, int(channels))
    d = max(0.0, float(duration_s))
    if ch == 1:
        return int(round(48.8 + 0.3609 * d))
    if ch == 2:
        return int(round(42.6 + 1.0986 * d))
    # Extrapolation for 3+ channels (no measurement on this machine).
    return int(round(RAM_INTERCEPT_MB + RAM_MB_PER_SECOND_PER_CHANNEL_POWER * (ch ** RAM_CHANNEL_EXPONENT) * d))


def ram_guard_limit_mb(avail_mb: Optional[int] = None) -> int:
    """Largest estimate a single unit may claim: min(4096, max(0, avail - 768)).

    Measured 2026-10-30: a 10-minute stereo unit peaks at 702 MB and a 30-minute
    one at 2020 MB, so the old `25% of free RAM` rule (675 MB here) refused
    units the machine handles comfortably. Reserving a flat 768 MB for the rest
    of the machine is both simpler and better matched to the measurements.
    """
    if avail_mb is None:
        avail_mb = get_available_ram_mb()
    if avail_mb is None:
        return RAM_GUARD_MAX_MB
    return max(0, min(RAM_GUARD_MAX_MB, int(avail_mb) - RAM_GUARD_RESERVE_MB))


def guard_admits(duration_s: float, channels: int, avail_mb: Optional[int] = None) -> Tuple[bool, int, int]:
    """(admitted, estimate_with_headroom_mb, limit_mb) for logging and tests."""
    est = int(round(RAM_FIT_HEADROOM * estimate_ram_mb(duration_s, channels)))
    limit = ram_guard_limit_mb(avail_mb)
    return est <= limit, est, limit


def get_available_ram_mb() -> Optional[int]:
    """Available physical RAM in MB via GlobalMemoryStatusEx (stdlib only; no psutil)."""
    try:
        kernel32 = ctypes.WinDLL("kernel32")  # type: ignore[attr-defined]

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(stat)
        if kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return int(stat.ullAvailPhys / (1024 * 1024))
    except Exception as exc:  # non-Windows or restricted environment
        logger.debug("تعذر قراءة الذاكرة المتاحة: %s", exc)
    return None


def compute_n48(duration_s: float) -> int:
    """Original content length at 48 kHz in samples (`round(duration * 48000)`)."""
    return max(0, round(duration_s * 48000))


# ---------------------------------------------------------------------------
# Tail integrity (WP1)
# ---------------------------------------------------------------------------

def build_pad_filter(pad_s: float = PAD_SECONDS, pad_peak: float = PAD_PEAK,
                     layout: str = "mono") -> str:
    """lavfi chain producing the non-silent pad.

    Digital silence is not usable: the tool zeroes output frames whose input
    frame is silent, which is exactly what destroyed the last 30 ms. White noise
    at -55 dBFS RMS keeps every input frame non-silent.
    """
    return (f"aevalsrc={pad_peak:.9f}*(2*random(0)-1):s=48000:d={pad_s:g}"
            f",aformat=channel_layouts={layout}")


def build_padded_input_filter(downmix_mono: bool = False, src_channels: int = 2,
                              pad_s: float = PAD_SECONDS) -> str:
    """`[0:a] -> 48 kHz s16 (+optional downmix) + noise pad` as one filter graph."""
    pre = ""
    if downmix_mono and src_channels >= 2:
        pre = "pan=mono|c0=0.5*c0+0.5*c1,"
    out_channels = 1 if (downmix_mono or src_channels == 1) else src_channels
    layout = "mono" if out_channels == 1 else ("stereo" if out_channels == 2 else f"{out_channels}c")
    return (f"[0:a]{pre}aresample=48000[a];"
            f"{build_pad_filter(pad_s=pad_s, layout=layout)}[n];"
            f"[a][n]concat=n=2:v=0:a=1[out]")


def pcm16_rms_from_wav(path: Path, max_seconds: float = 0.1,
                       from_end: bool = True) -> Optional[float]:
    """RMS (0..1) of the first/last `max_seconds` of a 16-bit PCM WAV, no deps.

    Used to detect a damaged tail: the tool writes digital zeros there, so the
    measured RMS is exactly 0.0 when the loss happened.
    """
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        return None
    pos, channels, bits, data_off, data_size = 12, None, None, None, None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        csize = int.from_bytes(raw[pos + 4:pos + 8], "little")
        if cid == b"fmt ":
            channels = int.from_bytes(raw[pos + 10:pos + 12], "little")
            bits = int.from_bytes(raw[pos + 22:pos + 24], "little")
        elif cid == b"data":
            data_off, data_size = pos + 8, csize
            break
        pos += 8 + csize + (csize & 1)
    if data_off is None or not channels or not bits or bits != 16:
        return None
    end = min(len(raw), data_off + data_size)
    frame_bytes = 2 * channels
    n_frames = (end - data_off) // frame_bytes
    if n_frames <= 0:
        return None
    n = min(n_frames, max(1, int(max_seconds * 48000)))
    start = data_off + (n_frames - n) * frame_bytes if from_end else data_off
    chunk = raw[start:start + n * frame_bytes]
    samples = memoryview(chunk).cast("h")
    if not samples:
        return None
    total = 0.0
    for v in samples:
        total += (v / 32768.0) ** 2
    return math.sqrt(total / len(samples))


def tail_is_damaged(source_tail_rms: Optional[float], output_tail_rms: Optional[float],
                    silence_floor: float = 1e-7) -> bool:
    """True when the source tail is audible but the output tail is digital silence."""
    if source_tail_rms is None or output_tail_rms is None:
        return False
    return source_tail_rms > 0.001 and output_tail_rms <= silence_floor


def crossfade_tail_filter(n48: int, crossfade_samples: int = TAIL_RESTORE_CROSSFADE_SAMPLES) -> str:
    """Filter restoring the last `DF_DELAY_SAMPLES` from the source, 5 ms crossfade.

    Inputs: 0 = denoised output (length n48), 1 = conformed source (length n48).
    Output: denoised for [0, n48-d), then a crossfade into the source for the
    final d samples, so `samples_out == samples_in` and there is no hole.
    """
    d = DF_DELAY_SAMPLES
    cut = max(0, n48 - d)
    cf = max(1, int(crossfade_samples))
    return (
        f"[0:a]atrim=end_sample={cut},asetpts=N/SR/TB[head];"
        f"[0:a]atrim=start_sample={cut},asetpts=N/SR/TB[td];"
        f"[1:a]atrim=start_sample={cut},asetpts=N/SR/TB[ts];"
        f"[td]afade=t=out:st=0:d={cf / 48000:.9f}:curve=tri[tdf];"
        f"[ts]afade=t=in:st=0:d={cf / 48000:.9f}:curve=tri[tsf];"
        f"[tdf][tsf]amix=inputs=2:duration=longest:normalize=0[blend];"
        f"[head][blend]concat=n=2:v=0:a=1[out]"
    )


def _probe_audio_source(src: Path, ffprobe: str | Path) -> Dict:
    """Probe an audio file and return the fields the pipeline needs."""
    info = probe_media(src, ffprobe_path=Path(ffprobe))
    audio = info.get("audio_streams", [])
    if not audio:
        raise DenoiseError(f"لا يحتوي الملف على أي مسار صوت: {src}")
    first = audio[0]
    fmt_name = str(info.get("format_name", ""))
    sample_rate = first.get("sample_rate")
    stream = info["raw"]["streams"][0] if info.get("raw", {}).get("streams") else first
    codec = first.get("codec_name") or stream.get("codec_name")
    # Prefer the stream's own duration when present (WAV: exact sample count / rate)
    duration = None
    for candidate in (stream.get("duration"), info.get("duration"), first.get("duration")):
        try:
            if candidate is not None and float(candidate) > 0:
                duration = float(candidate)
                break
        except (TypeError, ValueError):
            continue
    return {
        "container": fmt_name,
        "codec": codec,
        "sample_rate": int(sample_rate) if sample_rate else None,
        "channels": int(first.get("channels") or stream.get("channels") or 1),
        "duration": duration or 0.0,
        "bits": stream.get("bits_per_raw_sample") or stream.get("bits_per_sample"),
        "n48": compute_n48(duration or 0.0),
    }


def _encoder_for(codec: str) -> str:
    encoder = CODEC_TO_ENCODER.get(str(codec).lower())
    if not encoder:
        raise DenoiseError(f"ترميز الصوت غير مدعوم لعملية تقليل الضوضاء: {codec}")
    return encoder


def _bitrate_for_stream(stream: Dict, probe_info: Dict) -> Optional[str]:
    """Approximate a lossy stream's bitrate (stream-level first, then format)."""
    br = stream.get("bit_rate")
    if not br and probe_info.get("raw", {}).get("format"):
        br = probe_info["raw"]["format"].get("bit_rate")
    if not br:
        return None
    try:
        kbps = int(br) // 1000
        return f"{kbps}k" if kbps > 0 else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------

class _CancelState:
    """Shared cancellation flag usable with or without a caller-provided Event."""

    def __init__(self, cancel_event: Optional[threading.Event]) -> None:
        self._event = cancel_event if cancel_event is not None else threading.Event()
        self.cancelled_local = False

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or self.cancelled_local

    def cancel(self) -> None:
        self.cancelled_local = True
        self._event.set()

    def check(self, message: str) -> None:
        if self.cancelled:
            raise InterruptedError(message)


_CANCEL_MSG = "تم إلغاء عملية تقليل الضوضاء بناءً على طلبك."


def _wait_or_cancel(proc: subprocess.Popen, state: _CancelState, poll_s: float = 0.1) -> int:
    """Poll a child process so cancellation can kill it promptly (no orphan left)."""
    while True:
        rc = proc.poll()
        if rc is not None:
            return rc
        if state.cancelled:
            logger.info("إلغاء: إنهاء العملية الفرعية (PID: %s)...", proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=1.0)
                except subprocess.TimeoutExpired:
                    pass
            raise InterruptedError(_CANCEL_MSG)
        time.sleep(poll_s)


def _run_ffmpeg(cmd: List[str], state: _CancelState, timeout_s: float = 180.0) -> None:
    """Run a short ffmpeg step with polling-based cancellation."""
    # Statistics lines print every ~0.5 s to stderr; drained only after exit,
    # so silence them to keep the 64 KB pipe buffer from filling (deadlock).
    cmd = [cmd[0], "-nostats", "-loglevel", "error"] + cmd[1:]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        stdin=subprocess.DEVNULL,
        creationflags=CREATE_NO_WINDOW,
    )
    try:
        deadline = time.monotonic() + timeout_s
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if state.cancelled:
                proc.terminate()
                try:
                    proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    try:
                        proc.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        pass
                raise InterruptedError(_CANCEL_MSG)
            if time.monotonic() > deadline:
                proc.kill()
                raise DenoiseError(f"انتهت المدة المسموحة لأداة ffmpeg: {' '.join(cmd[:3])}...")
            time.sleep(0.1)
    finally:
        if proc.stderr:
            proc.stderr.close()
    if rc != 0:
        err = b""
        try:
            err = proc.stderr.read() if proc.stderr else b""
        except Exception:
            pass
        raise DenoiseError(f"فشل أمر ffmpeg أثناء تقليل الضوضاء (رمز {rc}).\n{err.decode('utf-8', 'replace')[-800:]}")


# ---------------------------------------------------------------------------
# Core single-file denoise (plan section 6)
# ---------------------------------------------------------------------------

def denoise_wav(
    src: Path | str,
    dst: Path | str,
    strength_db: float = 100,
    *,
    exe: str | Path,
    ffmpeg: str | Path,
    ffprobe: str | Path,
    auto_convert: bool = True,
    restore_original_rate: bool = True,
    downmix_mono: bool = False,
    target_codec: Optional[str] = None,
    target_bitrate: Optional[str] = None,
    cancel_event: Optional[threading.Event] = None,
    temp_base: Optional[Path | str] = None,
) -> Dict:
    """Denoise one audio file to `dst` (media file, same codec family by default).

    One input file per call; a private temp dir with short ASCII names per call.
    Returns a dict with measured facts: {"rtf": float, "clip_events": int,
    "durations_in_s": float, "padded_samples": int, "df_out_samples": int}.

    Raises:
    - DenoiseSkipped: strength <= 0 (feature off) or non-conformant source with
      auto_convert=False (an explanatory message is attached).
    - DenoiseError: the tool or an ffmpeg step failed.
    - InterruptedError: caller requested cancellation.
    """
    state = _CancelState(cancel_event)
    src_path = Path(src).resolve()
    dst_path = Path(dst).resolve()

    # 1. Strength validation (plan section 3, rule 4 / section 6 step 1).
    try:
        strength = float(strength_db)
    except (TypeError, ValueError):
        strength = float("nan")
    if math.isnan(strength) or not math.isfinite(strength):
        raise DenoiseError(f"قيمة قوة تقليل الضوضاء غير صالحة: {strength_db}")
    if strength <= 0:
        # Never launch the tool for off-strength (plan rule: no `-a 0`).
        raise DenoiseSkipped(f"قوة تقليل الضوضاء صفر أو أقل ({strength_db})؛ سيُترك الصوت دون تغيير.")
    strength = min(strength, 100.0)
    state.check(_CANCEL_MSG)

    if not src_path.is_file():
        raise FileNotFoundError(f"ملف الصوت غير موجود: {src_path}")
    if not Path(exe).is_file():
        raise DenoiseError(f"أداة تقليل الضوضاء غير موجودة: {exe}")

    # 2. Probe the source (plan section 6 step 2).
    meta = _probe_audio_source(src_path, ffprobe)
    n48 = meta["n48"]
    original_rate = meta["sample_rate"] or 48000
    conformant = is_conformant_wav(meta)
    if not conformant and not auto_convert:
        raise DenoiseSkipped(
            "الملف ليس 48 كيلوهرتز / 16 بت WAV؛ فعّل التحويل التلقائي لاستخدام تقليل الضوضاء "
            f"(ترميز {meta['codec']} بمعدل {original_rate})."
        )
    state.check(_CANCEL_MSG)

    # 4. Private temp dir; conform input: 48 kHz s16 + a NON-SILENT noise pad.
    with tempfile.TemporaryDirectory(prefix="dfn_", dir=str(temp_base) if temp_base else None) as tmp_name:
        tmp = Path(tmp_name)
        in_wav = tmp / "in.wav"
        # Free-disk guard: in.wav + the tool output + the trimmed final coexist.
        # Measured 2026-10-30: a 30-min stereo unit needs ~1368 MB (in.wav is
        # 4 bytes/sample/channel, and the tool writes a second copy).
        try:
            tmp_parent = Path(temp_base) if temp_base else Path(tempfile.gettempdir())
            s16_bytes = int(n48 * meta["channels"] * 2)
            required = s16_bytes * 4 + 50 * 1024 * 1024
            free_bytes = shutil.disk_usage(str(tmp_parent)).free
            if free_bytes < required:
                raise DenoiseError(
                    f"المساحة الحرّة على القرص غير كافية لتقليل الضوضاء "
                    f"(مطلوب ≈ {required / (1024 * 1024):.0f} MB، المتوفر {free_bytes / (1024 * 1024):.0f} MB "
                    f"على القرص: {tmp_parent})."
                )
        except OSError:
            logger.debug("تعذر فحص المساحة الحرّة قبل تقليل الضوضاء.")
        # WP1: the pad must not be digital silence, or the tool zeroes the last
        # 1440 samples of real content (measured; see module docstring).
        filter_chain = build_padded_input_filter(
            downmix_mono=downmix_mono, src_channels=meta["channels"]
        )
        cmd = [
            str(ffmpeg), "-nostdin", "-y",
            "-vn",
            "-i", str(src_path),
            "-filter_complex", filter_chain,
            "-map", "[out]",
            "-c:a", "pcm_s16le",
            str(in_wav),
        ]
        _run_ffmpeg(cmd, state)

        in_meta = _probe_audio_source(in_wav, ffprobe)
        padded_samples = wav_file_sample_count(in_wav)
        in_duration = in_meta["duration"]
        out_channels = int(in_meta["channels"])

        # 5. Run deep-filter with stdout/stderr redirected to files (deadlock-safe).
        out_dir = tmp / "out"
        df_data = tmp / "df_stdout.log"
        df_err_p = tmp / "df_stderr.log"
        cmd = build_df_args(exe, in_wav, out_dir, strength)
        cmd.append("-v")
        logger.info(
            "بدء تقليل الضوضاء لـ %s (القوة %.0f dB, النوع %s)",
            src_path.name, strength, meta["codec"],
        )
        logger.debug("أمر الأداة: %s", " ".join(cmd))
        t0 = time.perf_counter()
        with open(df_data, "wb") as fo, open(df_err_p, "wb") as fe:
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=fo,
                    stderr=fe,
                    stdin=subprocess.DEVNULL,
                    creationflags=CREATE_NO_WINDOW,
                    cwd=str(tmp),
                )
            except OSError as exc:
                raise DenoiseError(f"فشل تشغيل أداة deep-filter: {exc}") from exc
            df_rc = _wait_or_cancel(proc, state)
        wall_s = time.perf_counter() - t0

        stderr_tail = _read_tail(df_err_p, max_lines=20)
        if df_rc != 0 or not (out_dir / in_wav.name).is_file():
            raise _map_df_failure(df_rc, stderr_tail)

        df_out = out_dir / in_wav.name
        df_out_meta = _probe_audio_source(df_out, ffprobe)
        df_out_samples = wav_file_sample_count(df_out)
        expected_len = padded_samples - DF_DELAY_SAMPLES
        if abs(df_out_samples - expected_len) > 2:
            logger.warning(
                "طول مخرج الأداة غير متوقع: %d مقابل %d المفترض (padded-1440). قد تكون سلوكيات الأداة تغيّرت.",
                df_out_samples, expected_len,
            )

        # WP1 content check: a length check alone cannot see a zeroed tail.
        # Compare the audible tail of the source with the tail the tool produced.
        source_tail_rms = pcm16_rms_from_wav(in_wav, max_seconds=0.1, from_end=True)
        output_tail_rms = pcm16_rms_from_wav(df_out, max_seconds=0.1, from_end=True)
        tail_restored = False
        if tail_is_damaged(source_tail_rms, output_tail_rms):
            logger.warning(
                "خرج الأداة ينتهي بصمت رقمي بينما نهاية المصدر مسموعة (RMS %.5f)؛ "
                "تُستعاد آخر %d عينة من المصدر مع تداخل 5 مللي ثانية.",
                source_tail_rms, DF_DELAY_SAMPLES,
            )
            tail_restored = True

        # 7. Clipping WARNs are one note, never an error (plan section 3 rule 8).
        stderr_text = df_err_p.read_bytes().decode("utf-8", errors="replace") if df_err_p.is_file() else ""
        clip_events = stderr_text.count(CLIPPING_WARN_TEXT)
        if clip_events:
            logger.warning(
                "ملاحظة: رُصدت %d حالة تشبع (clipping) أثناء المعالجة؛ الخرج لم يتأثر.", clip_events,
            )
        rtf_match = _RTF_LINE_RE.search(_read_tail(df_data, max_lines=200) + "\n" + stderr_text)
        measured_rtf = float(rtf_match.group(1)) if rtf_match else round(wall_s / max(in_duration, 1e-6), 4)

        # 8. Finalize with one ffmpeg call: exact length + optional rate restore.
        # When the tool damaged the tail, splice the source back for the last
        # DF_DELAY_SAMPLES instead of shipping a hole.
        codec = _encoder_for(target_codec or meta["codec"])
        out_format = format_for_suffix(dst_path.suffix)
        dst_tmp = dst_path.parent / (dst_path.name + ".tmp")
        dst_tmp.parent.mkdir(parents=True, exist_ok=True)
        tail_filters = []
        if restore_original_rate and original_rate != 48000:
            tail_filters.append(f"aresample={original_rate}")
        if tail_restored:
            graph = crossfade_tail_filter(n48)
            if tail_filters:
                graph = graph + ";[out]" + ",".join(tail_filters) + "[o]"
                out_label = "[o]"
            else:
                out_label = "[out]"
            final_cmd = [
                str(ffmpeg), "-nostdin", "-y",
                "-i", str(df_out),
                "-i", str(in_wav),
                "-filter_complex", graph,
                "-map", out_label,
            ]
        else:
            filters = [f"atrim=end_sample={n48}", f"apad=whole_len={n48}", *tail_filters]
            final_cmd = [
                str(ffmpeg), "-nostdin", "-y",
                "-i", str(df_out),
                "-filter:a", ",".join(filters),
            ]
        final_cmd += ["-c:a", codec, "-f", out_format]
        if target_bitrate:
            final_cmd += ["-b:a", target_bitrate]
        final_cmd.append(str(dst_tmp))
        _run_ffmpeg(final_cmd, state)
        # 9. Atomic replace.
        os.replace(dst_tmp, dst_path)

    if clip_events == 0:
        logger.info("اكتمل تقليل الضوضاء لـ %s (%.1f ث, RTF %s)", dst_path.name, in_duration, measured_rtf)
    return {
        "rtf": measured_rtf,
        "clip_events": clip_events,
        "in_duration_s": in_meta["duration"],
        "padded_samples": padded_samples,
        "df_out_samples": df_out_samples,
        "df_out_expected": expected_len,
        "trimmed": n48,
        "wall_s": round(wall_s, 2),
        "channels_out": out_channels,
        "tail_restored": tail_restored,
        "source_tail_rms": source_tail_rms,
        "output_tail_rms": output_tail_rms,
    }


def wav_file_sample_count(path: Path) -> int:
    """Exact sample count of a RIFF/WAVE file by parsing its chunks (no deps)."""
    raw = path.read_bytes()
    if raw[0:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise DenoiseError(f"الملف ليس بصيغة WAVE صالحة: {path}")
    pos = 12
    channels = bits = block_align = None
    data_size = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        csize = int.from_bytes(raw[pos + 4:pos + 8], "little")
        if cid == b"fmt ":
            channels = int.from_bytes(raw[pos + 10:pos + 12], "little")
            bits = int.from_bytes(raw[pos + 22:pos + 24], "little")
            block_align = int.from_bytes(raw[pos + 20:pos + 22], "little")
        elif cid == b"data":
            data_size = csize
            break
        pos += 8 + csize + (csize & 1)
    if data_size is None:
        raise DenoiseError(f"تعذر إيجاد مقطع data في ملف WAV: {path}")
    align = block_align or (channels or 1) * ((bits or 16) // 8)
    return data_size // max(1, align)


def _read_tail(path: Path, max_lines: int) -> str:
    try:
        lines = path.read_bytes().decode("utf-8", errors="replace").splitlines()
        return "\n".join(lines[-max_lines:])
    except OSError:
        return ""


def _map_df_failure(rc: int, stderr_tail: str) -> DenoiseError:
    """Map deep-filter exit codes to Arabic diagnostics (plan section 6 step 6)."""
    if rc == DF_EXIT_OOM or "memory allocation of" in stderr_tail.lower():
        return DenoiseError(
            "نفدت الذاكرة المتاحة لأداة تقليل الضوضاء، لذلك لم تُعالج هذا الجزء. "
            "اقسم الملف إلى أجزاء أقصر (10 دقائق أو أقل) أو أغلق البرامج الأخرى ثم أعد المحاولة."
        )
    if rc == 101 or "TooWide" in stderr_tail:
        return DenoiseError(
            "أداة تقليل الضوضاء لا تدعم الصوت بعمق 24 أو 32 بت، ولم يتمكن التطبيق من تحويل هذا الملف. "
            "حوّل الملف إلى 48 كيلوهرتز / 16 بت WAV (أو فعّل التحويل التلقائي) ثم أعد المحاولة."
        )
    if rc == 1:
        cause = ""
        for line in stderr_tail.splitlines():
            if "Caused by" in line:
                cause = line.strip()
                break
        return DenoiseError(f"فشل تقليل الضوضاء للأداة (رمز 1): {cause or stderr_tail}")
    if rc == 2:
        return DenoiseError(f"خطأ داخلي: معاملات غير صالحة للأداة (رمز 2).\n{stderr_tail}")
    if rc < 0:
        return DenoiseError("أُلغيت المعالجة أثناء تشغيل الأداة.")
    return DenoiseError(f"فشل تقليل الضوضاء برمز {rc}.\n{(stderr_tail or 'لا تفاصيل إضافية.')[-800:]}")


# ---------------------------------------------------------------------------
# Dual-mono detection (WP3)
# ---------------------------------------------------------------------------

# The residual must be this far below the mid signal everywhere checked before
# the shortcut is allowed. All five real sources measure -113 dB or lower, but
# the rule stays conservative: any doubt means true stereo.
DUAL_MONO_MIN_RESIDUAL_DB = 60.0
DUAL_MONO_MAX_WINDOWS = 10
DUAL_MONO_WINDOW_SECONDS = 5.0


def channel_residual_db(ffmpeg: str | Path, src: Path, start_s: float, dur_s: float,
                        state: _CancelState) -> Optional[float]:
    """L-R residual level relative to the mid signal over one window, in dB.

    Returns None when the window has no measurable mid level (silence), which the
    caller must treat as "not provably dual-mono".
    """
    def rms_of(pan: str) -> Optional[float]:
        c = [str(ffmpeg), "-nostdin", "-v", "info", "-i", str(src),
             "-filter_complex",
             f"[0:a]atrim=start={start_s:.3f}:duration={dur_s:.3f},asetpts=N/SR/TB,"
             f"{pan},astats=metadata=1:reset=0[o]",
             "-map", "[o]", "-f", "null", "-"]
        p = subprocess.run(c, capture_output=True, stdin=subprocess.DEVNULL,
                           creationflags=CREATE_NO_WINDOW, check=False)
        if state.cancelled:
            raise InterruptedError(_CANCEL_MSG)
        text = p.stderr.decode("utf-8", errors="replace")
        vals = re.findall(r"RMS level dB:\s*(-?[0-9.]+|-inf)", text)
        if not vals:
            return None
        try:
            return float(vals[-1])
        except ValueError:
            return None

    mid_db = rms_of("pan=mono|c0=0.5*c0+0.5*c1")
    dif_db = rms_of("pan=mono|c0=0.5*c0-0.5*c1")
    if mid_db is None or dif_db is None:
        return None
    if mid_db < -90.0:          # window is essentially silent: no evidence
        return None
    if dif_db <= -200.0:
        return 200.0
    return mid_db - dif_db


def is_dual_mono(ffmpeg: str | Path, src: Path, channels: int, duration_s: float,
                 state: _CancelState) -> Tuple[bool, Optional[float]]:
    """(is_dual_mono, worst residual dB). Only for exactly 2 channels."""
    if channels != 2 or duration_s <= 0:
        return False, None
    n = DUAL_MONO_MAX_WINDOWS
    win = min(DUAL_MONO_WINDOW_SECONDS, max(1.0, duration_s / n))
    worst: Optional[float] = None
    for i in range(n):
        start = i * (duration_s - win) / max(1, n - 1) if n > 1 else 0.0
        start = max(0.0, min(start, max(0.0, duration_s - win)))
        try:
            res = channel_residual_db(ffmpeg, src, start, win, state)
        except InterruptedError:
            raise
        except Exception as exc:  # noqa: BLE001 - never fail the run for a probe
            logger.debug("تعذر فحص تطابق القناتين: %s", exc)
            return False, None
        if res is None:
            continue
        worst = res if worst is None else min(worst, res)
    if worst is None:
        return False, None
    return worst >= DUAL_MONO_MIN_RESIDUAL_DB, worst



@dataclass
class DenoisePartStatus:
    """Outcome of denoising one split part."""
    part: Path
    ok: bool
    reason: str = ""
    rtf: float = 0.0
    clip_events: int = 0
    skipped: bool = False
    wall_s: float = 0.0
    channels: int = 0
    dual_mono: bool = False
    residual_db: Optional[float] = None
    tail_restored: bool = False

    @property
    def outcome(self) -> str:
        if self.ok:
            return "denoised"
        return "skipped" if self.skipped else "failed"


@dataclass
class DenoiseReport:
    """Aggregate result of denoising a split batch."""
    statuses: List[DenoisePartStatus] = field(default_factory=list)
    measured_rtfs: Dict[int, List[float]] = field(default_factory=dict)
    warned_names: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0
    report_path: Optional[Path] = None

    @property
    def failures(self) -> List[DenoisePartStatus]:
        return [s for s in self.statuses if not s.ok and not s.skipped]

    @property
    def skips(self) -> List[DenoisePartStatus]:
        return [s for s in self.statuses if s.skipped]

    @property
    def denoised(self) -> List[DenoisePartStatus]:
        return [s for s in self.statuses if s.ok]

    def counts(self) -> Dict[str, int]:
        return {
            "denoised": len(self.denoised),
            "skipped": len(self.skips),
            "failed": len(self.failures),
            "total": len(self.statuses),
        }

    def summary_lines(self) -> List[str]:
        """User-facing summary: separate counts for denoised / skipped / failed."""
        lines: List[str] = []
        c = self.counts()
        if self.statuses:
            if c["denoised"] == 0:
                reason = next((s.reason for s in self.statuses if s.reason), "")
                lines.append(
                    "لم يتم تقليل الضوضاء لأي جزء"
                    + (f" (السبب: {reason})" if reason else ".")
                )
            else:
                parts = [f"تم تقليل الضوضاء لـ {c['denoised']} من {c['total']} جزء/أجزاء."]
                if c["skipped"]:
                    parts.append(f"تُخطّي {c['skipped']}.")
                if c["failed"]:
                    parts.append(f"فشل {c['failed']}.")
                lines.append(" ".join(parts))
        for s in self.skips:
            lines.append(f"تنبيه: لم يُعالج الجزء {s.part.name} ({s.reason}).")
        for s in self.failures:
            lines.append(f"تنبيه: تُرك الجزء {s.part.name} دون تقليل الضوضاء ({s.reason}).")
        for s in self.denoised:
            if s.clip_events:
                lines.append(
                    f"ملاحظة: مدخل الجزء {s.part.name} مشبع "
                    f"({s.clip_events} حالة تشبع رُصدت أثناء المعالجة)."
                )
        return lines

    def report_text(self) -> str:
        """Contents of `denoise_report.txt` written next to the parts."""
        c = self.counts()
        lines = [
            "# تقرير تقليل الضوضاء (DeepFilterNet)",
            f"# الأجزاء: {c['total']} | تمت المعالجة: {c['denoised']} | "
            f"تُخطّي: {c['skipped']} | فشل: {c['failed']}",
            f"# الزمن الكلي: {self.elapsed_s:.1f} ثانية",
            "#",
            "# part\toutcome\treason\twall_s\trtf\tclip_events\tchannels\tdual_mono\ttail_restored",
        ]
        for s in self.statuses:
            reason = (s.reason or "-").replace("\t", " ").replace("\n", " ")[:300]
            lines.append(
                f"{s.part.name}\t{s.outcome}\t{reason}\t{s.wall_s:.2f}\t{s.rtf:.4f}\t"
                f"{s.clip_events}\t{s.channels}\t{'yes' if s.dual_mono else 'no'}\t"
                f"{'yes' if s.tail_restored else 'no'}"
            )
        return "\n".join(lines) + "\n"


def _extract_audio_stream(
    part: Path,
    stream_index: int,
    out_wav: Path,
    ffmpeg: str | Path,
    ffprobe: str | Path,
    state: _CancelState,
) -> None:
    """Extract one audio stream to a PCM WAV that deep-filter can analyze losslessly.

    pcm_s16 stays s16; 24-bit and higher integers go to s32 (bit exact); floats
    stay f32. Deep-filter would panic on raw 24/32-bit data if fed directly,
    hence the re-encode rather than `-c copy`.
    """
    info = probe_media(part, ffprobe_path=Path(ffprobe))
    audio_streams = info.get("audio_streams", [])
    if not audio_streams:
        raise DenoiseError(f"الجزء لا يحتوي على مسار صوت: {part.name}")
    src_stream = audio_streams[min(stream_index, len(audio_streams) - 1)]
    sample_fmt = src_stream.get("sample_fmt", "s16")
    # Map decode formats to a container-friendly PCM that preserves precision.
    extract_codec = "pcm_s16le" if sample_fmt in ("s16", "u8") else (
        "pcm_f32le" if sample_fmt.startswith("flt") else "pcm_s32le"
    )
    cmd = [
        str(ffmpeg), "-nostdin", "-y",
        "-i", str(part),
        "-map", f"0:a:{stream_index}",
        "-vn", "-sn", "-dn",
        "-c:a", extract_codec,
        str(out_wav),
    ]
    _run_ffmpeg(cmd, state)


def _remux_part_with_denoised_audio(
    part: Path,
    denoised: Sequence[Path],
    denoise_encoders: Sequence[Tuple[str, Optional[str]]],
    original_audio_streams: int,
    ffmpeg: str | Path,
    ffprobe: str | Path,
    state: _CancelState,
) -> None:
    """Replace the audio of `part` with denoised versions, copying video/subs.

    Output goes to <part>.tmp.<ext>, is verified (stream counts + duration),
    then atomically replaces the part so a failed remux never destroys it.

    Stream order, language tags, titles, dispositions, metadata and chapters are
    preserved: the maps are built by walking the *original* stream list and
    substituting only the audio payloads that were denoised (WP7). ffmpeg copies
    stream metadata and dispositions automatically when a stream is mapped from
    the source with `-c copy`.
    """
    info = probe_media(part, ffprobe_path=Path(ffprobe))
    cmd: List[str] = [str(ffmpeg), "-nostdin", "-y", "-i", str(part)]
    for wav in denoised:
        cmd += ["-i", str(wav)]

    # Walk the original streams in order and substitute audio payloads.
    # Stream metadata (language, title, ...) does NOT follow a stream that is
    # re-encoded from a different input, so it is copied explicitly per stream.
    audio_seen = 0
    out_audio_idx = 0
    meta_args: List[str] = []
    for stream in info.get("all_streams", []):
        stype = (stream.get("codec_type") or "").lower()
        idx = stream.get("index")
        if stype == "audio":
            if audio_seen < len(denoised):
                cmd += ["-map", f"{audio_seen + 1}:a:0"]
                tags = stream.get("tags") or {}
                for key, value in tags.items():
                    if str(key).lower() in ("encoder", "duration", "duration-ms"):
                        continue
                    meta_args += [f"-metadata:s:a:{out_audio_idx}", f"{key}={value}"]
                disp = stream.get("disposition") or {}
                for key, value in disp.items():
                    if value:
                        meta_args += [f"-disposition:a:{out_audio_idx}", key]
                out_audio_idx += 1
            else:
                cmd += ["-map", f"0:{idx}"]
            audio_seen += 1
        elif stype == "data":
            # Data streams (timecode, GPS) break some muxers; the splitter drops
            # them too, so keep that behaviour.
            continue
        elif idx is not None:
            cmd += ["-map", f"0:{idx}"]

    # Copy everything that is not being re-encoded, and the container metadata
    # (chapters and global tags follow automatically unless -map_chapters is used).
    cmd += ["-c", "copy", "-map_chapters", "0", "-map_metadata", "0", *meta_args,
            "-f", format_for_suffix(part.suffix)]
    for idx, (encoder, bitrate) in enumerate(denoise_encoders[: len(denoised)]):
        cmd += [f"-c:a:{idx}", encoder]
        if bitrate:
            cmd += [f"-b:a:{idx}", bitrate]
    out_tmp = part.with_name(part.name + ".tmp")
    cmd.append(str(out_tmp))
    try:
        _run_ffmpeg(cmd, state)
    except BaseException:
        if out_tmp.exists():
            out_tmp.unlink(missing_ok=True)
        raise

    try:
        new_probe = probe_media(out_tmp, ffprobe_path=Path(ffprobe))
        src_probe_dur = info["duration"]
        dur_diff = abs((new_probe["duration"] or 0.0) - (src_probe_dur or 0.0))
        if len(new_probe.get("video_streams", [])) != len(info.get("video_streams", [])) or dur_diff > 0.15:
            raise DenoiseError(
                f"تحقق فاشل بعد إعادة الدمج لـ {part.name} "
                f"(فرق المدة {dur_diff:.2f} ث؛ الحد 0.15 ث)."
            )
        if len(new_probe.get("audio_streams", [])) != original_audio_streams:
            raise DenoiseError(f"تحقق فاشل بعد إعادة الدمج لـ {part.name}: عدد مسارات الصوت اختلف.")
        # WP7: the stream order must match the source exactly.
        src_types = [s.get("codec_type") for s in info.get("all_streams", [])
                     if (s.get("codec_type") or "") in ("video", "audio", "subtitle", "attachment")]
        new_types = [s.get("codec_type") for s in new_probe.get("all_streams", [])
                     if (s.get("codec_type") or "") in ("video", "audio", "subtitle", "attachment")]
        if src_types != new_types:
            raise DenoiseError(
                f"تحقق فاشل بعد إعادة الدمج لـ {part.name}: ترتيب المسارات تغيّر "
                f"({src_types} -> {new_types})."
            )
    except Exception as exc:
        if isinstance(exc, InterruptedError):
            if out_tmp.exists():
                out_tmp.unlink(missing_ok=True)
            raise
        out_tmp.unlink(missing_ok=True)
        if isinstance(exc, DenoiseError):
            raise
        raise DenoiseError(f"تعذر التحقق من الجزء بعد إعادة الدمج: {exc}") from exc
    os.replace(out_tmp, part)


def _restore_channels(wav: Path, channels: int, ffmpeg: str | Path, state: _CancelState) -> None:
    """Duplicate the denoised mono channel into every original channel (WP3).

    `pan=stereo|c0=c0|c1=c0` for 2 channels; a generic `pan` for more. The file is
    rewritten in place (temp + replace) so the caller's path stays valid.
    """
    if channels <= 1:
        return
    if channels == 2:
        pan = "pan=stereo|c0=c0|c1=c0"
    else:
        outs = "|".join(f"c{i}=c0" for i in range(channels))
        pan = f"pan={channels}c|{outs}"
    tmp = wav.with_name(wav.stem + "_ch.wav")
    cmd = [str(ffmpeg), "-nostdin", "-y", "-i", str(wav), "-filter:a", pan,
           "-c:a", "pcm_s16le", str(tmp)]
    _run_ffmpeg(cmd, state)
    os.replace(tmp, wav)


def choose_temp_base(parts_dir: Optional[Path], requested: Optional[str | Path] = None) -> Tuple[Path, bool]:
    """Pick the temp root (WP4): a private dir inside the parts folder.

    The output drive already has to hold the results, so it is the safest place
    for the 4-byte-per-sample temp copies. Falls back to the system temp when
    the parts folder cannot be written. Returns (path, created_by_us).
    """
    if requested:
        return Path(requested), False
    if parts_dir:
        candidate = Path(parts_dir) / f".dfn_tmp_{os.getpid()}"
        try:
            candidate.mkdir(parents=True, exist_ok=True)
            probe = candidate / ".write_test"
            probe.write_bytes(b"x")
            probe.unlink()
            return candidate, True
        except OSError as exc:
            logger.warning("تعذر استخدام مجلد الأجزاء كمجلد مؤقت (%s)؛ سيُستخدم المجلد المؤقت للنظام.", exc)
    return Path(tempfile.gettempdir()), False


def cleanup_temp_base(path: Path, created_by_us: bool) -> None:
    """Remove the private temp dir; leave user-supplied ones alone."""
    if not created_by_us:
        return
    try:
        shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass


def write_report_file(parts_dir: Path, report: DenoiseReport) -> Optional[Path]:
    """Write `denoise_report.txt` next to the parts (WP6). Never raises."""
    try:
        target = Path(parts_dir) / "denoise_report.txt"
        target.write_text(report.report_text(), encoding="utf-8")
        return target
    except OSError as exc:
        logger.warning("تعذر كتابة تقرير تقليل الضوضاء: %s", exc)
        return None


def denoise_parts(
    parts: Sequence[Tuple[Path, float]],
    *,
    settings: Dict,
    ffmpeg_exe: Optional[Path] = None,
    ffprobe_exe: Optional[Path] = None,
    deep_filter_exe: Optional[Path] = None,
    cancel_event: Optional[threading.Event] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> DenoiseReport:
    """Denoise split parts (audio-only or video parts) in place, per plan section 7.

    Parameters:
    - parts: list of (path, duration_seconds) from the splitter result.
    - settings: {"enabled", "strength_db", "auto_convert", "restore_original_rate",
                 "workers"} (extra settings ignored).
    - progress_callback(percent 0..100, message) — called from worker threads.

    Behavior per item:
    - DenoiseError/DenoiseSkipped → the item is left untouched and warned
      (fallback to the un-denoised audio; never a silent fake-denoised file).
    - RAM guard: a unit whose RAM estimate exceeds min(4096 MB, 25% available)
      is refused (the pipeline has no smaller unit; chunking needs approval).
    """
    report = DenoiseReport()
    t_all = time.perf_counter()
    ffmpeg_bin = Path(ffmpeg_exe or find_tool("ffmpeg") or "ffmpeg")
    ffprobe_bin = Path(ffprobe_exe or find_tool("ffprobe") or "ffprobe")

    exe = deep_filter_exe or find_deep_filter(settings.get("deep_filter_path"))
    if exe is None:
        raise DenoiseError(
            "أداة DeepFilterNet غير موجودة لتفعيل تقليل الضوضاء؛ ضع deep-filter.exe في PATH أو حدد مساراً."
        )
    logger.info("أداة deep-filter محلولة إلى: %s", exe)

    state = _CancelState(cancel_event)
    workers = 1
    try:
        workers = max(1, min(2, int(settings.get("workers", 1) or 1)))
    except (TypeError, ValueError):
        pass

    # WP4: temp lives next to the parts (same drive as the results) unless the
    # caller asked for something else. One dir for the whole batch.
    parts_dir = settings.get("parts_dir")
    if not parts_dir and parts:
        parts_dir = Path(parts[0][0]).parent
    temp_root, temp_created = choose_temp_base(
        Path(parts_dir) if parts_dir else None, settings.get("temp_base")
    )
    settings = dict(settings)
    settings["temp_base"] = str(temp_root)
    logger.info("مجلد العمل المؤقت لتقليل الضوضاء: %s", temp_root)

    total_audio_seconds = sum(max(0.0, float(dur or 0.0)) for _p, dur in parts)
    done_seconds_lock = threading.Lock()
    done_seconds = {"value": 0.0}

    def _report_progress() -> None:
        if progress_callback and total_audio_seconds > 0:
            pct = min(99.9, done_seconds["value"] / total_audio_seconds * 100.0)
            remaining = max(0.0, total_audio_seconds - done_seconds["value"])
            eta = remaining * _avg_rtf(settings)
            progress_callback(pct, f"متبقٍ تقريباً {eta:.0f} ثانية")

    def _process_part(part: Path, dur: float) -> DenoisePartStatus:
        t_part = time.perf_counter()
        try:
            state.check(_CANCEL_MSG)
            info = probe_media(part, ffprobe_path=ffprobe_bin)
            audio_streams = info["audio_streams"]
            # WP2: the estimate must use the *largest* stream and the effective
            # channel count (after the dual-mono shortcut), not just stream 0.
            src_channels = max(int(s.get("channels") or 1) for s in audio_streams) if audio_streams else 1
            effective_dur = float(info.get("duration") or dur or 0.0)
            if state.cancelled:
                raise InterruptedError(_CANCEL_MSG)

            # WP3: dual-mono detection (only meaningful for exactly 2 channels).
            shortcut_on = bool(settings.get("denoise_dual_mono_shortcut", True))
            dual = False
            residual_db: Optional[float] = None
            if shortcut_on and src_channels == 2:
                dual, residual_db = is_dual_mono(ffmpeg_bin, part, src_channels, effective_dur, state)
            eff_channels = 1 if dual else src_channels

            admitted, est, limit = guard_admits(effective_dur, eff_channels)
            logger.info(
                "حارس الذاكرة: الجزء %s مدة %.1f ث، قنوات فعالة %d%s، التقدير %.0f MB، "
                "الحد %d MB، المتاح %s MB -> %s",
                part.name, effective_dur, eff_channels,
                f" (ثنائي القناة متطابق، المتبقي {residual_db:.1f} dB)" if dual else "",
                est, limit, get_available_ram_mb(), "مقبول" if admitted else "مرفوض",
            )
            if not admitted:
                return DenoisePartStatus(
                    part=part, ok=False, channels=eff_channels, dual_mono=dual,
                    residual_db=residual_db,
                    reason=(f"الحجم المقدر للذاكرة ({est} MB) أكبر من الحد ({limit} MB)؛ "
                            f"قسم الملف لمدد أقصر"),
                )
            # One temp dir per part; one deep-filter call per audio stream.
            total_rtf = 0.0
            clip_events = 0
            tail_restored = False
            part_temp_dir = settings.get("temp_base")
            with tempfile.TemporaryDirectory(
                prefix="dfn_parts_",
                dir=str(part_temp_dir) if part_temp_dir else None,
            ) as tmp_name:
                tmp = Path(tmp_name)
                denoised: List[Path] = []
                encoders: List[Tuple[str, Optional[str]]] = []
                for idx, stream in enumerate(audio_streams):
                    src_stream = tmp / f"src_{idx}.wav"
                    _extract_audio_stream(part, idx, src_stream, ffmpeg_bin, ffprobe_bin, state)
                    dst_stream = tmp / f"denoised_{idx}.wav"
                    codec_name = str(stream.get("codec_name") or "pcm_s16le")
                    encoder = _encoder_for(codec_name)
                    bitrate = _bitrate_for_stream(stream, info)
                    # WP3: denoise the mono downmix, then restore the layout below.
                    stream_dual = dual and int(stream.get("channels") or 1) == 2
                    try:
                        denoise_result = denoise_wav(
                            src_stream, dst_stream,
                            strength_db=settings.get("strength_db", 100),
                            exe=exe,
                            ffmpeg=ffmpeg_bin,
                            ffprobe=ffprobe_bin,
                            auto_convert=bool(settings.get("auto_convert", True)),
                            restore_original_rate=bool(settings.get("restore_original_rate", True)),
                            downmix_mono=stream_dual,
                            temp_base=settings.get("temp_base"),
                            cancel_event=cancel_event,
                        )
                    except DenoiseSkipped as skip:
                        return DenoisePartStatus(part=part, ok=False, reason=str(skip),
                                                 skipped=True, channels=eff_channels,
                                                 dual_mono=dual, residual_db=residual_db,
                                                 wall_s=round(time.perf_counter() - t_part, 2))
                    if stream_dual:
                        _restore_channels(dst_stream, int(stream.get("channels") or 2),
                                          ffmpeg_bin, state)
                    denoised.append(dst_stream)
                    encoders.append((encoder, bitrate))
                    total_rtf = max(total_rtf, float(denoise_result.get("rtf") or 0.0))
                    clip_events += int(denoise_result.get("clip_events") or 0)
                    tail_restored = tail_restored or bool(denoise_result.get("tail_restored"))
                    report.measured_rtfs.setdefault(eff_channels, []).append(
                        float(denoise_result.get("rtf") or 0.0)
                    )
                if state.cancelled:
                    raise InterruptedError(_CANCEL_MSG)
                _remux_part_with_denoised_audio(
                    part, denoised, encoders, len(audio_streams), ffmpeg_bin, ffprobe_bin, state,
                )
            return DenoisePartStatus(part=part, ok=True, rtf=total_rtf,
                                     clip_events=clip_events, channels=eff_channels,
                                     dual_mono=dual, residual_db=residual_db,
                                     tail_restored=tail_restored,
                                     wall_s=round(time.perf_counter() - t_part, 2))
        except InterruptedError:
            raise
        except DenoiseError as exc:
            logger.error("فشل تقليل ضوضاء الجزء %s: %s", part.name, exc)
            return DenoisePartStatus(part=part, ok=False, reason=str(exc)[:200],
                                     wall_s=round(time.perf_counter() - t_part, 2))
        except Exception as exc:  # unexpected → audible-safe fallback
            logger.exception("خطأ غير متوقع أثناء تقليل ضوضاء الجزء %s: %s", part.name, exc)
            return DenoisePartStatus(part=part, ok=False, reason=f"خطأ غير متوقع: {exc}",
                                     wall_s=round(time.perf_counter() - t_part, 2))

    def _worker(items: Sequence[Tuple[Path, float]]) -> None:
        for part, dur in items:
            if state.cancelled:
                break
            status = _process_part(part, dur)
            report.statuses.append(status)
            with done_seconds_lock:
                try:
                    done_seconds["value"] += max(0.0, float(dur or 0.0))
                except (TypeError, ValueError):
                    pass
            _report_progress()
            if state.cancelled:
                break

    sequential = [list(p) for p in parts]
    if workers <= 1 or len(parts) <= 1:
        _worker(sequential)
    else:
        half = (len(sequential) + 1) // 2
        threads = [
            threading.Thread(target=_worker, args=(sequential[:half],), daemon=True),
            threading.Thread(target=_worker, args=(sequential[half:],), daemon=True),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    if state.cancelled and len(report.statuses) < len(parts):
        logger.warning("تم إلغاء تقليل الضوضاء؛ الأجزاء المتبقية تركت دون معالجة.")
    report.elapsed_s = round(time.perf_counter() - t_all, 2)
    # WP6: per-part report next to the parts (the merger ignores unknown files).
    if parts_dir:
        report.report_path = write_report_file(Path(parts_dir), report)
    # WP4: always remove the private temp dir, including after cancellation.
    cleanup_temp_base(temp_root, temp_created)
    return report


def _avg_rtf(settings: Optional[Dict]) -> float:
    """Session-moving-average RTF if the caller recorded past runs, else defaults."""
    if not settings:
        return DEFAULT_RTF_BY_CHANNELS[1]
    history = settings.get("rtf_history") or {}
    rates = []
    for ch_list in history.values():
        rates.extend(float(x) for x in ch_list)
    if rates:
        return sum(rates) / len(rates)
    return settings.get("rtf_default", DEFAULT_RTF_BY_CHANNELS[1])
