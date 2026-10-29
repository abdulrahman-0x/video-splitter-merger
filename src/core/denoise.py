"""DeepFilterNet noise-reduction wrapper for the splitter/merger app.

Wraps the external `deep-filter` command-line tool (DeepFilterNet3, tested with
0.5.6 on Windows) and exposes one function per call (one input file per call:

1. The tool only reads 8/16/22.05/32/44.1/48 kHz u8/s16/f32 WAV input safely and
   panics (exit 101, `TooWide`) on 24-bit and 32-bit PCM, which is the app's
   normal source format. Every input is therefore re-encoded by ffmpeg to
   48 kHz `pcm_s16le` inside a private temp dir (ASCII names only).
2. Without `-D` the enhanced output is delayed by exactly 1440 samples (30 ms
   at 48 kHz); with `-D` it is aligned but exactly 1440 samples shorter than the
   input (the tail is dropped). The pipeline therefore pads the input with
   0.1 s of silence (`apad`), runs with `-D`, then trims back to the exact
   original sample count (`atrim=end_sample=N48,apad=whole_len=N48`).
3. `-a 0` is a bit-exact passthrough that mis-aligns when combined with `-D`:
   strength <= 0 is treated as "feature off" and the tool is never launched.
4. The tool emits no progress; the caller estimates ETA with
   ETA = remaining audio duration * RTF (measured: mono ~0.15-0.19,
   stereo ~0.26 on the machine this was built on).

All measured behavior above (2026-10-29, deep-filter 0.5.6, ffmpeg 9.0.2) is
reproducible with the smoke scripts kept in the project report
`deepfilternet_implementation_report.md`.
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

# Extra silence appended before denoising so the dropped tail never removes
# real audio (4800 samples at 48 kHz, i.e. 0.1 s).
PAD_SECONDS = 0.1
PAD_SAMPLES_48K = int(PAD_SECONDS * 48000)

# Convergence-independent defaults when no RTF has been measured yet in this
# session (plan section 3, rule 6): mono / stereo, indexed by channel count.
DEFAULT_RTF_BY_CHANNELS = {1: 0.20, 2: 0.27}

# RAM estimate constant from the plan: ~0.45 MB/second per mono channel at
# 48 kHz s16 (measured: 60 min mono ~1424 MB peak working set).
RAM_MB_PER_SECOND_PER_CHANNEL = 0.45

# Upper bound for a single denoise unit and the share of available RAM it may claim.
RAM_GUARD_MAX_MB = 4096
RAM_GUARD_AVAIL_SHARE = 0.25

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
    """Plan section 3, rule 5 estimate: `ram_mb = duration * channels * 0.45`."""
    return int(duration_s * max(1, channels) * RAM_MB_PER_SECOND_PER_CHANNEL)


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


def ram_guard_limit_mb() -> int:
    """Plan section 3 rule 5: min(4096 MB, 25% of available RAM)."""
    avail = get_available_ram_mb()
    if avail is None:
        return RAM_GUARD_MAX_MB
    return min(RAM_GUARD_MAX_MB, int(avail * RAM_GUARD_AVAIL_SHARE))


def compute_n48(duration_s: float) -> int:
    """Original content length at 48 kHz in samples (`round(duration * 48000)`)."""
    return max(0, round(duration_s * 48000))


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

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

    # 4. Private temp dir; conform input: 48 kHz s16 + 0.1 s pad.
    with tempfile.TemporaryDirectory(prefix="dfn_", dir=str(temp_base) if temp_base else None) as tmp_name:
        tmp = Path(tmp_name)
        in_wav = tmp / "in.wav"
        # Free-disk guard (plan section 9): in.wav + DF output + trimmed final coexist.
        try:
            tmp_parent = Path(temp_base) if temp_base else Path(tempfile.gettempdir())
            s16_bytes = int(n48 * meta["channels"] * 2)
            required = s16_bytes * 4 + 50 * 1024 * 1024
            free_bytes = shutil.disk_usage(str(tmp_parent)).free
            if free_bytes < required:
                raise DenoiseError(
                    f"المساحة الحرّة على القرص غير كافية لتقليل الضوضاء "
                    f"(مطلوب ≈ {required / (1024 * 1024):.0f} MB، المتوفر {free_bytes / (1024 * 1024):.0f} MB)."
                )
        except OSError:
            logger.debug("تعذر فحص المساحة الحرّة قبل تقليل الضوضاء.")
        filter_chain = "aresample=48000,apad=pad_dur=0.1"
        cmd = [
            str(ffmpeg), "-nostdin", "-y",
            "-vn",
            "-i", str(src_path),
            "-map", "0:a:0",
            "-filter:a", filter_chain,
            "-c:a", "pcm_s16le",
            str(in_wav),
        ]
        if downmix_mono:
            cmd[cmd.index("-filter:a") + 1] = (
                "aresample=48000,pan=mono|c0=0.5*c0+0.5*c1,apad=pad_dur=0.1"
            )
        _run_ffmpeg(cmd, state)

        in_meta = _probe_audio_source(in_wav, ffprobe)
        padded_samples = wav_file_sample_count(in_wav)
        in_duration = in_meta["duration"]

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
        filters = [f"atrim=end_sample={n48}", f"apad=whole_len={n48}"]
        if restore_original_rate and original_rate != 48000:
            filters.append(f"aresample={original_rate}")
        codec = _encoder_for(target_codec or meta["codec"])
        out_format = format_for_suffix(dst_path.suffix)
        dst_tmp = dst_path.parent / (dst_path.name + ".tmp")
        dst_tmp.parent.mkdir(parents=True, exist_ok=True)
        final_cmd = [
            str(ffmpeg), "-nostdin", "-y",
            "-i", str(df_out),
            "-filter:a", ",".join(filters),
            "-c:a", codec,
            "-f", out_format,
        ]
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
    if rc == 101 or "TooWide" in stderr_tail:
        return DenoiseError(
            f"توقف تقليل الضوضاء: الحديد لا يقبل هذا العمق البت (رمز {rc}).\n{stderr_tail}"
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
# Pipeline integration (plan section 7): per part after splitting
# ---------------------------------------------------------------------------

@dataclass
class DenoisePartStatus:
    """Outcome of denoising one split part."""
    part: Path
    ok: bool
    reason: str = ""
    rtf: float = 0.0
    clip_events: int = 0
    skipped: bool = False


@dataclass
class DenoiseReport:
    """Aggregate result of denoising a split batch."""
    statuses: List[DenoisePartStatus] = field(default_factory=list)
    measured_rtfs: Dict[int, List[float]] = field(default_factory=dict)
    warned_names: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def failures(self) -> List[DenoisePartStatus]:
        return [s for s in self.statuses if not s.ok]

    def summary_lines(self) -> List[str]:
        lines: List[str] = []
        failed = [s for s in self.statuses if not s.ok]
        clipped = [s for s in self.statuses if s.ok and s.clip_events]
        if self.statuses:
            lines.append(f"تم تقليل الضوضاء بنجاح لـ {sum(1 for s in self.statuses if s.ok)} من {len(self.statuses)} جزء/أجزاء.")
        for s in failed:
            lines.append(f"تنبيه: تُرك الجزء {s.part.name} دون تقليل الضوضاء ({s.reason}).")
        for s in clipped:
            lines.append(f"ملاحظة: مدخل الجزء {s.part.name} مشبع ({s.clip_events} حالة تشبع رُصدت أثناء المعالجة).")
        return lines


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
    """
    info = probe_media(part, ffprobe_path=Path(ffprobe))
    cmd: List[str] = [str(ffmpeg), "-nostdin", "-y", "-i", str(part)]
    for wav in denoised:
        cmd += ["-i", str(wav)]
    # ?-mappers: video/subtitle/attachment maps degrade to no-ops for audio-only parts.
    cmd += ["-map", "0:v?", "-map", "0:s?", "-map", "0:t?"]
    for idx in range(len(denoised)):
        cmd += ["-map", f"{idx + 1}:a"]
    # Map back un-replaced audio streams (more streams than denoised) to preserve counts.
    handled_streams = min(len(denoised), original_audio_streams)
    if original_audio_streams > handled_streams:
        for extra in range(handled_streams, original_audio_streams):
            cmd += ["-map", f"0:a:{extra}?"]
    cmd += ["-c:v", "copy", "-c:s", "copy", "-c:t", "copy", "-f", format_for_suffix(part.suffix)]
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
            raise DenoiseError(f"تحقق فاشل بعد إعادة الدمج لـ {part.name}: عدد مسارات الصوت اختلّف.")
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
        try:
            state.check(_CANCEL_MSG)
            info = probe_media(part, ffprobe_path=ffprobe_bin)
            audio_streams = info["audio_streams"]
            channels = int(audio_streams[0].get("channels") or 2)
            effective_dur = float(info.get("duration") or dur or 0.0)
            guard = ram_guard_limit_mb()
            if state.cancelled:
                raise InterruptedError(_CANCEL_MSG)
            if estimate_ram_mb(effective_dur, channels) > guard:
                return DenoisePartStatus(
                    part=part, ok=False,
                    reason=f"الحجم المقدر للذاكرة أكبر من الحد ({guard} MB)؛ قسم الملف لمدد أقصر",
                )
            # One temp dir per part; one deep-filter call per audio stream.
            total_rtf = 0.0
            clip_events = 0
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
                    try:
                        denoise_result = denoise_wav(
                            src_stream, dst_stream,
                            strength_db=settings.get("strength_db", 100),
                            exe=exe,
                            ffmpeg=ffmpeg_bin,
                            ffprobe=ffprobe_bin,
                            auto_convert=bool(settings.get("auto_convert", True)),
                            restore_original_rate=bool(settings.get("restore_original_rate", True)),
                            temp_base=settings.get("temp_base"),
                            cancel_event=cancel_event,
                        )
                    except DenoiseSkipped as skip:
                        return DenoisePartStatus(part=part, ok=False, reason=str(skip), skipped=True)
                    denoised.append(dst_stream)
                    encoders.append((encoder, bitrate))
                    total_rtf = max(total_rtf, float(denoise_result.get("rtf") or 0.0))
                    clip_events += int(denoise_result.get("clip_events") or 0)
                    report.measured_rtfs.setdefault(channels, []).append(
                        float(denoise_result.get("rtf") or 0.0)
                    )
                if state.cancelled:
                    raise InterruptedError(_CANCEL_MSG)
                _remux_part_with_denoised_audio(
                    part, denoised, encoders, len(audio_streams), ffmpeg_bin, ffprobe_bin, state,
                )
            return DenoisePartStatus(part=part, ok=True, rtf=total_rtf, clip_events=clip_events)
        except InterruptedError:
            raise
        except DenoiseError as exc:
            logger.error("فشل تقليل ضوضاء الجزء %s: %s", part.name, exc)
            return DenoisePartStatus(part=part, ok=False, reason=str(exc)[:200])
        except Exception as exc:  # unexpected → audible-safe fallback
            logger.exception("خطأ غير متوقع أثناء تقليل ضوضاء الجزء %s: %s", part.name, exc)
            return DenoisePartStatus(part=part, ok=False, reason=f"خطأ غير متوقع: {exc}")

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
    return report


def _avg_rtf(settings: Optional[Dict]) -> float:
    """Session-moving-average RTF if the caller recorded past runs, else defaults."""
    if not settings:
        return 0.20
    history = settings.get("rtf_history") or {}
    rates = []
    for ch_list in history.values():
        rates.extend(float(x) for x in ch_list)
    if rates:
        return sum(rates) / len(rates)
    return settings.get("rtf_default", 0.20)
