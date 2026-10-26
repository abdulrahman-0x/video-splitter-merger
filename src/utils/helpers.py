"""Helper functions for time formatting, parsing, natural sorting, and path safety."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Tuple

# Supported video container extensions
SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov"}
BEST_EFFORT_VIDEO_EXTENSIONS = {".avi"}

# Supported audio container extensions
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac"}
BEST_EFFORT_AUDIO_EXTENSIONS = {".aac", ".ogg", ".opus"}

# Aggregated supported & best effort extensions
SUPPORTED_EXTENSIONS = SUPPORTED_VIDEO_EXTENSIONS | SUPPORTED_AUDIO_EXTENSIONS
BEST_EFFORT_EXTENSIONS = BEST_EFFORT_VIDEO_EXTENSIONS | BEST_EFFORT_AUDIO_EXTENSIONS
ALL_ALLOWED_EXTENSIONS = SUPPORTED_EXTENSIONS | BEST_EFFORT_EXTENSIONS

# Unicode digit translation maps
_ARABIC_INDIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
_EASTERN_ARABIC_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ASCII_DIGITS = "0123456789"
_DIGIT_TRANS_TABLE = str.maketrans(
    _ARABIC_INDIC_DIGITS + _EASTERN_ARABIC_DIGITS,
    _ASCII_DIGITS + _ASCII_DIGITS,
)


def normalize_numeric_string(text: str) -> str:
    """Normalize Arabic-Indic digits and decimal separators to standard ASCII.

    Converts:
    - Arabic-Indic digits (٠-٩) and Eastern digits (۰-۹) to 0-9.
    - Arabic decimal comma (٫ / U+066B) and standard comma (,) to dot (.).
    """
    cleaned = text.strip().translate(_DIGIT_TRANS_TABLE)
    cleaned = cleaned.replace("٫", ".").replace(",", ".")
    return cleaned


def parse_duration_minutes(text: str, min_seconds: float = 10.0) -> float:
    """Parse a user-entered duration in minutes.

    Accepts decimal numbers, commas, and Arabic digits.
    Raises ValueError with descriptive Arabic messages if input is invalid.
    """
    if not text or not text.strip():
        raise ValueError("يرجى إدخال مدة التقسيم بالدقائق.")

    normalized = normalize_numeric_string(text)

    try:
        minutes = float(normalized)
    except ValueError:
        raise ValueError(f"قيمة المدة غير صالحة: '{text}'. يرجى إدخال رقم صحيح أو عشري.")

    if not math.isfinite(minutes) or math.isnan(minutes):
        raise ValueError(f"قيمة المدة غير صالحة: '{text}'. يرجى إدخال رقم صحيح أو عشري.")

    if minutes <= 0:
        raise ValueError("يجب أن تكون مدة التقسيم أكبر من الصفر.")

    total_seconds = minutes * 60.0
    if total_seconds < min_seconds:
        raise ValueError(
            f"الحد الأدنى لمدة التقسيم هو {int(min_seconds)} ثوانٍ "
            f"({min_seconds / 60:.2g} دقيقة)."
        )

    return minutes


def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS format."""
    if seconds < 0:
        seconds = 0.0
    total_sec = int(round(seconds))
    hours = total_sec // 3600
    minutes = (total_sec % 3600) // 60
    secs = total_sec % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def parse_time_str(time_str: str) -> float:
    """Parse HH:MM:SS or HH:MM:SS.xxx into total seconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 3:
        h, m, s = parts
        return float(h) * 3600 + float(m) * 60 + float(s)
    if len(parts) == 2:
        m, s = parts
        return float(m) * 60 + float(s)
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(f"صيغة الوقت غير صالحة: {time_str}")


def natural_sort_key(string: str) -> list[Any]:
    """Sort key for natural alphanumeric sorting (e.g., part_2 before part_10)."""
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split(r"(\d+)", string)
    ]


def calculate_padding_width(total_parts: int) -> int:
    """Calculate minimum zero-padding width for part filenames.

    Returns max(3, digits_in_total_parts).
    For example:
    - 5 parts -> width 3 ('001' .. '005')
    - 999 parts -> width 3 ('001' .. '999')
    - 1500 parts -> width 4 ('0001' .. '1500')
    """
    digits = len(str(max(1, total_parts)))
    return max(3, digits)


def get_unique_path(target_path: Path, is_directory: bool = False) -> Path:
    """Generate a unique path by appending _2, _3, etc., if target already exists.

    For files: 'video_merged.mp4' -> 'video_merged_2.mp4'
    For dirs:  'video_parts'      -> 'video_parts_2'
    """
    if not target_path.exists():
        return target_path

    if is_directory or target_path.is_dir():
        parent = target_path.parent
        base_name = target_path.name
        counter = 2
        while True:
            candidate = parent / f"{base_name}_{counter}"
            if not candidate.exists():
                return candidate
            counter += 1
    else:
        parent = target_path.parent
        stem = target_path.stem
        suffix = target_path.suffix
        counter = 2
        while True:
            candidate = parent / f"{stem}_{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1


def escape_concat_path(path: str | Path) -> str:
    """Escape a file path for use in FFmpeg concat demuxer file list.

    FFmpeg concat demuxer requires:
    - Forward slashes.
    - Single quotes escaped as: '\\'' (close quote, escaped quote, re-open quote).
    Returns: 'escaped_path' (wrapped in single quotes).
    """
    posix_path = str(path).replace("\\", "/")
    escaped = posix_path.replace("'", r"'\''")
    return f"'{escaped}'"


def check_supported_format(path: str | Path) -> Tuple[bool, str]:
    """Check if the file format is supported (video or audio).

    Returns:
    - (True, "supported"): MP4, MKV, MOV, WAV, MP3, M4A, FLAC
    - (True, "best_effort"): AVI, AAC, OGG, OPUS
    - (False, "unsupported"): Any other format
    """
    suffix = Path(path).suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS:
        return True, "supported"
    if suffix in BEST_EFFORT_EXTENSIONS:
        return True, "best_effort"
    return False, "unsupported"


def is_audio_file(path: str | Path) -> bool:
    """Check if the file extension corresponds to a supported audio format."""
    suffix = Path(path).suffix.lower()
    return suffix in (SUPPORTED_AUDIO_EXTENSIONS | BEST_EFFORT_AUDIO_EXTENSIONS)


def is_video_file(path: str | Path) -> bool:
    """Check if the file extension corresponds to a supported video format."""
    suffix = Path(path).suffix.lower()
    return suffix in (SUPPORTED_VIDEO_EXTENSIONS | BEST_EFFORT_VIDEO_EXTENSIONS)


def get_media_category(path: str | Path) -> str:
    """Return category of media: 'video', 'audio', or 'unknown'."""
    if is_video_file(path):
        return "video"
    if is_audio_file(path):
        return "audio"
    return "unknown"


def is_path_length_safe(path: str | Path, max_length: int = 250) -> bool:
    """Check if the path length is within standard Windows MAX_PATH limits."""
    return len(str(Path(path).resolve())) <= max_length
