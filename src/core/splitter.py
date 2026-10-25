"""Video splitting engine using FFmpeg stream copy with keyframe alignment."""

from __future__ import annotations

import csv
import logging
import math
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from src.core.ffmpeg_manager import (
    check_free_space,
    find_tool,
    probe_media,
)
from src.core.runner import FFmpegRunner
from src.utils.helpers import (
    calculate_padding_width,
    check_supported_format,
    escape_concat_path,
    format_duration,
    get_unique_path,
    is_path_length_safe,
    natural_sort_key,
)

logger = logging.getLogger(__name__)


@dataclass
class PartInfo:
    """Information about a single split video part."""
    file_path: Path
    duration: float
    start_time: float
    end_time: float


@dataclass
class SplitResult:
    """Result of a successful video split operation."""
    parts_dir: Path
    parts: List[PartInfo]
    parts_list_file: Path
    total_original_duration: float
    requested_segment_duration: float
    has_deviation_warning: bool = False
    warning_message: str = ""
    created_files: List[Path] = field(default_factory=list)


def build_stream_map(streams: List[dict], container_suffix: str) -> List[str]:
    """Build FFmpeg -map arguments for video, audio, and subtitle streams.

    Explicitly excludes data streams (like timecode or GoPro GPS) to prevent MP4 muxer errors.
    Allows attachments for MKV containers.
    """
    map_args: List[str] = []
    is_mkv = container_suffix.lower() == ".mkv"

    allowed_types = {"video", "audio", "subtitle"}
    if is_mkv:
        allowed_types.add("attachment")

    for s in streams:
        codec_type = s.get("codec_type", "").lower()
        idx = s.get("index")
        if idx is not None and codec_type in allowed_types:
            map_args.extend(["-map", f"0:{idx}"])

    if not map_args:
        # Fallback if no streams matched specific criteria
        map_args = ["-map", "0:v", "-map", "0:a?", "-map", "0:s?"]

    return map_args


class VideoSplitter:
    """Orchestrates video splitting with safety pre-checks, progress, and cleanup."""

    def __init__(self, runner: Optional[FFmpegRunner] = None) -> None:
        self.runner = runner or FFmpegRunner()

    def cancel(self) -> None:
        """Cancel ongoing split operation."""
        self.runner.cancel()

    def split(
        self,
        input_path: Path | str,
        segment_duration_sec: float,
        output_parent_dir: Optional[Path | str] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        ffmpeg_exe: Optional[Path] = None,
        ffprobe_exe: Optional[Path] = None,
    ) -> SplitResult:
        """Split video file into segments of requested duration.

        Parameters:
        - input_path: Target video file path.
        - segment_duration_sec: Desired segment length in seconds.
        - output_parent_dir: Directory where `<video_name>_parts` folder will be created.
        - progress_callback: (percent: float, current_seconds: float)
        """
        source_path = Path(input_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"ملف الفيديو غير موجود: {source_path}")

        # 1. Format verification
        supported, status = check_supported_format(source_path)
        if not supported:
            raise ValueError(
                f"صيغة الفيديو '{source_path.suffix}' غير مدعومة. "
                "الصيغ المدعومة هي: MP4, MKV, MOV (وصيغة AVI بدعم تجريبي)."
            )

        # 2. Tools verification
        ffmpeg_bin = ffmpeg_exe or find_tool("ffmpeg")
        ffprobe_bin = ffprobe_exe or find_tool("ffprobe")
        if not ffmpeg_bin or not ffprobe_bin:
            raise RuntimeError("أدوات FFmpeg / FFprobe غير متوفرة لتنفيذ التقسيم.")

        # 3. Probe input media
        probe_data = probe_media(source_path, ffprobe_path=ffprobe_bin)
        total_duration = probe_data["duration"]
        if total_duration <= 0.0:
            raise ValueError("تعذر تحديد مدة الفيديو أو أن مدته تساوي صفراً.")

        if segment_duration_sec >= total_duration:
            raise ValueError(
                f"الفيديو أقصر من مدة التقسيم المحددة ({format_duration(segment_duration_sec)}).\n"
                f"مدة الفيديو الإجمالية: {format_duration(total_duration)}."
            )

        # 4. Storage and path checks
        source_size = source_path.stat().st_size
        dest_base_dir = Path(output_parent_dir).resolve() if output_parent_dir else source_path.parent
        has_space, free_bytes, _ = check_free_space(dest_base_dir, source_size)
        if not has_space:
            raise RuntimeError(
                f"المساحة الحرة على القرص غير كافية لتقسيم الفيديو.\n"
                f"المطلوب: {source_size / (1024*1024):.1f} ميجابايت، المتوفر: {free_bytes / (1024*1024):.1f} ميجابايت."
            )

        # Destination folder: <stem>_parts (with _2, _3 if already exists)
        target_parts_dir = get_unique_path(dest_base_dir / f"{source_path.stem}_parts", is_directory=True)
        if not is_path_length_safe(target_parts_dir, max_length=240):
            raise ValueError(
                f"مسار مجلد الحفظ الناتج طويل جداً ويتجاوز حدود نظام ويندوز:\n{target_parts_dir}"
            )

        # Prepare folder and tracked files
        target_parts_dir.mkdir(parents=True, exist_ok=True)
        dir_created_by_us = True
        created_files: List[Path] = []

        # 5. Build naming pattern and command
        expected_parts_count = math.ceil(total_duration / segment_duration_sec)
        pad_width = calculate_padding_width(expected_parts_count)
        part_name_pattern = f"{source_path.stem}_part_%0{pad_width}d{source_path.suffix}"
        part_pattern_path = target_parts_dir / part_name_pattern

        segment_list_csv = target_parts_dir / "segments_temp.csv"
        map_args = build_stream_map(probe_data["all_streams"], source_path.suffix)

        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-i", str(source_path),
            "-c", "copy",
            *map_args,
            "-f", "segment",
            "-segment_time", str(segment_duration_sec),
            "-reset_timestamps", "1",
            "-segment_list", str(segment_list_csv),
            "-segment_list_type", "csv",
            str(part_pattern_path),
        ]

        logger.info("بدء تقسيم الفيديو إلى مجلد: %s", target_parts_dir)

        try:
            self.runner.run(
                cmd=cmd,
                total_duration=total_duration,
                progress_callback=progress_callback,
            )
        except (Exception, KeyboardInterrupt) as exc:
            logger.warning("إلغاء أو فشل أثناء التقسيم. جاري تنظيف الملفات المنشأة...")
            # Clean up all created files in target_parts_dir
            for p in target_parts_dir.iterdir():
                try:
                    if p.is_file():
                        p.unlink(missing_ok=True)
                except Exception:
                    pass
            if dir_created_by_us:
                try:
                    target_parts_dir.rmdir()
                except Exception:
                    pass
            raise exc

        # 6. Read segments list and verify created parts
        parts_info: List[PartInfo] = []
        if segment_list_csv.is_file():
            try:
                with open(segment_list_csv, "r", encoding="utf-8", errors="replace") as csvfile:
                    reader = csv.reader(csvfile)
                    for row in reader:
                        if not row or len(row) < 3:
                            continue
                        name = row[0].strip()
                        start_t = float(row[1].strip())
                        end_t = float(row[2].strip())
                        part_file = target_parts_dir / name
                        if part_file.is_file():
                            dur = max(0.0, end_t - start_t)
                            parts_info.append(
                                PartInfo(
                                    file_path=part_file,
                                    duration=dur,
                                    start_time=start_t,
                                    end_time=end_t,
                                )
                            )
                            created_files.append(part_file)
            finally:
                segment_list_csv.unlink(missing_ok=True)

        # Fallback if csv wasn't populated or produced fewer files
        if not parts_info:
            found_parts = sorted(
                [p for p in target_parts_dir.glob(f"{source_path.stem}_part_*{source_path.suffix}")],
                key=lambda p: natural_sort_key(p.name),
            )
            for p in found_parts:
                part_probe = probe_media(p, ffprobe_path=ffprobe_bin)
                dur = part_probe["duration"]
                parts_info.append(PartInfo(file_path=p, duration=dur, start_time=0.0, end_time=dur))
                created_files.append(p)

        if not parts_info:
            raise RuntimeError("لم ينتج أمر FFmpeg أي أجزاء فيديو صالحة.")

        # 7. Generate parts_list.txt for future concat
        parts_list_file = target_parts_dir / "parts_list.txt"
        with open(parts_list_file, "w", encoding="utf-8") as pf:
            pf.write("# Concat list generated by Video Splitter & Merger\n")
            for part in parts_info:
                # Use relative filename with forward slashes and single-quote escaping
                escaped = escape_concat_path(part.file_path.name)
                pf.write(f"file {escaped}\n")
        created_files.append(parts_list_file)

        # 8. Check Keyframe deviation warning (AC5)
        # Deviation threshold: min(10% of requested, 30 seconds)
        max_deviation_thresh = min(segment_duration_sec * 0.10, 30.0)
        has_deviation_warning = False
        warning_msg = ""

        # Check all parts except the last one (last part holds the remaining duration)
        for i, part in enumerate(parts_info[:-1], start=1):
            dev = abs(part.duration - segment_duration_sec)
            if dev > max_deviation_thresh:
                has_deviation_warning = True
                warning_msg = (
                    f"ملاحظة: تباعد إطارات الـ Keyframes في الفيديو أدى إلى اختلاف مدة الجزء {i} "
                    f"({format_duration(part.duration)}) عن المدة المطلوبة "
                    f"بمقدار {dev:.1f} ثانية (أكبر من الحد {max_deviation_thresh:.1f} ثانية)."
                )
                logger.info("تنبيه انحراف المدة: %s", warning_msg)
                break

        return SplitResult(
            parts_dir=target_parts_dir,
            parts=parts_info,
            parts_list_file=parts_list_file,
            total_original_duration=total_duration,
            requested_segment_duration=segment_duration_sec,
            has_deviation_warning=has_deviation_warning,
            warning_message=warning_msg,
            created_files=created_files,
        )
