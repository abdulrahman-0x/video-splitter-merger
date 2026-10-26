"""Video merging engine using FFmpeg concat demuxer with pre-compatibility checks and post-merge verification."""

from __future__ import annotations

import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from src.core.ffmpeg_manager import (
    check_free_space,
    find_tool,
    probe_media,
)
from src.core.runner import FFmpegRunner
from src.utils.helpers import (
    check_supported_format,
    escape_concat_path,
    format_duration,
    get_unique_path,
    is_path_length_safe,
    natural_sort_key,
)

logger = logging.getLogger(__name__)


@dataclass
class MergeResult:
    """Result of a successful video merge operation."""
    output_path: Path
    total_duration: float
    parts_count: int
    duration_difference: float


def parse_concat_list(list_path: Path) -> List[Path]:
    """Parse a concat demuxer file list strictly.

    Extracts file paths, resolving relative paths against the list's parent directory.
    Validates that every file exists.
    """
    if not list_path.is_file():
        raise FileNotFoundError(f"ملف قائمة الأجزاء غير موجود: {list_path}")

    parent_dir = list_path.parent
    file_paths: List[Path] = []

    with open(list_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # Strict check: line must start with "file "
            if not line.startswith("file "):
                raise ValueError(
                    f"سطر غير صالح في قائمة الأجزاء (السطر {line_no}): '{line}'. "
                    "يجب أن تبدأ الأسطر بكلمة 'file '."
                )

            # Extract path (strip 'file ', strip outer quotes and unescape)
            path_part = line[5:].strip()
            if path_part.startswith("'") and path_part.endswith("'"):
                inner = path_part[1:-1].replace(r"'\''", "'").replace(r"\'", "'").replace(r"\\", "\\")
            elif path_part.startswith('"') and path_part.endswith('"'):
                inner = path_part[1:-1].replace(r'\"', '"').replace(r"\\", "\\")
            else:
                inner = path_part

            target_file = (parent_dir / inner).resolve() if not Path(inner).is_absolute() else Path(inner).resolve()
            if not target_file.is_file():
                raise FileNotFoundError(
                    f"الملف المشار إليه في القائمة (السطر {line_no}) غير موجود:\n{target_file}"
                )

            file_paths.append(target_file)

    if len(file_paths) < 2:
        raise ValueError(
            f"تحتوي القائمة على {len(file_paths)} ملف/ملفات فقط. "
            "يلزم وجود جزأين على الأقل لإتمام عملية الدمج."
        )

    return file_paths


def discover_parts_in_dir(parts_dir: Path) -> List[Path]:
    """Discover parts in a directory.

    If parts_list.txt exists, parses it; otherwise naturally sorts matching video files.
    """
    if not parts_dir.is_dir():
        raise NotADirectoryError(f"مجلد الأجزاء غير موجود: {parts_dir}")

    list_candidate = parts_dir / "parts_list.txt"
    if list_candidate.is_file():
        try:
            return parse_concat_list(list_candidate)
        except Exception as exc:
            logger.warning("فشل قراءة parts_list.txt في المجلد، سيتم البحث المباشر عن الملفات: %s", exc)

    # Search for all supported video parts in directory
    candidates: List[Path] = []
    for file_path in parts_dir.iterdir():
        if file_path.is_file():
            supported, _ = check_supported_format(file_path)
            if supported:
                candidates.append(file_path)

    if len(candidates) < 2:
        raise ValueError(
            f"تم العثور على {len(candidates)} ملف/ملفات فيديو فقط في المجلد. "
            "يلزم وجود جزأين على الأقل لإتمام عملية الدمج."
        )

    return sorted(candidates, key=lambda p: natural_sort_key(p.name))


def verify_parts_compatibility(
    part_paths: List[Path],
    ffprobe_bin: Path,
) -> Tuple[List[dict], float]:
    """Probe all parts and verify that codecs, dimensions, frame rates, and stream counts match.

    Returns:
    (probes_list, sum_of_durations)
    Raises ValueError on incompatibility.
    """
    probes: List[dict] = []
    total_parts_duration = 0.0

    for idx, path in enumerate(part_paths, start=1):
        probe = probe_media(path, ffprobe_path=ffprobe_bin)
        probes.append(probe)
        total_parts_duration += probe["duration"]

    base_probe = probes[0]
    base_file = part_paths[0].name
    base_videos = base_probe["video_streams"]
    base_audios = base_probe["audio_streams"]
    base_subtitles = base_probe["subtitle_streams"]
    is_audio_only = base_probe.get("is_audio_only", False) or (not base_videos and bool(base_audios))

    if not base_videos and not base_audios:
        raise ValueError(f"الجزء الأول ({base_file}) لا يحتوي على أي مسار وسائط صالح.")

    if is_audio_only:
        if not base_audios:
            raise ValueError(f"الجزء الأول ({base_file}) لا يحتوي على أي مسار صوتي صالح.")
        b_a = base_audios[0]
        b_a_codec = b_a.get("codec_name")
        b_a_rate = b_a.get("sample_rate")
        b_a_channels = b_a.get("channels")

        for idx, (path, p_info) in enumerate(zip(part_paths[1:], probes[1:]), start=2):
            cur_file = path.name
            cur_audios = p_info["audio_streams"]
            if not cur_audios:
                raise ValueError(f"الجزء {idx} ({cur_file}) لا يحتوي على أي مسار صوتي.")
            if len(cur_audios) != len(base_audios):
                raise ValueError(
                    f"عدم تطابق في مسارات الصوت: الجزء الأول ({base_file}) به {len(base_audios)} مسار، "
                    f"بينما الجزء {idx} ({cur_file}) به {len(cur_audios)} مسار."
                )
            c_a = cur_audios[0]
            if c_a.get("codec_name") != b_a_codec:
                raise ValueError(
                    f"اختلاف ترميز الصوت: الجزء الأول ({b_a_codec}) "
                    f"يختلف عن الجزء {idx} ({c_a.get('codec_name')})."
                )
            if c_a.get("sample_rate") != b_a_rate:
                raise ValueError(
                    f"اختلاف معدل العينات الصوتي (Sample Rate): الجزء الأول ({b_a_rate} Hz) "
                    f"يختلف عن الجزء {idx} ({c_a.get('sample_rate')} Hz)."
                )
            if c_a.get("channels") != b_a_channels:
                raise ValueError(
                    f"اختلاف قنوات الصوت: الجزء الأول ({b_a_channels} قنوات) "
                    f"يختلف عن الجزء {idx} ({c_a.get('channels')} قنوات)."
                )

        return probes, total_parts_duration

    # Video mode
    base_v = base_videos[0]
    b_v_codec = base_v.get("codec_name")
    b_v_width = base_v.get("width")
    b_v_height = base_v.get("height")
    b_v_fps = base_v.get("r_frame_rate") or base_v.get("avg_frame_rate")

    for idx, (path, p_info) in enumerate(zip(part_paths[1:], probes[1:]), start=2):
        cur_file = path.name
        cur_videos = p_info["video_streams"]
        cur_audios = p_info["audio_streams"]
        cur_subtitles = p_info["subtitle_streams"]

        # 1. Stream count matching
        if len(cur_videos) != len(base_videos):
            raise ValueError(
                f"عدم تطابق في مسارات الفيديو: الجزء الأول ({base_file}) به {len(base_videos)} مسار، "
                f"بينما الجزء {idx} ({cur_file}) به {len(cur_videos)} مسار."
            )
        if len(cur_audios) != len(base_audios):
            raise ValueError(
                f"عدم تطابق في مسارات الصوت: الجزء الأول ({base_file}) به {len(base_audios)} مسار، "
                f"بينما الجزء {idx} ({cur_file}) به {len(cur_audios)} مسار."
            )
        if len(cur_subtitles) != len(base_subtitles):
            raise ValueError(
                f"عدم تطابق في مسارات الترجمة: الجزء الأول ({base_file}) به {len(base_subtitles)} مسار، "
                f"بينما الجزء {idx} ({cur_file}) به {len(cur_subtitles)} مسار."
            )

        # 2. Primary video stream properties
        c_v = cur_videos[0]
        if c_v.get("codec_name") != b_v_codec:
            raise ValueError(
                f"اختلاف ترميز الفيديو: الجزء الأول ({b_v_codec}) يختلف عن الجزء {idx} ({c_v.get('codec_name')})."
            )
        if c_v.get("width") != b_v_width or c_v.get("height") != b_v_height:
            raise ValueError(
                f"اختلاف أبعاد الفيديو: الجزء الأول ({b_v_width}x{b_v_height}) "
                f"يختلف عن الجزء {idx} ({c_v.get('width')}x{c_v.get('height')})."
            )

        # 3. Primary audio stream properties (if present)
        if base_audios and cur_audios:
            b_a = base_audios[0]
            c_a = cur_audios[0]
            if b_a.get("codec_name") != c_a.get("codec_name"):
                raise ValueError(
                    f"اختلاف ترميز الصوت: الجزء الأول ({b_a.get('codec_name')}) "
                    f"يختلف عن الجزء {idx} ({c_a.get('codec_name')})."
                )
            if b_a.get("channels") != c_a.get("channels"):
                raise ValueError(
                    f"اختلاف قنوات الصوت: الجزء الأول ({b_a.get('channels')} قنوات) "
                    f"يختلف عن الجزء {idx} ({c_a.get('channels')} قنوات)."
                )

    return probes, total_parts_duration


def derive_merged_output_path(
    source_target: Path,
    sample_extension: str,
    output_parent_dir: Optional[Path] = None,
) -> Path:
    """Derive merged output path.

    Removes _parts or _parts_N suffix from folder name and places merged file adjacent to it.
    """
    if source_target.is_file():
        parts_dir = source_target.parent
    else:
        parts_dir = source_target

    base_dir = output_parent_dir.resolve() if output_parent_dir else parts_dir.parent
    folder_name = parts_dir.name

    # Strip _parts or _parts_\d+
    clean_name = re.sub(r"_parts(?:_\d+)?$", "", folder_name, flags=re.IGNORECASE)
    if not clean_name:
        clean_name = folder_name

    ext = sample_extension if sample_extension.startswith(".") else f".{sample_extension}"
    candidate = base_dir / f"{clean_name}_merged{ext}"
    return get_unique_path(candidate)


class VideoMerger:
    """Orchestrates video merging with pre-validation, atomic replace, and duration verification."""

    def __init__(self, runner: Optional[FFmpegRunner] = None) -> None:
        self.runner = runner or FFmpegRunner()

    def cancel(self) -> None:
        """Cancel ongoing merge operation."""
        self.runner.cancel()

    def merge(
        self,
        input_target: Path | str,
        output_parent_dir: Optional[Path | str] = None,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        ffmpeg_exe: Optional[Path] = None,
        ffprobe_exe: Optional[Path] = None,
    ) -> MergeResult:
        """Merge video parts into a single video file.

        Parameters:
        - input_target: Path to parts directory OR path to parts_list.txt.
        - output_parent_dir: Optional parent folder for the merged file.
        - progress_callback: (percent: float, current_seconds: float)
        """
        target = Path(input_target).resolve()
        if not target.exists():
            raise FileNotFoundError(f"المسار المحدد غير موجود: {target}")

        # 1. Discover parts
        if target.is_file():
            part_paths = parse_concat_list(target)
            parts_dir = target.parent
        else:
            part_paths = discover_parts_in_dir(target)
            parts_dir = target

        # 2. Check tools
        ffmpeg_bin = ffmpeg_exe or find_tool("ffmpeg")
        ffprobe_bin = ffprobe_exe or find_tool("ffprobe")
        if not ffmpeg_bin or not ffprobe_bin:
            raise RuntimeError("أدوات FFmpeg / FFprobe غير متوفرة لتنفيذ الدمج.")

        # 3. Pre-compatibility checks across parts
        probes, sum_durations = verify_parts_compatibility(part_paths, ffprobe_bin)
        base_probe = probes[0]
        sample_ext = part_paths[0].suffix

        # 4. Check disk space
        total_parts_size = sum(p.stat().st_size for p in part_paths)
        dest_parent = Path(output_parent_dir).resolve() if output_parent_dir else parts_dir.parent
        has_space, free_bytes, _ = check_free_space(dest_parent, total_parts_size)
        if not has_space:
            raise RuntimeError(
                f"المساحة الحرة على القرص غير كافية لدمج الأجزاء.\n"
                f"المطلوب: {total_parts_size / (1024*1024):.1f} ميجابايت، المتوفر: {free_bytes / (1024*1024):.1f} ميجابايت."
            )

        # 5. Determine final and temporary output paths
        final_output_path = derive_merged_output_path(parts_dir, sample_ext, dest_parent)
        if not is_path_length_safe(final_output_path, max_length=240):
            raise ValueError(
                f"مسار ملف الدمج الناتج يتجاوز الحد المسموح به في ويندوز:\n{final_output_path}"
            )

        partial_output_path = final_output_path.parent / f"{final_output_path.stem}.partial{final_output_path.suffix}"

        # 6. Prepare concat list
        temp_concat_list: Optional[Path] = None
        if target.is_file() and target.name.lower() == "parts_list.txt":
            concat_list_to_use = target
        else:
            temp_concat_list = parts_dir / "concat_temp.txt"
            with open(temp_concat_list, "w", encoding="utf-8") as tf:
                tf.write("# Temporary concat list for Video Splitter & Merger\n")
                for p in part_paths:
                    tf.write(f"file {escape_concat_path(p.resolve())}\n")
            concat_list_to_use = temp_concat_list

        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list_to_use.resolve()),
            "-map", "0",
            "-c", "copy",
            str(partial_output_path.resolve()),
        ]

        logger.info("بدء دمج %d جزء في: %s", len(part_paths), partial_output_path)

        try:
            self.runner.run(
                cmd=cmd,
                total_duration=sum_durations,
                progress_callback=progress_callback,
            )
        except (Exception, KeyboardInterrupt) as exc:
            logger.warning("إلغاء أو خطأ أثناء الدمج. جاري حذف الملف المؤقت...")
            if partial_output_path.exists():
                partial_output_path.unlink(missing_ok=True)
            raise exc
        finally:
            if temp_concat_list and temp_concat_list.exists():
                temp_concat_list.unlink(missing_ok=True)

        # 7. Post-merge verification (FR-002 AC5)
        if not partial_output_path.is_file() or partial_output_path.stat().st_size == 0:
            if partial_output_path.exists():
                partial_output_path.unlink(missing_ok=True)
            raise RuntimeError("فشل الدمج: لم ينتج ملف فيديو صالح أو أن حجم الملف الناتج صفر.")

        try:
            merged_probe = probe_media(partial_output_path, ffprobe_path=ffprobe_bin)
            merged_dur = merged_probe["duration"]
            dur_diff = abs(merged_dur - sum_durations)

            # Tolerance: <= 0.5 seconds vs sum of parts
            if dur_diff > 0.5:
                logger.error(
                    "فشل التحقق بعد الدمج: مدة الفيديو المدمج (%.2f ثانية) تختلف عن مجموع الأجزاء (%.2f ثانية) بمقدار %.2f ثانية.",
                    merged_dur,
                    sum_durations,
                    dur_diff,
                )
                partial_output_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"فشل التحقق بعد الدمج: مدة الفيديو المدمج ({format_duration(merged_dur)}) "
                    f"تختلف عن مجموع الأجزاء ({format_duration(sum_durations)}) بمقدار {dur_diff:.2f} ثانية "
                    f"(الحد المسموح به 0.5 ثانية)."
                )

            # Stream counts must match base part
            if len(merged_probe["video_streams"]) != len(base_probe["video_streams"]):
                partial_output_path.unlink(missing_ok=True)
                raise RuntimeError("فشل التحقق بعد الدمج: عدد مسارات الفيديو في الملف الناتج لا يطابق الأجزاء.")
            if len(merged_probe["audio_streams"]) != len(base_probe["audio_streams"]):
                partial_output_path.unlink(missing_ok=True)
                raise RuntimeError("فشل التحقق بعد الدمج: عدد مسارات الصوت في الملف الناتج لا يطابق الأجزاء.")
            if len(merged_probe["subtitle_streams"]) != len(base_probe["subtitle_streams"]):
                partial_output_path.unlink(missing_ok=True)
                raise RuntimeError("فشل التحقق بعد الدمج: عدد مسارات الترجمة في الملف الناتج لا يطابق الأجزاء.")

            # 8. Atomic rename upon success
            os.replace(partial_output_path, final_output_path)
            logger.info("تم إنجاز الدمج بنجاح: %s", final_output_path)

            return MergeResult(
                output_path=final_output_path,
                total_duration=merged_dur,
                parts_count=len(part_paths),
                duration_difference=dur_diff,
            )

        except Exception as exc:
            if partial_output_path.exists():
                partial_output_path.unlink(missing_ok=True)
            raise exc
