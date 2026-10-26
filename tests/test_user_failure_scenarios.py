"""Comprehensive test suite for all user failure, edge-case, and environmental error scenarios.

Adheres to Test Guard rules:
- Real synthetic test files and objects (no MagicMocks).
- Parametrization where inputs vary under identical setup.
- Asserts clear, user-facing error messages and safe cleanup side-effects.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from src.core.ffmpeg_manager import (
    check_free_space,
    extract_binaries_from_zip,
    probe_media,
)
from src.core.merger import (
    VideoMerger,
    discover_parts_in_dir,
    parse_concat_list,
    verify_parts_compatibility,
)
from src.core.splitter import VideoSplitter
from src.gui.app import App
from src.utils.helpers import (
    check_supported_format,
    get_unique_path,
    is_path_length_safe,
    parse_duration_minutes,
)
from tests.conftest import create_synthetic_video
from tests.test_ffmpeg_manager import create_mock_zip


# ============================================================================
# 1. User Input & Duration Parsing Scenarios
# ============================================================================
class TestUserInputScenarios:
    """Tests handling of invalid, malicious, or malformed user inputs."""

    @pytest.mark.parametrize(
        "empty_input",
        ["", "   ", "\t", "\n"],
    )
    def test_user_leaves_duration_field_blank(self, empty_input: str) -> None:
        with pytest.raises(ValueError, match="يرجى إدخال مدة التقسيم بالدقائق"):
            parse_duration_minutes(empty_input)

    @pytest.mark.parametrize(
        "garbage_text",
        [
            "abc",
            "عشر دقائق",
            "30 min",
            "12..5",
            "10:00",
            "!@#$%",
            "None",
            "NaN",
        ],
    )
    def test_user_enters_non_numeric_text_in_duration(self, garbage_text: str) -> None:
        with pytest.raises(ValueError, match="غير صالحة"):
            parse_duration_minutes(garbage_text)

    @pytest.mark.parametrize(
        "zero_or_negative",
        ["0", "0.0", "-1", "-0.5", "-٣٠", "٠"],
    )
    def test_user_enters_zero_or_negative_duration(self, zero_or_negative: str) -> None:
        with pytest.raises(ValueError, match="أكبر من الصفر"):
            parse_duration_minutes(zero_or_negative)

    @pytest.mark.parametrize(
        "too_short_text",
        ["0.05", "0.1", "0.15", "٠٫١"],  # Under 10 seconds (10s = 0.16667 min)
    )
    def test_user_enters_duration_under_ten_seconds_minimum(self, too_short_text: str) -> None:
        with pytest.raises(ValueError, match="الحد الأدنى لمدة التقسيم هو 10 ثوانٍ"):
            parse_duration_minutes(too_short_text)

    @pytest.mark.parametrize(
        "unsupported_ext",
        [
            "video.flv",
            "movie.wmv",
            "clip.webm",
            "audio.wma",
            "image.png",
            "doc.pdf",
            "archive.zip",
            "script.py",
        ],
    )
    def test_user_selects_unsupported_file_extension(self, unsupported_ext: str) -> None:
        is_supported, status = check_supported_format(unsupported_ext)
        assert is_supported is False
        assert status == "unsupported"

    def test_user_attempts_split_without_selecting_file_in_gui(self) -> None:
        app = App()
        app.update()
        try:
            app.split_view.selected_file = None
            app.split_view._on_start_click()
            assert "يرجى اختيار ملف فيديو صالح أولاً" in app.split_view.status_label.cget("text")
        finally:
            app.destroy()

    def test_user_attempts_merge_without_selecting_target_in_gui(self) -> None:
        app = App()
        app.update()
        try:
            app.merge_view.selected_target = None
            app.merge_view._on_start_click()
            assert "يرجى اختيار مجلد أجزاء" in app.merge_view.status_label.cget("text")
        finally:
            app.destroy()


# ============================================================================
# 2. File, Container & Path Problems
# ============================================================================
class TestFileAndPathProblems:
    """Tests handling of corrupted files, missing files, long paths, and pre-existing names."""

    def test_user_selects_nonexistent_video_file_for_split(self, tmp_path: Path) -> None:
        splitter = VideoSplitter()
        missing_file = tmp_path / "deleted_video.mp4"
        with pytest.raises(FileNotFoundError, match="غير موجود"):
            splitter.split(input_path=missing_file, segment_duration_sec=30.0)

    def test_user_selects_empty_corrupted_file(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        _, ffprobe_bin = require_ffmpeg
        fake_video = tmp_path / "corrupted.mp4"
        fake_video.write_bytes(b"This is not a real video file contents at all.")

        with pytest.raises(RuntimeError, match="تعذر قراءة ملف الفيديو"):
            probe_media(fake_video, ffprobe_path=ffprobe_bin)

    def test_user_selects_segment_duration_longer_than_video(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        short_video = tmp_path / "ten_second_video.mp4"
        create_synthetic_video(short_video, ffmpeg_bin, duration_sec=10)

        splitter = VideoSplitter()
        # Request 60s split on a 10s video
        with pytest.raises(ValueError, match="(الفيديو|الملف) أقصر من مدة التقسيم المحددة"):
            splitter.split(
                input_path=short_video,
                segment_duration_sec=60.0,
                ffmpeg_exe=ffmpeg_bin,
                ffprobe_exe=ffprobe_bin,
            )

    def test_user_destination_folder_already_exists_does_not_overwrite(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        video = tmp_path / "lecture.mp4"
        create_synthetic_video(video, ffmpeg_bin, duration_sec=20)

        # Pre-existing folders
        folder_1 = tmp_path / "lecture_parts"
        folder_1.mkdir()
        canary_file = folder_1 / "user_private_notes.txt"
        canary_file.write_text("Do not overwrite!", encoding="utf-8")

        folder_2 = tmp_path / "lecture_parts_2"
        folder_2.mkdir()

        splitter = VideoSplitter()
        res = splitter.split(
            input_path=video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Should automatically select _parts_3 without touching _parts or _parts_2
        assert res.parts_dir.name == "lecture_parts_3"
        assert canary_file.is_file()
        assert canary_file.read_text(encoding="utf-8") == "Do not overwrite!"

    def test_user_destination_merged_file_already_exists_does_not_overwrite(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        video = tmp_path / "meeting.mp4"
        create_synthetic_video(video, ffmpeg_bin, duration_sec=20)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Create existing merged file
        existing_merged = tmp_path / "meeting_merged.mp4"
        existing_merged.write_text("Previous merged version", encoding="utf-8")

        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Must not overwrite meeting_merged.mp4; should create meeting_merged_2.mp4
        assert merge_res.output_path.name == "meeting_merged_2.mp4"
        assert existing_merged.read_text(encoding="utf-8") == "Previous merged version"

    def test_user_path_with_complex_arabic_symbols_and_apostrophes(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        weird_name = "محاضرة 'مهمة جداً' (النسخة النهائية) - قسم [1] & اختبار.mp4"
        video = tmp_path / weird_name
        create_synthetic_video(video, ffmpeg_bin, duration_sec=20)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert split_res.parts_dir.is_dir()
        assert len(split_res.parts) == 2

        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert merge_res.output_path.is_file()

    def test_user_output_path_exceeding_windows_max_path_is_rejected(
        self, tmp_path: Path
    ) -> None:
        very_long_path = tmp_path / ("a" * 260) / "video.mp4"
        assert is_path_length_safe(very_long_path, max_length=240) is False


# ============================================================================
# 3. Merge Incompatibility & Corruption Scenarios
# ============================================================================
class TestMergeIncompatibilityScenarios:
    """Tests handling when parts to be merged are corrupted or incompatible."""

    def test_user_attempts_merge_with_only_one_part_in_folder(self, tmp_path: Path) -> None:
        single_part_dir = tmp_path / "solo_parts"
        single_part_dir.mkdir()
        (single_part_dir / "part_1.mp4").touch()

        with pytest.raises(ValueError, match="جزأين على الأقل"):
            discover_parts_in_dir(single_part_dir)

    def test_user_folder_has_extraneous_non_video_files_is_handled_safely(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        folder = tmp_path / "dirty_parts"
        folder.mkdir()

        # Create two real video parts
        p1 = folder / "my_part_1.mp4"
        p2 = folder / "my_part_2.mp4"
        create_synthetic_video(p1, ffmpeg_bin, duration_sec=5)
        create_synthetic_video(p2, ffmpeg_bin, duration_sec=5)

        # Add extraneous non-video files that a user might have dropped in
        (folder / "thumbnail.jpg").write_bytes(b"\xff\xd8\xff")
        (folder / "readme.txt").write_text("notes", encoding="utf-8")
        (folder / "subtitles.srt").write_text("1\n00:00:01 --> 00:00:02\nHi", encoding="utf-8")
        (folder / "ignore.doc").touch()

        # discover_parts_in_dir should filter to only the two video files
        discovered = discover_parts_in_dir(folder)
        assert len(discovered) == 2
        assert [p.name for p in discovered] == ["my_part_1.mp4", "my_part_2.mp4"]

        # Merge must succeed ignoring the non-video files
        merger = VideoMerger()
        res = merger.merge(folder, ffmpeg_exe=ffmpeg_bin, ffprobe_exe=ffprobe_bin)
        assert res.output_path.is_file()

    def test_user_concat_list_contains_invalid_syntax_line(self, tmp_path: Path) -> None:
        (tmp_path / "part_1.mp4").touch()
        corrupt_list = tmp_path / "parts_list.txt"
        corrupt_list.write_text(
            "# Valid comment\n"
            "file 'part_1.mp4'\n"
            "This line is completely broken!\n"
            "file 'part_2.mp4'\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="سطر غير صالح في قائمة الأجزاء"):
            parse_concat_list(corrupt_list)

    def test_user_concat_list_references_deleted_file(self, tmp_path: Path) -> None:
        folder = tmp_path / "parts"
        folder.mkdir()
        (folder / "part_1.mp4").touch()
        # part_2 is NOT created (simulating user deleted it)
        list_file = folder / "parts_list.txt"
        list_file.write_text("file 'part_1.mp4'\nfile 'part_2.mp4'\n", encoding="utf-8")

        with pytest.raises(FileNotFoundError, match="غير موجود"):
            parse_concat_list(list_file)

    def test_user_attempts_merge_parts_with_different_video_codecs(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: H.264
        v1 = tmp_path / "codec_h264.mp4"
        create_synthetic_video(v1, ffmpeg_bin, duration_sec=5)

        # Part 2: MPEG-4
        v2 = tmp_path / "codec_mpeg4.mp4"
        cmd = [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=duration=5",
            "-c:v", "mpeg4", "-c:a", "aac",
            str(v2),
        ]
        subprocess.run(cmd, check=True, creationflags=0x08000000)

        with pytest.raises(ValueError, match="اختلاف ترميز الفيديو"):
            verify_parts_compatibility([v1, v2], ffprobe_bin=ffprobe_bin)

    def test_user_attempts_merge_parts_with_different_audio_codecs(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: AAC audio
        v1 = tmp_path / "audio_aac.mp4"
        create_synthetic_video(v1, ffmpeg_bin, duration_sec=5)

        # Part 2: MP3 audio
        v2 = tmp_path / "audio_mp3.mp4"
        cmd = [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=duration=5",
            "-c:v", "libx264", "-c:a", "libmp3lame",
            str(v2),
        ]
        subprocess.run(cmd, check=True, creationflags=0x08000000)

        with pytest.raises(ValueError, match="اختلاف ترميز الصوت"):
            verify_parts_compatibility([v1, v2], ffprobe_bin=ffprobe_bin)

    def test_user_attempts_merge_parts_where_one_has_no_audio(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: Video + Audio
        v1 = tmp_path / "with_audio.mp4"
        create_synthetic_video(v1, ffmpeg_bin, duration_sec=5)

        # Part 2: Video only (no audio)
        v2 = tmp_path / "no_audio.mp4"
        cmd = [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-c:v", "libx264", "-an",
            str(v2),
        ]
        subprocess.run(cmd, check=True, creationflags=0x08000000)

        with pytest.raises(ValueError, match="عدم تطابق في مسارات الصوت"):
            verify_parts_compatibility([v1, v2], ffprobe_bin=ffprobe_bin)


# ============================================================================
# 4. Cancellation & Safe Cleanup Scenarios
# ============================================================================
class TestCancellationAndCleanupScenarios:
    """Tests clean termination of FFmpeg without leaving orphan processes or partial files."""

    def test_user_cancels_split_operation_midway_cleans_folder(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        video = tmp_path / "long_movie.mp4"
        create_synthetic_video(video, ffmpeg_bin, duration_sec=30)

        splitter = VideoSplitter()

        # Trigger cancellation after getting first progress
        def on_prog(pct: float, sec: float) -> None:
            if pct > 0:
                splitter.cancel()

        with pytest.raises(InterruptedError, match="تم إلغاء العملية"):
            splitter.split(
                input_path=video,
                segment_duration_sec=5.0,
                progress_callback=on_prog,
                ffmpeg_exe=ffmpeg_bin,
                ffprobe_exe=ffprobe_bin,
            )

        # The parts directory should be completely cleaned up
        parts_dir = tmp_path / "long_movie_parts"
        assert not parts_dir.exists()

    def test_user_cancels_merge_midway_deletes_partial_file_only(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        video = tmp_path / "source.mp4"
        create_synthetic_video(video, ffmpeg_bin, duration_sec=30)

        # First split into 3 parts
        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert len(split_res.parts) == 3

        # Now begin merge and cancel midway
        merger = VideoMerger()

        def on_prog(pct: float, sec: float) -> None:
            if pct > 0:
                merger.cancel()

        with pytest.raises(InterruptedError, match="تم إلغاء العملية"):
            merger.merge(
                input_target=split_res.parts_dir,
                progress_callback=on_prog,
                ffmpeg_exe=ffmpeg_bin,
                ffprobe_exe=ffprobe_bin,
            )

        # Merged partial file must NOT exist
        partial_file = tmp_path / "source_merged.partial.mp4"
        assert not partial_file.exists()

        # Crucial: All 3 parts must remain intact and not deleted!
        for p in split_res.parts:
            assert p.file_path.is_file()


# ============================================================================
# 5. Environmental & System Failures (Disk Space, SHA-256)
# ============================================================================
class TestEnvironmentalFailures:
    """Tests disk space limits, hash tampering, and sparse keyframe warnings."""

    def test_disk_space_insufficient_check(self, tmp_path: Path) -> None:
        # Require 1 Exabyte of free space
        has_space, free_bytes, req_bytes = check_free_space(tmp_path, 10**18)
        assert has_space is False
        assert free_bytes < req_bytes

    def test_tampered_downloaded_zip_rejected_by_sha256(self, tmp_path: Path) -> None:
        tampered_zip = tmp_path / "tampered_ffmpeg.zip"
        dest_bin = tmp_path / "bin"
        files = {"ffmpeg.exe": b"fake modified binary"}
        create_mock_zip(tampered_zip, files)

        official_sha = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            extract_binaries_from_zip(
                zip_path=tampered_zip,
                target_bin_dir=dest_bin,
                expected_sha256=official_sha,
            )

    def test_sparse_keyframes_produces_clear_deviation_warning(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg
        # Video with keyframe every 10 seconds (gop_size=250 at 25 fps)
        video = tmp_path / "sparse_gop.mp4"
        create_synthetic_video(video, ffmpeg_bin, duration_sec=30, gop_size=250, fps=25)

        splitter = VideoSplitter()
        # Request split every 3 seconds (cannot split at 3s due to keyframe at 10s)
        res = splitter.split(
            input_path=video,
            segment_duration_sec=3.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert res.has_deviation_warning is True
        assert "تباعد إطارات الـ Keyframes" in res.warning_message
        assert res.parts_dir.is_dir()
