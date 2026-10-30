"""Comprehensive user failure, edge-case, and environmental scenario tests for audio support.

Adheres strictly to Test Guard rules:
- Real synthetic audio files generated via FFmpeg lavfi sine (no MagicMocks).
- Tests behavior and user-observable outcomes (raised exceptions, file cleanup, output duration).
- One scenario per test, data-driven for variants (parametrized).
- Tests named as test_<scenario>_<expected_outcome>.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import pytest

from src.core.ffmpeg_manager import (
    find_tool,
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
    get_media_category,
    is_audio_file,
)
from tests.conftest import create_synthetic_audio, create_synthetic_video


class TestAudioUserFailureScenarios:
    """User failure and edge-case scenarios specific to audio splitting and merging."""

    def test_merge_rejects_audio_with_video_mismatch(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Regression test: Merging an audio-only file with a video file must be rejected immediately."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: Audio only (M4A with AAC)
        audio_part = tmp_path / "part_01.m4a"
        create_synthetic_audio(audio_part, ffmpeg_bin, duration_sec=5, sample_rate=44100, channels=2)

        # Part 2: Video file (MP4 with H.264 video and matching AAC audio)
        video_part = tmp_path / "part_02.mp4"
        cmd = [
            str(ffmpeg_bin), "-y",
            "-f", "lavfi", "-i", "testsrc=duration=5:size=320x240:rate=25",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=5:sample_rate=44100",
            "-c:v", "libx264", "-c:a", "aac",
            str(video_part),
        ]
        subprocess.run(cmd, check=True, creationflags=0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)

        with pytest.raises(ValueError, match="عدم تطابق في نوع الوسائط"):
            verify_parts_compatibility([audio_part, video_part], ffprobe_bin=ffprobe_bin)

    def test_merge_rejects_channel_count_mismatch(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """User attempts to merge stereo (2 channels) with mono (1 channel)."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        part_stereo = tmp_path / "stereo.wav"
        create_synthetic_audio(part_stereo, ffmpeg_bin, duration_sec=5, channels=2)

        part_mono = tmp_path / "mono.wav"
        create_synthetic_audio(part_mono, ffmpeg_bin, duration_sec=5, channels=1)

        with pytest.raises(ValueError, match="اختلاف قنوات الصوت"):
            verify_parts_compatibility([part_stereo, part_mono], ffprobe_bin=ffprobe_bin)

    def test_split_rejects_audio_shorter_than_requested_segment(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """User specifies segment duration (30s) greater than audio file duration (12s)."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        short_audio = tmp_path / "short_podcast.mp3"
        create_synthetic_audio(short_audio, ffmpeg_bin, duration_sec=12)

        splitter = VideoSplitter()
        with pytest.raises(ValueError, match="الملف أقصر من مدة التقسيم المحددة"):
            splitter.split(
                input_path=short_audio,
                segment_duration_sec=30.0,
                ffmpeg_exe=ffmpeg_bin,
                ffprobe_exe=ffprobe_bin,
            )

    @pytest.mark.parametrize(
        "corrupt_content",
        [
            b"",  # 0-byte file
            b"ID3\x03\x00\x00\x00NotARealMp3StreamCorruptedDataAtAll",
            b"RIFF\x24\x00\x00\x00WAVEfmt CorruptedWavHeaderWithoutPCM",
        ],
    )
    def test_probe_corrupted_or_empty_audio_file_raises_clear_error(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path, corrupt_content: bytes
    ) -> None:
        """User selects an empty or corrupted audio file."""
        _, ffprobe_bin = require_ffmpeg

        corrupt_file = tmp_path / "broken_audio.mp3"
        corrupt_file.write_bytes(corrupt_content)

        with pytest.raises(RuntimeError, match="تعذر قراءة ملف (الفيديو|الوسائط|الصوت)"):
            probe_media(corrupt_file, ffprobe_path=ffprobe_bin)

    def test_split_and_merge_audio_with_complex_arabic_and_special_characters_in_path(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Audio file path containing Arabic text, quotes, brackets, and ampersand."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        arabic_dir = tmp_path / "تسجيلات بودكاست (الحلقة 1) 'خاص جداً' & تجربة"
        arabic_dir.mkdir(parents=True)
        audio_file = arabic_dir / "حوار إذاعي [مباشر] #1.mp3"
        create_synthetic_audio(audio_file, ffmpeg_bin, duration_sec=20)

        # 1. Split
        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=audio_file,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert split_res.parts_dir.is_dir()
        assert len(split_res.parts) == 2
        for part in split_res.parts:
            assert part.file_path.is_file()
            assert part.file_path.stat().st_size > 0

        # 2. Merge back
        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert merge_res.output_path.is_file()
        assert abs(merge_res.total_duration - 20.0) <= 0.5

    def test_split_and_merge_audio_with_uneven_duration_avoids_micro_stubs(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Splitting 25-second audio into 10s parts must yield parts (10s, 10s, 5s) without micro-stubs."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        audio_file = tmp_path / "uneven_song.wav"
        create_synthetic_audio(audio_file, ffmpeg_bin, duration_sec=25)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=audio_file,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert len(split_res.parts) == 3
        # First two parts ~10s, last part ~5s
        assert abs(split_res.parts[0].duration - 10.0) <= 0.2
        assert abs(split_res.parts[1].duration - 10.0) <= 0.2
        assert abs(split_res.parts[2].duration - 5.0) <= 0.2

        # Verify roundtrip merge
        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert merge_res.parts_count == 3
        assert abs(merge_res.total_duration - 25.0) <= 0.2

    def test_merge_audio_directory_with_missing_part_in_concat_list_raises_filenotfound(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """parts_list.txt lists part_2.wav which the user deleted manually."""
        ffmpeg_bin, _ = require_ffmpeg

        folder = tmp_path / "audio_parts"
        folder.mkdir()
        p1 = folder / "rec_part_01.wav"
        p3 = folder / "rec_part_03.wav"
        create_synthetic_audio(p1, ffmpeg_bin, duration_sec=5)
        create_synthetic_audio(p3, ffmpeg_bin, duration_sec=5)

        concat_list = folder / "parts_list.txt"
        concat_list.write_text(
            f"file '{p1.name}'\nfile 'rec_part_02.wav'\nfile '{p3.name}'\n",
            encoding="utf-8",
        )

        with pytest.raises(FileNotFoundError, match="غير موجود"):
            parse_concat_list(concat_list)

    def test_discover_audio_parts_ignores_unrelated_files_and_preserves_natural_order(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Directory with audio parts and non-media clutter sorts naturally."""
        ffmpeg_bin, _ = require_ffmpeg

        folder = tmp_path / "mixed_folder"
        folder.mkdir()

        # Create audio parts with numbers 1, 2, 10
        p1 = folder / "audio_part_1.mp3"
        p2 = folder / "audio_part_2.mp3"
        p10 = folder / "audio_part_10.mp3"
        for p in (p1, p2, p10):
            create_synthetic_audio(p, ffmpeg_bin, duration_sec=3)

        # Extraneous files
        (folder / "cover.jpg").write_bytes(b"\xff\xd8\xff")
        (folder / "lyrics.lrc").write_text("[00:01] Hello", encoding="utf-8")
        (folder / "notes.txt").write_text("audio notes", encoding="utf-8")

        discovered = discover_parts_in_dir(folder)
        assert len(discovered) == 3
        # Natural order must put part_2 before part_10
        assert [f.name for f in discovered] == ["audio_part_1.mp3", "audio_part_2.mp3", "audio_part_10.mp3"]

    def test_cancel_audio_split_midway_cleans_up_created_directory(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Cancelling split on an audio file midway must clean up created folder and parts."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        audio = tmp_path / "long_audio.wav"
        create_synthetic_audio(audio, ffmpeg_bin, duration_sec=60)

        splitter = VideoSplitter()

        def on_prog(pct: float, sec: float) -> None:
            if pct > 0:
                splitter.cancel()

        with pytest.raises(InterruptedError, match="تم إلغاء العملية"):
            splitter.split(
                input_path=audio,
                segment_duration_sec=5.0,
                progress_callback=on_prog,
                ffmpeg_exe=ffmpeg_bin,
                ffprobe_exe=ffprobe_bin,
            )

        # Expected destination folder must be cleaned up
        parts_dir = tmp_path / "long_audio_parts"
        assert not parts_dir.exists()

    def test_cancel_audio_merge_midway_deletes_partial_file_only(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Cancelling audio merge midway must remove temporary partial file without touching parts."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        audio = tmp_path / "source.wav"
        create_synthetic_audio(audio, ffmpeg_bin, duration_sec=30)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=audio,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

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

        # Parts must remain intact
        for part in split_res.parts:
            assert part.file_path.is_file()

        # Partial output file must not exist
        partial_file = split_res.parts_dir.parent / "source_merged.part.wav"
        assert not partial_file.exists()

    @pytest.mark.parametrize(
        ("ext", "expected_category"),
        [
            (".aac", "audio"),
            (".ogg", "audio"),
            (".opus", "audio"),
        ],
    )
    def test_best_effort_audio_formats_recognition(
        self, ext: str, expected_category: str
    ) -> None:
        """Best effort audio formats are correctly identified as best_effort audio."""
        test_file = f"sample{ext}"
        supported, status = check_supported_format(test_file)
        assert supported is True
        assert status == "best_effort"
        assert get_media_category(test_file) == "audio"
        assert is_audio_file(test_file) is True

    def test_gui_split_view_displays_audio_metadata_on_selection(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Selecting an audio file in SplitView updates the metadata label.

        Drives the real `_on_browse_file` path (the file dialog is the one
        boundary stubbed out) and asserts on the label the user actually sees.
        """
        ffmpeg_bin, _ = require_ffmpeg

        audio = tmp_path / "lecture.mp3"
        create_synthetic_audio(audio, ffmpeg_bin, duration_sec=30, sample_rate=44100, channels=2)

        app = App()
        app.update()
        try:
            monkeypatch.setattr(
                "src.gui.split_view.filedialog.askopenfilename", lambda **_: str(audio)
            )
            app.split_view._on_browse_file()
            app.update()

            assert app.split_view.selected_file == audio.resolve()
            info_text = app.split_view.file_info_label.cget("text")
            # The label must report the audio specifics, not just the file name.
            assert "MP3" in info_text.upper()
            assert "44100" in info_text or "44.1" in info_text
        finally:
            app.destroy()

    def test_gui_merge_view_handles_audio_parts_compatibility_display(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        """Selecting audio parts in MergeView shows compatible audio parts text."""
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        folder = tmp_path / "audio_meeting_parts"
        folder.mkdir()
        p1 = folder / "meeting_part_01.wav"
        p2 = folder / "meeting_part_02.wav"
        create_synthetic_audio(p1, ffmpeg_bin, duration_sec=5)
        create_synthetic_audio(p2, ffmpeg_bin, duration_sec=5)

        app = App()
        app.update()
        try:
            app.merge_view._handle_selected_target(folder)
            app.update()

            status_text = app.merge_view.target_info_label.cget("text")
            assert "أجزاء صوتية متوافقة" in status_text
            assert "PCM" in status_text.upper() or "WAV" in status_text.upper()
        finally:
            app.destroy()
