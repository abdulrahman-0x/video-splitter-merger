"""Integration and unit tests for video splitting and merging engine."""

from pathlib import Path
import pytest

from src.core.merger import (
    VideoMerger,
    derive_merged_output_path,
    discover_parts_in_dir,
    parse_concat_list,
    verify_parts_compatibility,
)
from src.core.splitter import VideoSplitter, build_stream_map
from tests.conftest import create_synthetic_video


class TestSplitterMergerUnit:
    """Unit tests not requiring FFmpeg binaries."""

    def test_build_stream_map_excludes_data(self) -> None:
        mock_streams = [
            {"index": 0, "codec_type": "video", "codec_name": "h264"},
            {"index": 1, "codec_type": "audio", "codec_name": "aac"},
            {"index": 2, "codec_type": "data", "codec_name": "bin_data"},
            {"index": 3, "codec_type": "data", "codec_name": "tmcd"},
            {"index": 4, "codec_type": "subtitle", "codec_name": "subrip"},
        ]
        mp4_maps = build_stream_map(mock_streams, ".mp4")
        assert mp4_maps == ["-map", "0:0", "-map", "0:1", "-map", "0:4"]
        assert "-map 0:2" not in " ".join(mp4_maps)
        assert "-map 0:3" not in " ".join(mp4_maps)

    def test_build_stream_map_mkv_attachments(self) -> None:
        mock_streams = [
            {"index": 0, "codec_type": "video"},
            {"index": 1, "codec_type": "audio"},
            {"index": 2, "codec_type": "attachment"},
        ]
        mkv_maps = build_stream_map(mock_streams, ".mkv")
        assert mkv_maps == ["-map", "0:0", "-map", "0:1", "-map", "0:2"]

        # In MP4, attachments should not be included
        mp4_maps = build_stream_map(mock_streams, ".mp4")
        assert mp4_maps == ["-map", "0:0", "-map", "0:1"]

    def test_derive_merged_output_path(self, tmp_path: Path) -> None:
        parts_dir = tmp_path / "lecture_2026_parts"
        merged_path = derive_merged_output_path(parts_dir, ".mp4")
        assert merged_path == tmp_path / "lecture_2026_merged.mp4"

        # When _parts_2 exists
        parts_dir_2 = tmp_path / "lecture_2026_parts_2"
        merged_path_2 = derive_merged_output_path(parts_dir_2, ".mp4")
        assert merged_path_2 == tmp_path / "lecture_2026_merged.mp4"

    def test_parse_concat_list_invalid_lines(self, tmp_path: Path) -> None:
        bad_list = tmp_path / "bad_list.txt"
        bad_list.write_text("invalid format line here\n", encoding="utf-8")
        with pytest.raises(ValueError, match="سطر غير صالح"):
            parse_concat_list(bad_list)

    def test_parse_concat_list_fewer_than_two_files(self, tmp_path: Path) -> None:
        single_file = tmp_path / "part_1.mp4"
        single_file.touch()
        list_file = tmp_path / "parts_list.txt"
        list_file.write_text(f"file '{single_file.name}'\n", encoding="utf-8")

        with pytest.raises(ValueError, match="جزأين على الأقل"):
            parse_concat_list(list_file)


class TestSplitterMergerIntegration:
    """Integration tests executing FFmpeg on synthetic videos."""

    def test_split_and_merge_roundtrip(self, require_ffmpeg: tuple[Path, Path], tmp_path: Path) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # 1. Create a 30s synthetic video with keyframe every 1s
        orig_video = tmp_path / "sample_test.mp4"
        create_synthetic_video(
            orig_video,
            ffmpeg_path=ffmpeg_bin,
            duration_sec=30,
            gop_size=25,
            fps=25,
        )

        # 2. Split into 10-second parts (3 parts expected)
        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=orig_video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert split_res.parts_dir.is_dir()
        assert len(split_res.parts) == 3
        assert split_res.parts_list_file.is_file()

        # Check part files exist
        for p in split_res.parts:
            assert p.file_path.is_file()
            assert p.file_path.stat().st_size > 0
            # Durations should be roughly 10s
            assert 9.0 <= p.duration <= 11.5

        # 3. Merge parts back
        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert merge_res.output_path.is_file()
        assert merge_res.parts_count == 3
        # Merged duration vs sum of parts duration <= 0.5s
        assert merge_res.duration_difference <= 0.5
        # Merged duration vs original <= 1.0s
        assert abs(merge_res.total_duration - 30.0) <= 1.0

    def test_split_sparse_keyframes_deviation_warning(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Video with keyframe every 10s (gop_size=250 at 25fps)
        sparse_video = tmp_path / "sparse_keyframes.mp4"
        create_synthetic_video(
            sparse_video,
            ffmpeg_path=ffmpeg_bin,
            duration_sec=30,
            gop_size=250,
            fps=25,
        )

        # Ask to split every 4 seconds. Since keyframes are at 0s, 10s, 20s,
        # FFmpeg stream copy can only cut at keyframe (10s), deviating significantly from 4s.
        splitter = VideoSplitter()
        res = splitter.split(
            input_path=sparse_video,
            segment_duration_sec=4.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert res.has_deviation_warning is True
        assert "تباعد إطارات الـ Keyframes" in res.warning_message

    def test_split_unicode_spaces_and_quotes(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        arabic_video = tmp_path / "فيديو تجريبي 'خاص' 1.mp4"
        create_synthetic_video(
            arabic_video,
            ffmpeg_path=ffmpeg_bin,
            duration_sec=20,
            gop_size=25,
            fps=25,
        )

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=arabic_video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert split_res.parts_dir.is_dir()
        assert len(split_res.parts) == 2

        # Concat list must correctly parse even with quotes and spaces
        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_list_file,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert merge_res.output_path.is_file()

    def test_split_unique_directory_no_overwrite(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        video = tmp_path / "test_unique.mp4"
        create_synthetic_video(video, ffmpeg_path=ffmpeg_bin, duration_sec=20)

        # Pre-create the _parts folder
        existing_dir = tmp_path / "test_unique_parts"
        existing_dir.mkdir()
        dummy_file = existing_dir / "keep_me.txt"
        dummy_file.write_text("important data", encoding="utf-8")

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Must create _parts_2 instead of overwriting _parts
        assert split_res.parts_dir.name == "test_unique_parts_2"
        # Pre-existing folder and file must remain untouched
        assert dummy_file.is_file()
        assert dummy_file.read_text(encoding="utf-8") == "important data"

    def test_merge_mkv_multiple_streams(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # MKV with 2 audio tracks and 1 subtitle track
        mkv_video = tmp_path / "multitrack.mkv"
        create_synthetic_video(
            mkv_video,
            ffmpeg_path=ffmpeg_bin,
            duration_sec=20,
            extra_audio=True,
            include_subtitle=True,
        )

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=mkv_video,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Verify stream counts
        from src.core.ffmpeg_manager import probe_media
        merged_info = probe_media(merge_res.output_path, ffprobe_path=ffprobe_bin)
        assert len(merged_info["video_streams"]) == 1
        assert len(merged_info["audio_streams"]) == 2
        assert len(merged_info["subtitle_streams"]) == 1

    def test_merge_rejects_incompatible_dimensions(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Video 1: 320x240
        v1 = tmp_path / "video_1.mp4"
        create_synthetic_video(v1, ffmpeg_path=ffmpeg_bin, duration_sec=10)

        # Video 2: 640x480
        v2 = tmp_path / "video_2.mp4"
        cmd = [
            str(ffmpeg_bin),
            "-y",
            "-f", "lavfi", "-i", "testsrc=duration=10:size=640x480:rate=25",
            "-f", "lavfi", "-i", "sine=duration=10",
            "-c:v", "libx264", "-c:a", "aac",
            str(v2),
        ]
        import subprocess
        subprocess.run(cmd, check=True, creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0)

        # Direct compatibility check
        with pytest.raises(ValueError, match="اختلاف أبعاد الفيديو"):
            verify_parts_compatibility([v1, v2], ffprobe_bin=ffprobe_bin)
