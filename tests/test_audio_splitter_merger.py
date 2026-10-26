"""Tests for audio splitting and merging functionality (WAV, MP3, FLAC, M4A, etc.).

Adheres to Test Guard principles:
- Real synthetic audio files generated via FFmpeg lavfi sine.
- No MagicMocks.
- Testing behavior: stream copy accuracy, header preservation, compatibility checks.
"""

from pathlib import Path
import pytest

from src.core.ffmpeg_manager import probe_media
from src.core.merger import VideoMerger, verify_parts_compatibility
from src.core.splitter import VideoSplitter
from src.utils.helpers import (
    check_supported_format,
    get_media_category,
    is_audio_file,
    is_video_file,
)
from tests.conftest import create_synthetic_audio


class TestAudioHelpers:
    """Test format identification for audio formats."""

    @pytest.mark.parametrize(
        ("filename", "expected_supported", "expected_status", "expected_category"),
        [
            ("audio.wav", True, "supported", "audio"),
            ("music.mp3", True, "supported", "audio"),
            ("track.m4a", True, "supported", "audio"),
            ("album.flac", True, "supported", "audio"),
            ("voice.aac", True, "best_effort", "audio"),
            ("podcast.ogg", True, "best_effort", "audio"),
            ("recording.opus", True, "best_effort", "audio"),
            ("legacy.wma", False, "unsupported", "unknown"),
        ],
    )
    def test_audio_format_recognition(
        self,
        filename: str,
        expected_supported: bool,
        expected_status: str,
        expected_category: str,
    ) -> None:
        supported, status = check_supported_format(filename)
        assert supported == expected_supported
        assert status == expected_status
        assert get_media_category(filename) == expected_category
        assert is_audio_file(filename) == (expected_category == "audio")
        assert is_video_file(filename) is False


class TestAudioSplitterMergerIntegration:
    """Integration tests executing FFmpeg on synthetic audio files."""

    def test_split_and_merge_wav_pcm_roundtrip(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # 1. Create a 30-second WAV PCM audio file
        wav_file = tmp_path / "recording.wav"
        create_synthetic_audio(wav_file, ffmpeg_bin, duration_sec=30, sample_rate=44100, channels=2)

        # 2. Split into 10-second parts
        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=wav_file,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert split_res.is_audio_only is True
        assert len(split_res.parts) == 3
        assert split_res.parts_list_file.is_file()

        # In uncompressed PCM/WAV, cutting precision is sub-100ms (<= 0.1s)
        for p in split_res.parts:
            assert p.file_path.is_file()
            assert abs(p.duration - 10.0) <= 0.15

        # 3. Merge parts back
        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert merge_res.output_path.is_file()
        assert merge_res.output_path.suffix.lower() == ".wav"
        assert merge_res.parts_count == 3
        # Merged duration vs original <= 0.15s
        assert abs(merge_res.total_duration - 30.0) <= 0.15

        # Verify probe
        merged_probe = probe_media(merge_res.output_path, ffprobe_path=ffprobe_bin)
        assert merged_probe["is_audio_only"] is True
        assert len(merged_probe["audio_streams"]) == 1
        assert merged_probe["audio_streams"][0]["channels"] == 2

    def test_split_and_merge_mp3_roundtrip(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # 30-second MP3 file
        mp3_file = tmp_path / "song.mp3"
        create_synthetic_audio(mp3_file, ffmpeg_bin, duration_sec=30, sample_rate=44100, channels=2)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=mp3_file,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert split_res.is_audio_only is True
        assert len(split_res.parts) == 3

        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        assert merge_res.output_path.is_file()
        assert merge_res.output_path.suffix.lower() == ".mp3"
        assert abs(merge_res.total_duration - 30.0) <= 0.5

    def test_split_mp3_with_album_art_ignores_cover_art_as_video(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # MP3 file with embedded cover picture (attached_pic)
        art_mp3 = tmp_path / "album_track.mp3"
        create_synthetic_audio(art_mp3, ffmpeg_bin, duration_sec=20, with_cover_art=True)

        probe_before = probe_media(art_mp3, ffprobe_path=ffprobe_bin)
        # Verify it has an album art stream
        assert len(probe_before["album_art_streams"]) == 1
        assert probe_before["is_audio_only"] is True

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=art_mp3,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )

        # Crucial check: It must produce 2 parts and NOT get stuck on 1 part due to cover art
        assert len(split_res.parts) == 2
        for p in split_res.parts:
            assert p.file_path.is_file()
            assert p.file_path.stat().st_size > 0

    def test_split_and_merge_flac_lossless(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        flac_file = tmp_path / "lossless.flac"
        create_synthetic_audio(flac_file, ffmpeg_bin, duration_sec=20)

        splitter = VideoSplitter()
        split_res = splitter.split(
            input_path=flac_file,
            segment_duration_sec=10.0,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert len(split_res.parts) == 2

        merger = VideoMerger()
        merge_res = merger.merge(
            input_target=split_res.parts_dir,
            ffmpeg_exe=ffmpeg_bin,
            ffprobe_exe=ffprobe_bin,
        )
        assert merge_res.output_path.is_file()
        assert merge_res.output_path.suffix.lower() == ".flac"

    def test_merge_rejects_audio_sample_rate_mismatch(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: 44.1 kHz
        a1 = tmp_path / "part_44k.wav"
        create_synthetic_audio(a1, ffmpeg_bin, duration_sec=5, sample_rate=44100)

        # Part 2: 48 kHz
        a2 = tmp_path / "part_48k.wav"
        create_synthetic_audio(a2, ffmpeg_bin, duration_sec=5, sample_rate=48000)

        with pytest.raises(ValueError, match="اختلاف معدل العينات الصوتي"):
            verify_parts_compatibility([a1, a2], ffprobe_bin=ffprobe_bin)

    def test_merge_rejects_audio_codec_mismatch(
        self, require_ffmpeg: tuple[Path, Path], tmp_path: Path
    ) -> None:
        ffmpeg_bin, ffprobe_bin = require_ffmpeg

        # Part 1: WAV PCM
        a1 = tmp_path / "part_pcm.wav"
        create_synthetic_audio(a1, ffmpeg_bin, duration_sec=5)

        # Part 2: MP3
        a2 = tmp_path / "part_mp3.mp3"
        create_synthetic_audio(a2, ffmpeg_bin, duration_sec=5)

        with pytest.raises(ValueError, match="اختلاف ترميز الصوت"):
            verify_parts_compatibility([a1, a2], ffprobe_bin=ffprobe_bin)
