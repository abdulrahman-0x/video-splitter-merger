"""Unit tests for src/core/ffmpeg_manager.py."""

import hashlib
import zipfile
from pathlib import Path
import pytest

from src.core.ffmpeg_manager import (
    check_free_space,
    compute_file_sha256,
    extract_binaries_from_zip,
    find_tool,
    probe_media,
)


def create_mock_zip(zip_path: Path, files: dict[str, bytes]) -> str:
    """Create a mock zip file with the given files and return its SHA-256."""
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)

    hasher = hashlib.sha256()
    with open(zip_path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest().lower()


class TestFFmpegManager:
    """Tests for discovery, zip extraction security, hashing, and probing."""

    def test_find_tool_in_custom_bin(self, tmp_path: Path) -> None:
        custom_bin = tmp_path / "custom bin dir with spaces"
        custom_bin.mkdir()
        fake_ffmpeg = custom_bin / "ffmpeg.exe"
        fake_ffmpeg.write_bytes(b"mock ffmpeg binary")

        found = find_tool("ffmpeg", custom_bin_dir=custom_bin)
        assert found == fake_ffmpeg
        assert find_tool("nonexistent_tool", custom_bin_dir=custom_bin) is None

    def test_extract_binaries_success(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "test_ffmpeg.zip"
        dest_bin = tmp_path / "bin"

        files = {
            "ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe": b"fake ffmpeg executable",
            "ffmpeg-9.0.2-essentials_build/bin/ffprobe.exe": b"fake ffprobe executable",
            "ffmpeg-9.0.2-essentials_build/doc/readme.txt": b"some doc",
            "malicious_file.txt": b"should not be extracted",
        }
        valid_sha = create_mock_zip(zip_path, files)

        ffmpeg_out, ffprobe_out = extract_binaries_from_zip(
            zip_path=zip_path,
            target_bin_dir=dest_bin,
            expected_sha256=valid_sha,
        )

        assert ffmpeg_out.is_file()
        assert ffprobe_out.is_file()
        assert ffmpeg_out.read_bytes() == b"fake ffmpeg executable"
        assert ffprobe_out.read_bytes() == b"fake ffprobe executable"
        # Verify that only the two binaries are extracted, nothing else
        extracted_files = [p.name for p in dest_bin.iterdir()]
        assert set(extracted_files) == {"ffmpeg.exe", "ffprobe.exe"}

    def test_extract_binaries_sha256_mismatch(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "corrupt_ffmpeg.zip"
        dest_bin = tmp_path / "bin"

        files = {
            "bin/ffmpeg.exe": b"fake ffmpeg",
            "bin/ffprobe.exe": b"fake ffprobe",
        }
        create_mock_zip(zip_path, files)
        wrong_sha = "0000000000000000000000000000000000000000000000000000000000000000"

        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            extract_binaries_from_zip(
                zip_path=zip_path,
                target_bin_dir=dest_bin,
                expected_sha256=wrong_sha,
            )

    def test_extract_binaries_missing_executable(self, tmp_path: Path) -> None:
        zip_path = tmp_path / "missing_files.zip"
        dest_bin = tmp_path / "bin"

        files = {
            "bin/ffmpeg.exe": b"only ffmpeg, no ffprobe",
        }
        sha = create_mock_zip(zip_path, files)

        with pytest.raises(RuntimeError, match="لا يحتوي على ffmpeg.exe و ffprobe.exe"):
            extract_binaries_from_zip(
                zip_path=zip_path,
                target_bin_dir=dest_bin,
                expected_sha256=sha,
            )

    def test_check_free_space(self, tmp_path: Path) -> None:
        has_space, free_bytes, req = check_free_space(tmp_path, 1024)
        assert has_space is True
        assert free_bytes > 1024
        assert req == 1024

        # Unreasonably huge requirement
        has_space, _, _ = check_free_space(tmp_path, 10**18)
        assert has_space is False

    def test_probe_media_file_not_found(self, tmp_path: Path) -> None:
        non_existent = tmp_path / "does_not_exist.mp4"
        with pytest.raises(FileNotFoundError, match="غير موجود"):
            probe_media(non_existent)
