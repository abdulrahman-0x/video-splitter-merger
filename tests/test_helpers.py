"""Unit tests for src/utils/helpers.py."""

from pathlib import Path
import pytest

from src.utils.helpers import (
    calculate_padding_width,
    check_supported_format,
    escape_concat_path,
    format_duration,
    get_unique_path,
    is_path_length_safe,
    natural_sort_key,
    normalize_numeric_string,
    parse_duration_minutes,
    parse_time_str,
)


class TestHelpers:
    """Test helper functions for time, paths, numbers, and escaping."""

    def test_normalize_numeric_string(self) -> None:
        assert normalize_numeric_string("١٢٣") == "123"
        assert normalize_numeric_string("۱۲۳") == "123"
        assert normalize_numeric_string("  ٤٥٫٥  ") == "45.5"
        assert normalize_numeric_string("12,5") == "12.5"
        assert normalize_numeric_string("٠١٢٣٤٥٦٧٨٩") == "0123456789"

    def test_parse_duration_minutes_valid(self) -> None:
        assert parse_duration_minutes("30") == 30.0
        assert parse_duration_minutes(" 1.5 ") == 1.5
        assert parse_duration_minutes("١٫٥") == 1.5
        assert parse_duration_minutes("٢,٥") == 2.5
        assert parse_duration_minutes("٣٠") == 30.0
        # Exactly 10 seconds: 10 / 60 = 0.166666...
        assert parse_duration_minutes("0.16667") == pytest.approx(0.16667, rel=1e-3)

    def test_parse_duration_minutes_invalid_inputs(self) -> None:
        # Empty or whitespace
        with pytest.raises(ValueError, match="يرجى إدخال مدة التقسيم"):
            parse_duration_minutes("")
        with pytest.raises(ValueError, match="يرجى إدخال مدة التقسيم"):
            parse_duration_minutes("   ")

        # Non-numeric
        with pytest.raises(ValueError, match="غير صالحة"):
            parse_duration_minutes("abc")
        with pytest.raises(ValueError, match="غير صالحة"):
            parse_duration_minutes("12.34.56")

        # Zero or negative
        with pytest.raises(ValueError, match="أكبر من الصفر"):
            parse_duration_minutes("0")
        with pytest.raises(ValueError, match="أكبر من الصفر"):
            parse_duration_minutes("-5")

        # Less than 10 seconds (e.g. 0.1 minutes = 6 seconds)
        with pytest.raises(ValueError, match="الحد الأدنى"):
            parse_duration_minutes("0.1")

    def test_format_duration(self) -> None:
        assert format_duration(0) == "00:00:00"
        assert format_duration(65) == "00:01:05"
        assert format_duration(3661) == "01:01:01"
        assert format_duration(7200.4) == "02:00:00"
        assert format_duration(-10) == "00:00:00"

    def test_parse_time_str(self) -> None:
        assert parse_time_str("01:01:01") == 3661.0
        assert parse_time_str("02:30") == 150.0
        assert parse_time_str("45.5") == 45.5
        with pytest.raises(ValueError):
            parse_time_str("invalid:time:format:extra")

    def test_natural_sort_key(self) -> None:
        file_list = [
            "video_part_10.mp4",
            "video_part_2.mp4",
            "video_part_1.mp4",
            "video_part_20.mp4",
        ]
        sorted_list = sorted(file_list, key=natural_sort_key)
        assert sorted_list == [
            "video_part_1.mp4",
            "video_part_2.mp4",
            "video_part_10.mp4",
            "video_part_20.mp4",
        ]

    def test_calculate_padding_width(self) -> None:
        assert calculate_padding_width(1) == 3
        assert calculate_padding_width(5) == 3
        assert calculate_padding_width(999) == 3
        assert calculate_padding_width(1000) == 4
        assert calculate_padding_width(12345) == 5

    def test_get_unique_path_files(self, tmp_path: Path) -> None:
        file_a = tmp_path / "video_merged.mp4"
        assert get_unique_path(file_a) == file_a

        # Create file_a
        file_a.touch()
        unique_1 = get_unique_path(file_a)
        assert unique_1 == tmp_path / "video_merged_2.mp4"

        # Create unique_1
        unique_1.touch()
        unique_2 = get_unique_path(file_a)
        assert unique_2 == tmp_path / "video_merged_3.mp4"

    def test_get_unique_path_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "my_video_parts"
        assert get_unique_path(dir_a, is_directory=True) == dir_a

        dir_a.mkdir()
        unique_dir = get_unique_path(dir_a, is_directory=True)
        assert unique_dir == tmp_path / "my_video_parts_2"

        unique_dir.mkdir()
        unique_dir2 = get_unique_path(dir_a, is_directory=True)
        assert unique_dir2 == tmp_path / "my_video_parts_3"

    def test_escape_concat_path(self) -> None:
        simple = "video.mp4"
        assert escape_concat_path(simple) == "'video.mp4'"

        windows_path = r"C:\My Videos\part 1.mp4"
        assert escape_concat_path(windows_path) == "'C:/My Videos/part 1.mp4'"

        with_single_quote = r"C:\Videos\it's 'great'\part_1.mp4"
        expected = "'C:/Videos/it'\\''s '\\''great'\\''/part_1.mp4'"
        assert escape_concat_path(with_single_quote) == expected

    def test_check_supported_format(self) -> None:
        assert check_supported_format("video.mp4") == (True, "supported")
        assert check_supported_format("video.MKV") == (True, "supported")
        assert check_supported_format("video.mov") == (True, "supported")
        assert check_supported_format("video.avi") == (True, "best_effort")
        assert check_supported_format("video.wmv") == (False, "unsupported")
        assert check_supported_format("video.flv") == (False, "unsupported")
        assert check_supported_format("video.txt") == (False, "unsupported")

    def test_is_path_length_safe(self, tmp_path: Path) -> None:
        short_path = tmp_path / "short_path.mp4"
        assert is_path_length_safe(short_path, max_length=250) is True

        long_name = "a" * 300 + ".mp4"
        long_path = tmp_path / long_name
        assert is_path_length_safe(long_path, max_length=250) is False
