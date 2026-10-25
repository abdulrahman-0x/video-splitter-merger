"""Unit tests for src/utils/helpers.py adhering to Test Guard rules."""

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

    @pytest.mark.parametrize(
        ("raw_input", "expected"),
        [
            ("١٢٣", "123"),
            ("۱۲۳", "123"),
            ("  ٤٥٫٥  ", "45.5"),
            ("12,5", "12.5"),
            ("٠١٢٣٤٥٦٧٨٩", "0123456789"),
        ],
    )
    def test_normalize_numeric_string_converts_arabic_and_persian_digits(
        self, raw_input: str, expected: str
    ) -> None:
        assert normalize_numeric_string(raw_input) == expected

    @pytest.mark.parametrize(
        ("raw_text", "expected_minutes"),
        [
            ("30", 30.0),
            (" 1.5 ", 1.5),
            ("١٫٥", 1.5),
            ("٢,٥", 2.5),
            ("٣٠", 30.0),
        ],
    )
    def test_parse_duration_minutes_accepts_valid_formats(
        self, raw_text: str, expected_minutes: float
    ) -> None:
        assert parse_duration_minutes(raw_text) == expected_minutes

    def test_parse_duration_minutes_boundary_ten_seconds(self) -> None:
        # 10 seconds = 10/60 minutes = 0.16667
        assert parse_duration_minutes("0.16667") == pytest.approx(0.16667, rel=1e-3)

    @pytest.mark.parametrize(
        ("invalid_input", "error_substring"),
        [
            ("", "يرجى إدخال مدة التقسيم"),
            ("   ", "يرجى إدخال مدة التقسيم"),
            ("abc", "غير صالحة"),
            ("12.34.56", "غير صالحة"),
            ("0", "أكبر من الصفر"),
            ("-5", "أكبر من الصفر"),
            ("0.1", "الحد الأدنى"),
        ],
    )
    def test_parse_duration_minutes_rejects_invalid_inputs(
        self, invalid_input: str, error_substring: str
    ) -> None:
        with pytest.raises(ValueError, match=error_substring):
            parse_duration_minutes(invalid_input)

    @pytest.mark.parametrize(
        ("seconds", "expected_formatted"),
        [
            (0, "00:00:00"),
            (65, "00:01:05"),
            (3661, "01:01:01"),
            (7200.4, "02:00:00"),
            (-10, "00:00:00"),
        ],
    )
    def test_format_duration_formats_seconds_to_hh_mm_ss(
        self, seconds: float, expected_formatted: str
    ) -> None:
        assert format_duration(seconds) == expected_formatted

    @pytest.mark.parametrize(
        ("time_string", "expected_seconds"),
        [
            ("01:01:01", 3661.0),
            ("02:30", 150.0),
            ("45.5", 45.5),
        ],
    )
    def test_parse_time_str_parses_various_formats(
        self, time_string: str, expected_seconds: float
    ) -> None:
        assert parse_time_str(time_string) == expected_seconds

    def test_parse_time_str_invalid_format_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="صيغة الوقت غير صالحة"):
            parse_time_str("invalid:time:format:extra")

    def test_natural_sort_orders_part_numbers_logically(self) -> None:
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

    @pytest.mark.parametrize(
        ("total_parts", "expected_width"),
        [
            (1, 3),
            (5, 3),
            (999, 3),
            (1000, 4),
            (12345, 5),
        ],
    )
    def test_calculate_padding_width_scales_with_part_count(
        self, total_parts: int, expected_width: int
    ) -> None:
        assert calculate_padding_width(total_parts) == expected_width

    def test_get_unique_path_appends_counter_for_existing_files(self, tmp_path: Path) -> None:
        file_a = tmp_path / "video_merged.mp4"
        assert get_unique_path(file_a) == file_a

        file_a.touch()
        unique_1 = get_unique_path(file_a)
        assert unique_1 == tmp_path / "video_merged_2.mp4"

        unique_1.touch()
        unique_2 = get_unique_path(file_a)
        assert unique_2 == tmp_path / "video_merged_3.mp4"

    def test_get_unique_path_appends_counter_for_existing_directories(self, tmp_path: Path) -> None:
        dir_a = tmp_path / "my_video_parts"
        assert get_unique_path(dir_a, is_directory=True) == dir_a

        dir_a.mkdir()
        unique_dir = get_unique_path(dir_a, is_directory=True)
        assert unique_dir == tmp_path / "my_video_parts_2"

        unique_dir.mkdir()
        unique_dir2 = get_unique_path(dir_a, is_directory=True)
        assert unique_dir2 == tmp_path / "my_video_parts_3"

    @pytest.mark.parametrize(
        ("input_path", "expected_escaped"),
        [
            ("video.mp4", "'video.mp4'"),
            (r"C:\My Videos\part 1.mp4", "'C:/My Videos/part 1.mp4'"),
            (r"C:\Videos\it's 'great'\part_1.mp4", "'C:/Videos/it'\\''s '\\''great'\\''/part_1.mp4'"),
        ],
    )
    def test_escape_concat_path_handles_windows_slashes_and_single_quotes(
        self, input_path: str, expected_escaped: str
    ) -> None:
        assert escape_concat_path(input_path) == expected_escaped

    @pytest.mark.parametrize(
        ("filename", "expected_tuple"),
        [
            ("video.mp4", (True, "supported")),
            ("video.MKV", (True, "supported")),
            ("video.mov", (True, "supported")),
            ("video.avi", (True, "best_effort")),
            ("video.wmv", (False, "unsupported")),
            ("video.flv", (False, "unsupported")),
            ("video.txt", (False, "unsupported")),
        ],
    )
    def test_check_supported_format_identifies_containers(
        self, filename: str, expected_tuple: tuple[bool, str]
    ) -> None:
        assert check_supported_format(filename) == expected_tuple

    def test_is_path_length_safe_identifies_path_length_limits(self, tmp_path: Path) -> None:
        short_path = tmp_path / "short_path.mp4"
        assert is_path_length_safe(short_path, max_length=250) is True

        long_name = "a" * 300 + ".mp4"
        long_path = tmp_path / long_name
        assert is_path_length_safe(long_path, max_length=250) is False
