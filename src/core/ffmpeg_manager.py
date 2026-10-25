"""FFmpeg and FFprobe discovery, probing, verification, and secure download management."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import threading
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# Gyandev official essentials release build (pinned)
FFMPEG_VERSION = "9.0.2"
FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/packages/ffmpeg-9.0.2-essentials_build.zip"
FFMPEG_FALLBACK_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"
FFMPEG_SHA256 = "60f467265b1e312373dbcd92200c2618a74850f98d3d078e94296bb3fa2047ba"
LOCAL_BIN_DIR = Path("bin")

# Windows process creation flags to suppress console window
CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


def find_tool(name: str, custom_bin_dir: Optional[Path] = None) -> Optional[Path]:
    """Find a tool executable either in bin/ or in system PATH."""
    bin_dir = (custom_bin_dir or LOCAL_BIN_DIR).resolve()
    candidate_name = f"{name}.exe" if not name.lower().endswith(".exe") else name
    local_path = bin_dir / candidate_name

    if local_path.is_file():
        return local_path

    path_tool = shutil.which(candidate_name) or shutil.which(name)
    if path_tool:
        return Path(path_tool).resolve()

    return None


def get_ffmpeg_tools(custom_bin_dir: Optional[Path] = None) -> Tuple[Optional[Path], Optional[Path]]:
    """Return paths to ffmpeg and ffprobe.

    Checks:
    1. Local bin/ directory
    2. System PATH
    Returns (ffmpeg_path, ffprobe_path).
    """
    ffmpeg_path = find_tool("ffmpeg", custom_bin_dir)
    ffprobe_path = find_tool("ffprobe", custom_bin_dir)
    return ffmpeg_path, ffprobe_path


def is_ffmpeg_available(custom_bin_dir: Optional[Path] = None) -> bool:
    """Return True if both ffmpeg and ffprobe are available."""
    ffmpeg, ffprobe = get_ffmpeg_tools(custom_bin_dir)
    return ffmpeg is not None and ffprobe is not None


def probe_media(file_path: Path | str, ffprobe_path: Optional[Path] = None) -> Dict[str, Any]:
    """Inspect media file using ffprobe and return parsed metadata.

    Raises:
    - FileNotFoundError if file does not exist.
    - RuntimeError if ffprobe fails or is not available.
    - ValueError if media contains no valid streams.
    """
    target_path = Path(file_path).resolve()
    if not target_path.is_file():
        raise FileNotFoundError(f"ملف الفيديو غير موجود: {target_path}")

    probe_exe = ffprobe_path or find_tool("ffprobe")
    if not probe_exe:
        raise RuntimeError("أداة ffprobe غير متوفرة. يرجى تثبيتها أو وضعها في مجلد bin/.")

    cmd = [
        str(probe_exe),
        "-v", "error",
        "-show_format",
        "-show_streams",
        "-print_format", "json",
        str(target_path),
    ]

    logger.debug("تشغيل أمر ffprobe: %s", " ".join(cmd))

    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW,
            check=False,
            timeout=30,
        )
    except Exception as exc:
        raise RuntimeError(f"فشل تشغيل ffprobe لفحص الملف: {exc}") from exc

    if proc.returncode != 0:
        err_msg = proc.stderr.decode("utf-8", errors="ignore").strip()
        logger.error("خطأ ffprobe أثناء فحص %s: %s", target_path, err_msg)
        raise RuntimeError(f"تعذر قراءة ملف الفيديو عبر ffprobe:\n{err_msg}")

    try:
        data = json.loads(proc.stdout.decode("utf-8", errors="ignore"))
    except json.JSONDecodeError as exc:
        raise RuntimeError("فشل تحليل مخرجات ffprobe (صيغة JSON غير صالحة).") from exc

    streams = data.get("streams", [])
    format_info = data.get("format", {})

    video_streams = [s for s in streams if s.get("codec_type") == "video"]
    audio_streams = [s for s in streams if s.get("codec_type") == "audio"]
    subtitle_streams = [s for s in streams if s.get("codec_type") == "subtitle"]
    data_streams = [s for s in streams if s.get("codec_type") == "data"]
    attachment_streams = [s for s in streams if s.get("codec_type") == "attachment"]

    if not video_streams:
        raise ValueError("الملف المحدد لا يحتوي على أي مسار فيديو صالح.")

    # Determine duration
    duration = None
    if "duration" in format_info:
        try:
            duration = float(format_info["duration"])
        except (ValueError, TypeError):
            pass

    if duration is None:
        for s in video_streams + audio_streams:
            if "duration" in s:
                try:
                    duration = float(s["duration"])
                    break
                except (ValueError, TypeError):
                    continue

    if duration is None:
        duration = 0.0

    size_bytes = 0
    if "size" in format_info:
        try:
            size_bytes = int(format_info["size"])
        except (ValueError, TypeError):
            size_bytes = target_path.stat().st_size
    else:
        size_bytes = target_path.stat().st_size

    return {
        "file_path": target_path,
        "format_name": format_info.get("format_name", ""),
        "duration": duration,
        "size_bytes": size_bytes,
        "video_streams": video_streams,
        "audio_streams": audio_streams,
        "subtitle_streams": subtitle_streams,
        "data_streams": data_streams,
        "attachment_streams": attachment_streams,
        "all_streams": streams,
        "raw": data,
    }


def check_free_space(target_path: Path, required_bytes: int) -> Tuple[bool, int, int]:
    """Check if the disk containing target_path has at least required_bytes free.

    Returns:
    (has_enough_space, free_bytes, required_bytes)
    """
    directory = target_path if target_path.is_dir() else target_path.parent
    directory.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(directory)
    has_enough = usage.free >= required_bytes
    return has_enough, usage.free, required_bytes


def compute_file_sha256(file_path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(1024 * 1024):  # 1 MB chunks
            hasher.update(chunk)
    return hasher.hexdigest().lower()


def get_remote_file_size(url: str, timeout: int = 10) -> Optional[int]:
    """Query Content-Length header for a remote URL."""
    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            method="HEAD",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content_length = resp.headers.get("Content-Length")
            if content_length:
                return int(content_length)
    except Exception as exc:
        logger.warning("تعذر جلب حجم الملف عن بعد من %s: %s", url, exc)
    return None


def extract_binaries_from_zip(
    zip_path: Path,
    target_bin_dir: Path,
    expected_sha256: Optional[str] = None,
) -> Tuple[Path, Path]:
    """Securely verify SHA-256 and extract only ffmpeg.exe and ffprobe.exe into target_bin_dir.

    Never calls extractall() to prevent Zip-Slip vulnerabilities.
    """
    if expected_sha256:
        calculated_hash = compute_file_sha256(zip_path)
        if calculated_hash != expected_sha256.lower():
            raise ValueError(
                f"فشل التحقق من صحة الملف (SHA-256 mismatch).\n"
                f"المتوقع: {expected_sha256}\n"
                f"الفعلي:   {calculated_hash}"
            )

    target_bin_dir.mkdir(parents=True, exist_ok=True)
    extracted_ffmpeg: Optional[Path] = None
    extracted_ffprobe: Optional[Path] = None

    with zipfile.ZipFile(zip_path, "r") as zf:
        for member in zf.infolist():
            member_name = member.filename.replace("\\", "/")
            base_name = os.path.basename(member_name).lower()

            if base_name in ("ffmpeg.exe", "ffprobe.exe"):
                dest_file = target_bin_dir / base_name
                logger.info("استخراج %s إلى %s", base_name, dest_file)
                with zf.open(member) as source, open(dest_file, "wb") as dest:
                    shutil.copyfileobj(source, dest)

                if base_name == "ffmpeg.exe":
                    extracted_ffmpeg = dest_file
                elif base_name == "ffprobe.exe":
                    extracted_ffprobe = dest_file

    if not extracted_ffmpeg or not extracted_ffprobe:
        raise RuntimeError("أرشيف ZIP لا يحتوي على ffmpeg.exe و ffprobe.exe.")

    return extracted_ffmpeg, extracted_ffprobe


def download_ffmpeg_essentials(
    target_bin_dir: Optional[Path] = None,
    progress_callback: Optional[Callable[[float, int, int], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> Tuple[Path, Path]:
    """Download official Gyan.dev FFmpeg essentials zip, verify SHA-256, and extract binaries.

    Parameters:
    - target_bin_dir: Destination folder for executables (defaults to bin/).
    - progress_callback: (percentage_0_to_100, downloaded_bytes, total_bytes).
    - cancel_event: Event to signal early cancellation.
    """
    dest_bin = target_bin_dir or LOCAL_BIN_DIR
    dest_bin.mkdir(parents=True, exist_ok=True)
    temp_zip = dest_bin / "ffmpeg_download.tmp"

    urls_to_try = [FFMPEG_URL, FFMPEG_FALLBACK_URL]
    download_success = False
    last_error: Optional[Exception] = None

    for url in urls_to_try:
        if cancel_event and cancel_event.is_set():
            raise InterruptedError("تم إلغاء عملية تنزيل FFmpeg.")

        logger.info("محاولة تنزيل FFmpeg من: %s", url)
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.headers.get("Content-Length", 0))
                downloaded = 0
                chunk_size = 256 * 1024  # 256 KB chunks

                with open(temp_zip, "wb") as out_file:
                    while True:
                        if cancel_event and cancel_event.is_set():
                            raise InterruptedError("تم إلغاء عملية تنزيل FFmpeg.")

                        chunk = response.read(chunk_size)
                        if not chunk:
                            break

                        out_file.write(chunk)
                        downloaded += len(chunk)

                        if progress_callback and total_size > 0:
                            percent = min(100.0, (downloaded / total_size) * 100.0)
                            progress_callback(percent, downloaded, total_size)

            download_success = True
            break
        except InterruptedError:
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)
            raise
        except Exception as exc:
            logger.warning("فشل التنزيل من الرابط %s: %s", url, exc)
            last_error = exc
            if temp_zip.exists():
                temp_zip.unlink(missing_ok=True)

    if not download_success:
        raise RuntimeError(
            f"فشل تنزيل حزمة FFmpeg من الإنترنت. يرجى التحقق من اتصال الشبكة، "
            f"أو وضع ملفي ffmpeg.exe و ffprobe.exe يدوياً داخل مجلد '{dest_bin.resolve()}'.\n"
            f"الخطأ: {last_error}"
        )

    try:
        logger.info("جاري التحقق من بصمة SHA-256 للملف المنزَّل...")
        extracted_ffmpeg, extracted_ffprobe = extract_binaries_from_zip(
            zip_path=temp_zip,
            target_bin_dir=dest_bin,
            expected_sha256=FFMPEG_SHA256,
        )
        logger.info("تم التحقق واستخراج أدوات FFmpeg بنجاح.")
        return extracted_ffmpeg, extracted_ffprobe
    finally:
        if temp_zip.exists():
            temp_zip.unlink(missing_ok=True)
