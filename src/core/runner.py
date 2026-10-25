"""Process runner for FFmpeg with non-blocking progress parsing, cancellation, and stderr capture."""

from __future__ import annotations

import collections
import logging
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

CREATE_NO_WINDOW = 0x08000000 if hasattr(subprocess, "CREATE_NO_WINDOW") else 0


class FFmpegRunner:
    """Manages executing FFmpeg processes safely under Windows with progress tracking and clean cancellation."""

    def __init__(self) -> None:
        self.proc: Optional[subprocess.Popen] = None
        self._cancelled = threading.Event()
        self._stderr_lines: collections.deque[str] = collections.deque(maxlen=40)
        self._stderr_thread: Optional[threading.Thread] = None

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def cancel(self) -> None:
        """Signal cancellation and terminate the FFmpeg process immediately."""
        self._cancelled.set()
        if self.proc and self.proc.poll() is None:
            logger.warning("إرسال إشارة إلغاء لعملية FFmpeg (PID: %s)...", self.proc.pid)
            try:
                self.proc.terminate()
                # Give it a short moment to exit gracefully
                try:
                    self.proc.wait(timeout=1.5)
                except subprocess.TimeoutExpired:
                    logger.warning("العملية لم تستجب لـ terminate، جاري فرض الإنهاء بـ kill (PID: %s)...", self.proc.pid)
                    self.proc.kill()
                    self.proc.wait(timeout=1.0)
            except Exception as exc:
                logger.error("خطأ أثناء محاولة إنهاء عملية FFmpeg: %s", exc)

    def _read_stderr(self, pipe) -> None:
        """Read stderr in background thread to prevent buffer deadlocks."""
        try:
            for raw_line in iter(pipe.readline, b""):
                line = raw_line.decode("utf-8", errors="replace").rstrip()
                if line:
                    self._stderr_lines.append(line)
        except Exception:
            pass
        finally:
            pipe.close()

    def run(
        self,
        cmd: List[str],
        total_duration: float = 0.0,
        progress_callback: Optional[Callable[[float, float], None]] = None,
        working_dir: Optional[Path] = None,
    ) -> int:
        """Execute an FFmpeg command with progress parsing.

        Parameters:
        - cmd: Command list.
        - total_duration: Media duration in seconds (for calculating percentage).
        - progress_callback: Callable receiving (percentage: float [0..100], current_sec: float).
        - working_dir: Optional working directory.

        Raises:
        - InterruptedError: If operation was cancelled by user.
        - RuntimeError: If FFmpeg returns non-zero exit code.
        """
        self._cancelled.clear()
        self._stderr_lines.clear()

        # Add progress flags if not already present
        full_cmd = list(cmd)
        if "-progress" not in full_cmd:
            full_cmd.extend(["-progress", "pipe:1", "-nostats"])

        logger.info("بدء تشغيل FFmpeg: %s", " ".join(full_cmd))

        try:
            self.proc = subprocess.Popen(
                full_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                cwd=str(working_dir) if working_dir else None,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:
            logger.error("فشل إطلاق عملية FFmpeg: %s", exc)
            raise RuntimeError(f"فشل إطلاق FFmpeg: {exc}") from exc

        # Start background stderr reading thread
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(self.proc.stderr,),
            daemon=True,
        )
        self._stderr_thread.start()

        # Parse progress key=value from stdout
        last_progress_time = 0.0
        try:
            if self.proc.stdout:
                for raw_line in iter(self.proc.stdout.readline, b""):
                    if self._cancelled.is_set():
                        break

                    line = raw_line.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    if "=" in line:
                        key, val = line.split("=", 1)
                        key = key.strip()
                        val = val.strip()

                        if key == "out_time_us":
                            try:
                                out_sec = float(val) / 1_000_000.0
                                if out_sec < 0:
                                    out_sec = 0.0
                                last_progress_time = out_sec

                                if progress_callback and total_duration > 0:
                                    percent = min(100.0, (out_sec / total_duration) * 100.0)
                                    progress_callback(percent, out_sec)
                            except ValueError:
                                pass
                        elif key == "progress" and val == "end":
                            if progress_callback and total_duration > 0:
                                progress_callback(100.0, total_duration)

            self.proc.wait()
        finally:
            if self.proc.stdout:
                self.proc.stdout.close()
            if self._stderr_thread:
                self._stderr_thread.join(timeout=2.0)

        if self._cancelled.is_set():
            logger.info("تم إلغاء عملية FFmpeg بنجاح من قِبل المستخدم.")
            raise InterruptedError("تم إلغاء العملية بناءً على طلبك.")

        retcode = self.proc.returncode
        if retcode != 0:
            tail_stderr = "\n".join(list(self._stderr_lines)[-20:])
            logger.error("فشل أمر FFmpeg برمز الخطأ (%s).\nآخر رسائل الخطأ:\n%s", retcode, tail_stderr)
            raise RuntimeError(
                f"فشلت العملية أثناء معالجة الفيديو (رمز الخطأ: {retcode}).\n"
                f"التفاصيل:\n{tail_stderr or 'لم تُسجَّل رسائل خطأ إضافية.'}"
            )

        if progress_callback and total_duration > 0:
            progress_callback(100.0, total_duration)

        logger.info("اكتمل تشغيل FFmpeg بنجاح.")
        return 0
