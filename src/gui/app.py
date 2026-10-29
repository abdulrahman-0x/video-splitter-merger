"""Main CustomTkinter Application window for Video Splitter & Merger."""

from __future__ import annotations

import logging
import queue
import threading
import tkinter.messagebox as messagebox
from pathlib import Path
from typing import Any, Optional

import customtkinter as ctk

from src.core.ffmpeg_manager import (
    download_ffmpeg_essentials,
    get_ffmpeg_tools,
    is_ffmpeg_available,
)
from src.core.denoise import DenoiseError, denoise_parts
from src.core.merger import MergeResult, VideoMerger
from src.core.runner import FFmpegRunner
from src.core.splitter import SplitResult, VideoSplitter
from src.gui.merge_view import MergeView
from src.gui.split_view import SplitView

logger = logging.getLogger(__name__)

# Configure UI defaults
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")


class App(ctk.CTk):
    """Main desktop application window."""

    def __init__(self) -> None:
        super().__init__()

        self.title("مقسم ومدمج الفيديو | Video Splitter & Merger")
        self.geometry("860x720")
        self.minsize(740, 580)

        # Threading and communication
        self.msg_queue: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.active_thread: Optional[threading.Thread] = None
        self.active_runner: Optional[FFmpegRunner] = None
        self.active_task_type: Optional[str] = None
        self.active_splitter: Optional[VideoSplitter] = None
        self.active_merger: Optional[VideoMerger] = None
        self.denoise_cancel_event: Optional[threading.Event] = None
        # Session-only measured RTFs per channel count (the app has no settings persistence).
        self.denoise_rtf_history: dict[str, list[float]] = {"1": [], "2": []}

        self._build_ui()

        # Handle window closing safely
        self.protocol("WM_DELETE_WINDOW", self._on_window_close)

        # Start message queue poll loop
        self.after(100, self._process_queue)

        # Check FFmpeg on startup after window is displayed
        self.after(300, self._check_ffmpeg_on_startup)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        # Header Frame
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=20, pady=(15, 5))
        header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header,
            text="مقسم ومدمج الفيديو السريع (بدون إعادة ترميز)",
            font=ctk.CTkFont(size=18, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="تقسيم وتجميع ملفات الفيديو بسرعة وسائط التخزين بنمط Stream Copy ودون فقدان الجودة.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Tabview for Split and Merge
        self.tabview = ctk.CTkTabview(self, corner_radius=10)
        self.tabview.grid(row=1, column=0, sticky="nsew", padx=20, pady=(5, 15))

        self.tab_split = self.tabview.add("تقسيم الفيديو (Split)")
        self.tab_merge = self.tabview.add("دمج الأجزاء (Merge)")

        self.tab_split.columnconfigure(0, weight=1)
        self.tab_split.rowconfigure(0, weight=1)
        self.tab_merge.columnconfigure(0, weight=1)
        self.tab_merge.rowconfigure(0, weight=1)

        # Split View
        self.split_view = SplitView(
            master=self.tab_split,
            on_start_split=self._start_split_task,
            on_cancel_split=self._cancel_current_task,
            fg_color="transparent",
        )
        self.split_view.grid(row=0, column=0, sticky="nsew")

        # Merge View
        self.merge_view = MergeView(
            master=self.tab_merge,
            on_start_merge=self._start_merge_task,
            on_cancel_merge=self._cancel_current_task,
            fg_color="transparent",
        )
        self.merge_view.grid(row=0, column=0, sticky="nsew")

    def _check_ffmpeg_on_startup(self) -> None:
        """Verify FFmpeg availability on startup and prompt user to download if missing."""
        if is_ffmpeg_available():
            logger.info("تم العثور على أدوات FFmpeg و FFprobe بنجاح.")
            return

        logger.warning("أدوات FFmpeg غير متوفرة في النظام.")
        answer = messagebox.askyesno(
            title="تثبيت محرك المعالجة FFmpeg",
            message=(
                "لم يتم العثور على أدوات معالجة الفيديو (FFmpeg) على جهازك.\n\n"
                "هل ترغب في تنزيل وتثبيت الحزمة الرسمية المعتمدة تلقائياً من موقع Gyan.dev؟\n"
                "(الحجم: ~110 ميجابايت، يتم التنزيل مرة واحدة فقط داخل مجلد البرنامج)\n\n"
                "عند اختيار 'لا'، يمكنك وضع ملفي ffmpeg.exe و ffprobe.exe يدوياً داخل مجلد 'bin'."
            ),
        )
        if answer:
            self._start_ffmpeg_download_task()
        else:
            self.split_view.set_status("FFmpeg غير متوفر. يرجى وضع ffmpeg.exe في مجلد bin/", is_error=True)
            self.merge_view.set_status("FFmpeg غير متوفر. يرجى وضع ffmpeg.exe في مجلد bin/", is_error=True)

    def _start_ffmpeg_download_task(self) -> None:
        """Launch background download of FFmpeg."""
        self._set_app_busy(True, "DOWNLOAD")
        self.split_view.set_status("جاري تنزيل حزمة FFmpeg الرسمية...", is_error=False)

        def worker() -> None:
            try:
                def on_prog(percent: float, downloaded: int, total: int) -> None:
                    self.msg_queue.put(("DOWNLOAD_PROGRESS", (percent, downloaded, total)))

                download_ffmpeg_essentials(progress_callback=on_prog)
                self.msg_queue.put(("DOWNLOAD_SUCCESS", None))
            except Exception as exc:
                self.msg_queue.put(("DOWNLOAD_ERROR", str(exc)))

        self.active_thread = threading.Thread(target=worker, daemon=True)
        self.active_thread.start()

    def _start_split_task(
        self,
        input_file: Path,
        segment_duration_sec: float,
        output_dir: Optional[Path],
    ) -> None:
        """Launch background video split operation."""
        if not is_ffmpeg_available():
            self.split_view.set_status("أدوات FFmpeg غير متوفرة. تعذر بدء العملية.", is_error=True)
            return

        self._set_app_busy(True, "SPLIT")
        self.split_view.set_status("جاري تجهيز وتقسيم الفيديو...")
        self.split_view.summary_textbox.configure(state="normal")
        self.split_view.summary_textbox.delete("1.0", "end")
        self.split_view.summary_textbox.insert("1.0", "العملية جارية حالياً، يرجى الانتظار...\n")
        self.split_view.summary_textbox.configure(state="disabled")

        runner = FFmpegRunner()
        self.active_runner = runner
        splitter = VideoSplitter(runner=runner)
        self.active_splitter = splitter

        denoise_settings = self.split_view.get_denoise_settings()
        if denoise_settings["enabled"]:
            self.denoise_cancel_event = threading.Event()
            denoise_settings["rtf_history"] = self.denoise_rtf_history

        def worker() -> None:
            try:
                def on_prog(pct: float, cur_sec: float) -> None:
                    self.msg_queue.put(("SPLIT_PROGRESS", (pct, cur_sec)))

                result = splitter.split(
                    input_path=input_file,
                    segment_duration_sec=segment_duration_sec,
                    output_parent_dir=output_dir,
                    progress_callback=on_prog,
                )

                extra_lines: Optional[list[str]] = None
                if denoise_settings["enabled"]:
                    def on_denoise_prog(pct: float, message: str) -> None:
                        self.msg_queue.put(("DENOISE_PROGRESS", (pct, message)))

                    report = denoise_parts(
                        parts=[(p.file_path, p.duration) for p in result.parts],
                        settings=denoise_settings,
                        cancel_event=self.denoise_cancel_event,
                        progress_callback=on_denoise_prog,
                    )
                    for channels_key, rtfs in report.measured_rtfs.items():
                        history = self.denoise_rtf_history.setdefault(str(int(channels_key)), [])
                        history.extend(float(r) for r in rtfs)
                        history[:] = history[-8:]  # session cap
                    extra_lines = report.summary_lines() or None

                self.msg_queue.put(("SPLIT_SUCCESS", (result, extra_lines)))
            except InterruptedError:
                self.msg_queue.put(("SPLIT_CANCELLED", None))
            except Exception as exc:
                self.msg_queue.put(("SPLIT_ERROR", str(exc)))

        self.active_thread = threading.Thread(target=worker, daemon=True)
        self.active_thread.start()

    def _start_merge_task(self, target_input: Path, output_dir: Optional[Path]) -> None:
        """Launch background video merge operation."""
        if not is_ffmpeg_available():
            self.merge_view.set_status("أدوات FFmpeg غير متوفرة. تعذر بدء العملية.", is_error=True)
            return

        self._set_app_busy(True, "MERGE")
        self.merge_view.set_status("جاري فحص وتجميع الأجزاء...")
        self.merge_view.summary_textbox.configure(state="normal")
        self.merge_view.summary_textbox.delete("1.0", "end")
        self.merge_view.summary_textbox.insert("1.0", "العملية جارية حالياً، يرجى الانتظار...\n")
        self.merge_view.summary_textbox.configure(state="disabled")

        runner = FFmpegRunner()
        self.active_runner = runner
        merger = VideoMerger(runner=runner)
        self.active_merger = merger

        def worker() -> None:
            try:
                def on_prog(pct: float, cur_sec: float) -> None:
                    self.msg_queue.put(("MERGE_PROGRESS", (pct, cur_sec)))

                result = merger.merge(
                    input_target=target_input,
                    output_parent_dir=output_dir,
                    progress_callback=on_prog,
                )
                self.msg_queue.put(("MERGE_SUCCESS", result))
            except InterruptedError:
                self.msg_queue.put(("MERGE_CANCELLED", None))
            except Exception as exc:
                self.msg_queue.put(("MERGE_ERROR", str(exc)))

        self.active_thread = threading.Thread(target=worker, daemon=True)
        self.active_thread.start()

    def _cancel_current_task(self) -> None:
        """Cancel whatever background task is running."""
        if self.active_runner:
            self.active_runner.cancel()
        if self.denoise_cancel_event:
            self.denoise_cancel_event.set()
        if self.active_task_type == "SPLIT":
            self.split_view.set_status("جاري إلغاء العملية وتنظيف الملفات...", is_error=True)
        elif self.active_task_type == "MERGE":
            self.merge_view.set_status("جاري إلغاء عملية الدمج...", is_error=True)

    def _set_app_busy(self, is_busy: bool, task_type: Optional[str] = None) -> None:
        """Enable/disable controls across the application."""
        self.active_task_type = task_type if is_busy else None
        if not is_busy:
            self.active_runner = None
            self.active_splitter = None
            self.active_merger = None
            self.denoise_cancel_event = None

        self.split_view.set_running_state(is_busy)
        self.merge_view.set_running_state(is_busy)

    def _process_queue(self) -> None:
        """Poll thread messages and update the UI accordingly."""
        try:
            while True:
                msg_type, data = self.msg_queue.get_nowait()

                if msg_type == "SPLIT_PROGRESS":
                    pct, cur_sec = data
                    self.split_view.update_progress(pct, cur_sec)

                elif msg_type == "DENOISE_PROGRESS":
                    pct, message = data
                    self.split_view.set_denoise_progress(pct, message)

                elif msg_type == "SPLIT_SUCCESS":
                    self._set_app_busy(False)
                    result, extra_lines = data if isinstance(data, tuple) and len(data) == 2 else (data, None)
                    self.split_view.show_success_result(result, extra_lines=extra_lines)

                elif msg_type == "SPLIT_ERROR":
                    self._set_app_busy(False)
                    self.split_view.set_status(f"خطأ أثناء التقسيم: راجع logs/app.log", is_error=True)
                    messagebox.showerror(
                        title="خطأ أثناء التقسيم",
                        message=f"{data}\n\nيرجى مراجعة ملف السجلات logs/app.log للمزيد من التفاصيل.",
                    )

                elif msg_type == "SPLIT_CANCELLED":
                    self._set_app_busy(False)
                    self.split_view.set_status("تم إلغاء العملية وتنظيف الملفات المنشأة بنجاح.", is_error=True)
                    self.split_view.progress_bar.set(0)
                    self.split_view.percent_label.configure(text="0%")

                elif msg_type == "MERGE_PROGRESS":
                    pct, cur_sec = data
                    self.merge_view.update_progress(pct, cur_sec)

                elif msg_type == "MERGE_SUCCESS":
                    self._set_app_busy(False)
                    self.merge_view.show_success_result(data)

                elif msg_type == "MERGE_ERROR":
                    self._set_app_busy(False)
                    self.merge_view.set_status(f"خطأ أثناء الدمج: راجع logs/app.log", is_error=True)
                    messagebox.showerror(
                        title="خطأ أثناء الدمج",
                        message=f"{data}\n\nيرجى مراجعة ملف السجلات logs/app.log للمزيد من التفاصيل.",
                    )

                elif msg_type == "MERGE_CANCELLED":
                    self._set_app_busy(False)
                    self.merge_view.set_status("تم إلغاء عملية الدمج وحذف الملف المؤقت.", is_error=True)
                    self.merge_view.progress_bar.set(0)
                    self.merge_view.percent_label.configure(text="0%")

                elif msg_type == "DOWNLOAD_PROGRESS":
                    pct, dld, tot = data
                    self.split_view.update_progress(pct, 0)
                    mb_dld = dld / (1024 * 1024)
                    mb_tot = tot / (1024 * 1024)
                    self.split_view.status_label.configure(
                        text=f"جاري تنزيل FFmpeg ({mb_dld:.1f} / {mb_tot:.1f} MB)..."
                    )

                elif msg_type == "DOWNLOAD_SUCCESS":
                    self._set_app_busy(False)
                    self.split_view.set_status("تم تثبيت FFmpeg بنجاح! التطبيق جاهز للعمل.", is_success=True)
                    messagebox.showinfo(
                        title="اكتمل التثبيت",
                        message="تم تنزيل وتثبيت أدوات FFmpeg المعتمدة بنجاح. يمكنك الآن تقسيم ودمج الفيديو.",
                    )

                elif msg_type == "DOWNLOAD_ERROR":
                    self._set_app_busy(False)
                    self.split_view.set_status("فشل تنزيل FFmpeg. يرجى وضعه في bin/ يدوياً.", is_error=True)
                    messagebox.showerror(
                        title="فشل التنزيل",
                        message=f"{data}\n\nيمكنك تحميل ffmpeg.exe و ffprobe.exe يدوياً ووضعهما في مجلد bin.",
                    )

        except queue.Empty:
            pass
        finally:
            self.after(100, self._process_queue)

    def _on_window_close(self) -> None:
        """Prompt confirmation before closing if a task is active."""
        if self.active_task_type is not None:
            confirm = messagebox.askyesno(
                title="تأكيد الإغلاق",
                message="هناك عملية جارية حالياً.\nهل تريد إلغاء العملية الجارية وإغلاق البرنامج؟",
            )
            if not confirm:
                return

            logger.info("المستخدم طلب إغلاق النافذة أثناء عملية جارية. جاري الإلغاء...")
            self._cancel_current_task()
            if self.active_thread and self.active_thread.is_alive():
                self.active_thread.join(timeout=2.0)

        self.destroy()
