"""Split View UI component for CustomTkinter."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from src.core.ffmpeg_manager import probe_media
from src.core.splitter import SplitResult
from src.utils.helpers import (
    check_supported_format,
    format_duration,
    parse_duration_minutes,
)


class SplitView(ctk.CTkFrame):
    """View handling the video splitting interface, input selection, and results display."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        on_start_split: Callable[[Path, float, Optional[Path]], None],
        on_cancel_split: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)

        self.on_start_split = on_start_split
        self.on_cancel_split = on_cancel_split

        self.selected_file: Optional[Path] = None
        self.custom_output_dir: Optional[Path] = None
        self.last_result: Optional[SplitResult] = None
        self.media_duration: float = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # 1. File Selection Section
        file_frame = ctk.CTkFrame(self, corner_radius=10)
        file_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        file_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            file_frame,
            text="اختيار ملف الفيديو (Select Video File):",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        self.file_entry = ctk.CTkEntry(
            file_frame,
            placeholder_text="اضغط على زر الاستعراض لاختيار ملف الفيديو (MP4, MKV, MOV, AVI)...",
            state="readonly",
            height=36,
        )
        self.file_entry.grid(row=1, column=0, sticky="ew", padx=(15, 10), pady=(0, 10))

        self.browse_file_btn = ctk.CTkButton(
            file_frame,
            text="استعراض (Browse)...",
            width=130,
            height=36,
            command=self._on_browse_file,
        )
        self.browse_file_btn.grid(row=1, column=1, padx=(0, 15), pady=(0, 10))

        self.file_info_label = ctk.CTkLabel(
            file_frame,
            text="لم يتم اختيار ملف بعد.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self.file_info_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 10))

        # 2. Options Section
        options_frame = ctk.CTkFrame(self, corner_radius=10)
        options_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        options_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            options_frame,
            text="مدة كل جزء بالدقائق (Segment duration in minutes):",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=15, pady=(12, 4))

        self.duration_entry = ctk.CTkEntry(
            options_frame,
            placeholder_text="مثال: 30 أو 1.5",
            width=160,
            height=36,
        )
        self.duration_entry.grid(row=0, column=1, sticky="w", padx=10, pady=(12, 4))

        ctk.CTkLabel(
            options_frame,
            text="الحد الأدنى 10 ثوانٍ (0.17 دقيقة). الجزء الأخير سيحمل المدة المتبقية تلقائياً.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 8))

        # Destination folder
        ctk.CTkLabel(
            options_frame,
            text="مجلد الحفظ (Output Folder):",
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).grid(row=2, column=0, sticky="w", padx=15, pady=(4, 12))

        dest_subframe = ctk.CTkFrame(options_frame, fg_color="transparent")
        dest_subframe.grid(row=2, column=1, sticky="ew", padx=10, pady=(4, 12))
        dest_subframe.columnconfigure(0, weight=1)

        self.dest_label = ctk.CTkLabel(
            dest_subframe,
            text="تلقائي: مجلد باسم <اسم_الفيديو>_parts بجوار الأصل",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self.dest_label.grid(row=0, column=0, sticky="w")

        self.browse_dest_btn = ctk.CTkButton(
            dest_subframe,
            text="تغيير المكان...",
            width=110,
            height=28,
            command=self._on_browse_dest,
        )
        self.browse_dest_btn.grid(row=0, column=1, padx=(10, 0))

        # 3. Actions & Progress Section
        action_frame = ctk.CTkFrame(self, corner_radius=10)
        action_frame.grid(row=2, column=0, sticky="ew", padx=15, pady=5)
        action_frame.columnconfigure(0, weight=1)

        btn_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        btn_row.grid(row=0, column=0, sticky="ew", padx=15, pady=12)

        self.start_btn = ctk.CTkButton(
            btn_row,
            text="بدء تقسيم الفيديو الآن (Start Split)",
            font=ctk.CTkFont(size=14, weight="bold"),
            height=42,
            command=self._on_start_click,
        )
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 10))

        self.cancel_btn = ctk.CTkButton(
            btn_row,
            text="إلغاء العملية (Cancel)",
            font=ctk.CTkFont(size=13),
            fg_color="#D32F2F",
            hover_color="#B71C1C",
            height=42,
            state="disabled",
            command=self._on_cancel_click,
        )
        self.cancel_btn.pack(side="right")

        # Progress bar
        self.progress_bar = ctk.CTkProgressBar(action_frame)
        self.progress_bar.grid(row=1, column=0, sticky="ew", padx=15, pady=(5, 5))
        self.progress_bar.set(0)

        prog_labels_row = ctk.CTkFrame(action_frame, fg_color="transparent")
        prog_labels_row.grid(row=2, column=0, sticky="ew", padx=15, pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            prog_labels_row,
            text="جاهز للبدء.",
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.status_label.pack(side="left")

        self.percent_label = ctk.CTkLabel(
            prog_labels_row,
            text="0%",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.percent_label.pack(side="right")

        # 4. Results & Summary Section
        self.results_frame = ctk.CTkFrame(self, corner_radius=10)
        self.results_frame.grid(row=3, column=0, sticky="nsew", padx=15, pady=(5, 15))
        self.rowconfigure(3, weight=1)
        self.results_frame.columnconfigure(0, weight=1)
        self.results_frame.rowconfigure(1, weight=1)

        results_header = ctk.CTkFrame(self.results_frame, fg_color="transparent")
        results_header.grid(row=0, column=0, sticky="ew", padx=15, pady=(10, 5))
        results_header.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            results_header,
            text="ملخص التقسيم (Split Summary):",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        self.open_folder_btn = ctk.CTkButton(
            results_header,
            text="فتح مجلد الأجزاء (Open Folder)",
            width=180,
            height=30,
            state="disabled",
            command=self._on_open_folder,
        )
        self.open_folder_btn.grid(row=0, column=1, sticky="e")

        self.summary_textbox = ctk.CTkTextbox(
            self.results_frame,
            font=ctk.CTkFont(size=12),
            wrap="word",
        )
        self.summary_textbox.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 12))
        self.summary_textbox.insert("1.0", "ستظهر هنا تفاصيل الأجزاء الناتجة بعد اكتمال العملية بنجاح.\n")
        self.summary_textbox.configure(state="disabled")

    def _on_browse_file(self) -> None:
        filetypes = [
            ("ملفات الفيديو المدعومة", "*.mp4;*.mkv;*.mov;*.avi"),
            ("ملفات MP4", "*.mp4"),
            ("ملفات MKV", "*.mkv"),
            ("ملفات MOV", "*.mov"),
            ("ملفات AVI", "*.avi"),
            ("كافة الملفات", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="اختر ملف الفيديو للتقسيم",
            filetypes=filetypes,
        )
        if not chosen:
            return

        chosen_path = Path(chosen).resolve()
        supported, status = check_supported_format(chosen_path)
        if not supported:
            self.file_info_label.configure(
                text=f"صيغة غير مدعومة: {chosen_path.suffix}",
                text_color="#D32F2F",
            )
            return

        self.selected_file = chosen_path
        self.file_entry.configure(state="normal")
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, str(chosen_path))
        self.file_entry.configure(state="readonly")

        # Probe file info
        try:
            probe = probe_media(chosen_path)
            self.media_duration = probe["duration"]
            size_mb = probe["size_bytes"] / (1024 * 1024)
            best_effort_note = " (دعم تجريبي)" if status == "best_effort" else ""
            self.file_info_label.configure(
                text=f"المدة: {format_duration(self.media_duration)} | "
                f"الحجم: {size_mb:.1f} MB | الصيغة: {chosen_path.suffix.upper()}{best_effort_note}",
                text_color="gray" if status != "best_effort" else "#FFA000",
            )
        except Exception as exc:
            self.file_info_label.configure(
                text=f"تم اختيار الملف (تعذر فحص التفاصيل عبر ffprobe: {exc})",
                text_color="#FFA000",
            )

    def _on_browse_dest(self) -> None:
        chosen = filedialog.askdirectory(title="اختر مجلد الحفظ")
        if chosen:
            self.custom_output_dir = Path(chosen).resolve()
            self.dest_label.configure(
                text=f"مخصص: {self.custom_output_dir}",
                text_color="gray",
            )

    def _on_start_click(self) -> None:
        if not self.selected_file or not self.selected_file.is_file():
            self.set_status("يرجى اختيار ملف فيديو صالح أولاً.", is_error=True)
            return

        duration_text = self.duration_entry.get().strip()
        try:
            minutes = parse_duration_minutes(duration_text, min_seconds=10.0)
        except ValueError as err:
            self.set_status(str(err), is_error=True)
            return

        segment_seconds = minutes * 60.0
        if self.media_duration > 0 and segment_seconds >= self.media_duration:
            self.set_status(
                f"الفيديو أقصر من مدة التقسيم المحددة ({format_duration(segment_seconds)}). "
                f"مدة الفيديو: {format_duration(self.media_duration)}.",
                is_error=True,
            )
            return

        self.on_start_split(self.selected_file, segment_seconds, self.custom_output_dir)

    def _on_cancel_click(self) -> None:
        self.on_cancel_split()

    def _on_open_folder(self) -> None:
        if self.last_result and self.last_result.parts_dir.is_dir():
            try:
                os.startfile(str(self.last_result.parts_dir))
            except Exception as exc:
                self.set_status(f"تعذر فتح المجلد: {exc}", is_error=True)

    def set_running_state(self, is_running: bool) -> None:
        """Update widget states during/after operation."""
        state = "disabled" if is_running else "normal"
        self.start_btn.configure(state=state)
        self.browse_file_btn.configure(state=state)
        self.browse_dest_btn.configure(state=state)
        self.duration_entry.configure(state=state)
        self.cancel_btn.configure(state="normal" if is_running else "disabled")

    def update_progress(self, percent: float, current_time_sec: float) -> None:
        """Update progress bar and percentage display."""
        norm = max(0.0, min(1.0, percent / 100.0))
        self.progress_bar.set(norm)
        self.percent_label.configure(text=f"{percent:.1f}%")
        self.status_label.configure(
            text=f"جاري التقسيم... التوقيت المعالج: {format_duration(current_time_sec)}",
            text_color="gray",
        )

    def set_status(self, text: str, is_error: bool = False, is_success: bool = False) -> None:
        color = "#D32F2F" if is_error else ("#388E3C" if is_success else "gray")
        self.status_label.configure(text=text, text_color=color)

    def show_success_result(self, result: SplitResult) -> None:
        self.last_result = result
        self.open_folder_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self.percent_label.configure(text="100%")
        self.set_status("اكتمل تقسيم الفيديو بنجاح!", is_success=True)

        lines = [
            f"تم تقسيم الفيديو بنجاح إلى {len(result.parts)} جزء/أجزاء.",
            f"مجلد الحفظ: {result.parts_dir}",
            f"ملف قائمة الدمج التلقائية: {result.parts_list_file.name}",
            "",
            "الأجزاء المنتجة والمدد الفعلية:",
        ]
        for idx, part in enumerate(result.parts, start=1):
            lines.append(f"  الجزء {idx:02d}: {part.file_path.name} | المدة: {format_duration(part.duration)}")

        if result.has_deviation_warning:
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"تنبيه: {result.warning_message}")
            lines.append("السبب: التقسيم بدون إعادة ترميز (Stream Copy) يعتمد على نقاط Keyframes الأصلية.")
            lines.append("=" * 60)

        self.summary_textbox.configure(state="normal")
        self.summary_textbox.delete("1.0", "end")
        self.summary_textbox.insert("1.0", "\n".join(lines))
        self.summary_textbox.configure(state="disabled")
