"""Merge View UI component for CustomTkinter."""

from __future__ import annotations

import os
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from src.core.ffmpeg_manager import find_tool, probe_media
from src.core.merger import (
    MergeResult,
    discover_parts_in_dir,
    parse_concat_list,
    verify_parts_compatibility,
)
from src.utils.helpers import format_duration


class MergeView(ctk.CTkFrame):
    """View handling the video merging interface, part discovery, and results display."""

    def __init__(
        self,
        master: ctk.CTkBaseClass,
        on_start_merge: Callable[[Path, Optional[Path]], None],
        on_cancel_merge: Callable[[], None],
        **kwargs,
    ) -> None:
        super().__init__(master, **kwargs)

        self.on_start_merge = on_start_merge
        self.on_cancel_merge = on_cancel_merge

        self.selected_target: Optional[Path] = None
        self.custom_output_dir: Optional[Path] = None
        self.last_result: Optional[MergeResult] = None
        self.detected_parts_count: int = 0
        self.total_estimated_duration: float = 0.0

        self._build_ui()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # 1. Target Selection Section
        target_frame = ctk.CTkFrame(self, corner_radius=10)
        target_frame.grid(row=0, column=0, sticky="ew", padx=15, pady=(15, 10))
        target_frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            target_frame,
            text="اختيار الأجزاء المراد دمجها (Select Video Parts or parts_list.txt):",
            font=ctk.CTkFont(size=14, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=15, pady=(10, 5))

        self.target_entry = ctk.CTkEntry(
            target_frame,
            placeholder_text="اختر مجلد الأجزاء الناتج عن التقسيم، أو ملف parts_list.txt...",
            state="readonly",
            height=36,
        )
        self.target_entry.grid(row=1, column=0, sticky="ew", padx=(15, 10), pady=(0, 10))

        btn_box = ctk.CTkFrame(target_frame, fg_color="transparent")
        btn_box.grid(row=1, column=1, padx=(0, 15), pady=(0, 10))

        self.browse_folder_btn = ctk.CTkButton(
            btn_box,
            text="استعراض مجلد...",
            width=120,
            height=36,
            command=self._on_browse_folder,
        )
        self.browse_folder_btn.pack(side="left", padx=(0, 5))

        self.browse_list_btn = ctk.CTkButton(
            btn_box,
            text="اختيار قائمة (.txt)...",
            width=130,
            height=36,
            command=self._on_browse_list_file,
        )
        self.browse_list_btn.pack(side="left")

        self.target_info_label = ctk.CTkLabel(
            target_frame,
            text="لم يتم تحديد مجلد أجزاء أو قائمة بعد.",
            font=ctk.CTkFont(size=12),
            text_color="gray",
            anchor="w",
        )
        self.target_info_label.grid(row=2, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 10))

        # 2. Options Section (Custom output parent)
        options_frame = ctk.CTkFrame(self, corner_radius=10)
        options_frame.grid(row=1, column=0, sticky="ew", padx=15, pady=5)
        options_frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            options_frame,
            text="مكان حفظ الفيديو المدمج (Output Folder):",
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", padx=15, pady=12)

        dest_subframe = ctk.CTkFrame(options_frame, fg_color="transparent")
        dest_subframe.grid(row=0, column=1, sticky="ew", padx=10, pady=12)
        dest_subframe.columnconfigure(0, weight=1)

        self.dest_label = ctk.CTkLabel(
            dest_subframe,
            text="تلقائي: بجوار مجلد الأجزاء باسم <اسم_الفيديو>_merged",
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
            text="بدء دمج الأجزاء الآن (Start Merge)",
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
            text="ملخص الدمج (Merge Summary):",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w")

        btn_actions_box = ctk.CTkFrame(results_header, fg_color="transparent")
        btn_actions_box.grid(row=0, column=1, sticky="e")

        self.play_video_btn = ctk.CTkButton(
            btn_actions_box,
            text="تشغيل الفيديو (Play Video)",
            width=160,
            height=30,
            state="disabled",
            command=self._on_play_video,
        )
        self.play_video_btn.pack(side="left", padx=(0, 8))

        self.open_folder_btn = ctk.CTkButton(
            btn_actions_box,
            text="فتح المجلد (Open Folder)",
            width=150,
            height=30,
            state="disabled",
            command=self._on_open_folder,
        )
        self.open_folder_btn.pack(side="left")

        self.summary_textbox = ctk.CTkTextbox(
            self.results_frame,
            font=ctk.CTkFont(size=12),
            wrap="word",
        )
        self.summary_textbox.grid(row=1, column=0, sticky="nsew", padx=15, pady=(0, 12))
        self.summary_textbox.insert("1.0", "ستظهر هنا تفاصيل الفيديو المدمج بعد اكتمال العملية بنجاح.\n")
        self.summary_textbox.configure(state="disabled")

    def _on_browse_folder(self) -> None:
        chosen = filedialog.askdirectory(title="اختر مجلد أجزاء الفيديو")
        if chosen:
            self._handle_selected_target(Path(chosen).resolve())

    def _on_browse_list_file(self) -> None:
        chosen = filedialog.askopenfilename(
            title="اختر ملف قائمة الأجزاء (parts_list.txt)",
            filetypes=[("ملف قائمة Concat", "parts_list.txt;*.txt"), ("كافة الملفات", "*.*")],
        )
        if chosen:
            self._handle_selected_target(Path(chosen).resolve())

    def _on_browse_dest(self) -> None:
        chosen = filedialog.askdirectory(title="اختر مجلد الحفظ")
        if chosen:
            self.custom_output_dir = Path(chosen).resolve()
            self.dest_label.configure(
                text=f"مخصص: {self.custom_output_dir}",
                text_color="gray",
            )

    def _handle_selected_target(self, target: Path) -> None:
        self.selected_target = target
        self.target_entry.configure(state="normal")
        self.target_entry.delete(0, "end")
        self.target_entry.insert(0, str(target))
        self.target_entry.configure(state="readonly")

        try:
            if target.is_file():
                parts = parse_concat_list(target)
            else:
                parts = discover_parts_in_dir(target)

            self.detected_parts_count = len(parts)

            # Probe compatibility
            ffprobe_bin = find_tool("ffprobe")
            if ffprobe_bin:
                probes, sum_durations = verify_parts_compatibility(parts, ffprobe_bin)
                self.total_estimated_duration = sum_durations
                base_v = probes[0]["video_streams"][0]
                codec = base_v.get("codec_name", "N/A")
                w = base_v.get("width", "N/A")
                h = base_v.get("height", "N/A")

                self.target_info_label.configure(
                    text=f"تم العثور على {len(parts)} أجزاء متوافقة | "
                    f"الترميز: {codec} ({w}x{h}) | المدة الكلية المتوقعة: {format_duration(sum_durations)}",
                    text_color="gray",
                )
            else:
                self.target_info_label.configure(
                    text=f"تم العثور على {len(parts)} أجزاء.",
                    text_color="gray",
                )
        except Exception as exc:
            self.target_info_label.configure(
                text=f"خطأ أثناء فحص الأجزاء: {exc}",
                text_color="#D32F2F",
            )

    def _on_start_click(self) -> None:
        if not self.selected_target or not self.selected_target.exists():
            self.set_status("يرجى اختيار مجلد أجزاء أو قائمة صالحة أولاً.", is_error=True)
            return

        self.on_start_merge(self.selected_target, self.custom_output_dir)

    def _on_cancel_click(self) -> None:
        self.on_cancel_merge()

    def _on_open_folder(self) -> None:
        if self.last_result and self.last_result.output_path.is_file():
            try:
                os.startfile(str(self.last_result.output_path.parent))
            except Exception as exc:
                self.set_status(f"تعذر فتح المجلد: {exc}", is_error=True)

    def _on_play_video(self) -> None:
        if self.last_result and self.last_result.output_path.is_file():
            try:
                os.startfile(str(self.last_result.output_path))
            except Exception as exc:
                self.set_status(f"تعذر تشغيل الفيديو: {exc}", is_error=True)

    def set_running_state(self, is_running: bool) -> None:
        state = "disabled" if is_running else "normal"
        self.start_btn.configure(state=state)
        self.browse_folder_btn.configure(state=state)
        self.browse_list_btn.configure(state=state)
        self.browse_dest_btn.configure(state=state)
        self.cancel_btn.configure(state="normal" if is_running else "disabled")

    def update_progress(self, percent: float, current_time_sec: float) -> None:
        norm = max(0.0, min(1.0, percent / 100.0))
        self.progress_bar.set(norm)
        self.percent_label.configure(text=f"{percent:.1f}%")
        self.status_label.configure(
            text=f"جاري الدمج... التوقيت المعالج: {format_duration(current_time_sec)}",
            text_color="gray",
        )

    def set_status(self, text: str, is_error: bool = False, is_success: bool = False) -> None:
        color = "#D32F2F" if is_error else ("#388E3C" if is_success else "gray")
        self.status_label.configure(text=text, text_color=color)

    def show_success_result(self, result: MergeResult) -> None:
        self.last_result = result
        self.open_folder_btn.configure(state="normal")
        self.play_video_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self.percent_label.configure(text="100%")
        self.set_status("اكتمل دمج الأجزاء بنجاح وبدون إعادة ترميز!", is_success=True)

        size_mb = result.output_path.stat().st_size / (1024 * 1024)
        lines = [
            "تم دمج أجزاء الفيديو بنجاح فائق (Stream Copy بدون أي فقدان للجودة)!",
            f"الملف المدمج: {result.output_path}",
            f"الحجم الإجمالي: {size_mb:.1f} MB",
            f"المدة الكلية: {format_duration(result.total_duration)} ({result.total_duration:.2f} ثانية)",
            f"عدد الأجزاء المدمجة: {result.parts_count} جزء",
            f"دقة فرق التوقيت: {result.duration_difference:.2f} ثانية (ضمن الحد المسموح 0.5s)",
        ]

        self.summary_textbox.configure(state="normal")
        self.summary_textbox.delete("1.0", "end")
        self.summary_textbox.insert("1.0", "\n".join(lines))
        self.summary_textbox.configure(state="disabled")
