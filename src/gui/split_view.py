"""Split View UI component for CustomTkinter."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path
from tkinter import filedialog
from typing import Callable, Optional

import customtkinter as ctk

from src.core.ffmpeg_manager import probe_media
from src.core.splitter import SplitResult
from src.core import denoise as denoise_module
from src.utils.helpers import (
    check_supported_format,
    format_duration,
    normalize_numeric_string,
    parse_duration_minutes,
)

logger = logging.getLogger(__name__)


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

        # 2b. Noise reduction (DeepFilterNet) section
        ctk.CTkLabel(
            options_frame,
            text="تقليل الضوضاء (DeepFilterNet):",
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        ).grid(row=3, column=0, sticky="w", padx=15, pady=(4, 4))

        self.denoise_enabled_var = ctk.BooleanVar(value=False)
        self.denoise_checkbox = ctk.CTkCheckBox(
            options_frame,
            text="تشغيل تقليل الضوضاء للأجزاء (بطيء: يعيد ترميز الصوت فقط، والفيديو يُنسخ كما هو)",
            variable=self.denoise_enabled_var,
            command=self._on_denoise_toggle,
            font=ctk.CTkFont(size=12),
        )
        self.denoise_checkbox.grid(row=4, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 4))

        strength_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        strength_row.grid(row=5, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 2))

        ctk.CTkLabel(
            strength_row,
            text="القوة (dB):",
            font=ctk.CTkFont(size=12),
            anchor="w",
        ).pack(side="left")

        self.denoise_strength_entry = ctk.CTkEntry(
            strength_row,
            placeholder_text="12",
            width=70,
            height=30,
        )
        self.denoise_strength_entry.insert(0, "12")
        self.denoise_strength_entry.pack(side="left", padx=(6, 10))

        # Presets carry numeric labels only; the measured numbers live in the
        # tooltip/status text (WP8). Noise-floor change / loudness change.
        self.preset_hints = {
            "6": "6 dB: أرضية الضوضاء -6.1 dB، الجهارة -4.0 LU",
            "12": "12 dB: أرضية الضوضاء -12.0 dB، الجهارة -4.1 LU",
            "20": "20 dB: أرضية الضوضاء -20.0 dB، الجهارة -5.5 LU",
            "30": "30 dB: أرضية الضوضاء -29.7 dB، الجهارة -6.3 LU",
            "100": "100 dB: أرضية الضوضاء -54.8 dB، الجهارة -6.5 LU",
        }
        for preset in (6, 12, 20, 30, 100):
            preset_btn = ctk.CTkButton(
                strength_row,
                text=str(preset),
                width=44,
                height=28,
                command=lambda value=str(preset): self._on_strength_preset(value),
            )
            preset_btn.pack(side="left", padx=(0, 6))
            self._attach_tooltip(preset_btn, self.preset_hints[str(preset)])

        ctk.CTkLabel(
            options_frame,
            text="القيم فوق 70 تكاد تشبه 100. جرّب عينة قصيرة أولاً.",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        ).grid(row=6, column=0, columnspan=2, sticky="w", padx=15, pady=(0, 4))

        toggles_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        toggles_row.grid(row=7, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 2))

        self.denoise_auto_convert_var = ctk.BooleanVar(value=True)
        self.denoise_auto_convert_cb = ctk.CTkCheckBox(
            toggles_row,
            text="تحويل تلقائي إلى 48 كيلوهرتز / 16-بت عند الحاجة",
            variable=self.denoise_auto_convert_var,
            command=self._on_denoise_toggle,
            font=ctk.CTkFont(size=12),
        )
        self.denoise_auto_convert_cb.pack(side="left")

        self.denoise_restore_rate_var = ctk.BooleanVar(value=True)
        self.denoise_restore_rate_cb = ctk.CTkCheckBox(
            toggles_row,
            text="إرجاع معدل العينات الأصلي بعد المعالجة",
            variable=self.denoise_restore_rate_var,
            font=ctk.CTkFont(size=12),
        )
        self.denoise_restore_rate_cb.pack(side="left", padx=(16, 0))

        # WP3: dual-mono shortcut (default on).
        self.denoise_dual_mono_var = ctk.BooleanVar(value=True)
        self.denoise_dual_mono_cb = ctk.CTkCheckBox(
            options_frame,
            text="اكتشاف القناتين المتطابقتين ومعالجتهما كقناة واحدة (أسرع وأقل ذاكرة)",
            variable=self.denoise_dual_mono_var,
            font=ctk.CTkFont(size=12),
        )
        self.denoise_dual_mono_cb.grid(row=8, column=0, columnspan=2, sticky="w", padx=15, pady=(2, 6))

        df_status_row = ctk.CTkFrame(options_frame, fg_color="transparent")
        df_status_row.grid(row=9, column=0, columnspan=2, sticky="ew", padx=15, pady=(0, 10))

        self.denoise_status_label = ctk.CTkLabel(
            df_status_row,
            text="جاري فحص أداة تقليل الضوضاء...",
            font=ctk.CTkFont(size=11),
            text_color="gray",
            anchor="w",
        )
        self.denoise_status_label.pack(side="left")

        self.denoise_browse_btn = ctk.CTkButton(
            df_status_row,
            text="استعراض للملف...",
            width=130,
            height=28,
            state="normal",
            command=self._on_browse_deep_filter,
        )
        self.denoise_browse_btn.pack(side="right")

        self.denoise_redetect_btn = ctk.CTkButton(
            df_status_row,
            text="إعادة كشف",
            width=100,
            height=28,
            command=self._on_redetect_deep_filter,
        )
        self.denoise_redetect_btn.pack(side="right", padx=(0, 6))

        self._detect_deep_filter()

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
            ("كافة وسائط الفيديو والصوت المدعومة", "*.mp4;*.mkv;*.mov;*.avi;*.wav;*.mp3;*.m4a;*.flac;*.aac;*.ogg;*.opus"),
            ("ملفات الصوت (WAV, MP3, M4A, FLAC, AAC...)", "*.wav;*.mp3;*.m4a;*.flac;*.aac;*.ogg;*.opus"),
            ("ملفات الفيديو (MP4, MKV, MOV, AVI)", "*.mp4;*.mkv;*.mov;*.avi"),
            ("ملفات WAV", "*.wav"),
            ("ملفات MP3", "*.mp3"),
            ("ملفات MP4", "*.mp4"),
            ("ملفات MKV", "*.mkv"),
            ("كافة الملفات", "*.*"),
        ]
        chosen = filedialog.askopenfilename(
            title="اختر ملف الفيديو أو الصوت للتقسيم",
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
            
            if probe.get("is_audio_only"):
                audio_s = probe["audio_streams"][0] if probe["audio_streams"] else {}
                a_codec = audio_s.get("codec_name", "audio").upper()
                a_rate = audio_s.get("sample_rate", "N/A")
                a_channels = audio_s.get("channels", "N/A")
                info_text = (
                    f"المدة: {format_duration(self.media_duration)} | الحجم: {size_mb:.1f} MB | "
                    f"صوت {a_codec} ({a_rate} Hz, {a_channels}ch){best_effort_note}"
                )
            else:
                info_text = (
                    f"المدة: {format_duration(self.media_duration)} | "
                    f"الحجم: {size_mb:.1f} MB | الصيغة: {chosen_path.suffix.upper()}{best_effort_note}"
                )

            self.file_info_label.configure(
                text=info_text,
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

    # ------- DeepFilterNet (noise reduction) controls -------

    def _detect_deep_filter(self, override: Optional[str] = None) -> Optional[Path]:
        """Locate the tool, update the status line, and enable/disable the feature."""
        self.deep_filter_exe: Optional[Path] = denoise_module.find_deep_filter(override)
        if self.deep_filter_exe:
            version = denoise_module.get_version(self.deep_filter_exe)
            self.denoise_status_label.configure(
                text=f"تم العثور على الأداة {version or 'بإصدار غير معروف'} في {self.deep_filter_exe}",
                text_color="gray",
            )
            self.denoise_checkbox.configure(state="normal")
        else:
            self.denoise_status_label.configure(
                text="لم يتم العثور على deep-filter.exe؛ تقليل الضوضاء غير متاح (ضع الملف في PATH أو حدد مساراً).",
                text_color="#FFA000",
            )
            self.denoise_enabled_var.set(False)
            self.denoise_checkbox.configure(state="disabled")
        self._on_denoise_toggle()
        return self.deep_filter_exe

    def _on_redetect_deep_filter(self) -> None:
        self._detect_deep_filter(None)

    def _on_browse_deep_filter(self) -> None:
        chosen = filedialog.askopenfilename(
            title="اختر ملف deep-filter.exe",
            filetypes=[("الأدوات التنفيذية", "*.exe"), ("كافة الملفات", "*.*")],
        )
        if chosen:
            self._detect_deep_filter(chosen)

    def _on_denoise_toggle(self) -> None:
        if not getattr(self, "denoise_restore_rate_cb", None):
            return
        if self.denoise_enabled_var.get():
            rate_state = "normal" if self.denoise_auto_convert_var.get() else "disabled"
            self.denoise_restore_rate_cb.configure(state=rate_state)
        else:
            self.denoise_restore_rate_cb.configure(state="disabled")

    def _on_strength_preset(self, value: str) -> None:
        self.denoise_strength_entry.delete(0, "end")
        self.denoise_strength_entry.insert(0, value)

    def _attach_tooltip(self, widget: ctk.CTkBaseClass, text: str) -> None:
        """Minimal hover tooltip (no extra dependency)."""
        tip: dict = {"win": None}

        def show(_event=None) -> None:
            if tip["win"] is not None:
                return
            win = ctk.CTkToplevel(widget)
            win.wm_overrideredirect(True)
            win.attributes("-topmost", True)
            label = ctk.CTkLabel(
                win, text=text, font=ctk.CTkFont(size=11),
                fg_color="#2b2b2b", text_color="#f0f0f0", corner_radius=4,
            )
            label.pack(padx=6, pady=4)
            x = widget.winfo_rootx() + 10
            y = widget.winfo_rooty() - 34
            win.geometry(f"+{x}+{y}")
            tip["win"] = win

        def hide(_event=None) -> None:
            win = tip.get("win")
            if win is not None:
                try:
                    win.destroy()
                except Exception:  # noqa: BLE001
                    pass
                tip["win"] = None

        widget.bind("<Enter>", show, add="+")
        widget.bind("<Leave>", hide, add="+")
        widget.bind("<ButtonPress>", hide, add="+")

    def apply_settings(self, data: dict) -> None:
        """Push persisted settings into the widgets (WP5)."""
        try:
            self.denoise_enabled_var.set(bool(data.get("enabled", False)))
            self.denoise_auto_convert_var.set(bool(data.get("auto_convert", True)))
            self.denoise_restore_rate_var.set(bool(data.get("restore_original_rate", True)))
            self.denoise_dual_mono_var.set(bool(data.get("denoise_dual_mono_shortcut", True)))
            self.denoise_strength_entry.delete(0, "end")
            self.denoise_strength_entry.insert(0, f"{float(data.get('strength_db', 12.0)):g}")
            path = data.get("deep_filter_path")
            # Re-detect even without a saved path: detection already ran during
            # _build_ui, but setting `enabled` above would otherwise leave the
            # checkbox checked-and-disabled when the tool is not on PATH.
            self._detect_deep_filter(str(path) if path else None)
            self._on_denoise_toggle()
        except Exception as exc:  # noqa: BLE001 - never fail startup over settings
            logger.warning("تعذر تطبيق الإعدادات المحفوظة: %s", exc)

    def get_denoise_settings(self) -> dict:
        """Collect checkbox state into the settings dict consumed by src.core.denoise."""
        strength_text = normalize_numeric_string(self.denoise_strength_entry.get() or "")
        strength = 12.0
        strength_text = strength_text.strip()
        if strength_text:
            try:
                strength = float(strength_text)
            except ValueError:
                strength = float("nan")
        return {
            "enabled": bool(self.denoise_enabled_var.get()),
            "strength_db": strength,
            "auto_convert": bool(self.denoise_auto_convert_var.get()),
            "restore_original_rate": bool(self.denoise_restore_rate_var.get()),
            "denoise_dual_mono_shortcut": bool(self.denoise_dual_mono_var.get()),
            # Kept in the API for programmatic callers; the UI no longer exposes
            # it because 2 workers measured only 1.25x faster at 2x the RAM.
            "workers": 1,
            "deep_filter_path": str(self.deep_filter_exe) if self.deep_filter_exe else None,
        }

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

        denoise_settings = self.get_denoise_settings()
        if denoise_settings["enabled"]:
            if not self.deep_filter_exe:
                self.set_status(
                    "تعذر بدء العملية: تقليل الضوضاء مفعل لكن أداة deep-filter غير موجودة.",
                    is_error=True,
                )
                return
            # Reject NaN and +/-inf here too: `denoise_wav` refuses them only
            # after the whole split has run, so every part would fail.
            if not (0 < denoise_settings["strength_db"] < float("inf")):
                self.set_status(
                    "قوة تقليل الضوضاء غير صالحة: أدخل قيمة بين 1 و 100 (أو ألغِ تفعيل الخيار).",
                    is_error=True,
                )
                return
            above = denoise_settings["strength_db"] > 100
            if above:
                self.set_status(
                    "قوة تقليل الضوضاء أكبر من 100؛ سيتم استخدام 100 (الحد الأقصى للتأثير).",
                    is_error=False,
                )

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
        # DeepFilterNet controls
        tool_found = self.deep_filter_exe is not None
        denoise_state = state if tool_found else "disabled"
        self.denoise_checkbox.configure(state=denoise_state)
        self.denoise_auto_convert_cb.configure(state=denoise_state)
        self.denoise_dual_mono_cb.configure(state=denoise_state)
        self.denoise_strength_entry.configure(state=denoise_state)
        self.denoise_browse_btn.configure(state=denoise_state)
        self.denoise_redetect_btn.configure(state=denoise_state)
        if is_running or not self.denoise_enabled_var.get():
            self.denoise_restore_rate_cb.configure(state="disabled")
        else:
            self.denoise_restore_rate_cb.configure(
                state="normal" if self.denoise_auto_convert_var.get() else "disabled"
            )

    def update_progress(self, percent: float, current_time_sec: float) -> None:
        """Update progress bar and percentage display."""
        norm = max(0.0, min(1.0, percent / 100.0))
        self.progress_bar.set(norm)
        self.percent_label.configure(text=f"{percent:.1f}%")
        self.status_label.configure(
            text=f"جاري التقسيم... التوقيت المعالج: {format_duration(current_time_sec)}",
            text_color="gray",
        )

    def set_denoise_progress(self, percent: float, message: str) -> None:
        """Show noise-reduction progress (estimated, the tool emits none)."""
        norm = max(0.0, min(1.0, percent / 100.0))
        self.progress_bar.set(norm)
        self.percent_label.configure(text=f"{percent:.1f}%")
        self.status_label.configure(
            text=f"جاري تقليل الضوضاء (DeepFilterNet)... {message}",
            text_color="gray",
        )

    def set_status(self, text: str, is_error: bool = False, is_success: bool = False) -> None:
        color = "#D32F2F" if is_error else ("#388E3C" if is_success else "gray")
        self.status_label.configure(text=text, text_color=color)

    def show_success_result(
        self,
        result: SplitResult,
        extra_lines: Optional[list[str]] = None,
        denoise_warning: bool = False,
    ) -> None:
        self.last_result = result
        self.open_folder_btn.configure(state="normal")
        self.progress_bar.set(1.0)
        self.percent_label.configure(text="100%")
        media_name = "الملف الصوتي" if result.is_audio_only else "الفيديو"
        if denoise_warning:
            # The split succeeded; only the noise reduction had problems (WP6).
            self.set_status(
                f"اكتمل تقسيم {media_name}، لكن تقليل الضوضاء لم يكتمل لكل الأجزاء.",
                is_error=False,
            )
        else:
            self.set_status(f"اكتمل تقسيم {media_name} بنجاح!", is_success=True)

        lines = [
            f"تم تقسيم {media_name} بنجاح إلى {len(result.parts)} جزء/أجزاء.",
            f"مجلد الحفظ: {result.parts_dir}",
            f"ملف قائمة الدمج التلقائية: {result.parts_list_file.name}",
            "",
            "الأجزاء المنتجة والمدد الفعلية:",
        ]
        for idx, part in enumerate(result.parts, start=1):
            lines.append(f"  الجزء {idx:02d}: {part.file_path.name} | المدة: {format_duration(part.duration)}")

        lines.append("")
        if extra_lines:
            lines.extend(extra_lines)

        if result.has_deviation_warning:
            lines.append("")
            lines.append("=" * 60)
            lines.append(f"تنبيه: {result.warning_message}")
            lines.append("السبب: التقسيم بدون إعادة ترميز (Stream Copy) يعتمد على التوقيتات وحزم الإطارات الأصلية.")
            lines.append("=" * 60)

        self.summary_textbox.configure(state="normal")
        self.summary_textbox.delete("1.0", "end")
        self.summary_textbox.insert("1.0", "\n".join(lines))
        self.summary_textbox.configure(state="disabled")
