"""Versioned JSON settings for the app (WP5).

Lives under %LOCALAPPDATA%\\VideoSplitterMerger\\settings.json. Atomic write,
tolerant load: unknown keys are ignored, invalid values fall back to defaults,
and a corrupt file never raises.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

APP_DIR_NAME = "VideoSplitterMerger"
SETTINGS_FILE_NAME = "settings.json"
SETTINGS_VERSION = 1
RTF_HISTORY_CAP = 8

DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "strength_db": 12.0,
    "auto_convert": True,
    "restore_original_rate": True,
    "denoise_dual_mono_shortcut": True,
    "deep_filter_path": None,
    "rtf_history": {"1": [], "2": []},
}


def settings_dir() -> Path:
    """%LOCALAPPDATA%\\VideoSplitterMerger, falling back to the user profile."""
    base = os.environ.get("LOCALAPPDATA")
    if not base:
        base = os.environ.get("USERPROFILE") or str(Path.home())
    return Path(base) / APP_DIR_NAME


def settings_path() -> Path:
    return settings_dir() / SETTINGS_FILE_NAME


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        low = value.strip().lower()
        if low in ("1", "true", "yes", "on"):
            return True
        if low in ("0", "false", "no", "off"):
            return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return default


def _coerce_strength(value: Any, default: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v in (float("inf"), float("-inf")):
        return default
    return min(100.0, max(1.0, v))


def _coerce_rtf_history(value: Any) -> Dict[str, list]:
    out: Dict[str, list] = {"1": [], "2": []}
    if not isinstance(value, dict):
        return out
    for key, seq in value.items():
        k = str(key).strip()
        if k not in ("1", "2"):
            continue
        if not isinstance(seq, (list, tuple)):
            continue
        vals = []
        for item in seq:
            try:
                v = float(item)
            except (TypeError, ValueError):
                continue
            if 0.0 < v < 10.0:
                vals.append(round(v, 6))
        out[k] = vals[-RTF_HISTORY_CAP:]
    return out


def sanitize(raw: Any) -> Dict[str, Any]:
    """Return a complete, valid settings dict built from `raw` (never raises)."""
    data = raw if isinstance(raw, dict) else {}
    return {
        "enabled": _coerce_bool(data.get("enabled"), DEFAULTS["enabled"]),
        "strength_db": _coerce_strength(data.get("strength_db"), DEFAULTS["strength_db"]),
        "auto_convert": _coerce_bool(data.get("auto_convert"), DEFAULTS["auto_convert"]),
        "restore_original_rate": _coerce_bool(
            data.get("restore_original_rate"), DEFAULTS["restore_original_rate"]
        ),
        "denoise_dual_mono_shortcut": _coerce_bool(
            data.get("denoise_dual_mono_shortcut"), DEFAULTS["denoise_dual_mono_shortcut"]
        ),
        "deep_filter_path": (
            str(data["deep_filter_path"])
            if isinstance(data.get("deep_filter_path"), str) and data["deep_filter_path"].strip()
            else None
        ),
        "rtf_history": _coerce_rtf_history(data.get("rtf_history")),
    }


def load(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load settings; missing or corrupt files yield defaults."""
    target = Path(path) if path else settings_path()
    try:
        text = target.read_text(encoding="utf-8")
    except FileNotFoundError:
        return sanitize({})
    except OSError as exc:
        logger.warning("تعذر قراءة ملف الإعدادات %s: %s", target, exc)
        return sanitize({})
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        logger.warning("ملف الإعدادات تالف (%s)؛ سيتم استخدام الإعدادات الافتراضية.", exc)
        return sanitize({})
    if not isinstance(raw, dict):
        logger.warning("ملف الإعدادات ليس كائناً JSON؛ سيتم استخدام الإعدادات الافتراضية.")
        return sanitize({})
    if raw.get("version") not in (None, SETTINGS_VERSION):
        logger.info("إصدار ملف الإعدادات %s غير معروف؛ تُقرأ المفاتيح المعروفة فقط.",
                    raw.get("version"))
    return sanitize(raw)


def save(settings: Dict[str, Any], path: Optional[Path] = None) -> bool:
    """Atomically write settings. Returns True on success; never raises."""
    target = Path(path) if path else settings_path()
    payload = {"version": SETTINGS_VERSION, **sanitize(settings)}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix="settings_", suffix=".tmp", dir=str(target.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False, indent=2)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_name, target)
        except BaseException:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return True
    except OSError as exc:
        logger.warning("تعذر حفظ ملف الإعدادات %s: %s", target, exc)
        return False
