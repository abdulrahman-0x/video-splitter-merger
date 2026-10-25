"""GUI integration and smoke tests for Video Splitter & Merger."""

from __future__ import annotations

import customtkinter as ctk
import pytest

from src.gui.app import App
from src.gui.merge_view import MergeView
from src.gui.split_view import SplitView


@pytest.fixture
def app_instance():
    """Create and tear down an App instance for testing."""
    app = App()
    app.update()
    yield app
    try:
        app.destroy()
    except Exception:
        pass


class TestGUI:
    """Test GUI components and interaction states."""

    def test_app_initialization_and_tabs(self, app_instance: App) -> None:
        assert app_instance.title() == "مقسم ومدمج الفيديو | Video Splitter & Merger"
        assert isinstance(app_instance.split_view, SplitView)
        assert isinstance(app_instance.merge_view, MergeView)

    def test_split_view_validation_without_file(self, app_instance: App) -> None:
        split_view = app_instance.split_view
        split_view.selected_file = None
        split_view._on_start_click()
        assert "يرجى اختيار ملف فيديو صالح أولاً" in split_view.status_label.cget("text")

    def test_merge_view_validation_without_target(self, app_instance: App) -> None:
        merge_view = app_instance.merge_view
        merge_view.selected_target = None
        merge_view._on_start_click()
        assert "يرجى اختيار مجلد أجزاء" in merge_view.status_label.cget("text")

    def test_split_view_controls_state_toggle(self, app_instance: App) -> None:
        split_view = app_instance.split_view

        # Set to running state
        split_view.set_running_state(True)
        assert split_view.start_btn.cget("state") == "disabled"
        assert split_view.cancel_btn.cget("state") == "normal"

        # Restore
        split_view.set_running_state(False)
        assert split_view.start_btn.cget("state") == "normal"
        assert split_view.cancel_btn.cget("state") == "disabled"

    def test_merge_view_controls_state_toggle(self, app_instance: App) -> None:
        merge_view = app_instance.merge_view

        # Set to running state
        merge_view.set_running_state(True)
        assert merge_view.start_btn.cget("state") == "disabled"
        assert merge_view.cancel_btn.cget("state") == "normal"

        # Restore
        merge_view.set_running_state(False)
        assert merge_view.start_btn.cget("state") == "normal"
        assert merge_view.cancel_btn.cget("state") == "disabled"
