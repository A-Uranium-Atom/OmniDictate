"""
test_main_gui.py
================
Headless GUI tests for OmniDictateApp using pytest-qt.

Design decisions
----------------
* The project uses PySide6, NOT PyQt6.  All imports use PySide6.
* OmniDictateApp.__init__ calls start_hotkey_listener() which creates a real
  pynput listener thread.  We patch HotkeyWorker.start_listening to a no-op
  so tests don't require keyboard hook privileges.
* mock_settings fixture (from conftest.py) must be requested explicitly to
  prevent QSettings from touching the user's registry during tests.
* Tests that only verify UI wiring avoid commented-out assertions; if a method
  or widget name is uncertain it is clearly marked TODO rather than left as a
  bare ``pass``.
"""

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from main_gui import OmniDictateApp, DEFAULT_MODEL_SIZE


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app_instance(qtbot, mock_settings, mocker):
    """
    Construct OmniDictateApp with all OS-level side-effects neutralised:
    - QSettings replaced with in-memory mock (via mock_settings fixture)
    - pynput listener start stubbed out so no global keyboard hook is created
    """
    mocker.patch("hotkey_listener.HotkeyWorker.start_listening")
    window = OmniDictateApp()
    qtbot.addWidget(window)
    return window


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

class TestInitialState:
    def test_window_title(self, app_instance):
        assert app_instance.windowTitle() == "OmniDictate"

    def test_window_minimum_size(self, app_instance):
        min_size = app_instance.minimumSize()
        assert min_size.width() >= 500
        assert min_size.height() >= 400

    def test_stop_button_initially_disabled(self, app_instance):
        assert not app_instance.stop_button.isEnabled()

    def test_start_button_initially_enabled(self, app_instance):
        assert app_instance.start_button.isEnabled()

    def test_vad_toggle_default_state(self, app_instance):
        """VAD mode is on by default."""
        assert app_instance.vad_toggle_button.isChecked() is True

    def test_model_combo_default(self, app_instance):
        assert app_instance.model_combo.currentText() == DEFAULT_MODEL_SIZE


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

class TestSettingsPersistence:
    def test_model_change_saved(self, app_instance):
        """Changing the model combo writes the value via save_settings."""
        app_instance.model_combo.setCurrentText("small")
        app_instance.save_settings()

        # load_settings re-reads into loaded_settings after save
        assert app_instance.loaded_settings["model_size"] == "small"

    def test_silence_threshold_saved(self, app_instance):
        app_instance.silence_spinbox.setValue(750)
        app_instance.save_settings()
        assert app_instance.loaded_settings["silence_threshold"] == 750

    def test_filter_word_added_and_saved(self, qtbot, app_instance):
        initial_count = app_instance.filter_list.count()
        app_instance.filter_add_edit.setText("unique test phrase")
        qtbot.mouseClick(app_instance.filter_add_button, Qt.LeftButton)
        assert app_instance.filter_list.count() == initial_count + 1

    def test_filter_word_not_duplicated(self, qtbot, app_instance):
        app_instance.filter_add_edit.setText("duplicate check")
        qtbot.mouseClick(app_instance.filter_add_button, Qt.LeftButton)
        count_after_first = app_instance.filter_list.count()

        # Adding the exact same word again must be rejected
        app_instance.filter_add_edit.setText("duplicate check")
        qtbot.mouseClick(app_instance.filter_add_button, Qt.LeftButton)
        assert app_instance.filter_list.count() == count_after_first


# ---------------------------------------------------------------------------
# Signal handling / transcription display
# ---------------------------------------------------------------------------

class TestSignalHandling:
    def test_handle_transcription_appends_to_display(self, app_instance):
        """handle_transcription (the real slot name) appends text to the display."""
        app_instance.handle_transcription("Hello world")
        assert "Hello world" in app_instance.transcription_display.toPlainText()

    def test_handle_transcription_multiple_calls(self, app_instance):
        """Multiple transcriptions accumulate in the display."""
        app_instance.handle_transcription("First sentence")
        app_instance.handle_transcription("Second sentence")
        content = app_instance.transcription_display.toPlainText()
        assert "First sentence" in content
        assert "Second sentence" in content

    def test_update_status_shown_in_status_bar(self, app_instance):
        app_instance.update_status("Listening...")
        assert app_instance.statusBar.currentMessage() == "Listening..."

    def test_visualizer_clamped_at_max(self, app_instance):
        """Amplitude values above 1000 must be clamped to 1000."""
        app_instance.update_visualizer(99999.0)
        assert app_instance.visualizer.value() == 1000

    def test_vad_toggle_updates_button_text(self, qtbot, app_instance):
        """Toggling VAD mode changes the button label."""
        app_instance.vad_toggle_button.setChecked(False)
        app_instance.update_vad_button_style()
        assert "PTT" in app_instance.vad_toggle_button.text()

        app_instance.vad_toggle_button.setChecked(True)
        app_instance.update_vad_button_style()
        assert "VAD" in app_instance.vad_toggle_button.text()
