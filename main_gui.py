import sys
import time
from pathlib import Path

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import (
    QIcon,
    QTextCursor,
)
from PySide6.QtWidgets import (
    QApplication,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

try:
    from core.dictation_worker import DictationWorker
except ImportError as e:
    print(f"Error: Could not import DictationWorker: {e}")
    sys.exit(1)
try:
    from hotkey_listener import HotkeyWorker
except ImportError as e:
    print(f"Error: Could not import from hotkey_listener.py: {e}")
    sys.exit(1)
from ui import DictationPage, SettingsPage
from ui.icons import format_key_name

CONFIG_ORG = "OmniCorp"
CONFIG_APP = "OmniDictate"
DEFAULT_MODEL_SIZE = "large-v3"
DEFAULT_LANGUAGE = None
DEFAULT_VAD_ENABLED = True
DEFAULT_SILENCE_THRESHOLD = 500
DEFAULT_CHAR_DELAY = 0.02
DEFAULT_PTT_KEY_STR = "keyboard.Key.shift_r"
DEFAULT_RMS_THRESHOLD = 0.01
DEFAULT_HALLUCINATION_FILTER = "Medium"
DEFAULT_INSERTION_METHOD = "Paste"
DEFAULT_PASTE_DELAY = 0.3
DEFAULT_FILTER_WORDS = [
    "thank you",
    "thanks for watching",
    "thanks for listening",
    "i'm sorry",
    "subtitles by",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "you",
]


class OmniDictateApp(QMainWindow):
    """Main application window for OmniDictate."""

    ptt_signal = Signal(bool)
    settings_updated_signal = Signal(dict)

    def __init__(self) -> None:
        """Initialize the main window, load settings, and wire all signals."""
        super().__init__()
        self.setWindowTitle("OmniDictate")
        self.resize(800, 600)
        self.setMinimumSize(500, 400)
        self.settings = QSettings(CONFIG_ORG, CONFIG_APP)
        self.dictation_thread = None
        self.dictation_worker = None
        self.hotkey_thread = None
        self.hotkey_worker = None
        self.capture_hotkey_thread = None
        self.capture_hotkey_worker = None
        self.is_dictation_running = False
        self.setting_key_for = None
        self.original_button_text = ""
        self._is_stopping = False
        self.last_start_click_time = 0
        self.load_settings()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget()
        self.main_layout.addWidget(self.stack)
        self.dictation_page = DictationPage(
            model_display_text=f"Model: {self.loaded_settings.get('model_size', DEFAULT_MODEL_SIZE)}",
            is_vad_checked=self.loaded_settings.get("vad_enabled", DEFAULT_VAD_ENABLED),
        )
        self.settings_page = SettingsPage(loaded_settings=self.loaded_settings)

        self.stack.addWidget(self.dictation_page)
        self.stack.addWidget(self.settings_page)
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Ready")
        self.statusBar.hide()
        self.dictation_page.start_button.clicked.connect(self.start_dictation)
        self.dictation_page.stop_button.clicked.connect(self.stop_dictation)
        self.dictation_page.vad_toggle_button.clicked.connect(self.toggle_vad)
        self.dictation_page.settings_button.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.settings_page)
        )
        self.settings_page.back_button.clicked.connect(
            lambda: self.stack.setCurrentWidget(self.dictation_page)
        )
        self.dictation_page.copy_button.clicked.connect(self.copy_transcription)
        self.settings_page.filter_add_button.clicked.connect(self.add_filter_word)
        self.settings_page.filter_remove_button.clicked.connect(self.remove_filter_word)
        self.settings_page.set_ptt_key_button.clicked.connect(
            lambda: self.prepare_to_set_key("ptt")
        )
        self.settings_page.restore_defaults_button.clicked.connect(
            self.restore_default_settings
        )
        self.settings_page.model_combo.currentTextChanged.connect(self.save_settings)
        self.settings_page.language_combo.currentTextChanged.connect(self.save_settings)
        self.settings_page.silence_spinbox.valueChanged.connect(self.save_settings)
        self.settings_page.delay_spinbox.valueChanged.connect(self.save_settings)
        self.settings_page.rms_spinbox.valueChanged.connect(self.save_settings)
        self.settings_page.hallucination_combo.currentTextChanged.connect(
            self.save_settings
        )
        self.settings_page.insertion_combo.currentTextChanged.connect(
            self.save_settings
        )
        self.settings_page.paste_delay_spinbox.valueChanged.connect(self.save_settings)
        self.start_hotkey_listener()
        self.update_vad_button_style()
        print("GUI Initialized.")

    def load_settings(self) -> None:
        """Read persisted QSettings values into ``self.loaded_settings`` dict."""
        self.loaded_settings = {
            "model_size": self.settings.value("model_size", DEFAULT_MODEL_SIZE),
            "language": self.settings.value("language", DEFAULT_LANGUAGE),
            "vad_enabled": self.settings.value(
                "vad_enabled", DEFAULT_VAD_ENABLED, type=bool
            ),
            "silence_threshold": self.settings.value(
                "silence_threshold", DEFAULT_SILENCE_THRESHOLD, type=int
            ),
            "char_delay": self.settings.value(
                "char_delay", DEFAULT_CHAR_DELAY, type=float
            ),
            "ptt_key_str": self.settings.value("ptt_key_str", DEFAULT_PTT_KEY_STR),
            "rms_threshold": self.settings.value(
                "rms_threshold", DEFAULT_RMS_THRESHOLD, type=float
            ),
            "hallucination_filter": self.settings.value(
                "hallucination_filter", DEFAULT_HALLUCINATION_FILTER
            ),
            "insertion_method": self.settings.value(
                "insertion_method", DEFAULT_INSERTION_METHOD
            ),
            "paste_delay": self.settings.value(
                "paste_delay", DEFAULT_PASTE_DELAY, type=float
            ),
            "filter_words": self.settings.value("filter_words", DEFAULT_FILTER_WORDS),
        }
        valid_models = ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
        if self.loaded_settings["model_size"] not in valid_models:
            print(
                f"Warning: Invalid model size '{self.loaded_settings['model_size']}', defaulting to {DEFAULT_MODEL_SIZE}"
            )
            self.loaded_settings["model_size"] = DEFAULT_MODEL_SIZE
        valid_languages = [
            None,
            "",
            "None",
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "nl",
            "ru",
            "zh",
            "ja",
        ]
        if self.loaded_settings["language"] not in valid_languages:
            print(
                f"Warning: Invalid language '{self.loaded_settings['language']}', defaulting to {DEFAULT_LANGUAGE}"
            )
            self.loaded_settings["language"] = DEFAULT_LANGUAGE
        if not isinstance(self.loaded_settings["filter_words"], list):
            self.loaded_settings["filter_words"] = DEFAULT_FILTER_WORDS
        print("Settings loaded:", self.loaded_settings)

    def save_settings(self) -> None:
        """Read widget values, persist to QSettings, and propagate to the worker."""
        if self.setting_key_for:
            return
        print("Saving settings...")
        self.settings.setValue(
            "model_size", self.settings_page.model_combo.currentText()
        )
        idx = self.settings_page.language_combo.currentIndex()
        lang_code = (
            self.settings_page.language_combo.itemData(idx)
            if idx != -1
            else DEFAULT_LANGUAGE
        )
        self.settings.setValue("language", lang_code)
        self.settings.setValue(
            "vad_enabled", self.dictation_page.vad_toggle_button.isChecked()
        )
        self.settings.setValue(
            "silence_threshold", self.settings_page.silence_spinbox.value()
        )
        self.settings.setValue("char_delay", self.settings_page.delay_spinbox.value())
        self.settings.setValue(
            "ptt_key_str", self.loaded_settings.get("ptt_key_str", DEFAULT_PTT_KEY_STR)
        )
        self.settings.setValue("rms_threshold", self.settings_page.rms_spinbox.value())
        self.settings.setValue(
            "hallucination_filter", self.settings_page.hallucination_combo.currentText()
        )
        self.settings.setValue(
            "insertion_method", self.settings_page.insertion_combo.currentText()
        )
        self.settings.setValue(
            "paste_delay", self.settings_page.paste_delay_spinbox.value()
        )
        filter_words = [
            self.settings_page.filter_list.item(i).text()
            for i in range(self.settings_page.filter_list.count())
        ]
        self.settings.setValue("filter_words", filter_words)
        self.settings.sync()
        print("Settings saved.")
        self.load_settings()
        self.dictation_page.model_display_label.setText(
            f"Model: {self.loaded_settings['model_size']}"
        )
        if not self.is_dictation_running:
            self.restart_hotkey_listener()
        if self.dictation_worker:
            self.settings_updated_signal.emit(
                {
                    "model_size": self.loaded_settings["model_size"],
                    "language": self.loaded_settings["language"],
                    "silence_threshold": self.loaded_settings["silence_threshold"],
                    "char_delay": self.loaded_settings["char_delay"],
                    "vad_enabled": self.loaded_settings["vad_enabled"],
                    "filter_words": self.loaded_settings.get("filter_words", []),
                    "rms_threshold": self.loaded_settings["rms_threshold"],
                    "hallucination_filter": self.loaded_settings[
                        "hallucination_filter"
                    ],
                    "insertion_method": self.loaded_settings["insertion_method"],
                    "paste_delay": self.loaded_settings["paste_delay"],
                }
            )

    @Slot()
    def restore_default_settings(self) -> None:
        """Restore all settings to their default values and save them."""
        print("Restoring default settings...")
        self.settings_page.model_combo.setCurrentText(DEFAULT_MODEL_SIZE)
        index = self.settings_page.language_combo.findData(DEFAULT_LANGUAGE)
        self.settings_page.language_combo.setCurrentIndex(index if index != -1 else 0)
        self.dictation_page.vad_toggle_button.setChecked(DEFAULT_VAD_ENABLED)
        self.settings_page.silence_spinbox.setValue(DEFAULT_SILENCE_THRESHOLD)
        self.settings_page.delay_spinbox.setValue(DEFAULT_CHAR_DELAY)
        self.settings_page.rms_spinbox.setValue(DEFAULT_RMS_THRESHOLD)
        self.settings_page.hallucination_combo.setCurrentText(
            DEFAULT_HALLUCINATION_FILTER
        )
        self.settings_page.insertion_combo.setCurrentText(DEFAULT_INSERTION_METHOD)
        self.settings_page.paste_delay_spinbox.setValue(DEFAULT_PASTE_DELAY)
        self.settings.setValue("ptt_key_str", DEFAULT_PTT_KEY_STR)
        self.settings_page.ptt_key_display_label.setText(
            format_key_name(DEFAULT_PTT_KEY_STR)
        )
        self.settings_page.filter_list.clear()
        self.settings_page.filter_list.addItems(DEFAULT_FILTER_WORDS)
        self.update_vad_button_style()
        self.save_settings()
        QMessageBox.information(
            self, "Settings Restored", "Default settings restored and saved."
        )

    def add_filter_word(self) -> None:
        """Append the text in ``filter_add_edit`` to the filter word list if unique."""
        word = self.settings_page.filter_add_edit.text().strip()
        if word and (
            not self.settings_page.filter_list.findItems(
                word, Qt.MatchFlag.MatchExactly
            )
        ):
            self.settings_page.filter_list.addItem(QListWidgetItem(word))
            self.settings_page.filter_add_edit.clear()
            self.save_settings()

    def remove_filter_word(self) -> None:
        """Remove all selected items from the filter word list."""
        items = self.settings_page.filter_list.selectedItems()
        if not items:
            return
        for item in items:
            self.settings_page.filter_list.takeItem(
                self.settings_page.filter_list.row(item)
            )
        self.save_settings()

    def prepare_to_set_key(self, key_type: str) -> None:
        """Enter key-capture mode for the given setting type (e.g. ``"ptt"``)."""
        if self.is_dictation_running:
            QMessageBox.warning(self, "Warning", "Stop dictation first.")
            return
        if self.setting_key_for:
            QMessageBox.warning(
                self, "Warning", f"Already waiting for {self.setting_key_for} key."
            )
            return
        self.setting_key_for = key_type
        button = self.settings_page.set_ptt_key_button
        self.original_button_text = button.text()
        button.setText("Press Key...")
        button.setProperty("waitingInput", True)
        self.style().polish(button)
        self.set_other_controls_enabled(False)
        self.stop_hotkey_listener()
        self.capture_hotkey_thread = QThread(self)
        self.capture_hotkey_worker = HotkeyWorker(is_capture_mode=True)
        self.capture_hotkey_worker.moveToThread(self.capture_hotkey_thread)
        self.capture_hotkey_worker.key_captured_signal.connect(self.handle_key_capture)
        self.capture_hotkey_worker.error_signal.connect(self.handle_key_capture_error)
        self.capture_hotkey_thread.finished.connect(
            self.capture_hotkey_worker.deleteLater
        )
        self.capture_hotkey_thread.finished.connect(
            self.capture_hotkey_thread.deleteLater
        )
        self.capture_hotkey_thread.started.connect(
            self.capture_hotkey_worker.start_listening
        )
        self.capture_hotkey_thread.start()

    @Slot(object, str)
    def handle_key_capture(self, key_obj: object, key_str: str) -> None:
        """Store the captured key and finalize key-capture mode."""
        print(f"Captured {self.setting_key_for} key: {key_str}")
        if self.setting_key_for == "ptt":
            self.loaded_settings["ptt_key_str"] = key_str
            self.settings_page.ptt_key_display_label.setText(format_key_name(key_str))
        self.finish_setting_key()
        self.save_settings()

    @Slot(str)
    def handle_key_capture_error(self, error_msg: str) -> None:
        """Display a warning dialog and cancel key-capture mode on failure."""
        QMessageBox.warning(self, "Hotkey Error", f"Could not capture key: {error_msg}")
        self.finish_setting_key()
        self.start_hotkey_listener()

    def finish_setting_key(self) -> None:
        """Exit key-capture mode, restore button labels, and restart hotkey listener."""
        if not self.setting_key_for:
            return
        button = self.settings_page.set_ptt_key_button
        button.setText(
            self.original_button_text
            if hasattr(self, "original_button_text") and self.original_button_text
            else "Change"
        )
        button.setProperty("waitingInput", False)
        self.style().polish(button)
        self.setting_key_for = None
        self.set_other_controls_enabled(True)
        if self.capture_hotkey_worker:
            self.capture_hotkey_worker.stop_listening()
        if self.capture_hotkey_thread and self.capture_hotkey_thread.isRunning():
            self.capture_hotkey_thread.quit()
            self.capture_hotkey_thread.wait(500)
        self.capture_hotkey_worker = None
        self.capture_hotkey_thread = None

    def set_other_controls_enabled(self, enabled: bool) -> None:
        """Enable or disable start/stop and settings widgets during key capture."""
        self.dictation_page.start_button.setEnabled(
            enabled and (not self.is_dictation_running)
        )
        self.dictation_page.stop_button.setEnabled(
            enabled and self.is_dictation_running
        )
        self.set_config_enabled(enabled)
        self.settings_page.set_ptt_key_button.setEnabled(enabled)

    @Slot()
    def toggle_vad(self) -> None:
        """Toggle VAD/PTT mode from the button click and persist the setting."""
        is_checked = self.dictation_page.vad_toggle_button.isChecked()
        self.update_vad_button_style()
        self.save_settings()
        if self.dictation_worker and self.is_dictation_running:
            self.dictation_worker.set_vad_enabled(is_checked)

    def update_vad_button_style(self) -> None:
        """Update VAD button styling based on state."""
        is_checked = self.dictation_page.vad_toggle_button.isChecked()
        ptt_key_name = format_key_name(
            self.loaded_settings.get("ptt_key_str", DEFAULT_PTT_KEY_STR)
        )
        if is_checked:
            self.dictation_page.vad_toggle_button.setText("VAD: ON")
        else:
            self.dictation_page.vad_toggle_button.setText("PTT: ON")
        if not self.is_dictation_running:
            self.dictation_page.hint_label.setText(
                "Select PTT or VAD mode and click Start"
            )
            self.dictation_page.hint_label.setStyleSheet(
                "color: #888; font-style: italic;"
            )
        elif is_checked:
            self.dictation_page.hint_label.setText("Listening for speech...")
            self.dictation_page.hint_label.setStyleSheet(
                "color: #0A84FF; font-style: italic;"
            )
        else:
            self.dictation_page.hint_label.setText(f"Hold '{ptt_key_name}' to speak")
            self.dictation_page.hint_label.setStyleSheet(
                "color: #0A84FF; font-style: italic;"
            )
        self.dictation_page.vad_toggle_button.style().unpolish(
            self.dictation_page.vad_toggle_button
        )
        self.dictation_page.vad_toggle_button.style().polish(
            self.dictation_page.vad_toggle_button
        )

    @Slot(str)
    def update_status(self, status_text: str) -> None:
        """Update the status bar text."""
        self.statusBar.showMessage(status_text)
        if "Listening" in status_text:
            if self.last_start_click_time > 0:
                print(
                    f"Startup latency: {time.time() - self.last_start_click_time:.2f}s"
                )
                self.last_start_click_time = 0
            self.update_vad_button_style()

    @Slot(str)
    def handle_transcription(self, text: str) -> None:
        """Append transcribed text to the display."""
        current_text = self.dictation_page.transcription_display.toPlainText()
        prefix = (
            "\n" if current_text and (not current_text.endswith(("\n", " "))) else ""
        )
        if prefix == "" and current_text and (not current_text.endswith(" ")):
            prefix = " "
        self.dictation_page.transcription_display.insertPlainText(prefix + text.strip())
        self.dictation_page.transcription_display.moveCursor(QTextCursor.End)

    @Slot(float)
    def update_visualizer(self, amplitude: float) -> None:
        """Update the audio visualizer progress bar."""
        val = int(amplitude)
        if val > 1000:
            val = 1000
        self.dictation_page.visualizer.setValue(val)

    @Slot(str)
    def show_error(self, error_text: str) -> None:
        """Display error messages."""
        print(f"GUI Error: {error_text}")
        self.update_status("Error")
        QMessageBox.critical(self, "OmniDictate Error", error_text)
        if self.is_dictation_running:
            self.stop_dictation()
        else:
            self.reset_ui_after_stop()

    @Slot(str)
    def show_warning(self, warning_text: str) -> None:
        """Display warning messages."""
        print(f"GUI Warning: {warning_text}")
        self.statusBar.showMessage(f"Warning: {warning_text}", 5000)

    @Slot()
    def copy_transcription(self) -> None:
        """Copy transcription to clipboard."""
        clipboard = QApplication.clipboard()
        text_to_copy = self.dictation_page.transcription_display.toPlainText()
        clipboard.setText(text_to_copy)
        self.statusBar.showMessage("Transcription copied to clipboard!", 2000)

    def _ensure_worker_created(self) -> None:
        """Create the worker and thread once. They persist until app close."""
        if self.dictation_thread and self.dictation_worker:
            return
        if self.dictation_thread and self.dictation_thread.isRunning():
            print("Error: Previous thread still running. Aborting.")
            QMessageBox.critical(
                self,
                "Error",
                "Previous dictation process still running. Please wait or restart.",
            )
            return
        self.dictation_worker = None
        self.dictation_thread = None
        print("Creating persistent dictation worker and thread...")
        self.dictation_thread = QThread(self)
        self.dictation_worker = DictationWorker(
            gui_wid=int(self.winId()),
            model_size=self.loaded_settings["model_size"],
            language=self.loaded_settings["language"],
            vad_enabled=self.loaded_settings["vad_enabled"],
            silence_threshold=self.loaded_settings["silence_threshold"],
            silence_duration=0.5,
            char_delay=self.loaded_settings["char_delay"],
            filter_words=self.loaded_settings["filter_words"],
            rms_threshold=self.loaded_settings["rms_threshold"],
            hallucination_filter=self.loaded_settings["hallucination_filter"],
            insertion_method=self.loaded_settings["insertion_method"],
            paste_delay=self.loaded_settings["paste_delay"],
        )
        self.dictation_worker.moveToThread(self.dictation_thread)
        self.dictation_worker.status_updated.connect(self.update_status)
        self.dictation_worker.transcription_ready.connect(self.handle_transcription)
        self.dictation_worker.error_occurred.connect(self.show_error)
        self.dictation_worker.warning_occurred.connect(self.show_warning)
        self.dictation_worker.audio_level.connect(self.update_visualizer)
        self.dictation_worker.auto_restart_requested.connect(self._handle_auto_restart)
        self.ptt_signal.connect(self.dictation_worker.set_ptt_state)
        self.settings_updated_signal.connect(self.dictation_worker.update_settings)
        self.dictation_thread.finished.connect(self.dictation_worker.deleteLater)
        self.dictation_thread.finished.connect(self.dictation_thread.deleteLater)
        self.dictation_thread.finished.connect(self._on_worker_destroyed)
        self.dictation_thread.start()
        print("Persistent worker thread started.")

    def start_dictation(self) -> None:
        """Start the dictation process."""
        if self.is_dictation_running:
            print("Dictation is already running.")
            return
        self.last_start_click_time = time.time()
        self.save_settings()
        print(
            f"Attempting to start dictation with model: {self.loaded_settings['model_size']}"
        )
        self._ensure_worker_created()
        if not self.dictation_worker:
            return
        from PySide6.QtCore import QMetaObject

        QMetaObject.invokeMethod(self.dictation_worker, "start_processing")
        self.update_status("Initializing...")
        self.dictation_page.start_button.setEnabled(False)
        self.dictation_page.stop_button.setEnabled(True)
        self.set_config_enabled(False)
        self.is_dictation_running = True
        self.update_vad_button_style()
        print("Dictation started.")

    def stop_dictation(self) -> None:
        """Stop the dictation process."""
        if (
            not self.is_dictation_running
            and self.dictation_page.start_button.isEnabled()
        ):
            print("Stop called but already stopped.")
            return
        if self._is_stopping:
            return
        self._is_stopping = True
        print("GUI requesting stop...")
        self.update_status("Stopping...")
        self.dictation_page.stop_button.setEnabled(False)
        if self.dictation_worker:
            from PySide6.QtCore import QMetaObject

            QMetaObject.invokeMethod(self.dictation_worker, "stop_processing")
        self.is_dictation_running = False
        self.reset_ui_after_stop()
        self.dictation_page.visualizer.setValue(0)
        self.update_vad_button_style()
        self._is_stopping = False
        print("Dictation stopped. Worker and model remain in memory.")

    @Slot()
    def _handle_auto_restart(self) -> None:
        """Called when the worker detects a stream failure (typically after sleep/wake).

        Performs a full Stop → delayed Start cycle to allow Windows audio drivers to
        fully reinitialize before the new stream is opened.

        The 5-second delay is intentional: Windows AudioSrv needs time to re-apply
        microphone volume/boost settings to any new WASAPI session after wake.
        A shorter delay causes the stream to open during a zero-gain window.
        """
        if not self.is_dictation_running:
            return
        print("GUI: Auto-restart triggered. Stopping dictation...")
        self.update_status("Recovering audio after sleep...")
        self.stop_dictation()
        driver_settle_ms = 5000
        print(f"GUI: Will restart dictation in {driver_settle_ms // 1000}s...")
        QTimer.singleShot(driver_settle_ms, self._auto_restart_start)

    @Slot()
    def _auto_restart_start(self) -> None:
        """Deferred callback that performs the Start half of auto-restart."""
        if self.is_dictation_running:
            return
        print("GUI: Auto-restart: starting dictation now.")
        self.start_dictation()

    def _destroy_worker(self) -> None:
        """Fully destroy the persistent worker and thread (for app close)."""
        if self.dictation_worker:
            from PySide6.QtCore import QMetaObject

            QMetaObject.invokeMethod(self.dictation_worker, "stop_processing")
        if self.dictation_thread and self.dictation_thread.isRunning():
            self.dictation_thread.quit()
            if not self.dictation_thread.wait(2000):
                print("Warning: Dictation thread didn't finish quitting.")

    @Slot()
    def _on_worker_destroyed(self) -> None:
        print("Dictation worker/thread destroyed.")
        self.dictation_worker = None
        self.dictation_thread = None

    def reset_ui_after_stop(self) -> None:
        """Reset UI elements after dictation stops."""
        self.dictation_page.start_button.setEnabled(True)
        self.dictation_page.stop_button.setEnabled(False)
        self.set_config_enabled(True)
        self.update_status("Idle")

    def set_config_enabled(self, enabled: bool) -> None:
        """Enable or disable configuration UI."""
        self.settings_page.model_combo.setEnabled(enabled)
        self.settings_page.language_combo.setEnabled(enabled)
        self.settings_page.silence_spinbox.setEnabled(enabled)
        self.settings_page.delay_spinbox.setEnabled(enabled)
        self.settings_page.filter_list.setEnabled(enabled)
        self.settings_page.filter_add_edit.setEnabled(enabled)
        self.settings_page.filter_add_button.setEnabled(enabled)
        self.settings_page.filter_remove_button.setEnabled(enabled)
        self.settings_page.set_ptt_key_button.setEnabled(enabled)
        self.settings_page.restore_defaults_button.setEnabled(enabled)

    def start_hotkey_listener(self) -> None:
        """Start the background hotkey listener."""
        self.stop_hotkey_listener()
        print("Starting hotkey listener thread...")
        self.hotkey_thread = QThread(self)
        self.hotkey_worker = HotkeyWorker(
            ptt_key_str=self.loaded_settings.get("ptt_key_str", DEFAULT_PTT_KEY_STR),
            is_capture_mode=False,
        )
        self.hotkey_worker.moveToThread(self.hotkey_thread)
        self.hotkey_worker.ptt_pressed_signal.connect(self.on_ptt_pressed)
        self.hotkey_worker.ptt_released_signal.connect(self.on_ptt_released)
        self.hotkey_worker.error_signal.connect(self.handle_hotkey_error)
        self.hotkey_thread.started.connect(self.hotkey_worker.start_listening)
        self.hotkey_thread.finished.connect(self.hotkey_worker.deleteLater)
        self.hotkey_thread.finished.connect(self.hotkey_thread.deleteLater)
        self.hotkey_thread.start()

    def stop_hotkey_listener(self) -> None:
        """Stop the background hotkey listener."""
        if self.hotkey_worker:
            self.hotkey_worker.stop_listening()
        if self.hotkey_thread and self.hotkey_thread.isRunning():
            self.hotkey_thread.quit()
            if not self.hotkey_thread.wait(1000):
                print("Warning: Hotkey thread did not stop gracefully.")
        self.hotkey_worker = None
        self.hotkey_thread = None

    def restart_hotkey_listener(self) -> None:
        """Restart the hotkey listener to apply new bindings."""
        print("Restarting hotkey listener with updated keys...")
        self.start_hotkey_listener()

    @Slot()
    def on_ptt_pressed(self) -> None:
        """Handle global PTT press event."""
        if self.is_dictation_running:
            self.ptt_signal.emit(True)
            if not self.dictation_page.vad_toggle_button.isChecked():
                self.dictation_page.hint_label.setText("Listening...")
                self.dictation_page.hint_label.setStyleSheet(
                    "color: #30D158; font-style: italic; font-weight: bold;"
                )

    @Slot()
    def on_ptt_released(self) -> None:
        """Handle global PTT release event."""
        if self.is_dictation_running:
            self.ptt_signal.emit(False)
            if not self.dictation_page.vad_toggle_button.isChecked():
                ptt_key_name = format_key_name(
                    self.loaded_settings.get("ptt_key_str", DEFAULT_PTT_KEY_STR)
                )
                self.dictation_page.hint_label.setText(
                    f"Hold '{ptt_key_name}' to speak"
                )
                self.dictation_page.hint_label.setStyleSheet(
                    "color: #0A84FF; font-style: italic;"
                )

    @Slot(str)
    def handle_hotkey_error(self, error_msg: str) -> None:
        """Display hotkey listener errors to the user."""
        QMessageBox.warning(
            self,
            "Hotkey Listener Error",
            f"Error in hotkey listener: {error_msg}\nListener might need restarting.",
        )

    def closeEvent(self, event: object) -> None:
        """Ensure threads are stopped when the window is closed."""
        print("Close event triggered.")
        self.save_settings()
        if self.is_dictation_running:
            self.stop_dictation()
        self._destroy_worker()
        self.stop_hotkey_listener()
        if isinstance(self.hotkey_thread, QThread) and self.hotkey_thread.isRunning():
            print("Waiting for hotkey thread...")
            start_wait = time.time()
            while self.hotkey_thread.isRunning() and time.time() - start_wait < 0.7:
                QApplication.processEvents()
                time.sleep(0.05)
            if self.hotkey_thread.isRunning():
                print("Warning: Hotkey thread still running.")
        event.accept()


if __name__ == "__main__":
    try:
        import ctypes

        myappid = "omnicorp.omnidictate.gui.2.0.2"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception as e:
        print(f"Error setting AppUserModelID: {e}")
    app = QApplication(sys.argv)
    try:
        basedir = Path(__file__).parent
        icon_path = basedir / "icon.ico"
        if icon_path.exists():
            app_icon = QIcon(str(icon_path))
            app.setWindowIcon(app_icon)
            print(f"Application icon set from: {icon_path}")
        else:
            print(f"Warning: Icon file not found at {icon_path}")
    except Exception as e:
        print(f"Error setting application icon: {e}")
    try:
        style_path = Path(__file__).parent / "style.qss"
        with open(style_path) as f:
            _style = f.read()
            app.setStyleSheet(_style)
        print("Stylesheet applied.")
    except FileNotFoundError:
        print(f"Stylesheet '{style_path}' not found.")
    except Exception as e:
        print(f"Error loading stylesheet: {e}")
    window = OmniDictateApp()
    window.show()
    sys.exit(app.exec())
