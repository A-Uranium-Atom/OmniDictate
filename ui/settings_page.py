"""Settings page widget — model, hotkey, advanced, and filter configuration."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEFAULT_CHAR_DELAY,
    DEFAULT_FILTER_WORDS,
    DEFAULT_HALLUCINATION_FILTER,
    DEFAULT_INSERTION_METHOD,
    DEFAULT_LANGUAGE,
    DEFAULT_MODEL_SIZE,
    DEFAULT_PASTE_DELAY,
    DEFAULT_PTT_KEY_STR,
    DEFAULT_RMS_THRESHOLD,
    DEFAULT_SILENCE_THRESHOLD,
)
from ui.icons import format_key_name


class SettingsPage(QWidget):
    """The settings configuration view.

    Public Attributes (accessed by the main window for signal wiring):
        back_button: Returns to the dictation page.
        model_combo: Whisper model selector.
        language_combo: Language selector.
        silence_spinbox: Silence threshold control.
        delay_spinbox: Character delay control.
        ptt_key_display_label: Shows current PTT key name.
        set_ptt_key_button: Opens key capture dialog.
        rms_spinbox: Audio sensitivity control.
        hallucination_combo: Hallucination filter level selector.
        insertion_combo: Insertion method selector.
        paste_delay_spinbox: Paste delay control.
        filter_list: List of hallucination filter words.
        filter_add_edit: Input for new filter words.
        filter_add_button: Adds the word in filter_add_edit.
        filter_remove_button: Removes the selected filter word.
        restore_defaults_button: Restores all settings to defaults.

    Args:
        loaded_settings: Dict of persisted settings from QSettings.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        loaded_settings: dict[str, object],
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the SettingsPage."""
        super().__init__(parent)
        self._settings = loaded_settings

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(10)

        # --- Header ---
        header_layout = QHBoxLayout()
        self.back_button = QPushButton("← Back")
        self.back_button.setObjectName("backButton")
        self.back_button.setFixedSize(100, 40)
        self.back_button.setCursor(Qt.PointingHandCursor)

        settings_title = QLabel("Settings")
        settings_title.setObjectName("headerTitle")
        settings_title.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(self.back_button)
        header_layout.addStretch()
        header_layout.addWidget(settings_title)
        header_layout.addStretch()
        header_layout.addSpacing(100)  # Balance
        layout.addLayout(header_layout)

        # --- Scroll Area for Settings ---
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("background: transparent;")

        content_widget = QWidget()
        content_widget.setStyleSheet("background: transparent;")
        scroll.setWidget(content_widget)

        # Grid Layout for Settings Content
        grid = QGridLayout(content_widget)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(18)
        grid.setContentsMargins(10, 10, 20, 10)

        row = 0
        row = self._build_model_section(grid, row)
        row = self._build_hotkey_section(grid, row)
        row = self._build_advanced_section(grid, row)
        row = self._build_filter_section(grid, row)

        grid.addItem(
            QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding), row, 0
        )

        self.restore_defaults_button = QPushButton("Restore Defaults")
        self.restore_defaults_button.setStyleSheet(
            "color: #888; background: transparent; border: 1px solid #444;"
        )
        grid.addWidget(self.restore_defaults_button, row + 1, 0, 1, 4)

        layout.addWidget(scroll)

    def _build_model_section(self, grid: QGridLayout, row: int) -> int:
        grid.addWidget(QLabel("AI Model", objectName="sectionHeader"), row, 0, 1, 2)
        row += 1

        grid.addWidget(
            QLabel(
                "Whisper Model:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            0,
        )
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"]
        )
        self.model_combo.setCurrentText(
            self._settings.get("model_size", DEFAULT_MODEL_SIZE)
        )
        grid.addWidget(self.model_combo, row, 1)

        # Language Selection
        grid.addWidget(
            QLabel(
                "Language:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            2,
        )
        self.language_combo = QComboBox()

        # Populate languages
        languages = [
            ("Auto Detect", None),
            ("English", "en"),
            ("Spanish", "es"),
            ("French", "fr"),
            ("German", "de"),
            ("Italian", "it"),
            ("Portuguese", "pt"),
            ("Dutch", "nl"),
            ("Russian", "ru"),
            ("Chinese", "zh"),
            ("Japanese", "ja"),
        ]
        for name, code in languages:
            self.language_combo.addItem(name, code)

        # Set current selection
        current_lang_code = self._settings.get("language", DEFAULT_LANGUAGE)
        index = self.language_combo.findData(current_lang_code)
        if index != -1:
            self.language_combo.setCurrentIndex(index)
        else:
            self.language_combo.setCurrentIndex(0)  # Default to Auto Detect if unknown
        grid.addWidget(self.language_combo, row, 3)
        row += 1

        grid.addWidget(
            QLabel(
                "Silence Threshold:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            0,
        )
        self.silence_spinbox = QSpinBox()
        self.silence_spinbox.setRange(50, 3000)
        self.silence_spinbox.setSingleStep(50)
        self.silence_spinbox.setValue(
            self._settings.get("silence_threshold", DEFAULT_SILENCE_THRESHOLD)
        )
        grid.addWidget(self.silence_spinbox, row, 1)

        grid.addWidget(
            QLabel(
                "Typing Delay (s):",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            2,
        )
        self.delay_spinbox = QDoubleSpinBox()
        self.delay_spinbox.setRange(0.0, 0.1)
        self.delay_spinbox.setSingleStep(0.005)
        self.delay_spinbox.setDecimals(3)
        self.delay_spinbox.setValue(
            self._settings.get("char_delay", DEFAULT_CHAR_DELAY)
        )
        grid.addWidget(self.delay_spinbox, row, 3)
        row += 1

        return row

    def _build_hotkey_section(self, grid: QGridLayout, row: int) -> int:
        grid.addWidget(QLabel("Hotkeys", objectName="sectionHeader"), row, 0, 1, 2)
        row += 1

        grid.addWidget(
            QLabel(
                "PTT Hotkey:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            0,
        )
        self.ptt_key_display_label = QLabel(
            format_key_name(self._settings.get("ptt_key_str", DEFAULT_PTT_KEY_STR))
        )
        self.ptt_key_display_label.setStyleSheet("color: #0A84FF; font-weight: bold;")
        self.set_ptt_key_button = QPushButton("Change")
        self.set_ptt_key_button.setCursor(Qt.PointingHandCursor)
        grid.addWidget(self.ptt_key_display_label, row, 1)
        grid.addWidget(self.set_ptt_key_button, row, 2)
        row += 1

        return row

    def _build_advanced_section(self, grid: QGridLayout, row: int) -> int:
        grid.addWidget(QLabel("Advanced", objectName="sectionHeader"), row, 0, 1, 2)
        row += 1

        # Audio Sensitivity (RMS Threshold)
        grid.addWidget(
            QLabel(
                "Audio Sensitivity:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            0,
        )
        self.rms_spinbox = QDoubleSpinBox()
        self.rms_spinbox.setRange(0.001, 0.1)
        self.rms_spinbox.setSingleStep(0.005)
        self.rms_spinbox.setDecimals(3)
        self.rms_spinbox.setValue(
            self._settings.get("rms_threshold", DEFAULT_RMS_THRESHOLD)
        )
        self.rms_spinbox.setToolTip(
            "Minimum audio energy to trigger transcription. Higher = less sensitive to background noise. Recommended: 0.010"
        )
        grid.addWidget(self.rms_spinbox, row, 1)

        # Hallucination Filter Level (same row, columns 2-3)
        grid.addWidget(
            QLabel(
                "Hallucination Filter:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            2,
        )
        self.hallucination_combo = QComboBox()
        self.hallucination_combo.addItems(["Low", "Medium", "High"])
        self.hallucination_combo.setCurrentText(
            self._settings.get("hallucination_filter", DEFAULT_HALLUCINATION_FILTER)
        )
        self.hallucination_combo.setToolTip(
            "Controls how aggressively phantom text from silence is suppressed. Low = permissive, High = aggressive. Recommended: Medium"
        )
        grid.addWidget(self.hallucination_combo, row, 3)
        row += 1

        # Insertion Method
        grid.addWidget(
            QLabel(
                "Insertion Method:",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            0,
        )
        self.insertion_combo = QComboBox()
        self.insertion_combo.addItems(["Paste", "Typing"])
        self.insertion_combo.setCurrentText(
            self._settings.get("insertion_method", DEFAULT_INSERTION_METHOD)
        )
        self.insertion_combo.setToolTip(
            "Paste: Instant text input using clipboard (recommended for long text). Typing: Emulate keystrokes."
        )
        grid.addWidget(self.insertion_combo, row, 1)

        grid.addWidget(
            QLabel(
                "Paste Delay (s):",
                objectName="settingLabel",
                alignment=Qt.AlignRight | Qt.AlignVCenter,
            ),
            row,
            2,
        )
        self.paste_delay_spinbox = QDoubleSpinBox()
        self.paste_delay_spinbox.setRange(0.1, 1.0)
        self.paste_delay_spinbox.setSingleStep(0.05)
        self.paste_delay_spinbox.setDecimals(2)
        self.paste_delay_spinbox.setValue(
            self._settings.get("paste_delay", DEFAULT_PASTE_DELAY)
        )
        self.paste_delay_spinbox.setToolTip(
            "Time to wait after pasting before restoring the clipboard. Increase if apps like Gemini paste old text."
        )
        grid.addWidget(self.paste_delay_spinbox, row, 3)
        row += 1

        return row

    def _build_filter_section(self, grid: QGridLayout, row: int) -> int:
        grid.addWidget(
            QLabel(
                "Filter Words:",
                objectName="settingLabel",
                alignment=Qt.AlignLeft | Qt.AlignVCenter,
            ),
            row,
            0,
            1,
            4,
        )
        row += 1
        self.filter_list = QListWidget()
        self.filter_list.addItems(
            self._settings.get("filter_words", DEFAULT_FILTER_WORDS)
        )
        self.filter_list.setFixedHeight(180)
        self.filter_list.setSpacing(0)
        grid.addWidget(self.filter_list, row, 0, 1, 4)
        row += 1

        filter_controls = QHBoxLayout()
        self.filter_add_edit = QLineEdit()
        self.filter_add_edit.setPlaceholderText("Enter phrase...")
        self.filter_add_button = QPushButton("Add")
        self.filter_remove_button = QPushButton("Remove")
        filter_controls.addWidget(self.filter_add_edit)
        filter_controls.addWidget(self.filter_add_button)
        filter_controls.addWidget(self.filter_remove_button)
        grid.addLayout(filter_controls, row, 0, 1, 4)
        row += 1

        return row
