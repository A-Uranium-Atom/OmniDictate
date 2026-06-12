"""Dictation page widget — transcription display, controls, and visualizer."""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ui.icons import create_gear_icon


class DictationPage(QWidget):
    """The main dictation view with transcription display and control dock.

    Public Attributes (accessed by the main window for signal wiring):
        model_display_label: Shows the active Whisper model name.
        settings_button: Opens the settings page.
        transcription_display: Read-only text area showing transcriptions.
        hint_label: Context-sensitive instruction text.
        visualizer: Audio level progress bar (0–1000).
        vad_toggle_button: Toggles VAD/PTT mode.
        start_button: Starts dictation.
        stop_button: Stops dictation (initially disabled).
        copy_button: Copies transcription text to clipboard.

    Args:
        model_display_text: Initial text for the model label.
        is_vad_checked: Initial checked state for the VAD toggle.
        parent: Optional parent widget.
    """

    def __init__(
        self,
        model_display_text: str = "Model: large-v3",
        is_vad_checked: bool = True,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize the DictationPage."""
        super().__init__(parent)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(15)

        # --- Header ---
        header_layout = QHBoxLayout()
        title = QLabel("OmniDictate")
        title.setObjectName("headerTitle")

        self.model_display_label = QLabel(model_display_text)
        self.model_display_label.setObjectName("settingLabel")
        self.model_display_label.setStyleSheet(
            "color: #888; font-size: 10pt; margin-right: 10px;"
        )

        self.settings_button = QPushButton()
        self.settings_button.setIcon(create_gear_icon())
        self.settings_button.setIconSize(QSize(40, 40))
        self.settings_button.setObjectName("iconButton")
        self.settings_button.setToolTip("Settings")

        header_layout.addWidget(title)
        header_layout.addStretch()
        header_layout.addWidget(self.model_display_label)
        header_layout.addWidget(self.settings_button)
        layout.addLayout(header_layout)

        # --- Main Text Area ---
        self.transcription_display = QTextEdit()
        self.transcription_display.setObjectName("transcriptionDisplay")
        self.transcription_display.setReadOnly(True)
        self.transcription_display.setPlaceholderText("Ready.")
        layout.addWidget(self.transcription_display)

        # --- Context Hint ---
        self.hint_label = QLabel("")
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setAlignment(Qt.AlignCenter)
        self.hint_label.setFixedHeight(30)
        layout.addWidget(self.hint_label)

        # --- Audio Visualizer ---
        self.visualizer = QProgressBar()
        self.visualizer.setObjectName("audioVisualizer")
        self.visualizer.setFixedHeight(4)
        self.visualizer.setTextVisible(False)
        self.visualizer.setRange(0, 1000)
        self.visualizer.setValue(0)
        layout.addWidget(self.visualizer)

        # --- Bottom Controls Dock ---
        control_dock = QFrame()
        control_dock.setObjectName("controlDock")
        control_dock.setFixedHeight(80)
        dock_layout = QHBoxLayout(control_dock)
        dock_layout.setContentsMargins(15, 0, 15, 0)
        dock_layout.setSpacing(15)

        self.vad_toggle_button = QPushButton("VAD Mode")
        self.vad_toggle_button.setCheckable(True)
        self.vad_toggle_button.setChecked(is_vad_checked)
        self.vad_toggle_button.setFixedSize(110, 45)
        self.vad_toggle_button.setObjectName("modeButton")

        self.start_button = QPushButton("Start")
        self.start_button.setFixedSize(100, 45)
        self.start_button.setObjectName("startButton")

        self.stop_button = QPushButton("Stop")
        self.stop_button.setFixedSize(100, 45)
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)

        self.copy_button = QPushButton("Copy")
        self.copy_button.setFixedSize(80, 45)

        dock_layout.addWidget(self.vad_toggle_button)
        dock_layout.addStretch()
        dock_layout.addWidget(self.start_button)
        dock_layout.addWidget(self.stop_button)
        dock_layout.addStretch()
        dock_layout.addWidget(self.copy_button)

        layout.addWidget(control_dock)
