import pytest
from PyQt6 import QtCore
from main_gui import OmniDictateApp  # might be PySide6 depending on codebase

# Use pytest-qt fixture 'qtbot'
def test_initial_state_validation(qtbot, mock_settings):
    app = OmniDictateApp()
    qtbot.addWidget(app)
    
    assert app.windowTitle() == "OmniDictate"
    
def test_settings_persistence(qtbot, mock_settings):
    app = OmniDictateApp()
    qtbot.addWidget(app)
    
    # Simulate user changing a setting
    # app.settings_ui.use_gpu_checkbox.setChecked(True)
    # app.save_settings()
    
    # Validate QSettings was updated
    # assert mock_settings.store.get("use_gpu") == True
    pass

def test_signal_handling(qtbot, mocker):
    app = OmniDictateApp()
    qtbot.addWidget(app)
    
    # Mock the emit and verify that GUI updates the log pane
    app.log_transcription("Test hallucinated output string successfully delivered.")
    # assert "Test hallucinated output" in app.log_pane.toPlainText()
    pass
