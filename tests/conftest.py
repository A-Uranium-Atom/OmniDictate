import pytest
from PySide6.QtCore import QSettings

@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Mock QSettings so we don't overwrite the user's actual settings during tests.
    """
    class MockSettings:
        def __init__(self, *args, **kwargs):
            self.store = {}
        def setValue(self, key, value):
            self.store[key] = value
        def value(self, key, default_value=None):
            return self.store.get(key, default_value)
            
    monkeypatch.setattr("main_gui.QSettings", MockSettings)

@pytest.fixture
def mock_faster_whisper(monkeypatch):
    """
    Mock faster-whisper Model to avoid downloading/loading the 
    real NN model during test execution.
    """
    class MockSegment:
        def __init__(self, text):
            self.text = text

    class MockModel:
        def __init__(self, model_size_or_path, device="cpu", compute_type="float32"):
            self.model_size_or_path = model_size_or_path
            self.device = device
            self.compute_type = compute_type
            self.transcribe_result = [MockSegment("Mocked transcription output.")]

        def transcribe(self, audio, **kwargs):
            return self.transcribe_result, None
            
    monkeypatch.setattr("core_logic.WhisperModel", MockModel)
