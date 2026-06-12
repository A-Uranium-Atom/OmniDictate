import pytest


@pytest.fixture
def mock_settings(monkeypatch) -> None:
    """
    Mock QSettings so we don't overwrite the user's actual settings during
    tests.  Request this fixture explicitly in any test that constructs
    OmniDictateApp; do NOT use autouse because some tests don't need it and
    the monkeypatch timing can clash with module-level teardown.
    """

    class MockSettings:
        def __init__(self, *args, **kwargs) -> None:
            self.store = {}

        def setValue(self, key, value) -> None:
            self.store[key] = value

        def value(self, key, default_value=None, type=None):  # noqa: A002
            raw = self.store.get(key, default_value)
            if type is not None and raw is not None:
                try:
                    return type(raw)
                except (ValueError, TypeError):
                    return default_value
            return raw

        def sync(self) -> None:
            pass

    monkeypatch.setattr("main_gui.QSettings", MockSettings)


@pytest.fixture
def mock_faster_whisper(monkeypatch) -> None:
    """
    Mock the faster-whisper WhisperModel to avoid downloading / loading the
    real model during test execution.  Request this fixture explicitly in any
    test that constructs a DictationWorker.
    """

    class MockSegment:
        def __init__(self, text, no_speech_prob=0.01) -> None:
            self.text = text
            # core.dictation_worker._transcription_task reads this attribute
            self.no_speech_prob = no_speech_prob

    class MockModel:
        def __init__(
            self,
            model_size_or_path,
            device="cpu",
            compute_type="float32",
            local_files_only=False,
        ) -> None:
            self.model_size_or_path = model_size_or_path
            self.device = device
            self.compute_type = compute_type
            self._result = [MockSegment("Mocked transcription output.")]

        def transcribe(self, audio, **kwargs):
            return iter(self._result), None

    monkeypatch.setattr("core.dictation_worker.WhisperModel", MockModel)
