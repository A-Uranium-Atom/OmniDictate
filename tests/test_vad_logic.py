"""Tests for voice activity detection state machine."""

from __future__ import annotations

import numpy as np

from config import CHUNK_SIZE, DictationSettings
from core.dictation_worker import DictationWorker


def _make_worker(
    mock_faster_whisper: object,
    gui_wid: int = 0,
    **overrides: object,
) -> DictationWorker:
    """Return a DictationWorker with sensible test defaults.

    Args:
        mock_faster_whisper: The conftest fixture that patches WhisperModel.
        gui_wid: Fake window handle.
        **overrides: Any DictationSettings field to override.

    Returns:
        A fully initialized DictationWorker with model loaded.
    """
    defaults: dict[str, object] = {
        "model_size": "tiny",
        "language": "en",
        "is_vad_enabled": True,
        "silence_threshold": 500,
        "char_delay": 0.0,
        "filter_words": ["thank you", "subtitles by amara.org"],
        "rms_threshold": 0.01,
        "hallucination_filter": "Medium",
        "insertion_method": "Paste",
    }
    defaults.update(overrides)
    settings = DictationSettings(**defaults)
    worker = DictationWorker(gui_wid=gui_wid, settings=settings)
    worker.load_model()
    return worker


class TestVADLogic:
    def test_silence_does_not_start_recording(
        self, qtbot, mock_faster_whisper, mocker
    ) -> None:
        worker = _make_worker(mock_faster_whisper, silence_threshold=500)
        silent_audio = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()

        worker._is_running = True
        worker.audio_queue.put(silent_audio)
        worker._check_audio_queue()
        assert worker.is_recording is False
        assert len(worker.audio_buffer) == 0

    def test_loud_audio_starts_vad_recording(
        self, qtbot, mock_faster_whisper, mocker
    ) -> None:
        worker = _make_worker(mock_faster_whisper, silence_threshold=500)
        loud_audio = np.full(CHUNK_SIZE, 30000, dtype=np.int16).tobytes()

        worker._is_running = True
        worker.audio_queue.put(loud_audio)
        worker._check_audio_queue()
        assert worker.is_recording is True
        assert worker.is_vad_active is True
        assert len(worker.audio_buffer) > 0
