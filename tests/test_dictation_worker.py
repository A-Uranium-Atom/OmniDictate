"""Tests for DictationWorker transcription pipeline and hallucination filtering."""

from __future__ import annotations

import numpy as np

from config import SAMPLE_RATE, DictationSettings
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


def _audio_int16(amplitude: float, duration_s: float = 0.5) -> np.ndarray:
    """Return an int16 audio buffer at the given amplitude (0.0 to 1.0)."""
    n_samples = int(SAMPLE_RATE * duration_s)
    return np.full(n_samples, int(amplitude * 32767), dtype=np.int16)


class TestHallucinationFiltering:
    def test_filter_word_exact_match_suppressed(
        self, qtbot, mock_faster_whisper, mocker, capfd
    ) -> None:
        worker = _make_worker(mock_faster_whisper)
        mock_segment = mocker.Mock()
        mock_segment.text = "thank you."
        mock_segment.no_speech_prob = 0.01
        mocker.patch.object(
            worker.model, "transcribe", return_value=([mock_segment], None)
        )
        worker.audio_buffer.append(np.frombuffer(b"fake_audio", dtype=np.int16))
        worker._process_audio_buffer()
        worker.transcription_executor.shutdown(wait=True)
        out, err = capfd.readouterr()
        assert "Filtered out hallucination: 'thank you.'" in out

    def test_real_speech_emits_signal(self, qtbot, mock_faster_whisper, mocker) -> None:
        worker = _make_worker(mock_faster_whisper)
        mock_segment = mocker.Mock()
        mock_segment.text = "Hello world."
        mock_segment.no_speech_prob = 0.01
        mocker.patch.object(
            worker.model, "transcribe", return_value=([mock_segment], None)
        )
        with qtbot.waitSignal(worker.transcription_ready, timeout=1000) as blocker:
            worker.audio_buffer.append(np.frombuffer(b"fake_audio", dtype=np.int16))
            worker._process_audio_buffer()
        assert blocker.args[0] == "Hello world."

    def test_repeated_hallucination_suppressed_after_threshold(
        self, qtbot, mock_faster_whisper, mocker, capfd
    ) -> None:
        import concurrent.futures

        worker = _make_worker(mock_faster_whisper)
        mock_segment = mocker.Mock()
        mock_segment.text = "Some hallucination."
        mock_segment.no_speech_prob = 0.01

        mock_info = mocker.Mock()
        mock_info.language_probability = 0.9
        mocker.patch.object(
            worker.model, "transcribe", return_value=([mock_segment], mock_info)
        )

        # First call
        worker.audio_buffer.append(np.frombuffer(b"fake_audio", dtype=np.int16))
        worker._process_audio_buffer()
        worker.transcription_executor.shutdown(wait=True)

        # Second call
        worker.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        worker.audio_buffer.append(np.frombuffer(b"fake_audio", dtype=np.int16))
        worker._process_audio_buffer()
        worker.transcription_executor.shutdown(wait=True)

        # Third call (should be suppressed)
        worker.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        worker.audio_buffer.append(np.frombuffer(b"fake_audio", dtype=np.int16))
        worker._process_audio_buffer()
        worker.transcription_executor.shutdown(wait=True)

        out, err = capfd.readouterr()
        assert "Filtered out repeated hallucination:" in out

    def test_rms_gate_skips_silent_audio(
        self, qtbot, mock_faster_whisper, mocker
    ) -> None:
        worker = _make_worker(mock_faster_whisper, rms_threshold=0.05)
        silent_audio = _audio_int16(0.01)
        mock_task = mocker.patch.object(worker, "_transcription_task")

        worker.audio_buffer.append(
            np.frombuffer(silent_audio.tobytes(), dtype=np.int16)
        )
        worker._process_audio_buffer()
        mock_task.assert_not_called()
