"""Extended tests for the refactored OmniDictate core modules.

Covers gaps identified during the Stage 5 audit:
  - PowerMonitor no-op behaviour when win32gui is absent.
  - DictationSettings->DictationWorker.update_settings round-trip.
  - InjectorSettings sync when settings change.
  - config.py: MAX_RECORDING_SECONDS type, DictationSettings mutable default safety.
  - text_injector: clipboard error during set causes fallback.
"""

from __future__ import annotations

from config import (
    CHUNK_SIZE,
    HALLUCINATION_PRESETS,
    MAX_RECORDING_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
    DictationSettings,
    HallucinationLevel,
    InsertionMethod,
    ModelSize,
)
from core.dictation_worker import DictationWorker
from core.text_injector import InjectorSettings, paste_text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_worker(
    mock_faster_whisper: object,
    gui_wid: int = 0,
    **overrides: object,
) -> DictationWorker:
    """Return a DictationWorker with sensible test defaults.

    Args:
        mock_faster_whisper: The conftest fixture that patches WhisperModel.
        gui_wid: Fake window handle (HWND).
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


# ---------------------------------------------------------------------------
# Config module
# ---------------------------------------------------------------------------


class TestConfigIntegrity:
    """Verify config.py numeric constants match spec values."""

    def test_sample_rate_is_16000(self) -> None:
        """SAMPLE_RATE must match the Whisper input requirement."""
        assert SAMPLE_RATE == 16_000

    def test_chunk_size_derivation(self) -> None:
        """CHUNK_SIZE must equal SAMPLE_RATE * CHUNK_DURATION rounded to int."""
        from config import CHUNK_DURATION

        assert CHUNK_SIZE == int(SAMPLE_RATE * CHUNK_DURATION)

    def test_max_recording_seconds_is_numeric(self) -> None:
        """MAX_RECORDING_SECONDS must be a positive int or float."""
        assert isinstance(MAX_RECORDING_SECONDS, (int, float))
        assert MAX_RECORDING_SECONDS > 0

    def test_silence_duration_value(self) -> None:
        """SILENCE_DURATION must be 0.5 seconds per spec section 3.1."""
        assert SILENCE_DURATION == 0.5

    def test_hallucination_presets_all_have_required_keys(self) -> None:
        """Every preset must expose no_speech_threshold and log_prob_threshold."""
        required_keys = {"no_speech_threshold", "log_prob_threshold"}
        for level, preset in HALLUCINATION_PRESETS.items():
            assert required_keys <= set(preset.keys()), (
                f"Preset for {level} is missing keys: "
                f"{required_keys - set(preset.keys())}"
            )

    def test_hallucination_presets_thresholds_are_descending(self) -> None:
        """HIGH aggressiveness must have a lower no_speech_threshold than LOW."""
        low_thresh = HALLUCINATION_PRESETS[HallucinationLevel.LOW][
            "no_speech_threshold"
        ]
        high_thresh = HALLUCINATION_PRESETS[HallucinationLevel.HIGH][
            "no_speech_threshold"
        ]
        assert high_thresh < low_thresh


class TestDictationSettingsIsolation:
    """Verify DictationSettings mutable default safety (spec section 1.1.9)."""

    def test_filter_words_default_is_a_copy(self) -> None:
        """Mutating one instance filter_words must not affect a new instance."""
        settings_a = DictationSettings()
        settings_b = DictationSettings()
        settings_a.filter_words.append("injected")
        assert "injected" not in settings_b.filter_words

    def test_settings_fields_match_defaults(self) -> None:
        """Default-constructed DictationSettings must match the DEFAULT_* constants."""
        from config import (
            DEFAULT_CHAR_DELAY,
            DEFAULT_HALLUCINATION_FILTER,
            DEFAULT_INSERTION_METHOD,
            DEFAULT_MODEL_SIZE,
            DEFAULT_PASTE_DELAY,
            DEFAULT_RMS_THRESHOLD,
            DEFAULT_SILENCE_THRESHOLD,
            DEFAULT_VAD_ENABLED,
        )

        settings = DictationSettings()
        assert settings.model_size == DEFAULT_MODEL_SIZE
        assert settings.is_vad_enabled == DEFAULT_VAD_ENABLED
        assert settings.silence_threshold == DEFAULT_SILENCE_THRESHOLD
        assert settings.char_delay == DEFAULT_CHAR_DELAY
        assert settings.rms_threshold == DEFAULT_RMS_THRESHOLD
        assert settings.hallucination_filter == DEFAULT_HALLUCINATION_FILTER
        assert settings.insertion_method == DEFAULT_INSERTION_METHOD
        assert settings.paste_delay == DEFAULT_PASTE_DELAY


# ---------------------------------------------------------------------------
# DictationWorker construction
# ---------------------------------------------------------------------------


class TestDictationWorkerConstruction:
    """Verify that DictationWorker.__init__ correctly maps DictationSettings fields."""

    def test_defaults_are_applied_when_settings_is_none(
        self, mock_faster_whisper: object
    ) -> None:
        """Passing settings=None must fall back to DictationSettings() defaults."""
        worker = DictationWorker(gui_wid=0)
        worker.load_model()
        assert worker.model_size == ModelSize.LARGE_V3
        assert worker._is_vad_enabled is True

    def test_filter_words_normalised_to_lowercase(
        self, mock_faster_whisper: object
    ) -> None:
        """filter_words must be stored as a set of lowercase-stripped strings."""
        worker = _make_worker(
            mock_faster_whisper,
            filter_words=["  Thank You  ", "SUBSCRIBE"],
        )
        assert "thank you" in worker.filter_words
        assert "subscribe" in worker.filter_words

    def test_silence_frames_computed_correctly(
        self, mock_faster_whisper: object
    ) -> None:
        """silence_frames must equal int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)."""
        worker = _make_worker(mock_faster_whisper)
        expected = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        assert worker.silence_frames == expected

    def test_injector_settings_initialised_from_dictation_settings(
        self, mock_faster_whisper: object
    ) -> None:
        """_injector_settings must mirror the DictationSettings insertion fields."""
        worker = _make_worker(
            mock_faster_whisper,
            insertion_method="Typing",
            char_delay=0.05,
            paste_delay=0.4,
        )
        assert worker._injector_settings.insertion_method == InsertionMethod.TYPING
        assert worker._injector_settings.char_delay == 0.05
        assert worker._injector_settings.paste_delay == 0.4


# ---------------------------------------------------------------------------
# DictationWorker.update_settings
# ---------------------------------------------------------------------------


class TestUpdateSettings:
    """Verify the update_settings dataclass round-trip (spec section 3.2.3)."""

    def test_update_settings_syncs_injector_settings(
        self, mock_faster_whisper: object
    ) -> None:
        """After update_settings, _injector_settings must reflect the new values."""
        worker = _make_worker(mock_faster_whisper)

        new_settings = DictationSettings(
            insertion_method=InsertionMethod.TYPING,
            char_delay=0.07,
            paste_delay=0.6,
        )
        worker.update_settings(new_settings)

        assert worker._injector_settings.insertion_method == InsertionMethod.TYPING
        assert worker._injector_settings.char_delay == 0.07
        assert worker._injector_settings.paste_delay == 0.6

    def test_update_settings_normalises_filter_words(
        self, mock_faster_whisper: object
    ) -> None:
        """filter_words in the new settings must be normalised to lowercase."""
        worker = _make_worker(mock_faster_whisper)
        new_settings = DictationSettings(filter_words=["  Bad Word  ", "NOISE"])
        worker.update_settings(new_settings)

        assert "bad word" in worker.filter_words
        assert "noise" in worker.filter_words

    def test_update_settings_updates_hallucination_filter(
        self, mock_faster_whisper: object
    ) -> None:
        """hallucination_filter attribute must be updated by update_settings."""
        worker = _make_worker(mock_faster_whisper, hallucination_filter="Low")
        new_settings = DictationSettings(hallucination_filter=HallucinationLevel.HIGH)
        worker.update_settings(new_settings)
        assert worker.hallucination_filter == HallucinationLevel.HIGH

    def test_update_settings_does_not_reload_model_when_size_unchanged(
        self, mock_faster_whisper: object, mocker: object
    ) -> None:
        """load_model must NOT be called when model_size is unchanged."""
        worker = _make_worker(mock_faster_whisper, model_size="tiny")
        spy = mocker.patch.object(worker, "load_model")

        same_settings = DictationSettings(model_size=ModelSize.TINY)
        worker.update_settings(same_settings)

        spy.assert_not_called()

    def test_update_settings_reloads_model_when_size_changes(
        self, mock_faster_whisper: object, mocker: object
    ) -> None:
        """load_model must be called with force_reload=True when model_size changes."""
        worker = _make_worker(mock_faster_whisper, model_size="tiny")
        spy = mocker.patch.object(worker, "load_model")

        new_settings = DictationSettings(model_size=ModelSize.BASE)
        worker.update_settings(new_settings)

        spy.assert_called_once_with(force_reload=True)


# ---------------------------------------------------------------------------
# PowerMonitor
# ---------------------------------------------------------------------------


class TestPowerMonitorNoop:
    """Verify PowerMonitor is a safe no-op when win32gui is unavailable."""

    def test_start_is_noop_without_win32(self, monkeypatch: object) -> None:
        """When _HAS_WIN32 is False, start() must return immediately without starting thread.

        We monkeypatch the module-level flag so this test runs on any platform.
        """
        import core.power_monitor as pm_module

        monkeypatch.setattr(pm_module, "_HAS_WIN32", False)

        callback_called: list[bool] = []
        monitor = pm_module.PowerMonitor(on_resume=lambda: callback_called.append(True))
        monitor.start()

        assert not monitor._thread.is_alive()
        assert callback_called == []

    def test_stop_is_safe_when_hwnd_is_none(self) -> None:
        """stop() must not raise when _hwnd is None (monitor never started)."""
        from core.power_monitor import PowerMonitor

        monitor = PowerMonitor(on_resume=lambda: None)
        monitor.stop()  # Must not raise


# ---------------------------------------------------------------------------
# InjectorSettings
# ---------------------------------------------------------------------------


class TestInjectorSettings:
    """Verify InjectorSettings default values match spec section 2.3.2."""

    def test_default_insertion_method_is_paste(self) -> None:
        """InjectorSettings must default to InsertionMethod.PASTE."""
        s = InjectorSettings()
        assert s.insertion_method == InsertionMethod.PASTE

    def test_default_char_delay(self) -> None:
        """char_delay default must be 0.02 seconds."""
        s = InjectorSettings()
        assert s.char_delay == 0.02

    def test_default_paste_delay(self) -> None:
        """paste_delay default must be 0.3 seconds."""
        s = InjectorSettings()
        assert s.paste_delay == 0.3


# ---------------------------------------------------------------------------
# paste_text: error paths
# ---------------------------------------------------------------------------


class TestPasteTextEdgeCases:
    """Additional edge cases for paste_text not covered by test_text_injector.py."""

    def test_paste_returns_false_when_open_clipboard_fails(
        self, mocker: object
    ) -> None:
        """If OpenClipboard raises during backup, paste_text must return False."""
        mock_kb = mocker.patch("core.text_injector.Controller")
        mocker.patch(
            "win32clipboard.OpenClipboard",
            side_effect=OSError("Access denied"),
        )
        mocker.patch("win32clipboard.CloseClipboard")

        result = paste_text("hello", mock_kb, paste_delay=0.0)
        assert result is False

    def test_paste_returns_false_when_set_clipboard_data_fails(
        self, mocker: object
    ) -> None:
        """If SetClipboardData raises when writing new text, paste_text returns False."""
        mock_kb = mocker.patch("core.text_injector.Controller")

        mocker.patch("win32clipboard.OpenClipboard")
        mocker.patch("win32clipboard.CloseClipboard")
        mocker.patch("win32clipboard.EmptyClipboard")
        mocker.patch("win32clipboard.EnumClipboardFormats", return_value=0)
        mocker.patch(
            "win32clipboard.SetClipboardData",
            side_effect=OSError("Cannot set"),
        )
        mocker.patch("win32clipboard.CF_UNICODETEXT", 13)

        result = paste_text("hello", mock_kb, paste_delay=0.0)
        assert result is False

    def test_paste_calls_warning_callback_on_file_drop(self, mocker: object) -> None:
        """warning_callback must be called when CF_HDROP is detected."""
        mock_kb = mocker.patch("core.text_injector.Controller")
        mocker.patch("win32clipboard.OpenClipboard")
        mocker.patch("win32clipboard.CloseClipboard")

        def mock_enum(format_id: int) -> int:
            if format_id == 0:
                return 15  # CF_HDROP value
            return 0

        mocker.patch("win32clipboard.EnumClipboardFormats", side_effect=mock_enum)
        mocker.patch("config.CF_HDROP", 15)

        warnings_received: list[str] = []
        result = paste_text(
            "hello",
            mock_kb,
            paste_delay=0.0,
            warning_callback=lambda msg: warnings_received.append(msg),
        )

        assert result is False
        assert len(warnings_received) == 1
        assert "clipboard" in warnings_received[0].lower()


# ---------------------------------------------------------------------------
# VAD state machine: PTT mode
# ---------------------------------------------------------------------------


class TestPTTStateMachine:
    """Verify that PTT key presses gate recording independently from VAD."""

    def test_ptt_press_starts_non_vad_recording(
        self, qtbot: object, mock_faster_whisper: object
    ) -> None:
        """set_ptt_state(True) must enable recording without VAD being active."""
        import numpy as np

        worker = _make_worker(mock_faster_whisper, is_vad_enabled=False)
        worker._is_running = True

        worker.set_ptt_state(True)
        worker.audio_queue.put(np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes())
        worker._check_audio_queue()

        assert worker._is_ptt_active is True
        assert worker.is_recording is True
        assert worker.is_vad_active is False

    def test_ptt_release_stops_recording(
        self, qtbot: object, mock_faster_whisper: object
    ) -> None:
        """set_ptt_state(False) after a press must stop recording and clear PTT flag."""
        import numpy as np

        worker = _make_worker(mock_faster_whisper, is_vad_enabled=False)
        worker._is_running = True

        worker.set_ptt_state(True)
        worker.audio_queue.put(np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes())
        worker._check_audio_queue()
        assert worker.is_recording is True

        worker.set_ptt_state(False)
        assert worker._is_ptt_active is False
        assert worker.is_recording is False


# ---------------------------------------------------------------------------
# DictationWorker: drain audio queue helper
# ---------------------------------------------------------------------------


class TestDrainAudioQueue:
    """Verify the _drain_audio_queue helper returns and clears all pending frames."""

    def test_drain_returns_all_queued_frames(self, mock_faster_whisper: object) -> None:
        """All frames put into audio_queue must be returned and queue becomes empty."""
        import numpy as np

        worker = _make_worker(mock_faster_whisper)
        frame_a = np.zeros(CHUNK_SIZE, dtype=np.int16).tobytes()
        frame_b = np.ones(CHUNK_SIZE, dtype=np.int16).tobytes()

        worker.audio_queue.put(frame_a)
        worker.audio_queue.put(frame_b)

        drained = worker._drain_audio_queue()

        assert len(drained) == 2
        assert drained[0] == frame_a
        assert drained[1] == frame_b
        assert worker.audio_queue.empty()

    def test_drain_on_empty_queue_returns_empty_list(
        self, mock_faster_whisper: object
    ) -> None:
        """_drain_audio_queue must return [] when the queue is empty."""
        worker = _make_worker(mock_faster_whisper)
        assert worker._drain_audio_queue() == []
