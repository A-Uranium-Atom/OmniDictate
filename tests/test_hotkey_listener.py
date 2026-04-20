"""
test_hotkey_listener.py
=======================
Unit tests for hotkey_listener.HotkeyWorker.

Signal names verified against hotkey_listener.py:
  ptt_pressed_signal  — emitted on PTT key press
  ptt_released_signal — emitted on PTT key release
  key_captured_signal — emitted in capture mode
  error_signal        — emitted on listener failure

Design decisions
----------------
* We stub out ``pynput.keyboard.Listener`` so no real OS-level keyboard hook
  is created during testing (avoids privilege requirements on CI).
* HotkeyWorker is instantiated without starting the listener thread;
  _on_press / _on_release are called directly to keep tests synchronous.
* ``_is_running`` is set to True manually before calling handler methods
  because both guard on that flag first.
"""

from unittest.mock import MagicMock, patch

import pytest
from pynput import keyboard

from hotkey_listener import HotkeyWorker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ptt_worker():
    """Return a HotkeyWorker configured for PTT mode, listener not started."""
    with patch("hotkey_listener.keyboard.Listener"):
        worker = HotkeyWorker(ptt_key_str="keyboard.Key.shift_r", capture_mode=False)
    worker._is_running = True
    return worker


@pytest.fixture
def capture_worker():
    """Return a HotkeyWorker in capture mode, listener not started."""
    with patch("hotkey_listener.keyboard.Listener"):
        worker = HotkeyWorker(capture_mode=True)
    worker._is_running = True
    return worker


# ---------------------------------------------------------------------------
# PTT mode
# ---------------------------------------------------------------------------

class TestPTTMode:
    def test_ptt_press_emits_pressed_signal(self, ptt_worker, mocker):
        mock_pressed = mocker.Mock()
        ptt_worker.ptt_pressed_signal.connect(mock_pressed)

        ptt_worker._on_press(keyboard.Key.shift_r)

        mock_pressed.assert_called_once()

    def test_ptt_release_emits_released_signal(self, ptt_worker, mocker):
        mock_released = mocker.Mock()
        ptt_worker.ptt_released_signal.connect(mock_released)

        ptt_worker._on_release(keyboard.Key.shift_r)

        mock_released.assert_called_once()

    def test_wrong_key_does_not_emit_pressed(self, ptt_worker, mocker):
        """Pressing a key that is NOT the configured PTT key must not emit."""
        mock_pressed = mocker.Mock()
        ptt_worker.ptt_pressed_signal.connect(mock_pressed)

        ptt_worker._on_press(keyboard.Key.space)

        mock_pressed.assert_not_called()

    def test_wrong_key_does_not_emit_released(self, ptt_worker, mocker):
        mock_released = mocker.Mock()
        ptt_worker.ptt_released_signal.connect(mock_released)

        ptt_worker._on_release(keyboard.Key.ctrl_l)

        mock_released.assert_not_called()

    def test_handlers_ignored_when_not_running(self, ptt_worker, mocker):
        """With _is_running=False no signals should fire."""
        ptt_worker._is_running = False
        mock_pressed  = mocker.Mock()
        ptt_worker.ptt_pressed_signal.connect(mock_pressed)
        mock_released = mocker.Mock()
        ptt_worker.ptt_released_signal.connect(mock_released)

        ptt_worker._on_press(keyboard.Key.shift_r)
        ptt_worker._on_release(keyboard.Key.shift_r)

        mock_pressed.assert_not_called()
        mock_released.assert_not_called()


# ---------------------------------------------------------------------------
# Capture mode
# ---------------------------------------------------------------------------

class TestCaptureMode:
    def test_capture_mode_emits_key_captured_signal(self, capture_worker, mocker):
        mock_captured = mocker.Mock()
        capture_worker.key_captured_signal.connect(mock_captured)

        capture_worker._on_press(keyboard.Key.shift_r)

        mock_captured.assert_called_once()

    def test_capture_mode_returns_false_to_stop_listener(self, capture_worker):
        """_on_press in capture mode must return False to stop the listener."""
        result = capture_worker._on_press(keyboard.Key.space)
        assert result is False

    def test_capture_mode_does_not_emit_ptt_signals(self, capture_worker, mocker):
        mock_pressed  = mocker.Mock()
        capture_worker.ptt_pressed_signal.connect(mock_pressed)
        mock_released = mocker.Mock()
        capture_worker.ptt_released_signal.connect(mock_released)

        capture_worker._on_press(keyboard.Key.shift_r)
        capture_worker._on_release(keyboard.Key.shift_r)

        mock_pressed.assert_not_called()
        mock_released.assert_not_called()


# ---------------------------------------------------------------------------
# Key parsing
# ---------------------------------------------------------------------------

class TestKeyParsing:
    def test_parse_named_key(self):
        with patch("hotkey_listener.keyboard.Listener"):
            worker = HotkeyWorker(ptt_key_str="keyboard.Key.ctrl_l", capture_mode=False)
        assert worker.ptt_key == keyboard.Key.ctrl_l

    def test_parse_char_key(self):
        with patch("hotkey_listener.keyboard.Listener"):
            worker = HotkeyWorker(
                ptt_key_str="keyboard.KeyCode.from_char('a')", capture_mode=False
            )
        assert worker.ptt_key == keyboard.KeyCode.from_char("a")

    def test_parse_invalid_key_falls_back_to_default(self, mocker):
        """A bad key string should fall back to shift_r and emit error_signal."""
        with patch("hotkey_listener.keyboard.Listener"):
            worker = HotkeyWorker(ptt_key_str="not_a_valid_key_format", capture_mode=False)
        # Default fallback is keyboard.Key.shift_r
        assert worker.ptt_key == keyboard.Key.shift_r

    def test_key_to_string_named_key(self):
        with patch("hotkey_listener.keyboard.Listener"):
            worker = HotkeyWorker(capture_mode=True)
        result = worker.key_to_string(keyboard.Key.shift_r)
        assert result == "keyboard.Key.shift_r"

    def test_key_to_string_char_key(self):
        with patch("hotkey_listener.keyboard.Listener"):
            worker = HotkeyWorker(capture_mode=True)
        result = worker.key_to_string(keyboard.KeyCode.from_char("z"))
        assert "from_char" in result
        assert "z" in result
