"""Tests for clipboard paste and text injection."""

from __future__ import annotations

from config import DictationSettings
from core.dictation_worker import DictationWorker
from core.text_injector import paste_text


def _make_worker(
    mock_faster_whisper: object,
    gui_wid: int = 0,
    **overrides: object,
) -> DictationWorker:
    """Return a DictationWorker with sensible test defaults."""
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


class TestClipboardPaste:
    def test_paste_restores_original_clipboard(self, mocker) -> None:
        mock_kb = mocker.patch("core.text_injector.Controller")

        mock_open = mocker.patch("win32clipboard.OpenClipboard")
        mock_close = mocker.patch("win32clipboard.CloseClipboard")
        mock_empty = mocker.patch("win32clipboard.EmptyClipboard")
        mock_set_data = mocker.patch("win32clipboard.SetClipboardData")

        def mock_enum(format_id):
            if format_id == 0:
                return 13
            return 0

        mocker.patch("win32clipboard.EnumClipboardFormats", side_effect=mock_enum)
        mocker.patch(
            "win32clipboard.GetClipboardData",
            return_value=b"original_data",
        )

        mocker.patch("win32clipboard.CF_UNICODETEXT", 13)

        success = paste_text("new_text", mock_kb, paste_delay=0.0)

        assert success is True
        mock_open.assert_called()
        mock_empty.assert_called()
        mock_set_data.assert_any_call(13, "new_text")
        mock_set_data.assert_any_call(13, b"original_data")
        mock_close.assert_called()

    def test_paste_falls_back_on_file_drop_format(self, mocker) -> None:
        mock_kb = mocker.patch("core.text_injector.Controller")
        mocker.patch("win32clipboard.OpenClipboard")
        mocker.patch("win32clipboard.CloseClipboard")

        def mock_enum(format_id):
            if format_id == 0:
                return 15
            return 0

        mocker.patch("win32clipboard.EnumClipboardFormats", side_effect=mock_enum)
        mocker.patch("config.CF_HDROP", 15)

        success = paste_text("new_text", mock_kb, paste_delay=0.0)

        assert success is False

    def test_typing_loop_skips_own_window(
        self, mocker, qtbot, mock_faster_whisper
    ) -> None:
        worker = _make_worker(mock_faster_whisper, gui_wid=12345)

        mocker.patch(
            "core.text_injector.ctypes.windll.user32.GetForegroundWindow",
            return_value=12345,
        )
        mock_kb = mocker.patch("core.text_injector.Controller")
        mock_paste = mocker.patch("core.text_injector.paste_text", return_value=True)

        mocker.patch("pythoncom.CoInitializeEx")
        mocker.patch("pythoncom.CoUninitialize")
        worker.text_queue.put("Hello")

        worker.stop_typing_event.set()

        worker._typing_loop()

        mock_paste.assert_not_called()
        mock_kb.type.assert_not_called()
