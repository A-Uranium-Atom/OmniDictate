"""Text injection via clipboard paste or keystroke emulation."""

from __future__ import annotations

import ctypes
import queue
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pynput.keyboard import Controller, Key

from config import CF_HDROP, InsertionMethod

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass
class InjectorSettings:
    """Mutable settings container shared between DictationWorker and the typing loop.

    The worker creates one instance and passes it to ``run_typing_loop``.
    When the user changes settings mid-session, the worker updates these
    fields in-place. The typing loop reads them on each iteration.
    Thread-safety relies on the GIL for atomic reads of these simple types.

    Attributes:
        insertion_method: Whether to paste via clipboard or type character-by-character.
        char_delay: Seconds between each keystroke in typing mode.
        paste_delay: Seconds to wait after Ctrl+V before restoring the clipboard.
    """

    insertion_method: InsertionMethod = InsertionMethod.PASTE
    char_delay: float = 0.02
    paste_delay: float = 0.3


def paste_text(  # noqa: C901
    text: str,
    keyboard_controller: Controller,
    paste_delay: float,
    warning_callback: Callable[[str], None] | None = None,
) -> bool:
    """Inject text via clipboard paste with full backup and restore.

    Follows the AGENTS.md injection lifecycle:
    1. Backup all current clipboard formats.
    2. Set new text as CF_UNICODETEXT.
    3. Simulate Ctrl+V.
    4. Wait ``paste_delay`` seconds for the target app to process.
    5. Restore the original clipboard contents.

    If CF_HDROP (file drag) is detected in the clipboard, immediately
    returns False to signal the caller to fall back to typing.

    Args:
        text: The string to paste.
        keyboard_controller: A pynput keyboard Controller instance.
        paste_delay: Seconds to wait after Ctrl+V before restoring.
        warning_callback: Optional callback for non-fatal warnings.

    Returns:
        True if paste succeeded, False if caller should fall back to typing.
    """
    import win32clipboard  # Local import: Windows-only, matches existing pattern

    clipboard_backup: dict[int, object] = {}
    is_fallback_needed: bool = False

    try:
        win32clipboard.OpenClipboard()
        format_id = win32clipboard.EnumClipboardFormats(0)
        while format_id != 0:
            if format_id == CF_HDROP:
                is_fallback_needed = True
                break
            try:
                data = win32clipboard.GetClipboardData(format_id)
                if data is not None:
                    clipboard_backup[format_id] = data
            except Exception:
                is_fallback_needed = True
                break
            format_id = win32clipboard.EnumClipboardFormats(format_id)
    except Exception:
        is_fallback_needed = True
    finally:
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass

    if is_fallback_needed:
        if warning_callback is not None:
            warning_callback(
                "Complex clipboard object detected. Falling back to typing to protect clipboard."
            )
        return False

    try:
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
        win32clipboard.CloseClipboard()
    except Exception as e:
        print(f"Error setting clipboard: {e}")
        try:
            win32clipboard.CloseClipboard()
        except Exception:
            pass
        return False

    time.sleep(0.05)
    keyboard_controller.press(Key.ctrl)
    keyboard_controller.press("v")
    keyboard_controller.release("v")
    keyboard_controller.release(Key.ctrl)
    time.sleep(paste_delay)

    if clipboard_backup:
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            for fmt, data in clipboard_backup.items():
                try:
                    win32clipboard.SetClipboardData(fmt, data)
                except Exception:
                    pass
        except Exception:
            pass
        finally:
            try:
                win32clipboard.CloseClipboard()
            except Exception:
                pass

    return True


def run_typing_loop(
    text_queue: queue.Queue[str],
    stop_event: threading.Event,
    is_running: Callable[[], bool],
    gui_wid: int,
    settings: InjectorSettings,
    error_callback: Callable[[str], None],
    warning_callback: Callable[[str], None],
) -> None:
    """Thread target: consumes text from a queue and injects it into the active window.

    Polls ``text_queue`` with a 0.5s timeout. For each item:
    1. Checks the foreground window is not OmniDictate (prevents self-injection).
    2. If insertion_method is Paste, attempts clipboard injection via ``paste_text``.
    3. If paste fails or insertion_method is Typing, falls back to keystroke emulation.

    Initializes COM (pythoncom) for win32clipboard access and uninitializes on exit.

    Args:
        text_queue: Queue of strings to inject. Populated by the transcription pipeline.
        stop_event: Threading event; set when the worker wants the loop to exit.
        is_running: Callable returning True while the worker is active.
        gui_wid: Window handle (HWND) of the OmniDictate main window.
        settings: Shared mutable settings; read each iteration for current values.
        error_callback: Called with error message strings on failure.
        warning_callback: Called with warning message strings (e.g. clipboard fallback).
    """
    print(f"Typing thread started, ID: {threading.get_ident()}")
    import pythoncom

    pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
    try:
        try:
            keyboard_controller = Controller()
        except ImportError:
            error_callback("pynput not installed.")
            return

        while is_running() and not stop_event.is_set():
            try:
                text_to_type = text_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            except Exception as e:
                error_callback(f"Typing queue error: {e}")
                time.sleep(0.1)
                continue

            try:
                hwnd = ctypes.windll.user32.GetForegroundWindow()
                if hwnd == gui_wid:
                    print("Skipping typing: OmniDictate window active.")
                    continue

                is_paste_successful: bool = False
                if settings.insertion_method == InsertionMethod.PASTE:
                    is_paste_successful = paste_text(
                        text_to_type,
                        keyboard_controller,
                        settings.paste_delay,
                        warning_callback,
                    )

                if not is_paste_successful:
                    if settings.char_delay <= 0.001:
                        keyboard_controller.type(text_to_type)
                    else:
                        for char in text_to_type:
                            if not is_running() or stop_event.is_set():
                                break
                            keyboard_controller.press(char)
                            keyboard_controller.release(char)
                            time.sleep(settings.char_delay)
            except Exception as e:
                error_callback(f"Error inserting text: {e}")
    finally:
        pythoncom.CoUninitialize()
        print(f"Typing thread exiting, ID: {threading.get_ident()}")
