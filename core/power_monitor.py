"""Windows power-event monitor for sleep/wake detection."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    import win32con
    import win32gui

    _HAS_WIN32: bool = True
except ImportError:
    _HAS_WIN32 = False


class PowerMonitor:
    """Monitors Windows power events via a hidden message-only window.

    Runs a background daemon thread with a Win32 message pump. When
    ``PBT_APMRESUMEAUTOMATIC`` or ``PBT_APMRESUMESUSPEND`` is received,
    ``on_resume`` is invoked immediately from the monitor thread.

    If ``win32gui`` is not available (non-Windows or missing pywin32),
    ``start()`` becomes a safe no-op and logs a warning.

    Args:
        on_resume: Callback invoked on system wake. Must be thread-safe
            (e.g. emitting a Qt signal, which is safe by design).
    """

    def __init__(self, on_resume: Callable[[], None]) -> None:
        """Initialize the PowerMonitor with a callback for system resume."""
        self._on_resume = on_resume
        self._hwnd: int | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="PowerMonitorThread"
        )

    def start(self) -> None:
        """Start the hidden window thread to listen for power events."""
        if not _HAS_WIN32:
            print(
                "PowerMonitor: win32gui not available; sleep/wake detection disabled."
            )
            return
        self._thread.start()
        print("PowerMonitor: Started.")

    def stop(self) -> None:
        """Stop the hidden window thread and clean up resources."""
        if self._hwnd is not None:
            try:
                win32gui.PostMessage(self._hwnd, win32con.WM_QUIT, 0, 0)
            except Exception:
                pass

    def _wndproc(self, hwnd: int, msg: int, wparam: int, lparam: int) -> int:
        if msg == win32con.WM_POWERBROADCAST:
            if wparam in (
                win32con.PBT_APMRESUMEAUTOMATIC,
                win32con.PBT_APMRESUMESUSPEND,
            ):
                print("PowerMonitor: System wake detected.")
                try:
                    self._on_resume()
                except Exception as exc:
                    print(f"PowerMonitor: Error in on_resume callback: {exc}")
        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _run(self) -> None:
        try:
            wc = win32gui.WNDCLASS()
            wc.lpfnWndProc = self._wndproc
            wc.lpszClassName = "OmniDictatePowerMonitor"
            atom = win32gui.RegisterClass(wc)
            self._hwnd = win32gui.CreateWindow(
                atom, "PowerMonitor", 0, 0, 0, 0, 0, 0, 0, 0, None
            )
            win32gui.PumpMessages()
        except Exception as exc:
            print(f"PowerMonitor: Thread error: {exc}")
