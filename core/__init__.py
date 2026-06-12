"""Core processing package for OmniDictate.

Submodules:
    power_monitor    — Windows sleep/wake detection daemon.
    text_injector    — Clipboard paste and character-by-character typing.
    dictation_worker — Audio capture, VAD, and transcription.
"""

from core.dictation_worker import DictationWorker

__all__ = ["DictationWorker"]
