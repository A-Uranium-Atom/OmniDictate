"""Shared configuration constants, enums, and data structures for OmniDictate."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class ModelSize(StrEnum):
    """Valid Whisper model identifiers accepted by faster-whisper."""

    LARGE_V3_TURBO = "large-v3-turbo"
    LARGE_V3 = "large-v3"
    MEDIUM = "medium"
    SMALL = "small"
    BASE = "base"
    TINY = "tiny"


class HallucinationLevel(StrEnum):
    """Controls aggressiveness of no-speech probability filtering."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class InsertionMethod(StrEnum):
    """Text delivery mechanism to the target application window."""

    PASTE = "Paste"
    TYPING = "Typing"


CONFIG_ORG: str = "OmniCorp"
CONFIG_APP: str = "OmniDictate"

SAMPLE_RATE: int = 16_000
CHUNK_DURATION: float = 0.05
CHUNK_SIZE: int = int(SAMPLE_RATE * CHUNK_DURATION)  # = 800
MAX_RECORDING_SECONDS: int = 30
SILENCE_DURATION: float = 0.5  # Seconds of silence before VAD stops recording

CF_HDROP: int = 15  # Win32 clipboard format for dragged file lists

HALLUCINATION_PRESETS: dict[HallucinationLevel, dict[str, float]] = {
    HallucinationLevel.LOW: {
        "no_speech_threshold": 0.8,
        "log_prob_threshold": -1.5,
    },
    HallucinationLevel.MEDIUM: {
        "no_speech_threshold": 0.6,
        "log_prob_threshold": -1.0,
    },
    HallucinationLevel.HIGH: {
        "no_speech_threshold": 0.4,
        "log_prob_threshold": -0.7,
    },
}

DEFAULT_MODEL_SIZE: str = ModelSize.LARGE_V3
DEFAULT_LANGUAGE: None = None
DEFAULT_VAD_ENABLED: bool = True
DEFAULT_SILENCE_THRESHOLD: int = 500
DEFAULT_CHAR_DELAY: float = 0.02
DEFAULT_PTT_KEY_STR: str = "keyboard.Key.shift_r"
DEFAULT_RMS_THRESHOLD: float = 0.01
DEFAULT_HALLUCINATION_FILTER: str = HallucinationLevel.MEDIUM
DEFAULT_INSERTION_METHOD: str = InsertionMethod.PASTE
DEFAULT_PASTE_DELAY: float = 0.3

DEFAULT_FILTER_WORDS: list[str] = [
    "thank you",
    "thanks for watching",
    "thanks for listening",
    "i'm sorry",
    "subtitles by",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "you",
]


def get_punctuation_char(punctuation_name: str) -> str | None:
    """Return the punctuation character for a spoken command name.

    Args:
        punctuation_name: The verbal command (e.g. "question mark", "comma").

    Returns:
        The corresponding single character, or None if not recognized.
    """
    punctuation_map: dict[str, str] = {
        "question mark": "?",
        "exclamation mark": "!",
        "comma": ",",
        "period": ".",
        "full stop": ".",
        "colon": ":",
        "semicolon": ";",
        "open parenthesis": "(",
        "close parenthesis": ")",
        "open bracket": "[",
        "close bracket": "]",
        "open brace": "{",
        "close brace": "}",
        "hyphen": "-",
        "dash": "-",
        "underscore": "_",
        "plus": "+",
        "equals": "=",
        "at": "@",
        "hash": "#",
        "dollar": "$",
        "percent": "%",
        "caret": "^",
        "ampersand": "&",
        "asterisk": "*",
    }
    return punctuation_map.get(punctuation_name.lower())


@dataclass
class DictationSettings:
    """Structured configuration passed between the GUI and DictationWorker.

    Replaces the untyped ``dict`` previously used by
    ``settings_updated_signal``. Every field has a safe default matching the
    application's original default values.
    """

    model_size: ModelSize = ModelSize.LARGE_V3
    language: str | None = None
    is_vad_enabled: bool = True
    silence_threshold: int = 500
    char_delay: float = 0.02
    ptt_key_str: str = DEFAULT_PTT_KEY_STR
    rms_threshold: float = 0.01
    hallucination_filter: HallucinationLevel = HallucinationLevel.MEDIUM
    insertion_method: InsertionMethod = InsertionMethod.PASTE
    paste_delay: float = 0.3
    filter_words: list[str] = field(default_factory=lambda: list(DEFAULT_FILTER_WORDS))
