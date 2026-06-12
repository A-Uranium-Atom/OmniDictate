"""Tests for config module constants, enums, and data structures."""

from __future__ import annotations

from config import (
    CF_HDROP,
    DEFAULT_FILTER_WORDS,
    HALLUCINATION_PRESETS,
    DictationSettings,
    HallucinationLevel,
    InsertionMethod,
    ModelSize,
    get_punctuation_char,
)


class TestEnums:
    def test_model_size_str_equality(self) -> None:
        """StrEnum members must compare equal to their string values."""
        assert ModelSize.LARGE_V3 == "large-v3"
        assert ModelSize.TINY == "tiny"

    def test_hallucination_level_values(self) -> None:
        assert list(HallucinationLevel) == ["Low", "Medium", "High"]

    def test_insertion_method_values(self) -> None:
        assert list(InsertionMethod) == ["Paste", "Typing"]


class TestDictationSettingsDefaults:
    def test_default_construction(self) -> None:
        """DictationSettings() with no args must match all DEFAULT_* constants."""
        settings = DictationSettings()
        assert settings.model_size == ModelSize.LARGE_V3
        assert settings.language is None
        assert settings.is_vad_enabled is True
        assert settings.filter_words == DEFAULT_FILTER_WORDS

    def test_filter_words_defensive_copy(self) -> None:
        """Each instance must get its own copy of the default list."""
        s1 = DictationSettings()
        s2 = DictationSettings()
        s1.filter_words.append("extra")
        assert "extra" not in s2.filter_words


class TestGetPunctuationChar:
    def test_known_punctuation(self) -> None:
        assert get_punctuation_char("question mark") == "?"
        assert get_punctuation_char("comma") == ","

    def test_case_insensitive(self) -> None:
        assert get_punctuation_char("PERIOD") == "."

    def test_unknown_returns_none(self) -> None:
        assert get_punctuation_char("banana") is None


class TestHallucinationPresets:
    def test_all_levels_present(self) -> None:
        for level in HallucinationLevel:
            assert level in HALLUCINATION_PRESETS

    def test_medium_values(self) -> None:
        preset = HALLUCINATION_PRESETS[HallucinationLevel.MEDIUM]
        assert preset["no_speech_threshold"] == 0.6
        assert preset["log_prob_threshold"] == -1.0

    def test_string_key_access(self) -> None:
        """StrEnum keys must be accessible via plain string."""
        assert (
            HALLUCINATION_PRESETS["Medium"]
            is HALLUCINATION_PRESETS[HallucinationLevel.MEDIUM]
        )


class TestConstants:
    def test_cf_hdrop_value(self) -> None:
        assert CF_HDROP == 15
