"""UI component package for OmniDictate.

Submodules:
    icons           — Vector icon rendering and key name formatting.
    dictation_page  — Main dictation page widget.
    settings_page   — Settings/configuration page widget.
"""

from ui.dictation_page import DictationPage
from ui.settings_page import SettingsPage

__all__ = ["DictationPage", "SettingsPage"]
