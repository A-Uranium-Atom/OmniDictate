# OmniDictate Codebase Audit Report

This report summarizes violations of the **Global AI Agent Guidelines** and the project-specific **AGENTS.md** file identified during the source code audit conducted on April 25, 2026.

## Executive Summary

The codebase is functional and architecturally sound in terms of concurrency and core logic (using `QThread`, `ThreadPoolExecutor`, and proper Windows API hooks). However, it significantly violates modularity, type safety, and documentation standards. The most critical issues are the excessive lengths of `main_gui.py` and `core_logic.py`, and the near-total absence of type annotations and docstrings.

---

## 1. High Severity Violations

| File | Line | Rule Violated | Violation Description | Recommended Refactoring |
| :--- | :--- | :--- | :--- | :--- |
| `main_gui.py` | - | File length < 500 lines | The file is **959 lines** long (increased from 864 in previous audit). | Split `OmniDictateApp` into smaller components. **Goal**: Modularize UI setup into separate widget class files or modules. |
| `core_logic.py` | - | File length < 500 lines | The file is **614 lines** long (increased from 552 in previous audit). | Move `_PowerMonitor` and `_paste_text` / `_typing_loop` logic into dedicated modules (e.g., `power_monitor.py`, `text_injector.py`). |
| `main_gui.py` | 318 | Function length < 50 lines | `setup_settings_page` is **161 lines** long. | Extract sections (Model settings, Audio settings, Filter words) into smaller helper functions or a dedicated `SettingsWidget` class. |
| `main_gui.py` | 226 | Function length < 50 lines | `setup_dictation_page` is **91 lines** long. | Break down UI construction into logical blocks (Header, Display Area, Controls). |
| `main_gui.py` | 65 | Function length < 50 lines | `__init__` is **76 lines** long. | Move signal connections and initial state setup into dedicated methods. |
| `core_logic.py` | 507 | Function length < 50 lines | `_paste_text` is **72 lines** long. | Decompose into `_backup_clipboard`, `_perform_paste`, and `_restore_clipboard`. |
| `All Files` | - | Full type annotations | Nearly all methods in `main_gui.py`, `core_logic.py`, and `hotkey_listener.py` lack type hints for parameters and return types. | Add explicit type hints for all parameters and return values (e.g., `def start_dictation(self) -> None:`). |
| `All Files` | - | Google-style docstrings | Most public classes and methods lack docstrings. Documentation is particularly sparse for complex internal logic. | Implement full Google-style docstrings for all public interfaces. |

---

## 2. Medium Severity Violations

| File | Line | Rule Violated | Violation Description | Recommended Refactoring |
| :--- | :--- | :--- | :--- | :--- |
| `main_gui.py` | 936-938, 950-951 | Path management (`pathlib`) | Usage of `os.path.dirname`, `os.path.join`, and `os.path.exists`. | Replace with `pathlib.Path(__file__).parent`. |
| `compress_video.py` | 9, 48 | Path management (`pathlib`) | Usage of `os.path.exists` and `os.getsize`. | Replace with `pathlib.Path`. |
| `core_logic.py` | 259 | Function length < 50 lines | `start_processing` is **51 lines** long. | Extract queue clearing and audio stream initialization into sub-methods. |
| `core_logic.py` | 106 | Class complexity | `DictationWorker` handles audio capture, VAD logic, model loading, and typing loop orchestration. | Decouple audio capture and VAD into a `StreamProcessor` and model inference into an `InferenceEngine`. |

---

## 3. Low Severity Violations

| File | Line | Rule Violated | Violation Description | Recommended Refactoring |
| :--- | :--- | :--- | :--- | :--- |
| `core_logic.py` | 47 | Google-style docstring | `get_punctuation_char` has a docstring but lacks the "Args" and "Returns" sections. | Update to strict Google format. |
| `hotkey_listener.py`| 14 | Type safety | `__init__` parameters have default values but no type hints. | Add type hints. |
| `main_gui.py` | 142 | Docstring consistency | `create_gear_icon` docstring is brief and lacks detail. | Expand to explain parameters and output. |

---

## 4. Compliance Check: AGENTS.md Safeguards

| Requirement | Status | Evidence |
| :--- | :--- | :--- |
| **Clipboard Backup/Restore** | ✅ PASS | Implemented in `core_logic.py`: `_paste_text` (lines 512-575). |
| **CF_HDROP Bailout** | ✅ PASS | Implemented in `core_logic.py`: `_paste_text` (lines 521-523). |
| **Self-Injection Prevention** | ✅ PASS | Implemented in `core_logic.py`: `_typing_loop` (line 593). |
| **Power Management Hooks** | ✅ PASS | Implemented in `core_logic.py`: `_PowerMonitor` (lines 54-103) and `main_gui.py`: `_handle_auto_restart` (lines 790-813). |
| **Hardware Fallback Chain** | ✅ PASS | Implemented in `core_logic.py`: `load_model` (lines 224-252). |
| **Memory Monitoring** | ✅ PASS | Implemented in `core_logic.py`: `_finalize_transcription` (lines 485-491). |
| **Max Recording Seconds** | ✅ PASS | Implemented in `core_logic.py`: `CHUNK_DURATION` / `MAX_RECORDING_SECONDS` check (lines 397-402). |

## 5. Resolved Violations

The following violations have been successfully addressed:

| Rule Violated | Resolution Details | Date Resolved |
| :--- | :--- | :--- |
| **Package Management** (Missing `pyproject.toml`) | Migrated from `requirements.txt` to `uv` + `pyproject.toml`. Pinned all dependencies in `uv.lock`. | 2026-05-04 |
| **Pre-Commit Automation** | Implemented `pre-commit` framework executing `ruff check`, `ruff format`, and `pytest` automatically. | 2026-05-04 |

---

## Conclusion

The OmniDictate codebase is robust but requires a significant "cleanup" phase to align with the modularity and documentation standards defined in the project's own guidelines. With Phase 1 (Environment & Tooling) completed, the priority should be **refactoring `main_gui.py` into multiple files** and **adding comprehensive type hinting** across the entire project. This audit incorporates historical violations noted on 2026-04-20 and reflects the current, further degraded state of modularity (increased file lengths).
