# Changelog

All notable changes to this project will be documented in this file.

## [2.2.3] - 2026-05-04

### Added
- **`pyproject.toml`**: Introduced as the single canonical source of truth for all
  dependency management, replacing the `requirements.txt` / `requirements-dev.txt` workflow.
  Managed via `uv` (≥0.11.8). Closes audit finding Section 1 (Package Manager).
- **`uv.lock`**: Deterministic lock file pinning all 91 transitive dependencies for
  fully reproducible installs across machines and CI environments.
- **Ruff linting & formatting**: Configured via `[tool.ruff]` in `pyproject.toml`.
  Rule set: `E`, `W`, `F`, `I`, `B`, `UP`, `ANN`, `D` (Google convention), `N`, `C90`.
  Line length: 88. McCabe complexity: 15. Relaxed: `ANN401`, `E501`, `D100`, `D104`.
- **Pre-commit hook** (`.pre-commit-config.yaml`): Enforces `ruff check`, `ruff format --check`,
  and `pytest` automatically on every `git commit`. Closes audit finding Section 5 (Pre-Commit).
- **`archive/`**: Legacy `requirements.txt` and `requirements-dev.txt` moved here for
  historical reference (e.g. PyInstaller / Inno Setup workflows).

### Changed
- **`pytest.ini`**: Deleted; settings migrated to `[tool.pytest.ini_options]` in `pyproject.toml`.
- **`.gitignore`**: De-duplicated and extended with `.venv/`, `.ruff_cache/`, `.pytest_cache/`.
- **Dev dependencies**: Added `ruff` and `pre-commit` to the `[dependency-groups] dev` group.

## [2.2.2] - 2026-04-22

### Added
- **Configurable Paste Delay**: Introduced a new `Paste Recovery Delay (ms)` setting in the Advanced menu. This allows users to fine-tune the delay before restoring the original clipboard content, resolving race conditions in asynchronous applications like Google Gemini.

### Changed
- **Clipboard Injection**: Refactored `DictationWorker` to utilize the user-defined `paste_delay` (defaulting to 300ms) for improved robustness across varied text input environments.

## [2.2.1] - 2026-04-20

### Added
- **AI Agent Guidelines**: Expanded `AGENTS.md` with technical safeguards for self-injection prevention, power management (sleep/wake recovery), memory-safe audio buffering, and halluncination filtering.

## [2.2.0] - 2026-04-19

### Added
- **Automated Test Suite**: Implemented a comprehensive test framework using `pytest`, `pytest-qt`, and `pytest-mock`.
  - Unit and integration tests for `DictationWorker` (audio processing, RMS gating, hallucination filtering).
  - Headless GUI state and signal tests for `OmniDictateApp`.
  - Mocked OS-level listeners and key string parsers for `HotkeyWorker`.
- **Manual Testing Protocol**: Authored `docs/manual_testing_protocol.md` for high-risk OS interactions (Admin elevation, hardware latency).
- **Environment**: Added `pytest.ini` for root configuration and `requirements-dev.txt` for testing dependencies.

### Fixed
- **Hotkey Listener**: Resolved a `TypeError` in `hotkey_listener.py` where keys with `None` virtual keycodes (`vk`) would cause a crash during string conversion.
- **Signal Handling**: Fixed a reliability issue in PySide6 signal mocking by implementing proper event loop pumping in tests.

## [2.1.0] - 2026-04-13

### Added
- Implemented `PowerMonitor` using Windows `WM_POWERBROADCAST` API for robust system wake-from-sleep detection.
- Added automatic recovery logic to restart the audio stream after system resume.
- Added a 3-second delay target to allow audio drivers to settle upon system wake.
- Added a 60-second safety limit on continuous recording segments to prevent GPU task timeouts (TDR).
- Introduced `transcription_executor` for offloaded, non-blocking audio processing.

### Changed
- **Audio Core**: Increased `CHUNK_DURATION` from 20ms to 50ms to significantly reduce CPU overhead and callback latency.
- **Threading**: Migrated transcription logic to a dedicated background thread pool, ensuring the GUI and audio levels remain responsive during processing.

### Fixed
- Fixed critical "all-zeros" audio input bug caused by stale PortAudio device handles after system sleep/wake cycles.
- Fixed `Audio Callback Error: input overflow` by offloading heavy transcription tasks from the primary audio worker thread.
- Resolved `qt.qpa.screen` errors (`0xe0000225`) caused by GPU/CPU exhaustion during excessively long recordings.

## [2.0.1] - 2026-04-05

### Added
- Added `psutil` dependency for future hardware monitoring capabilities.
- Added a directory tracking for `docs/` and a comprehensive `codebase_context.md` for better project context.

### Changed
- **Dependencies**: Migrated from CUDA 12.6 to CUDA 12.4 for broader hardware compatibility and refined PyTorch 2.6.0+cu124 support.
- **UI**: Improved button interaction styles in `style.qss`. Replaced flaky `transform: translateY(1px)` with stable `padding-top: 2px` for button-pressed states.
- **Performance**: Optimized the core typing loop in `core_logic.py`. Added a high-speed bypass for minimal-delay typing and refined the message queue processing.
- **Lifecycle**: Streamlined application shutdown procedure. Removed inefficient polling during the close event in `main_gui.py` and implemented a cleaner worker thread teardown.

### Fixed
- Fixed a potential race condition and performance lag during intensive typing tasks by optimizing the typing event loop.
- Resolved an issue where worker processes might persist in the background after the main GUI was closed.
