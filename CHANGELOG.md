# Changelog

All notable changes to this project will be documented in this file.

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
