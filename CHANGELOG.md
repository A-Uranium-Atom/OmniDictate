# Changelog

All notable changes to this project will be documented in this file.

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
