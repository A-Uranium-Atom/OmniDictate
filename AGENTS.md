# OmniDictate Project Architecture Rules

## Tech Stack Context
* **Core Framework:** PySide6 (Qt)
* **Audio Interfacing:** sounddevice (PortAudio)
* **Inference Engine:** faster-whisper (CTranslate2)
* **OS Intercepts:** pynput, pywin32 (win32clipboard, win32gui), psutil

## 1. Concurrency & UI Threading
* **Strict Decoupling:** Never block the main PySide6 UI event loop in `main_gui.py`.
* **Inference Isolation:** Heavy operations (tensor translation, memory bounding) MUST occur in isolated `QThread` and `ThreadPoolExecutor` sub-threads in `core_logic.py`.
* **Daemonization:** Global inputs via `hotkey_listener.py` must run as a non-blocking daemon thread, transmitting events back via Qt Signals.
* **Fallback Chain:** Respect the CTranslate2 initialization hardware fallback chain: CUDA `float16` -> CUDA `float32` -> CPU `int8`.

## 2. Clipboard & OS Injection Safeguards
* **Injection Lifecycle:** The `_typing_loop` must follow this exact sequence:
  1. Validate window context `!=` OmniDictate.
  2. Backup current clipboard memory.
  3. Insert text to clipboard and trigger `Ctrl+V` injection via `pynput`.
  4. Restore the existing clipboard memory structurally.
* **Data Corruption Prevention:** If `CF_HDROP` (dragged files) is detected in the clipboard, you must bail out of standard injection and fallback to character-by-character keypress emulation.

## 3. State Management
* Avoid reloading the ML models globally unless hardware parameters (like `model_size`) are explicitly changed by the user.
* Route configuration state changes to workers dynamically via `settings_updated_signal(dict)`.

## 4. Documentation & Version Control
* **Agent-to-Agent Communication:** When writing documentation, assume the reader could be another AI agent. Use highly structured, consistent formatting. Provide clear technical reasoning (the "why"), explicit file paths, and detailed context so the next agent or human can seamlessly pick up where you left off without ambiguity.
* **Changelog & Versioning:** After completing a task, decide if it warrants a version bump. If so:
  1. Update `CHANGELOG.md` with specific changes, what was fixed/added, and the technical reasoning.
  2. Synchronize the new version string in `pyproject.toml` (`version = "..."`).
  3. Synchronize the new version string in `OmniDictate_Setup.iss` (`AppVersion` and `OutputBaseFilename`).
* **Dependency Lockfile:** If you add or modify dependencies in `pyproject.toml`, you MUST run `uv lock` to regenerate the `uv.lock` file.
* **Codebase Context:** If a feature or architectural pattern changes, update `docs/codebase_context.md`. Treat this file as the definitive Source of Truth. Seamlessly integrate changes with consistent detail, ensuring future agents can easily map the project's data flow.
* **README:** For major user-facing changes or new features, update `README.md`. Follow its current visual style and keep installation/usage instructions accurate.