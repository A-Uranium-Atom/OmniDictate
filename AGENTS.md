# OmniDictate Project Architecture Rules

## Tech Stack Context
* **Core Framework:** PySide6 (Qt)
* **Audio Interfacing:** sounddevice (PortAudio)
* **Inference Engine:** faster-whisper (CTranslate2)
* **OS Intercepts:** pynput, pywin32 (win32clipboard, win32gui), psutil

## 1. Mandatory Context Bootstrapping
* **Read Before You Act:** Before making *any* code or documentation change, read `docs/codebase_context.md` in full. This file is the authoritative Source of Truth for the project's architecture, data flow, threading model, and tooling. Do not rely on training data or prior assumptions.
* **Resolve Conflicts Toward the Context Doc:** If any detail in your memory conflicts with what is written in `docs/codebase_context.md`, treat the context doc as correct and flag the discrepancy in your response.

## 2. Concurrency & UI Threading
* **Strict Decoupling:** Never block the main PySide6 UI event loop in `main_gui.py`.
* **Inference Isolation:** Heavy operations (tensor translation, memory bounding) MUST occur in isolated `QThread` and `ThreadPoolExecutor` sub-threads in `core/dictation_worker.py`.
* **Daemonization:** Global inputs via `hotkey_listener.py` must run as a non-blocking daemon thread, transmitting events back via Qt Signals.
* **Fallback Chain:** Respect the CTranslate2 initialization hardware fallback chain: CUDA `float16` -> CUDA `float32` -> CPU `int8`.

## 3. Clipboard & OS Injection Safeguards
* **Injection Lifecycle:** The `_typing_loop` must follow this exact sequence:
  1. Validate window context `!=` OmniDictate.
  2. Backup current clipboard memory.
  3. Insert text to clipboard and trigger `Ctrl+V` injection via `pynput`.
  4. Restore the existing clipboard memory structurally.
* **Data Corruption Prevention:** If `CF_HDROP` (dragged files) is detected in the clipboard, you must bail out of standard injection and fallback to character-by-character keypress emulation.

## 4. State Management
* Avoid reloading the ML models globally unless hardware parameters (like `model_size`) are explicitly changed by the user.
* Route configuration state changes to workers dynamically via `settings_updated_signal(dict)`.

## 5. Documentation & Version Control
* **Agent-to-Agent Communication:** When writing documentation, assume the reader could be another AI agent. Use highly structured, consistent formatting. Provide clear technical reasoning (the "why"), explicit file paths, and detailed context so the next agent or human can seamlessly pick up where you left off without ambiguity.
* **Changelog & Versioning:** After completing a task, decide if it warrants a version bump. If so:
  1. Update `CHANGELOG.md` with specific changes, what was fixed/added, and the technical reasoning.
  2. Synchronize the new version string in `pyproject.toml` (`version = "..."`).
  3. Synchronize the new version string in `OmniDictate_Setup.iss` (`AppVersion` and `OutputBaseFilename`).
* **Dependency Lockfile:** If you add or modify dependencies in `pyproject.toml`, you MUST run `uv lock` to regenerate the `uv.lock` file.
* **Codebase Context:** If a feature or architectural pattern changes, update `docs/codebase_context.md`. Treat this file as the definitive Source of Truth. Every update must: (1) explain the *why* behind the design decision, not just the *what*; (2) reference the exact file path, class, method, or signal involved; (3) update the Data Flow section if the change affects how data moves between threads or components.
* **README:** For major user-facing changes or new features, update `README.md`. Follow its current visual style and keep installation/usage instructions accurate.

## 6. Testing & Pre-Commit Discipline
* **Run Tests First:** Before proposing any code change, confirm the existing test suite passes with `uv run pytest`. This establishes a clean baseline. Never skip this step.
* **Write Tests for New Logic:** Any new function, class, or data-flow branch in `core/dictation_worker.py`, `main_gui.py`, or `hotkey_listener.py` MUST have a corresponding test in `tests/`. Mock OS-level dependencies using the patterns already established in `tests/conftest.py`.
* **Pre-Commit Is Mandatory:** All commits run `ruff check` → `ruff format --check` → `pytest` automatically via `.pre-commit-config.yaml`. Do NOT use `git commit --no-verify` except in genuine emergencies. Do not suppress linter warnings with `# noqa` unless accompanied by an explicit inline comment explaining why.

## 7. Qt Signal / Slot Contract
* **Cross-Thread Calls via Signals Only:** Never call a method on a `QThread` or `QObject` worker directly from a different thread. All cross-thread communication MUST go through Qt signals and slots, which are thread-safe by design.
* **No Direct UI Mutations from Workers:** Worker threads (`DictationWorker`, `HotkeyWorker`) must never touch any `QWidget` property directly. Emit a signal and let `main_gui.py` perform the update on the main thread.
* **Settings Propagation Pattern:** Push runtime configuration changes to workers via `settings_updated_signal(dict)`. Do not re-instantiate a worker to change a setting — this drops the audio stream.

## 8. Dependency & Environment Hygiene
* **`pyproject.toml` Is the Source of Truth:** Never install packages with bare `pip install`. All dependency changes go through `uv add <package>` (runtime) or `uv add --group dev <package>` (dev-only).
* **Do Not Touch Archive Files:** `archive/requirements.txt` and `archive/requirements-dev.txt` exist for PyInstaller/Inno Setup reference only. Do not modify them.
* **Verify Package Authenticity:** Before adding any new dependency, confirm it exists on PyPI and is not a hallucinated package name. Prefer packages already in the dependency graph over introducing new ones.