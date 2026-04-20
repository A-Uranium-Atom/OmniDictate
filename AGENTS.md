# OmniDictate Project AI Agent Guidelines (AGENTS.md)

## 1. Project Overview & Tech Stack
You are assisting with **OmniDictate**, a Windows-based, real-time speech-to-text desktop application designed for secure, on-device transcription.
* **Language:** Python 3.x
* **Package Manager:** `uv`
* **GUI Framework:** `PySide6` (Qt)
* **Audio Capture:** `sounddevice` (PortAudio)
* **Inference Engine:** `faster-whisper` (`CTranslate2`)
* **OS/Hardware Interfacing:** `pynput`, `pywin32` (`win32clipboard`, `win32gui`), `psutil`
* **Testing:** `pytest`, `pytest-qt`
* **Linting & Formatting:** `Ruff`

## 2. Architecture & Concurrency Rules
OmniDictate relies on a strictly decoupled architecture. Never block the main UI event loop.
* **Presentation Layer (`main_gui.py`):** Runs on the main thread. Only handle UI updates, `QSettings` state persistence, and worker orchestration here.
* **Inference Engine (`core_logic.py`):** Heavy operations (matrix math, model loading) MUST occur in isolated `QThread` and `ThreadPoolExecutor` sub-threads. Maintain the hardware fallback chain strictly: CUDA `float16` -> CUDA `float32` -> CPU `int8`.
* **Global Inputs (`hotkey_listener.py`):** Always run `pynput` as a non-blocking daemon thread, transmitting events back via Qt Signals.
* **I/O Boundaries:** Maintain a clear boundary between I/O-bound operations and CPU-bound tasks. Do not mix synchronous and asynchronous I/O within the same scope.

## 3. Project-Specific Safeguards
* **Clipboard Operations:** The `_typing_loop` must always backup the clipboard, simulate text injection (`Ctrl+V`), and restore the previous memory state. If `CF_HDROP` (dragged files) is detected, you must bail out to character-by-character keypress emulation to prevent data corruption.
* **Self-Injection Prevention:** The `_typing_loop` must always validate that the active foreground window is *not* the OmniDictate application itself prior to pasting text into external windows.
* **Power Management:** Do not alter the `_PowerMonitor` daemon or Windows `PBT_APMRESUMEAUTOMATIC` hooks. PortAudio driver streams must be explicitly torn down and delayed-restarted after a system sleep/wake cycle to prevent dead handles.
* **Memory & Buffer Limits:** Audio chunk accumulation must strictly adhere to `MAX_RECORDING_SECONDS` constraints (default 60s). Never buffer indefinitely to protect RAM and GPU execution limits.
* **Quality Filtering:** Ensure output hallucination filters (comparing against `filter_words` and tracking redundant repetitive strings) are executed prior to pushing text to the final `text_queue`.
* **State Management:** Always use `settings_updated_signal(dict)` to push configuration changes dynamically to workers. Avoid reloading the ML models unless hardware parameters (like `model_size`) change.

## 4. Code Quality & Style
* **Type Hinting:** Include full and explicit type annotations for all function and method signatures.
* **Docstrings:** Require Google-style docstrings for all public functions, classes, and modules.
* **Path Management:** Strictly use `pathlib` for all file system operations; avoid `os.path`.
* **Modularity & Size Constraints:**
    * Keep functions focused on a single purpose, strictly under 50 lines.
    * Maintain a cyclomatic complexity score of under 10 for all functions.
    * Prefer creating new files over extending existing ones. Keep files under 500 lines. If a file or function must exceed these limits for explicit clarity, it is an exception, not the rule.
* **Immutability:** Never use mutable default arguments (e.g., `def func(items=[])` is forbidden).
* **Linting:** Use `Ruff` for combined formatting and linting. Conform strictly to PEP 8.

## 5. Testing & Commits
* **Testing Infrastructure:** Write tests using functional `pytest` styles. Utilize the existing `conftest.py` dependency stubs for headless, mocked hardware/clipboard testing.
* **Test Coverage:** Maintain a minimum of 80% code coverage for any newly introduced modules.
* **Pre-Commit Hook:** Before suggesting any commit, or marking a task as complete, you must run: `uv run ruff check . && uv run pytest`.

## 6. Execution & Planning
* **Implementation Plans:** Before writing major code, output a 4-step plan: 
    1. **Issue:** Define the problem.
    2. **Approach:** Propose the technical architecture.
    3. **Execution:** Outline step-by-step file modifications.
    4. **Feedback Loop:** Detail how the change will be tested.