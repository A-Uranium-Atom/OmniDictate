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