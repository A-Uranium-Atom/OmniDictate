# OmniDictate Codebase Context

## Overview
OmniDictate is a Windows-based, real-time speech-to-text desktop application designed for secure, on-device transcription with sub-second latency. It captures audio input from the user's microphone, processes it locally using the Faster Whisper Large Language Model, and intelligently injects the transcribed text directly into the user's active window context. The application provides an elegant graphical interface built with Qt (`PySide6`) and strictly separates its presentation layer from its asynchronous hardware/inference engines to prevent UI blocking.

## High-Level Architecture & Threading Model
The architecture is heavily decoupled using Python threads and Qt's event loop. It relies on four primary domains to ensure uninterrupted real-time audio capture and seamless window text output.

1. **Presentation & Orchestration Layer (`main_gui.py`)** runs on the main thread.
2. **Inference & Processing Engine (`core_logic.py`)** runs in an isolated `QThread` with sub-threads for task execution, audio polling, and typing.
3. **Global Input Monitor (`hotkey_listener.py`)** runs a daemon thread to monitor OS-level input hooks without blocking.
4. **Testing Infrastructure (`tests/`)** utilizes dependency mocking for offline, headless regression tests.

---

### 1. Presentation & Orchestration (`main_gui.py`)
This is the application's lifecycle manager. It houses the `OmniDictateApp` class (extending `QMainWindow`).

**Key Responsibilities:**
- **UI Management**: Houses a Stacked Widget switching between the primary Dictation Page (transcription log, settings toggle, visual timer, and audio levels) and the Settings Page.
- **State Persistence**: Serializes and initializes user parameters via `QSettings` (e.g., model size, hallucination thresholds, VAD preferences, insertion methods, and custom filter words).
- **Thread Orchestration**: Responsible for instantiating the `DictationWorker` and `HotkeyWorker`. It cleanly starts, signals, and terminates these workers via Qt `Signals` and `Slots`. 
- **Dynamic Updates**: If a user updates settings mid-session, this class emits a `settings_updated_signal(dict)` that updates internal parameters for the workers on-the-fly, reloading models only if necessary (e.g., when `model_size` changes).

---

### 2. Inference & Processing Engine (`core_logic.py`)
This is the heavy hitter of the application, managing raw microphone data, matrix operations, and OS-level keyboard/clipboard simulations. It centers around the `DictationWorker` (extending `QObject`).

**Sub-Components & Logic:**
- **Model Loading & Fallbacks**: Utilizes `CTranslate2` under-the-hood. It implements an intelligent hardware fallback chain: attempts CUDA `float16` -> falls back to CUDA `float32` (if precision crashes) -> falls back to CPU `int8`.
- **Audio Capture (`sounddevice`)**: Ingests raw `int16` buffer chunks in `50ms` fragments continuously on a PortAudio callback background thread. Pushes these non-blockingly into a Python `queue.Queue()`.
- **VAD & PTT Logic**: 
  - Runs a `QTimer` polling the audio queue.
  - Calculates the Root Mean Square (RMS) amplitude of audio chunks.
  - If amplitude passes `rms_threshold` (Voice Activity Detection mode) or if the Push-to-Talk (PTT) key is pressed down, audio chunks accumulate in `audio_buffer`.
  - When speech concludes (based on a `silence_frames` timeout) or hits the `MAX_RECORDING_SECONDS` (60s limit), the buffer is dispatched.
- **Transcription Context Offloading**: To prevent the audio stream and event loop from hanging, model transcription executes inside a `ThreadPoolExecutor`.
- **Post-Processing Filtering**:
  - Validates `no_speech_prob` per-segment to strip noise.
  - Filters out known Whisper artifacts using the user-defined `filter_words` array.
  - Detects if the model repeatedly hallucinates the exact same phrase multiple times in a row, actively scrubbing redundant strings.
- **Typing & Insertion Thread (`_typing_loop`)**:
  - A detached background thread consuming textual outputs via `text_queue`.
  - Polling active window via `ctypes.windll.user32.GetForegroundWindow()` to prevent recursive inserting into the OmniDictate app itself.
  - **Clipboard Injection**: Backs up the user's active clipboard via `win32clipboard`, places the new text, simulates a `Ctrl+V` keystroke via `pynput`, and immediately restores all prior rich-text/file formats to the clipboard context. If the clipboard has complex dragged files (`CF_HDROP`), it bails out to protect data and falls back to character-by-character keypress emulation.
- **Power Management (`_PowerMonitor`)**: A daemon thread hooking into Windows `win32gui` messages to explicitly track `PBT_APMRESUMEAUTOMATIC`. It forces an automated tear-down and delayed reboot of the audio driver queue after a system wakes from sleep to prevent dead PortAudio handles.

---

### 3. Global Input Monitor (`hotkey_listener.py`)
Decoupled OS-level key interception to trigger application behavior without active window focus.

**Key Responsibilities:**
- **`HotkeyWorker`**: A background daemon that spins up a `pynput.keyboard.Listener`.
- Listens for specific user-mapped Push-To-Talk actions globally.
- Transmits events immediately back to the main thread via decoupled Qt signals (`ptt_pressed_signal`, `ptt_released_signal`).
- Exposes a `capture_mode` configuration utilized by `main_gui.py` to allow end-users to easily remap new keystrokes on the fly.

---

### 4. Testing Infrastructure (`tests/`)
A completely isolated automated test suite to validate regression on CI pipelines that lack audio hardware or GPUs.
- **Frameworks Frameworks**: Utilizes `pytest` / `pytest-qt`.
- **Dependency Stubs**: The `conftest.py` file fully mocks out the `sounddevice` stream, Windows Clipboard hooks, and Faster-Whisper.
- **GUI Validations (`test_main_gui.py`)**: Tests initial variable configurations, ensuring `QSettings` load, asserting UI layout updates when migrating between VAD/PTT settings.
- **Inference Emulation (`test_core_logic.py`)**: Tests logic loops such as VAD duration limits, repetition/hallucination filtering, and safely bypassing complex clipboard formats. 
- **Hotkey Validation (`test_hotkey_listener.py`)**: Confirms `pynput` correctly translates keycodes to human-readable strings and appropriately binds PTT handlers.

---

## Data Flow (End-to-End Transcription)
For AI agents assessing the flow of a single input request, the data travels sequentially across the following isolated concurrently running processes:

1. **Hardware Ingestion**: `sounddevice` receives unblocked raw `int16` fragments -> queues into `audio_queue`.
2. **Buffer Accumulation**: `DictationWorker` measures RMS > Threshold. Successive fragments form the `audio_buffer`.
3. **Execution Delegate**: Silence timeout is reached -> Array concatenates to `float32` -> submitted to `ThreadPoolExecutor`.
4. **CTranslate2 Evaluation**: The Faster Whisper underlying ML model transcribes strings.
5. **Quality Filtering**: Output strings are checked against the artifact lists and repetition buffers.
6. **Cross-Thread Transport**: Emitted to `main_gui.py` to render the display log, and enqueued to `text_queue`.
7. **OS Interception**: `_typing_loop` validates window context `!=` OmniDictate. 
8. **Final Delivery**: `win32clipboard` backs up memory -> text inserted to clipboard -> `Ctrl+V` injected via `pynput` -> existing clipboard memory structurally restored.

## Major Dependencies & Purposes
- **PySide6**: Core lifecycle, threading abstraction, layout constraints, signal routing, and application visual components.
- **sounddevice**: Raw audio hardware interfacing over PortAudio endpoints.
- **faster-whisper (CTranslate2)**: The actual C++ optimized CPU/CUDA engine managing tensor translation with memory-efficient bounds.
- **torch / numpy**: Vector and array manipulation structures fed into the ASR modeling pipeline.
- **pynput**: Keyboard macro simulation for typing injection and daemonized background hotkey catching for PTT.
- **pywin32 (`win32gui`, `win32clipboard`)**: Explicit, low-level OS mapping utilized for system-sleep triggers and robust text injection to external graphical windows.
- **psutil**: Memory diagnostics utilized to track RAM leaks and garbage collection bottlenecks.
- **pytest / pytest-qt**: Testing ecosystem providing mocked interfaces for Qt lifecycle assertions without rendering X11 contexts.
