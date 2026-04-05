# OmniDictate Codebase Context

## Overview
OmniDictate is a Windows-based, real-time speech-to-text desktop application. It captures audio input from the user's microphone, translates it into text securely and rapidly via an on-device local Large Language Model (Faster Whisper), and intelligently pastes or types the output seamlessly into the user's active window context. The application provides an elegant GUI built with Qt (`PySide6`) and runs inference operations asynchronously to keep latency sub-second.

## Architecture

The system is divided into four main asynchronous domains to prevent GUI freezing and ensure uninterrupted real-time audio capturing and text output:

1. **`main_gui.py` (Presentation & Orchestration Layer)**  
   - Implements `OmniDictateApp` extending from `QMainWindow`.
   - Manages UI events, visual updates (audio levels, text transcription view log), and the settings interface.
   - Saves user parameters persistently using `QSettings`.
   - Serves as the central manager orchestrating various uncoupled workers via Qt's `Signals` and `Slots`.

2. **`core_logic.py` (Inference & Processing Engine)**  
   - Hosts the `DictationWorker` class, running in its own dedicated `QThread`.
   - Manages model loading, ensuring fallback capability to CUDA (Float16, Float32) or CPU (Int8).
   - Responsible for real-time audio ingestion parsing, managing buffers, Voice Activity Detection logic (VAD or Push-To-Talk).
   - Feeds finalized text items to a dedicated text queue while notifying the UI.
   - Spawns a secondary `_typing_loop` thread that connects to `win32clipboard` and `ctypes`. This performs context-aware automated typing / robust clipboard drop-in insertion (Paste) of text to whatever application window is in the foreground.

3. **`hotkey_listener.py` (Global Input Monitor)**  
   - Hosts `HotkeyWorker` running a background daemon listener out-of-band via `pynput.keyboard.Listener`.
   - Safely listens globally for the bound macro/key triggers (like specifically mapped Push-To-Talk keys).
   - Decoupled from the GUI logic — communicates state updates entirely asynchronously through Qt Signals.

4. **Auxiliary Utility: `compress_video.py`**
   - A standalone utility script utilizing `moviepy` that helps in transcoding and size-reducing MP4 videos independently. (Likely used to prepare release demos or assets).

## Data Flow
The process flows sequentially but completely overlapping across threads:
1. **Audio Capture**: `sounddevice` pushes raw `int16` `bytes` in small 0.02s chunks immediately into a fast multiprocessing queue (`audio_queue`).
2. **Audio Check Timer**: The DictationWorker frequently polls the queue to measure RMS amplitude. If amplitude meets thresholds (or PTT is pressed), chunks are buffered. When speech halts (silence_frames timeout exceeded), the buffer is dispatched as a task to Whisper.
3. **Transcription & VAD Filter**: Audio float normalized chunks run through `faster_whisper`. Integrated thresholds (`no_speech_threshold`, `log_prob`) act as the first layer suppressing phantom hallucinated outputs.
4. **Post-Processing Filtering**: Regex patterns and defined repeated-hallucination lists are applied to scrub the resulting string for any known artifacts (e.g., "thank you", "subtitles by").
5. **Output Processing**:  
   - The string is emitted to `main_gui.py` via `transcription_ready` to update the log pane.
   - The string is enqueued into `text_queue`.
6. **Window Delivery**: The typing sub-loop (via `pynput` / `win32`) polls `text_queue`, validates the active target window (rejecting text if OmniDictate is actively focused), securely backs up the user's pre-existing clipboard, dispatches a quick simulated `Ctrl+V`, then immediately restores the clipboard.

## Major Dependencies
- **PySide6**: Application framework, lifecycle management, and UI.
- **sounddevice**: Blocking-free continuous audio ingestion.
- **faster-whisper (CTranslate2)**: Under-the-hood neural engine powering transcription.
- **numpy / torch**: Core array transformations and tensor mapping required by ASR routines.
- **pynput / pywin32**: Advanced global OS-level keyhooks, window targeting, and secure clipboard modifications.
- **psutil**: Performance profiling / Diagnostics tracking.
