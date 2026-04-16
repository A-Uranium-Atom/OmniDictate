# OmniDictate - Manual Testing Protocol

Because OmniDictate interacts intensely with OS-level hooks, external audio hardware, and global state (the Windows Clipboard), automated testing via `pytest` cannot cover all functional realities. 

Before any major release, a developer MUST perform the following manual testing checklist on actual hardware.

## 1. Hardware & Latency Checks

### VAD & PTT Responsiveness
- **Action**: Speak naturally in a quiet room, then speak with a loud fan or television running in the background.
- **What to look for**: Ensure the Voice Activity Detection (VAD) audio bar in the GUI responds accurately only to your voice. Ensure it does not continuously queue transcription requests when only background noise is present.
- **Action**: Switch to Push-To-Talk (PTT) mode in the Settings.
- **What to look for**: Ensure the audio bar and transcription ONLY activates when the assigned hotkey is held down.

### Inference Latency Calibration
- **Action**: Go to Settings -> Transcription. Toggle "Use GPU" ON (if hardware permits). Speak a fast 5-second sentence.
- **What to look for**: The transcription should appear in the target window within < 1.0 second of completing the sentence.
- **Action**: Toggle "Use GPU" OFF (Fallback to CPU Int8). Speak the same sentence.
- **What to look for**: The transcription will be slower, but ensure it completes successfully without crashing `main_gui.py`.

### Memory Leak Observation
- **Action**: Leave the application running and actively dictating (or pressing PTT frequently) for over 30 minutes.
- **What to look for**: Open Windows Task Manager (or use `psutil`). Monitor the RAM usage of the OmniDictate process. The RAM should plateau after the initial model loading. If RAM usage climbs continuously by several megabytes every minute without stopping, there is a queue block or memory leak that must be addressed.

---

## 2. OS-Level Intercepts

### Target App Fidelity
- **Action**: Attempt to dictate text into the following active applications:
  1. Notepad (Standard Text)
  2. Microsoft Word / Google Docs (Rich Text)
  3. A Web Browser Search Bar (Chrome/Firefox)
  4. A Fullscreen Application/Game (Optional, but recommended)
- **What to look for**: In all scenarios, the text should be pasted cleanly into the active cursor location without deleting existing text or lagging the application.

### Admin Elevation Blocking
- **Action**: Launch a known application "As Administrator" (e.g., elevated Command Prompt or Task Manager). Click into it to make it the active, foreground window.
- **Action**: Attempt to use the Push-To-Talk hotkey.
- **What to look for**: Windows inherently blocks un-elevated background hooks from triggering when an elevated app is in the foreground. Verify whether PTT functions here. If it does not, ensure this known behavior is documented for the end-user (or test if running OmniDictate as Admin resolves it).

### Clipboard Persistence Guarantee
- **Action**: Take a screenshot (or copy a large block of code) to your Windows Clipboard. Press `Win + V` to ensure it is the most recent item.
- **Action**: Open Notepad. Speak a sentence using OmniDictate so that it types "Hello, this is a test." 
- **Action**: Immediately press `Ctrl + V` manually on your keyboard.
- **What to look for**: The screenshot or code block you originally copied MUST be pasted. 
If the text "Hello, this is a test" is pasted instead, the `win32clipboard` backup/restore loop has failed and polluted the user's global clipboard. **This is a critical failure.**
