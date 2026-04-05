# core_logic.py

# Standard library imports
import time
import threading
import sys
import os
import queue
import psutil
import re

# Third-party library imports
import numpy as np
import torch
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard # Keep for Controller in typing thread
from pynput.keyboard import Controller, Key

# PySide6 imports for threading and signals
from PySide6.QtCore import QObject, Signal, QTimer, Slot

# --- Configuration Constants ---
SAMPLE_RATE = 16000
CHUNK_DURATION = 0.02
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
# CHAR_DELAY passed via __init__

# Hallucination filter presets: (no_speech_threshold, log_prob_threshold)
HALLUCINATION_LEVELS = {
    "Low":    {"no_speech_threshold": 0.8, "log_prob_threshold": -1.5},
    "Medium": {"no_speech_threshold": 0.6, "log_prob_threshold": -1.0},  # Recommended
    "High":   {"no_speech_threshold": 0.4, "log_prob_threshold": -0.7},
}

# --- Helper Functions ---
def get_punctuation_char(punctuation_name):
    """Returns the punctuation character based on a verbal command."""
    pmap = {"question mark": "?", "exclamation mark": "!", "comma": ",", "period": ".", "full stop": ".", "colon": ":", "semicolon": ";", "open parenthesis": "(", "close parenthesis": ")", "open bracket": "[", "close bracket": "]", "open brace": "{", "close brace": "}", "hyphen": "-", "dash": "-", "underscore": "_", "plus": "+", "equals": "=", "at": "@", "hash": "#", "dollar": "$", "percent": "%", "caret": "^", "ampersand": "&", "asterisk": "*"}
    return pmap.get(punctuation_name.lower())


# --- Worker Class ---
class DictationWorker(QObject):
    status_updated = Signal(str)
    transcription_ready = Signal(str)
    error_occurred = Signal(str)
    warning_occurred = Signal(str)
    audio_level = Signal(float)

    def __init__(self, gui_wid, model_size="large-v3", language="en", vad_enabled=True,
                 silence_threshold=500, silence_duration=0.5, char_delay=0.02,
                 filter_words=None, rms_threshold=0.01, hallucination_filter="Medium",
                 insertion_method="Paste", parent=None):
        super().__init__(parent)
        self.gui_wid = gui_wid
        self.model_size = model_size
        # Ensure language is None if "None" string or empty (for Auto Detect)
        if language in ["None", ""]: language = None
        self.language_code = language
        self._vad_enabled = vad_enabled
        self.silence_threshold = silence_threshold
        self.silence_frames = int(silence_duration * SAMPLE_RATE / CHUNK_SIZE)
        self.char_delay = char_delay
        self.filter_words = set(word.lower().strip() for word in filter_words) if filter_words else set()
        
        # Hallucination prevention settings
        self.rms_threshold = rms_threshold
        self.hallucination_filter = hallucination_filter
        self.insertion_method = insertion_method
        self._last_transcript = ""
        self._repeat_count = 0
        
        print(f"Worker Init: Filter Words={len(self.filter_words)}, RMS Threshold={self.rms_threshold}, Hallucination Filter={self.hallucination_filter}, Insertion Method={self.insertion_method}")

        self.model = None
        self.audio_stream = None
        self._is_running = False
        self._ptt_active = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.recording = False
        self.audio_buffer = []
        self.vad_active = False
        self.frames_since_speech = 0
        self.typing_thread_instance = None
        self.stop_typing_event = threading.Event()
        self.audio_check_timer = QTimer(self)
        self.audio_check_timer.timeout.connect(self._check_audio_queue)
        self.audio_check_interval = 50

    # --- Public Slots ---
    @Slot(bool)
    def set_vad_enabled(self, enabled: bool):
        if self._vad_enabled != enabled:
            print(f"Setting VAD Enabled: {enabled}")
            self._vad_enabled = enabled
            if not enabled and self.vad_active:
                self.recording = False; self.vad_active = False; self.audio_buffer = []
                if self._is_running: self.status_updated.emit("Listening...")

    @Slot(bool)
    def set_ptt_state(self, is_pressed: bool):
        self._ptt_active = is_pressed
        if not is_pressed and self.recording and not self.vad_active:
            print("Recording stopped (PTT Release). Transcribing...")
            self.recording = False
            self._process_audio_buffer()

    @Slot(dict)
    def update_settings(self, settings: dict):
        """Dynamically update worker settings. Only reloads model if model_size changed."""
        print("Worker: Applying dynamic settings update.")
        if "language" in settings:
            lang = settings["language"]
            self.language_code = None if lang in ["None", ""] else lang
        if "silence_threshold" in settings:
            self.silence_threshold = settings["silence_threshold"]
        if "char_delay" in settings:
            self.char_delay = settings["char_delay"]
        if "filter_words" in settings:
            self.filter_words = set(word.lower().strip() for word in settings["filter_words"])
        if "vad_enabled" in settings:
            self.set_vad_enabled(settings["vad_enabled"])
        if "rms_threshold" in settings:
            self.rms_threshold = settings["rms_threshold"]
        if "hallucination_filter" in settings:
            self.hallucination_filter = settings["hallucination_filter"]
        if "insertion_method" in settings:
            self.insertion_method = settings["insertion_method"]
        new_model = settings.get("model_size")
        if new_model and new_model != self.model_size:
            print(f"Model size changed: {self.model_size} -> {new_model}. Reloading model...")
            self.model_size = new_model
            if self.model:
                self.load_model(force_reload=True)

    # --- Core Logic Methods ---
    def load_model(self, force_reload=False):
        if self.model and not force_reload: return True
        if self.model and force_reload: print(f"Force reloading model '{self.model_size}'..."); del self.model; self.model = None;
        if torch.cuda.is_available(): print("Clearing CUDA cache..."); torch.cuda.empty_cache()
        
        try:
            self.status_updated.emit(f"Loading model '{self.model_size}'...")
            if not self.model_size: raise ValueError("Model size empty.")
            
            model_path = self.model_size
            if self.model_size == "large-v3-turbo":
                model_path = "deepdml/faster-whisper-large-v3-turbo-ct2"

            # Attempt 1: Default (CUDA + float16 if available)
            try:
                use_cuda = torch.cuda.is_available()
                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"
                print(f"Attempting to load model on {device} with {compute_type}...")
                self.model = WhisperModel(model_path, device=device, compute_type=compute_type, local_files_only=False)
                status_msg = f"Model '{self.model_size}' loaded on {device.upper()} ({compute_type})."
                print(status_msg); self.status_updated.emit(status_msg)
                return True
            except Exception as e:
                if "float16" in str(e) and device == "cuda":
                    print(f"Float16 failed on CUDA: {e}. Retrying with float32...")
                    # Fallback 1: CUDA + float32
                    try:
                        compute_type = "float32"
                        self.model = WhisperModel(model_path, device="cuda", compute_type=compute_type, local_files_only=False)
                        status_msg = f"Model '{self.model_size}' loaded on CUDA (float32 fallback)."
                        print(status_msg); self.status_updated.emit(status_msg)
                        return True
                    except Exception as e2:
                        print(f"Float32 on CUDA failed: {e2}. Falling back to CPU...")
                
                # Fallback 2: CPU + int8 (Final Resort)
                print("Falling back to CPU (int8)...")
                self.model = WhisperModel(model_path, device="cpu", compute_type="int8", local_files_only=False)
                status_msg = f"Model '{self.model_size}' loaded on CPU (int8 fallback)."
                print(status_msg); self.status_updated.emit(status_msg)
                return True

        except Exception as e: 
            error_msg = f"Error loading model: {e}"
            print(error_msg); self.error_occurred.emit(error_msg); self.model = None; return False

    @Slot()
    def start_processing(self):
        if self._is_running: return
        if not self.load_model(force_reload=False): self.error_occurred.emit("Model failed to load."); return

        self._is_running = True; self.status_updated.emit("Starting...")
        self.audio_buffer = []; self.recording = False; self.vad_active = False; self.frames_since_speech = 0

        # Clear queues
        print("Clearing queues...")
        while True:
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
            except Exception as e_q: print(f"Error clearing audio queue item: {e_q}"); break
        while True:
            try: self.text_queue.get_nowait()
            except queue.Empty: break
            except Exception as e_q: print(f"Error clearing text queue item: {e_q}"); break
        print("Queues cleared.")

        self.stop_typing_event.clear()
        if self.typing_thread_instance and self.typing_thread_instance.is_alive(): print("Warning: Typing thread still alive?")
        self.typing_thread_instance = threading.Thread(target=self._typing_loop, daemon=True)
        self.typing_thread_instance.start()

        try:
            device_info = sd.query_devices(kind='input')
            self.status_updated.emit(f"Using device: {device_info['name']}")
            self.audio_stream = sd.InputStream(samplerate=SAMPLE_RATE, blocksize=CHUNK_SIZE, device=None, channels=1, dtype='int16', callback=self._audio_callback)
            self.audio_stream.start()
            self.status_updated.emit("Listening...")
            self.audio_check_timer.start(self.audio_check_interval)
        except sd.PortAudioError as pae: error_msg = f"PortAudio Error: {pae}"; print(error_msg); self.error_occurred.emit(error_msg); self.stop_processing()
        except Exception as e: error_msg = f"Audio stream error: {e}"; print(error_msg); self.error_occurred.emit(error_msg); self.stop_processing()

    @Slot()
    def stop_processing(self):
        if not self._is_running: return
        print("Stopping worker processing..."); self.status_updated.emit("Stopping...")
        self._is_running = False; self.audio_check_timer.stop()

        if self.audio_stream:
            try: self.audio_stream.abort(); self.audio_stream.close(); print("Audio stream stopped.")
            except Exception as e: print(f"Error stopping audio stream: {e}")
            finally: self.audio_stream = None

        self.stop_typing_event.set()
        if self.typing_thread_instance and self.typing_thread_instance.is_alive():
            print("Waiting for typing thread to finish...")
            self.typing_thread_instance.join(timeout=1.5)
            if self.typing_thread_instance.is_alive(): print("Warning: Typing thread did not stop gracefully.")
        self.typing_thread_instance = None

        # Clear queues again
        print("Clearing queues...")
        while True:
            try: self.audio_queue.get_nowait()
            except queue.Empty: break
            except Exception as e: break
        while True:
            try: self.text_queue.get_nowait()
            except queue.Empty: break
            except Exception as e: break
        print("Queues cleared.")

        # Model is intentionally kept loaded in memory for instant restart.
        # It will only be reloaded when model_size changes via update_settings().

        self.recording = False; self.vad_active = False; self.audio_buffer = []
        print("Worker processing stopped."); self.status_updated.emit("Idle")

    # --- Internal Methods ---
    def _audio_callback(self, indata, frames, time, status):
        if status: error_msg = f"Audio Callback Error: {status}"; print(error_msg, file=sys.stderr)
        if self._is_running: self.audio_queue.put(bytes(indata))

    @Slot()
    def _check_audio_queue(self):
        if not self._is_running: return
        try:
            processed_chunk_count = 0; max_chunks_per_cycle = 5
            while not self.audio_queue.empty() and processed_chunk_count < max_chunks_per_cycle:
                raw_audio_chunk = self.audio_queue.get_nowait(); processed_chunk_count += 1
                try: chunk_np = np.frombuffer(raw_audio_chunk, dtype=np.int16); amplitude = np.abs(chunk_np).mean()
                except Exception as e: print(f"Error VAD chunk: {e}"); continue

                self.audio_level.emit(amplitude)

                if self._ptt_active:
                    if not self.recording: self.status_updated.emit("Recording (PTT)..."); self.recording = True; self.vad_active = False; self.audio_buffer = []
                    self.audio_buffer.append(chunk_np); self.frames_since_speech = 0; continue
                elif self._vad_enabled:
                    if not self.recording:
                        if amplitude > self.silence_threshold: self.status_updated.emit("Recording (VAD)..."); self.recording = True; self.vad_active = True; self.audio_buffer = []; self.audio_buffer.append(chunk_np); self.frames_since_speech = 0
                    elif self.recording and self.vad_active:
                        if amplitude > self.silence_threshold: self.frames_since_speech = 0; self.audio_buffer.append(chunk_np)
                        else:
                            self.frames_since_speech += 1
                            if self.frames_since_speech > self.silence_frames: self.status_updated.emit("Transcribing (VAD)..."); self.recording = False; self.vad_active = False; self._process_audio_buffer()
        except queue.Empty: pass
        except Exception as e: error_msg = f"Audio check loop error: {e}"; print(error_msg); self.error_occurred.emit(error_msg)

    def _process_audio_buffer(self):
        if not self.audio_buffer: return
        buffer_copy = list(self.audio_buffer); self.audio_buffer = []
        try: audio_data = np.concatenate(buffer_copy); audio_float32 = audio_data.astype(np.float32) / 32768.0
        except ValueError: print("Error concatenating buffer copy."); return
        if audio_float32.size == 0: print("Concatenated audio empty."); return

        # --- Layer 1: Pre-transcription RMS energy gate ---
        rms_energy = np.sqrt(np.mean(audio_float32 ** 2))
        if rms_energy < self.rms_threshold:
            print(f"Skipping transcription: buffer RMS too low ({rms_energy:.4f} < {self.rms_threshold})")
            if self._is_running and not self.recording and not self._ptt_active: self.status_updated.emit("Listening...")
            return

        start_time = time.time(); transcribed_text = ""
        try:
            if not self.model: self.error_occurred.emit("Model not loaded."); return

            # --- Layer 2: Silero VAD + no_speech/log_prob thresholds ---
            h_level = HALLUCINATION_LEVELS.get(self.hallucination_filter, HALLUCINATION_LEVELS["Medium"])
            segments, info = self.model.transcribe(
                audio_float32,
                beam_size=5,
                language=self.language_code,
                temperature=0.0,
                condition_on_previous_text=False,
                vad_filter=True,
                no_speech_threshold=h_level["no_speech_threshold"],
                log_prob_threshold=h_level["log_prob_threshold"],
            )

            # --- Layer 2.5: Per-segment confidence filtering ---
            segments_list = list(segments)
            good_segments = [s for s in segments_list if s.no_speech_prob < h_level["no_speech_threshold"]]
            transcribed_text = "".join(s.text for s in good_segments)
        except Exception as e: error_msg = f"Transcription error: {e}"; print(error_msg); self.error_occurred.emit(error_msg); return
        finally:
             if self._is_running and not self.recording and not self._ptt_active: self.status_updated.emit("Listening...")
        end_time = time.time()

        processed_text = transcribed_text.strip()

        # --- Layer 3: Improved post-transcription filtering ---
        # Normalized matching (strip trailing punctuation before comparing)
        text_normalized = processed_text.lower().strip().rstrip('.!?,;: ')
        if any(text_normalized == fw.rstrip('.!?,;: ') for fw in self.filter_words):
            print(f"Filtered out hallucination: '{processed_text}'"); return

        # Repetition detection — same text 3+ times in a row is almost certainly hallucination
        if text_normalized and text_normalized == self._last_transcript:
            self._repeat_count += 1
            if self._repeat_count >= 2:
                print(f"Filtered out repeated hallucination: '{processed_text}' (x{self._repeat_count + 1})")
                return
        else:
            self._repeat_count = 0
        self._last_transcript = text_normalized

        if processed_text:
            print(f"Transcribed: {processed_text} (Latency: {end_time - start_time:.2f}s)")
            try:
                process = psutil.Process(os.getpid())
                ram_mb = process.memory_info().rss / (1024 * 1024)
                vram_mb = torch.cuda.memory_reserved() / (1024 * 1024) if torch.cuda.is_available() else 0
                print(f"Memory Usage - RAM: {ram_mb:.1f} MB | VRAM: {vram_mb:.1f} MB")
            except Exception as mem_e:
                print(f"Error checking memory: {mem_e}")
                
            self.transcription_ready.emit(processed_text)

            text_lower = processed_text.lower(); is_command = False
            if not is_command:
                punc_match = re.match(r"^(question mark|exclamation mark|comma|period|full stop|colon|semicolon|open parenthesis|close parenthesis|open bracket|close bracket|open brace|close brace|hyphen|dash|underscore|plus|equals|at|hash|dollar|percent|caret|ampersand|asterisk)[.?!]?$", text_lower.strip())
                if punc_match:
                    punc_char = get_punctuation_char(punc_match.group(1))
                    if punc_char:
                        self.text_queue.put(punc_char)
                        print(f"Queued punctuation: {punc_char}")
                        is_command = True
            
            if not is_command: self.text_queue.put(processed_text + " ")

    def _paste_text(self, text, keyboard_controller):
        import time
        import win32clipboard
        from pynput import keyboard

        clipboard_backup = {}
        fallback_to_typing = False

        # 1. Backup all clipboard formats
        try:
            win32clipboard.OpenClipboard()
            format_id = win32clipboard.EnumClipboardFormats(0)
            while format_id:
                # 15 is CF_HDROP (File copy). Hard to safely restore in raw python.
                if format_id == 15:
                    fallback_to_typing = True
                    break
                try:
                    data = win32clipboard.GetClipboardData(format_id)
                    if data is not None:
                        clipboard_backup[format_id] = data
                except Exception:
                    fallback_to_typing = True
                    break
                format_id = win32clipboard.EnumClipboardFormats(format_id)
        except Exception:
            fallback_to_typing = True
        finally:
            try: win32clipboard.CloseClipboard()
            except: pass

        if fallback_to_typing:
            self.warning_occurred.emit("Complex clipboard object detected. Falling back to typing to protect clipboard.")
            return False # Signal calling loop to use typing method

        # 2. Set new text
        try:
            win32clipboard.OpenClipboard()
            win32clipboard.EmptyClipboard()
            win32clipboard.SetClipboardData(win32clipboard.CF_UNICODETEXT, text)
            win32clipboard.CloseClipboard()
        except Exception as e:
            print(f"Error setting clipboard: {e}")
            try: win32clipboard.CloseClipboard()
            except: pass
            return False
            
        # 3. Simulate Ctrl+V
        time.sleep(0.05) # Small delay for OS to register new clipboard
        keyboard_controller.press(keyboard.Key.ctrl)
        keyboard_controller.press('v')
        keyboard_controller.release('v')
        keyboard_controller.release(keyboard.Key.ctrl)
        time.sleep(0.1) # Let the app process the paste

        # 4. Restore backup
        if clipboard_backup:
            try:
                win32clipboard.OpenClipboard()
                win32clipboard.EmptyClipboard()
                for fmt, data in clipboard_backup.items():
                    # Catch individual format restore errors so we don't abort midway
                    try: win32clipboard.SetClipboardData(fmt, data)
                    except Exception: pass 
            except Exception:
                pass
            finally:
                try: win32clipboard.CloseClipboard()
                except: pass
        
        return True

    def _typing_loop(self):
        print("Typing thread started, ID:", threading.get_ident())
        import pythoncom
        pythoncom.CoInitializeEx(pythoncom.COINIT_MULTITHREADED)
        try:
            import ctypes, time
            try: from pynput import keyboard; keyboard_controller = keyboard.Controller()
            except ImportError: print("ERROR: pynput not installed."); self.error_occurred.emit("pynput not installed."); return

            while self._is_running and not self.stop_typing_event.is_set():
                try:
                    text_to_type = self.text_queue.get(timeout=0.5)
                    try:
                        hwnd = ctypes.windll.user32.GetForegroundWindow()
                        if hwnd == self.gui_wid: print("Skipping typing: OmniDictate window active."); continue
                        
                        paste_successful = False
                        if self.insertion_method == "Paste":
                            paste_successful = self._paste_text(text_to_type, keyboard_controller)
                        
                        if not paste_successful:
                            # Optimized typing logic
                            if self.char_delay <= 0.001:
                                keyboard_controller.type(text_to_type)
                            else:
                                for char in text_to_type:
                                    if not self._is_running or self.stop_typing_event.is_set(): break
                                    keyboard_controller.press(char)
                                    keyboard_controller.release(char)
                                    time.sleep(self.char_delay)
                    except Exception as e: error_msg = f"Error inserting text: {e}"; print(error_msg); self.error_occurred.emit(error_msg)
                except queue.Empty: continue
                except Exception as e: error_msg = f"Typing queue error: {e}"; print(error_msg); self.error_occurred.emit(error_msg); time.sleep(0.1)
        finally:
            pythoncom.CoUninitialize()
            print("Typing thread exiting, ID:", threading.get_ident())