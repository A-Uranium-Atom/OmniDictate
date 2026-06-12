"""Audio capture, voice activity detection, and Whisper transcription worker."""

from __future__ import annotations

import concurrent.futures
import os
import queue
import re
import sys
import threading
import time

import numpy as np
import psutil
import sounddevice as sd
import torch
from faster_whisper import WhisperModel
from PySide6.QtCore import QObject, QTimer, Signal, Slot

from config import (
    CHUNK_DURATION,
    CHUNK_SIZE,
    HALLUCINATION_PRESETS,
    MAX_RECORDING_SECONDS,
    SAMPLE_RATE,
    SILENCE_DURATION,
    DictationSettings,
    get_punctuation_char,
)
from core.power_monitor import PowerMonitor
from core.text_injector import InjectorSettings, run_typing_loop


class DictationWorker(QObject):
    """Worker handling the audio capture and transcription background process."""

    status_updated = Signal(str)
    transcription_ready = Signal(str)
    error_occurred = Signal(str)
    warning_occurred = Signal(str)
    audio_level = Signal(float)
    auto_restart_requested = Signal()

    def __init__(self, gui_wid: int, settings: DictationSettings | None = None) -> None:
        """Initialize the dictation worker.

        Args:
            gui_wid: HWND of the OmniDictate main window (prevents self-injection).
            settings: Configuration dataclass. Defaults to ``DictationSettings()``
                if None is passed.
        """
        super().__init__()
        if settings is None:
            settings = DictationSettings()

        self.gui_wid = gui_wid
        self.model_size = settings.model_size
        self.language_code = settings.language
        self._is_vad_enabled = settings.is_vad_enabled
        self.silence_threshold = settings.silence_threshold
        self.silence_frames = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)
        self.char_delay = settings.char_delay
        self.filter_words = set(word.lower().strip() for word in settings.filter_words)
        self.rms_threshold = settings.rms_threshold
        self.hallucination_filter = settings.hallucination_filter
        self.insertion_method = settings.insertion_method
        self.paste_delay = settings.paste_delay

        self._injector_settings = InjectorSettings(
            insertion_method=self.insertion_method,
            char_delay=self.char_delay,
            paste_delay=self.paste_delay,
        )

        self._last_transcript = ""
        self._repeat_count = 0

        self.model = None
        self.audio_stream = None
        self._is_running = False
        self._is_ptt_active = False
        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_recording = False
        self.audio_buffer = []
        self.is_vad_active = False
        self.frames_since_speech = 0
        self.typing_thread_instance = None
        self.stop_typing_event = threading.Event()
        self.audio_check_timer = QTimer(self)
        self.audio_check_timer.timeout.connect(self._check_audio_queue)
        self.audio_check_interval = 100

        self.overflow_count = 0
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1
        )
        self.is_transcribing = False
        self._power_monitor: PowerMonitor | None = None

    def set_vad_enabled(self, enabled: bool) -> None:
        """Set the voice activity detection state."""
        if self._is_vad_enabled != enabled:
            print(f"Setting VAD Enabled: {enabled}")
            self._is_vad_enabled = enabled
            if not enabled and self.is_vad_active:
                self.is_recording = False
                self.is_vad_active = False
                self.audio_buffer = []
                if self._is_running:
                    self.status_updated.emit("Listening...")

    def set_ptt_state(self, is_pressed: bool) -> None:
        """Set the push-to-talk state."""
        self._is_ptt_active = is_pressed
        if not is_pressed and self.is_recording and not self.is_vad_active:
            print("Recording stopped (PTT Release). Transcribing...")
            self.is_recording = False
            self._process_audio_buffer()

    @Slot(object)
    def update_settings(self, settings: DictationSettings) -> None:
        """Apply a full settings snapshot from the GUI.

        Compares model_size to detect changes requiring a model reload.
        Updates the shared ``InjectorSettings`` for the typing thread.

        Args:
            settings: Complete settings state from the GUI.
        """
        print("DictationWorker: Received settings update.")
        is_model_changed = settings.model_size != self.model_size

        self.model_size = settings.model_size
        self.language_code = settings.language
        self.silence_threshold = settings.silence_threshold
        self.char_delay = settings.char_delay
        self.filter_words = set(word.lower().strip() for word in settings.filter_words)
        self.rms_threshold = settings.rms_threshold
        self.hallucination_filter = settings.hallucination_filter
        self.insertion_method = settings.insertion_method
        self.paste_delay = settings.paste_delay

        self._injector_settings.insertion_method = settings.insertion_method
        self._injector_settings.char_delay = settings.char_delay
        self._injector_settings.paste_delay = settings.paste_delay

        if settings.is_vad_enabled != self._is_vad_enabled:
            self.set_vad_enabled(settings.is_vad_enabled)

        if is_model_changed and self.model:
            print(f"Model size changed to {self.model_size}. Reloading...")
            self.load_model(force_reload=True)

    def load_model(self, force_reload: bool = False) -> None:
        """Load the Whisper model."""
        if self.model and not force_reload:
            return True
        if self.model and force_reload:
            print(f"Force reloading model '{self.model_size}'...")
            del self.model
            self.model = None
        if torch.cuda.is_available():
            print("Clearing CUDA cache...")
            torch.cuda.empty_cache()

        try:
            self.status_updated.emit(f"Loading model '{self.model_size}'...")
            if not self.model_size:
                raise ValueError("Model size empty.")

            model_path = self.model_size
            if self.model_size == "large-v3-turbo":
                model_path = "deepdml/faster-whisper-large-v3-turbo-ct2"

            # Attempt 1: Default (CUDA + float16 if available)
            try:
                use_cuda = torch.cuda.is_available()
                device = "cuda" if use_cuda else "cpu"
                compute_type = "float16" if use_cuda else "int8"
                print(f"Attempting to load model on {device} with {compute_type}...")
                self.model = WhisperModel(
                    model_path,
                    device=device,
                    compute_type=compute_type,
                    local_files_only=False,
                )
                status_msg = f"Model '{self.model_size}' loaded on {device.upper()} ({compute_type})."
                print(status_msg)
                self.status_updated.emit(status_msg)
                return True
            except Exception as e:
                if "float16" in str(e) and device == "cuda":
                    print(f"Float16 failed on CUDA: {e}. Retrying with float32...")
                    # Fallback 1: CUDA + float32
                    try:
                        compute_type = "float32"
                        self.model = WhisperModel(
                            model_path,
                            device="cuda",
                            compute_type=compute_type,
                            local_files_only=False,
                        )
                        status_msg = f"Model '{self.model_size}' loaded on CUDA (float32 fallback)."
                        print(status_msg)
                        self.status_updated.emit(status_msg)
                        return True
                    except Exception as e2:
                        print(f"Float32 on CUDA failed: {e2}. Falling back to CPU...")

                # Fallback 2: CPU + int8 (Final Resort)
                print("Falling back to CPU (int8)...")
                self.model = WhisperModel(
                    model_path,
                    device="cpu",
                    compute_type="int8",
                    local_files_only=False,
                )
                status_msg = f"Model '{self.model_size}' loaded on CPU (int8 fallback)."
                print(status_msg)
                self.status_updated.emit(status_msg)
                return True

        except Exception as e:
            error_msg = f"Error loading model: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            self.model = None
            return False

    @Slot()
    def start_processing(self) -> None:
        """Start processing audio."""
        if self._is_running:
            return
        if not self.load_model(force_reload=False):
            self.error_occurred.emit("Model failed to load.")
            return

        self._is_running = True
        self.status_updated.emit("Starting...")
        self.audio_buffer = []
        self.is_recording = False
        self.is_vad_active = False
        self.frames_since_speech = 0

        # Clear queues
        print("Clearing queues...")
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as e_q:
                print(f"Error clearing audio queue item: {e_q}")
                break
        while True:
            try:
                self.text_queue.get_nowait()
            except queue.Empty:
                break
            except Exception as e_q:
                print(f"Error clearing text queue item: {e_q}")
                break
        print("Queues cleared.")

        self.stop_typing_event.clear()
        if self.typing_thread_instance and self.typing_thread_instance.is_alive():
            print("Warning: Typing thread still alive?")
        self.typing_thread_instance = threading.Thread(
            target=self._typing_loop, daemon=True
        )
        self.typing_thread_instance.start()

        try:
            device_info = sd.query_devices(kind="input")
            self.status_updated.emit(f"Using device: {device_info['name']}")
            self.overflow_count = 0
            self.audio_stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                blocksize=CHUNK_SIZE,
                device=None,
                channels=1,
                dtype="int16",
                callback=self._audio_callback,
            )
            self.audio_stream.start()
            self.status_updated.emit("Listening...")
            self.audio_check_timer.start(self.audio_check_interval)
        except sd.PortAudioError as pae:
            error_msg = f"PortAudio Error: {pae}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            self.stop_processing()
        except Exception as e:
            error_msg = f"Audio stream error: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
            self.stop_processing()

        # Start Windows power-event monitor so we know exactly when the system wakes
        self._power_monitor = PowerMonitor(on_resume=self._on_system_resume)
        self._power_monitor.start()

    def _on_system_resume(self) -> None:
        """Called by _PowerMonitor the instant Windows fires PBT_APMRESUMEAUTOMATIC.

        Signals the GUI to perform a full stop → start cycle after a driver-settle delay.
        """
        print("PowerMonitor: System wake detected. Requesting auto-restart...")
        self.auto_restart_requested.emit()

    @Slot()
    def stop_processing(self) -> None:
        """Stop processing audio."""
        if not self._is_running:
            return
        print("Stopping worker processing...")
        self.status_updated.emit("Stopping...")
        self._is_running = False
        self.audio_check_timer.stop()

        if self.audio_stream:
            try:
                self.audio_stream.abort()
                self.audio_stream.close()
                print("Audio stream stopped.")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.audio_stream = None

        self.stop_typing_event.set()
        if self.typing_thread_instance and self.typing_thread_instance.is_alive():
            print("Waiting for typing thread to finish...")
            self.typing_thread_instance.join(timeout=1.5)
            if self.typing_thread_instance.is_alive():
                print("Warning: Typing thread did not stop gracefully.")
        self.typing_thread_instance = None

        # Stop the power monitor thread
        if self._power_monitor:
            self._power_monitor.stop()
            self._power_monitor = None

        # Clear queues again
        print("Clearing queues...")
        while True:
            try:
                self.audio_queue.get_nowait()
            except queue.Empty:
                break
            except Exception:
                break
        # Cleanly shut down transcription executor if still running
        if self.transcription_executor:
            self.transcription_executor.shutdown(wait=False)
            self.transcription_executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=1
            )

        self.is_recording = False
        self.is_vad_active = False
        self.audio_buffer = []
        print("Worker processing stopped.")
        self.status_updated.emit("Idle")

    # --- Internal Methods ---
    def _drain_audio_queue(self) -> list[bytes]:
        """Remove and return all pending frames from the audio queue.

        Returns:
            List of raw audio byte buffers, one per frame.
        """
        frames: list[bytes] = []
        while not self.audio_queue.empty():
            try:
                frames.append(self.audio_queue.get_nowait())
            except queue.Empty:
                break
        return frames

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            if status.input_overflow:
                self.overflow_count += 1
                if (
                    self.overflow_count % 10 == 0
                ):  # Log every 10th overflow to avoid spamming
                    print(
                        f"Audio Callback Warning: Input overflow x{self.overflow_count}",
                        file=sys.stderr,
                    )
            else:
                print(f"Audio Callback Error: {status}", file=sys.stderr)

        if self._is_running:
            self.audio_queue.put(bytes(indata))

    @Slot()
    def _check_audio_queue(self) -> None:  # noqa: C901
        if not self._is_running:
            return
        try:
            # Priority: Stream is no longer active (hardware handle severed).
            if self.audio_stream and not self.audio_stream.active:
                print(
                    "Stability: Stream is no longer active; requesting auto-restart..."
                )
                self.auto_restart_requested.emit()
                return

            processed_chunk_count = 0
            for raw_audio_chunk in self._drain_audio_queue():
                processed_chunk_count += 1
                try:
                    chunk_np = np.frombuffer(raw_audio_chunk, dtype=np.int16)
                    amplitude = np.abs(chunk_np).mean()
                except Exception as e:
                    print(f"Error VAD chunk: {e}")
                    continue

                self.audio_level.emit(amplitude)

                if self._is_ptt_active:
                    if not self.is_recording:
                        self.status_updated.emit("Recording (PTT)...")
                        self.is_recording = True
                        self.is_vad_active = False
                        self.audio_buffer = []
                    self.audio_buffer.append(chunk_np)
                    self.frames_since_speech = 0
                    continue
                elif self._is_vad_enabled:
                    if not self.is_recording:
                        if amplitude > self.silence_threshold:
                            self.status_updated.emit("Recording (VAD)...")
                            self.is_recording = True
                            self.is_vad_active = True
                            self.audio_buffer = []
                            self.audio_buffer.append(chunk_np)
                            self.frames_since_speech = 0
                    elif self.is_recording and self.is_vad_active:
                        if amplitude > self.silence_threshold:
                            self.frames_since_speech = 0
                            self.audio_buffer.append(chunk_np)
                        else:
                            self.frames_since_speech += 1
                            is_silent = self.frames_since_speech > self.silence_frames
                            is_too_long = (
                                len(self.audio_buffer) * CHUNK_DURATION
                            ) > MAX_RECORDING_SECONDS

                            if is_silent or is_too_long:
                                if is_too_long:
                                    print(
                                        "Safety: Recording reached limit, force-transcribing."
                                    )
                                self.status_updated.emit("Transcribing...")
                                self.is_recording = False
                                self.is_vad_active = False
                                self._process_audio_buffer()

            # Overflow guard: too many overflows in a row indicate a degraded stream
            if self.overflow_count > 50 and not self.is_transcribing:
                print(
                    f"Stability: Excessive overflows ({self.overflow_count}); requesting auto-restart."
                )
                self.auto_restart_requested.emit()
        except queue.Empty:
            pass
        except Exception as e:
            error_msg = f"Audio check loop error: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)

    def _process_audio_buffer(self) -> None:
        if not self.audio_buffer:
            return
        buffer_copy = list(self.audio_buffer)
        self.audio_buffer = []
        try:
            audio_data = np.concatenate(buffer_copy)
            audio_float32 = audio_data.astype(np.float32) / 32768.0
        except ValueError:
            print("Error concatenating buffer copy.")
            return
        if audio_float32.size == 0:
            print("Concatenated audio empty.")
            return

        # --- Layer 1: Pre-transcription RMS energy gate ---
        rms_energy = np.sqrt(np.mean(audio_float32**2))
        if rms_energy < self.rms_threshold:
            print(
                f"Skipping transcription: buffer RMS too low ({rms_energy:.4f} < {self.rms_threshold})"
            )
            if self._is_running and not self.is_recording and not self._is_ptt_active:
                self.status_updated.emit("Listening...")
            return

        h_level = HALLUCINATION_PRESETS.get(
            self.hallucination_filter, HALLUCINATION_PRESETS["Medium"]
        )

        # Offload transcription to background thread to keep queue consumer responsive
        self.is_transcribing = True
        self.transcription_executor.submit(
            self._transcription_task, audio_float32, h_level
        )

    def _transcription_task(
        self, audio_float32: np.ndarray, h_level: dict
    ) -> list[object]:
        """The heavy transcription workload running in a background executor."""
        start_time = time.time()
        transcribed_text = ""
        try:
            if not self.model:
                return

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
            good_segments = [
                s
                for s in segments_list
                if s.no_speech_prob < h_level["no_speech_threshold"]
            ]
            transcribed_text = "".join(s.text for s in good_segments)

            self._finalize_transcription(transcribed_text, time.time() - start_time)
        except Exception as e:
            error_msg = f"Transcription error: {e}"
            print(error_msg)
            self.error_occurred.emit(error_msg)
        finally:
            self.is_transcribing = False
            if self._is_running and not self.is_recording and not self._is_ptt_active:
                self.status_updated.emit("Listening...")

    def _finalize_transcription(self, transcribed_text: str, latency: float) -> None:
        """Post-processing and UI updates after background transcription finishes."""
        processed_text = transcribed_text.strip()

        # --- Layer 3: Improved post-transcription filtering ---
        # Normalized matching (strip trailing punctuation before comparing)
        text_normalized = processed_text.lower().strip().rstrip(".!?,;: ")
        if any(text_normalized == fw.rstrip(".!?,;: ") for fw in self.filter_words):
            print(f"Filtered out hallucination: '{processed_text}'")
            return

        # Repetition detection — same text 3+ times in a row is almost certainly hallucination
        if text_normalized and text_normalized == self._last_transcript:
            self._repeat_count += 1
            if self._repeat_count >= 2:
                print(
                    f"Filtered out repeated hallucination: '{processed_text}' (x{self._repeat_count + 1})"
                )
                return
        else:
            self._repeat_count = 0
        self._last_transcript = text_normalized

        if processed_text:
            print(f"Transcribed: {processed_text} (Latency: {latency:.2f}s)")
            try:
                process = psutil.Process(os.getpid())
                ram_mb = process.memory_info().rss / (1024 * 1024)
                print(f"Memory Usage - RAM: {ram_mb:.1f} MB")
            except Exception as mem_e:
                print(f"Error checking memory: {mem_e}")

            self.transcription_ready.emit(processed_text)

            text_lower = processed_text.lower()
            is_command = False
            if not is_command:
                punc_match = re.match(
                    r"^(question mark|exclamation mark|comma|period|full stop|colon|semicolon|open parenthesis|close parenthesis|open bracket|close bracket|open brace|close brace|hyphen|dash|underscore|plus|equals|at|hash|dollar|percent|caret|ampersand|asterisk)[.?!]?$",
                    text_lower.strip(),
                )
                if punc_match:
                    punc_char = get_punctuation_char(punc_match.group(1))
                    if punc_char:
                        self.text_queue.put(punc_char)
                        print(f"Queued punctuation: {punc_char}")
                        is_command = True

            if not is_command:
                self.text_queue.put(processed_text + " ")

    def _typing_loop(self) -> None:
        """Delegate to the extracted text injection loop."""
        run_typing_loop(
            text_queue=self.text_queue,
            stop_event=self.stop_typing_event,
            is_running=lambda: self._is_running,
            gui_wid=self.gui_wid,
            settings=self._injector_settings,
            error_callback=lambda msg: self.error_occurred.emit(msg),
            warning_callback=lambda msg: self.warning_occurred.emit(msg),
        )
