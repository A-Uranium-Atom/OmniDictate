import re

with open("core_logic.py", encoding="utf-8") as f:
    core_content = f.read()

out_lines = []
out_lines.append(
    '"""Audio capture, voice activity detection, and Whisper transcription worker."""'
)
out_lines.append("from __future__ import annotations")
out_lines.append("import concurrent.futures")
out_lines.append("import os")
out_lines.append("import queue")
out_lines.append("import re")
out_lines.append("import sys")
out_lines.append("import threading")
out_lines.append("import time")
out_lines.append("import numpy as np")
out_lines.append("import psutil")
out_lines.append("import sounddevice as sd")
out_lines.append("import torch")
out_lines.append("from faster_whisper import WhisperModel")
out_lines.append("from PySide6.QtCore import QObject, QTimer, Signal, Slot")
out_lines.append(
    "from config import CHUNK_DURATION, CHUNK_SIZE, HALLUCINATION_PRESETS, MAX_RECORDING_SECONDS, SAMPLE_RATE, SILENCE_DURATION, DictationSettings, get_punctuation_char"
)
out_lines.append("from core.power_monitor import PowerMonitor")
out_lines.append("from core.text_injector import InjectorSettings, run_typing_loop")

# Extract DictationWorker class body
worker_body_match = re.search(
    r"class DictationWorker\(QObject\):(.*?)$", core_content, re.DOTALL
)
worker_body = worker_body_match.group(1)

# Extract signals
signals = re.search(
    r"^\s+status_updated = Signal\(str\).*?auto_restart_requested = Signal\(\)",
    worker_body,
    re.DOTALL | re.MULTILINE,
).group(0)

# Make class
out_lines.append("class DictationWorker(QObject):")
out_lines.append(signals)

# __init__
init_code = '''
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
        self.filter_words = set(
            word.lower().strip() for word in settings.filter_words
        )
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
        self.transcription_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.is_transcribing = False
        self._power_monitor: PowerMonitor | None = None
'''
out_lines.append(init_code)

update_settings_code = '''
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
        self.filter_words = set(
            word.lower().strip() for word in settings.filter_words
        )
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
'''

drain_audio_queue_code = '''
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
'''


def extract_method(name: str) -> str | None:
    """Extract method body from worker_body given its name."""
    pattern = (
        r"( +)(@Slot\(\)\n +)?def "
        + name
        + r"\(.*?$((?:\n(?:\1 .*|\s*))*)(?=\n +@|\n +def|\Z)"
    )
    match = re.search(pattern, worker_body, re.MULTILINE)
    if match:
        return match.group(0)
    return None


methods = {
    "set_vad_enabled": (
        "def set_vad_enabled(self, enabled: bool):",
        'def set_vad_enabled(self, enabled: bool) -> None:\n        """Set the voice activity detection state."""',
    ),
    "set_ptt_state": (
        "def set_ptt_state(self, is_pressed: bool):",
        'def set_ptt_state(self, is_pressed: bool) -> None:\n        """Set the push-to-talk state."""',
    ),
    "load_model": (
        "def load_model(self, force_reload=False):",
        'def load_model(self, force_reload: bool = False) -> None:\n        """Load the Whisper model."""',
    ),
    "start_processing": (
        "def start_processing(self):",
        'def start_processing(self) -> None:\n        """Start processing audio."""',
    ),
    "_on_system_resume": (
        "def _on_system_resume(self):",
        "def _on_system_resume(self) -> None:",
    ),
    "stop_processing": (
        "def stop_processing(self):",
        'def stop_processing(self) -> None:\n        """Stop processing audio."""',
    ),
    "_audio_callback": (
        "def _audio_callback(self, indata, frames, time_info, status):",
        "def _audio_callback(self, indata: np.ndarray, frames: int, time_info: object, status: sd.CallbackFlags) -> None:",
    ),
    "_process_audio_buffer": (
        "def _process_audio_buffer(self):",
        "def _process_audio_buffer(self) -> None:",
    ),
    "_transcription_task": (
        "def _transcription_task(self, audio_float32, h_level):",
        "def _transcription_task(self, audio_float32: np.ndarray, h_level: dict) -> list[object]:",
    ),
    "_finalize_transcription": (
        "def _finalize_transcription(self, transcribed_text, latency):",
        "def _finalize_transcription(self, transcribed_text: str, latency: float) -> None:",
    ),
    "_typing_loop": None,
}

out_lines.append(extract_method("set_vad_enabled").replace(*methods["set_vad_enabled"]))
out_lines.append(extract_method("set_ptt_state").replace(*methods["set_ptt_state"]))
out_lines.append(update_settings_code)
out_lines.append(extract_method("load_model").replace(*methods["load_model"]))
out_lines.append(
    extract_method("start_processing").replace(*methods["start_processing"])
)
out_lines.append(
    extract_method("_on_system_resume").replace(*methods["_on_system_resume"])
)
out_lines.append(extract_method("stop_processing").replace(*methods["stop_processing"]))
out_lines.append(drain_audio_queue_code)
out_lines.append(extract_method("_audio_callback").replace(*methods["_audio_callback"]))

_check_audio_queue = extract_method("_check_audio_queue")
_check_audio_queue = _check_audio_queue.replace(
    "def _check_audio_queue(self):", "def _check_audio_queue(self) -> None:"
)
_check_audio_queue = re.sub(
    r"while not self.audio_queue.empty\(\).*?raw_audio_chunk = self.audio_queue.get_nowait\(\); processed_chunk_count \+= 1",
    "for raw_audio_chunk in self._drain_audio_queue():\n                processed_chunk_count += 1",
    _check_audio_queue,
    flags=re.DOTALL,
)
out_lines.append(_check_audio_queue)

out_lines.append(
    extract_method("_process_audio_buffer").replace(*methods["_process_audio_buffer"])
)
out_lines.append(
    extract_method("_transcription_task").replace(*methods["_transcription_task"])
)
out_lines.append(
    extract_method("_finalize_transcription").replace(
        *methods["_finalize_transcription"]
    )
)
out_lines.append(extract_method("_typing_loop"))

with open("core/dictation_worker.py", "w", encoding="utf-8") as f:
    f.write("\n".join(out_lines))
print("Created dictation_worker.py")
