"""Локальный интерфейс: микрофон -> распознавание -> голос F5.

Модель намеренно не импортируется в главном потоке: ``VoiceEngine`` создаётся и
используется только в worker-потоке.  Это оставляет окно отзывчивым во время
загрузки модели и инференса.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np


TARGET_SAMPLE_RATE = 16_000


class GenerationGate:
    """Отменяет результаты уже начатой работы без принудительной остановки ML."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._generation = 0

    def next(self) -> int:
        with self._lock:
            self._generation += 1
            return self._generation

    def cancel(self) -> int:
        return self.next()

    def current(self, generation: int) -> bool:
        with self._lock:
            return generation == self._generation


class PhraseSegmenter:
    """Потокобезопасный сегментатор по RMS; не зависит от аудиоустройства."""

    def __init__(
        self,
        sample_rate: int = TARGET_SAMPLE_RATE,
        threshold: float = 0.015,
        min_speech_seconds: float = 0.25,
        end_silence_seconds: float = 0.7,
        max_phrase_seconds: float = 12.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.end_silence_seconds = end_silence_seconds
        self.max_phrase_seconds = max_phrase_seconds
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        with self._lock:
            self._chunks: list[np.ndarray] = []
            self._active_samples = 0
            self._silence_samples = 0
            self._speech_samples = 0

    def configure(self, *, threshold: float, end_silence_seconds: float) -> None:
        with self._lock:
            self.threshold = max(0.0001, float(threshold))
            self.end_silence_seconds = max(0.05, float(end_silence_seconds))

    def feed(self, samples: np.ndarray) -> list[np.ndarray]:
        data = np.asarray(samples, dtype=np.float32).reshape(-1)
        if not len(data):
            return []
        rms = float(np.sqrt(np.mean(np.square(data, dtype=np.float32))))
        with self._lock:
            is_voice = rms >= self.threshold
            if not self._chunks:
                if not is_voice:
                    return []
                self._chunks = [data.copy()]
                self._active_samples = len(data)
                self._speech_samples = len(data)
                self._silence_samples = 0
                return []

            self._chunks.append(data.copy())
            self._active_samples += len(data)
            if is_voice:
                self._speech_samples += len(data)
                self._silence_samples = 0
            else:
                self._silence_samples += len(data)

            stop_for_silence = self._silence_samples >= int(self.end_silence_seconds * self.sample_rate)
            stop_for_length = self._active_samples >= int(self.max_phrase_seconds * self.sample_rate)
            if stop_for_silence or stop_for_length:
                return self._finish_locked()
            return []

    def flush(self) -> list[np.ndarray]:
        with self._lock:
            return self._finish_locked()

    def _finish_locked(self) -> list[np.ndarray]:
        chunks, speech = self._chunks, self._speech_samples
        self._chunks = []
        self._active_samples = self._silence_samples = self._speech_samples = 0
        if not chunks or speech < int(self.min_speech_seconds * self.sample_rate):
            return []
        return [np.concatenate(chunks).astype(np.float32, copy=False)]

# класс делает невозможным изменение полей после создания экземпляра, что обеспечивает неизменяемость данных об аудиоустройстве.
@dataclass(frozen=True)
class AudioDevice:
    index: int
    label: str
    default_rate: int


class VoiceApp:
    def __init__(self, root) -> None:
        import tkinter as tk
        from tkinter import messagebox, ttk

        self.root = root
        self.tk, self.ttk, self.messagebox = tk, ttk, messagebox
        self.root.title("DJ Arbuz — голосовой ассистент")
        self.root.minsize(690, 490)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=100)
        self.capture_queue: queue.Queue[tuple[int, int, np.ndarray]] = queue.Queue(maxsize=32)
        self.jobs: queue.Queue[tuple] = queue.Queue(maxsize=3)
        self.playback_active = threading.Event()
        self._output_lock = threading.Lock()
        self.running = False
        self.closing = False
        self.gate = GenerationGate()
        self._active_generation = 0
        self._generation_lock = threading.Lock()
        self.segmenter = PhraseSegmenter()
        self.input_stream = None
        self.sd = None
        self.capture_thread: Optional[threading.Thread] = None
        self.worker_thread: Optional[threading.Thread] = None
        self.engine = None
        self.engine_loaded = False
        self.devices: list[AudioDevice] = []
        self._input_devices: list[AudioDevice] = []
        self._output_devices: list[AudioDevice] = []
        self._config_lock = threading.Lock()
        self._steps = 24
        self._output_device: Optional[AudioDevice] = None

        self.status_var = tk.StringVar(value="Выберите микрофон и нажмите «Старт».")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.threshold_var = tk.DoubleVar(value=0.015)
        self.silence_var = tk.DoubleVar(value=0.70)
        self.steps_var = tk.IntVar(value=24)
        self._build()
        self.refresh_devices()
        self.root.after(50, self._poll_events)

    def _build(self) -> None:
        ttk = self.ttk
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except self.tk.TclError:
            pass
        bg, panel, fg, accent = "#20242b", "#2a303a", "#eef2f7", "#54b6e8"
        self.root.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=panel)
        style.configure("TFrame", background=bg)
        style.configure("Card.TLabelframe", background=panel, foreground=fg)
        style.configure("Card.TLabelframe.Label", background=panel, foreground=fg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", padding=(9, 5))
        style.configure("Accent.TButton", background=accent, foreground="#10212b")
        style.map("Accent.TButton", background=[("active", "#78c9ef")])

        outer = ttk.Frame(self.root, padding=14)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        devices = ttk.LabelFrame(outer, text=" Аудиоустройства ", style="Card.TLabelframe", padding=10)
        devices.grid(row=0, column=0, sticky="ew")
        devices.columnconfigure(1, weight=1)
        ttk.Label(devices, text="Вход").grid(row=0, column=0, sticky="w", padx=(0, 8), pady=3)
        self.input_box = ttk.Combobox(devices, textvariable=self.input_var, state="readonly")
        self.input_box.grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Label(devices, text="Выход").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=3)
        self.output_box = ttk.Combobox(devices, textvariable=self.output_var, state="readonly")
        self.output_box.bind("<<ComboboxSelected>>", self._store_processing_config)
        self.output_box.grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(devices, text="Обновить", command=self.refresh_devices).grid(row=0, column=2, rowspan=2, padx=(8, 0))

        settings = ttk.Frame(outer, padding=(0, 10, 0, 8))
        settings.grid(row=1, column=0, sticky="ew")
        ttk.Label(settings, text="Чувствительность RMS").grid(row=0, column=0, sticky="w")
        ttk.Scale(settings, from_=0.003, to=0.08, variable=self.threshold_var, command=self._apply_segment_settings).grid(row=1, column=0, sticky="ew", padx=(0, 20))
        ttk.Label(settings, text="Конец фразы: тишина (с)").grid(row=0, column=1, sticky="w")
        ttk.Scale(settings, from_=0.2, to=2.0, variable=self.silence_var, command=self._apply_segment_settings).grid(row=1, column=1, sticky="ew")
        settings.columnconfigure(0, weight=1)
        settings.columnconfigure(1, weight=1)

        controls = ttk.Frame(outer)
        controls.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        self.start_button = ttk.Button(controls, text="Старт", style="Accent.TButton", command=self.start)
        self.start_button.pack(side="left")
        self.stop_button = ttk.Button(controls, text="Стоп", command=self.stop, state="disabled")
        self.stop_button.pack(side="left", padx=7)
        ttk.Label(controls, text="Качество (шаги)").pack(side="left", padx=(15, 4))
        for steps in (16, 24, 32):
            ttk.Radiobutton(controls, text=str(steps), value=steps, variable=self.steps_var).pack(side="left", padx=3)
        self.steps_var.trace_add("write", self._store_processing_config)

        ttk.Label(outer, text="Распознанный текст / текст для ручной проверки").grid(row=3, column=0, sticky="w")
        self.text = self.tk.Text(outer, height=9, wrap="word", bg=panel, fg=fg, insertbackground=fg, relief="flat", padx=9, pady=8)
        self.text.grid(row=4, column=0, sticky="nsew", pady=(4, 8))
        outer.rowconfigure(4, weight=1)
        ttk.Button(outer, text="Озвучить текст", command=self.manual_synthesize).grid(row=5, column=0, sticky="w")
        ttk.Label(outer, textvariable=self.status_var, wraplength=650).grid(row=6, column=0, sticky="w", pady=(10, 0))

    def _emit(self, kind: str, value: object) -> None:
        try:
            self.events.put_nowait((kind, value))
        except queue.Full:
            pass

    def _apply_segment_settings(self, _unused=None) -> None:
        self.segmenter.configure(threshold=self.threshold_var.get(), end_silence_seconds=self.silence_var.get())

    def _store_processing_config(self, *_unused) -> None:
        output = self._select(self.output_var.get(), getattr(self, "_output_devices", []))
        with self._config_lock:
            self._steps = int(self.steps_var.get())
            self._output_device = output

    def _processing_config(self) -> tuple[int, Optional[AudioDevice]]:
        with self._config_lock:
            return self._steps, self._output_device

    def _set_active_generation(self, generation: int) -> None:
        with self._generation_lock:
            self._active_generation = generation

    def _get_active_generation(self) -> int:
        with self._generation_lock:
            return self._active_generation

    def _sounddevice(self):
        if self.sd is None:
            import sounddevice as sounddevice
            self.sd = sounddevice
        return self.sd

    def refresh_devices(self) -> None:
        try:
            sd = self._sounddevice()
            raw_devices, hostapis = sd.query_devices(), sd.query_hostapis()
            inputs, outputs = [], []
            for index, info in enumerate(raw_devices):
                host = hostapis[info["hostapi"]]["name"]
                label = f"{index}: {info['name']}  [{host}]"
                rate = max(1, int(info["default_samplerate"]))
                if info["max_input_channels"] > 0:
                    inputs.append(AudioDevice(index, label, rate))
                if info["max_output_channels"] > 0:
                    outputs.append(AudioDevice(index, label, rate))
            self.devices = inputs + outputs
            self._input_devices, self._output_devices = inputs, outputs
            self.input_box["values"] = [x.label for x in inputs]
            self.output_box["values"] = [x.label for x in outputs]
            if inputs and self.input_var.get() not in self.input_box["values"]:
                self.input_var.set(inputs[0].label)
            if outputs and self.output_var.get() not in self.output_box["values"]:
                self.output_var.set(outputs[0].label)
            self._store_processing_config()
            self.status_var.set(f"Найдено: входов {len(inputs)}, выходов {len(outputs)}.")
        except Exception as exc:
            self.status_var.set(f"Не удалось получить устройства: {exc}")

    @staticmethod
    def _select(label: str, choices: list[AudioDevice]) -> Optional[AudioDevice]:
        return next((item for item in choices if item.label == label), None)

    def start(self) -> None:
        if self.running:
            return
        input_device = self._select(self.input_var.get(), self._input_devices)
        if input_device is None:
            self.status_var.set("Выберите доступный микрофон.")
            return
        self.segmenter.reset()
        generation = self.gate.next()
        self._set_active_generation(generation)
        try:
            sd = self._sounddevice()
            self.input_stream = sd.InputStream(
                device=input_device.index, channels=1, samplerate=input_device.default_rate,
                dtype="float32",
                callback=lambda indata, frames, time, status: self._audio_callback(
                    generation, input_device.default_rate, indata, frames, time, status
                ),
            )
            self.input_stream.start()
        except Exception as exc:
            self.input_stream = None
            self.status_var.set(f"Микрофон не запущен: {exc}")
            return
        self.running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._start_threads()
        self._put_job(("load", generation))
        self.status_var.set("Слушаю. Во время озвучивания захват временно приостановлен.")

    def _start_threads(self) -> None:
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._capture_loop, name="audio-capture", daemon=True)
            self.capture_thread.start()
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.worker_thread = threading.Thread(target=self._worker_loop, name="voice-engine", daemon=True)
            self.worker_thread.start()

    def _audio_callback(self, generation: int, source_rate: int, indata, _frames, _time, status) -> None:
        if status:
            self._emit("status", f"Аудиовход: {status}")
        if self.closing or not self.running or self.playback_active.is_set():
            return
        try:
            self.capture_queue.put_nowait((generation, source_rate, indata[:, 0].copy()))
        except queue.Full:
            self._emit("status", "Аудиобуфер переполнен — часть сигнала пропущена.")

    def _capture_loop(self) -> None:
        # It lives through start/stop cycles.  Every frame carries the stream's
        # generation, so a callback from a just-closed stream cannot leak into
        # the next listening session.
        while not self.closing:
            try:
                generation, input_rate, audio = self.capture_queue.get(timeout=0.15)
            except queue.Empty:
                continue
            if not self.running or not self.gate.current(generation) or self.playback_active.is_set():
                self.segmenter.reset()
                continue
            try:
                speech = self._resample(audio, input_rate, TARGET_SAMPLE_RATE)
                for phrase in self.segmenter.feed(speech):
                    if not self.running or not self.gate.current(generation) or self.playback_active.is_set():
                        self.segmenter.reset()
                        break
                    steps, output = self._processing_config()
                    self._put_job(("phrase", generation, phrase, steps, output))
            except Exception as exc:
                self._emit("status", f"Ошибка обработки микрофона: {exc}")

    @staticmethod
    def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return np.asarray(audio, dtype=np.float32)
        from math import gcd
        from scipy.signal import resample_poly
        divisor = gcd(source_rate, target_rate)
        return resample_poly(np.asarray(audio, dtype=np.float32), target_rate // divisor, source_rate // divisor).astype(np.float32)

    def _put_job(self, job: tuple) -> None:
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self._emit("status", "Очередь обработки занята; фраза пропущена.")

    def _engine_status(self, message: str) -> None:
        self._emit("status", str(message))

    def _ensure_engine_loaded(self) -> None:
        """Retries load on the next job when a previous initialization failed."""
        if self.engine_loaded:
            return
        self._emit("status", "Загрузка модели…")
        if self.engine is None:
            from engine import VoiceEngine  # heavy backend is isolated here
            self.engine = VoiceEngine(Path(__file__).resolve().parent, self._engine_status)
        self.engine.load()
        self.engine_loaded = True
        self._emit("status", "Модель готова.")

    def _worker_loop(self) -> None:
        while not self.closing:
            try:
                job = self.jobs.get(timeout=0.2)
            except queue.Empty:
                continue
            if job[0] == "quit":
                return
            try:
                self._ensure_engine_loaded()
                kind, generation = job[0], job[1]
                if not self.gate.current(generation):
                    continue
                if kind == "phrase":
                    self._emit("status", "Распознавание…")
                    text = self.engine.transcribe(job[2])
                    if not self.gate.current(generation):
                        continue
                    self._emit("recognized", text)
                    if text.strip():
                        self._synthesize_and_play(text, generation, job[3], job[4])
                elif kind == "manual":
                    self._synthesize_and_play(job[2], generation, job[3], job[4])
            except Exception as exc:
                self._emit("status", f"Ошибка движка: {exc}")

    def _synthesize_and_play(self, text: str, generation: int, steps: int, output: Optional[AudioDevice]) -> None:
        if not self.gate.current(generation):
            return
        self._emit("status", "Синтез речи…")
        audio, rate = self.engine.synthesize(text, steps=steps)
        if not self.gate.current(generation):
            return
        if output is None:
            self._emit("status", "Нет устройства вывода для воспроизведения.")
            return
        try:
            sd = self._sounddevice()
            play_rate = int(rate)
            try:
                sd.check_output_settings(device=output.index, channels=1, dtype="float32", samplerate=play_rate)
            except Exception:
                play_rate = output.default_rate
                audio = self._resample(audio, int(rate), play_rate)
            if not self.gate.current(generation):
                return
            # Start and Stop share this lock.  If Stop wins, the generation
            # check below prevents a late sounddevice.play call; if play wins,
            # Stop calls sd.stop after the non-blocking stream is started.
            with self._output_lock:
                if not self.gate.current(generation):
                    return
                self._clear_capture_queue()
                self.segmenter.reset()
                self.playback_active.set()
                self._emit("status", "Озвучивание…")
                sd.play(np.asarray(audio, dtype=np.float32), samplerate=play_rate, device=output.index, blocking=False)
            sd.wait()
            if self.gate.current(generation):
                self._emit("status", "Слушаю.")
        except Exception as exc:
            self._emit("status", f"Воспроизведение не выполнено: {exc}")
        finally:
            self.playback_active.clear()

    def manual_synthesize(self) -> None:
        content = self.text.get("1.0", "end-1c").strip()
        if not content:
            self.status_var.set("Введите текст для озвучивания.")
            return
        # A manual preview should not invalidate the still-open microphone
        # stream.  Outside a listening session it starts a fresh generation.
        if self.running:
            generation = self._get_active_generation()
        else:
            generation = self.gate.next()
            self._set_active_generation(generation)
        steps, output = self._processing_config()
        self._start_threads()
        self._put_job(("manual", generation, content, steps, output))
        self.stop_button.configure(state="normal")
        self.status_var.set("Текст поставлен в очередь на озвучивание.")

    def _clear_capture_queue(self) -> None:
        while True:
            try:
                self.capture_queue.get_nowait()
            except queue.Empty:
                return

    def stop(self) -> None:
        self.gate.cancel()
        self.running = False
        self.segmenter.reset()
        self._clear_capture_queue()
        try:
            if self.input_stream is not None:
                self.input_stream.stop()
                self.input_stream.close()
        except Exception as exc:
            self.status_var.set(f"Микрофон остановлен с предупреждением: {exc}")
        finally:
            self.input_stream = None
        with self._output_lock:
            try:
                if self.sd is not None:
                    self.sd.stop()
            except Exception:
                pass
        self.playback_active.clear()
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("Остановлено. Готово к новому запуску.")

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "status":
                    self.status_var.set(str(value))
                elif kind == "recognized":
                    self.text.delete("1.0", "end")
                    self.text.insert("1.0", str(value))
        except queue.Empty:
            pass
        if not self.closing:
            self.root.after(50, self._poll_events)

    def close(self) -> None:
        if self.closing:
            return
        self.closing = True
        self.stop()
        self._put_job(("quit",))
        # Потоки daemon; короткие join не блокируют закрытие, даже если ML-драйвер завис.
        for thread in (self.capture_thread, self.worker_thread):
            if thread and thread.is_alive():
                thread.join(timeout=0.3)
        self.root.destroy()


def main() -> None:
    import tkinter as tk
    root = tk.Tk()
    VoiceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
