import queue
import threading

import numpy as np

from app import AudioDevice, GenerationGate, PhraseSegmenter, VoiceApp


def test_phrase_emits_after_configured_silence() -> None:
    segmenter = PhraseSegmenter(sample_rate=100, threshold=0.1, min_speech_seconds=0.25, end_silence_seconds=0.7)
    assert segmenter.feed(np.full(30, 0.2, dtype=np.float32)) == []
    emitted = segmenter.feed(np.zeros(70, dtype=np.float32))
    assert len(emitted) == 1
    assert emitted[0].dtype == np.float32
    assert len(emitted[0]) == 100


def test_short_noise_is_discarded_and_maximum_duration_closes_phrase() -> None:
    segmenter = PhraseSegmenter(sample_rate=100, threshold=0.1, min_speech_seconds=0.25, end_silence_seconds=0.7, max_phrase_seconds=0.5)
    assert segmenter.feed(np.full(20, 0.2, dtype=np.float32)) == []
    assert segmenter.feed(np.zeros(70, dtype=np.float32)) == []
    assert segmenter.feed(np.full(30, 0.2, dtype=np.float32)) == []
    emitted = segmenter.feed(np.full(20, 0.2, dtype=np.float32))
    assert len(emitted) == 1
    assert len(emitted[0]) == 50


def test_generation_gate_rejects_late_work_after_stop() -> None:
    gate = GenerationGate()
    started = gate.next()
    assert gate.current(started)
    gate.cancel()
    assert not gate.current(started)


def test_engine_load_is_retried_after_a_failed_attempt() -> None:
    class Engine:
        attempts = 0

        def load(self):
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("temporary model error")

    app = VoiceApp.__new__(VoiceApp)
    app.engine = Engine()
    app.engine_loaded = False
    app._emit = lambda *_: None
    with np.testing.assert_raises(RuntimeError):
        app._ensure_engine_loaded()
    assert not app.engine_loaded
    app._ensure_engine_loaded()
    assert app.engine_loaded
    assert app.engine.attempts == 2


def test_cancel_between_synthesis_and_play_prevents_late_playback() -> None:
    class Engine:
        def synthesize(self, text, steps):
            return np.ones(20, dtype=np.float32), 24_000

    class SoundDevice:
        def __init__(self, gate):
            self.gate = gate
            self.play_calls = 0

        def check_output_settings(self, **_kwargs):
            self.gate.cancel()  # models a Stop event immediately before play

        def play(self, **_kwargs):
            self.play_calls += 1

        def wait(self):
            raise AssertionError("wait must not be reached after cancellation")

    app = VoiceApp.__new__(VoiceApp)
    app.gate = GenerationGate()
    generation = app.gate.next()
    app.engine = Engine()
    app.playback_active = threading.Event()
    app._output_lock = threading.Lock()
    app.capture_queue = queue.Queue()
    app.segmenter = PhraseSegmenter()
    app._emit = lambda *_: None
    fake_sd = SoundDevice(app.gate)
    app._sounddevice = lambda: fake_sd

    app._synthesize_and_play("проверка", generation, 24, AudioDevice(0, "test", 24_000))
    assert fake_sd.play_calls == 0
    assert not app.playback_active.is_set()
