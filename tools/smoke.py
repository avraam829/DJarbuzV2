"""Exercise real checkpoints and ASR without capturing the user's microphone."""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from engine import VoiceEngine

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

engine = VoiceEngine(ROOT, lambda message: print(message, flush=True))
t = time.perf_counter()
engine.load()
loaded = time.perf_counter() - t
audio, rate = sf.read(ROOT / "assets/training/dataset/wavs/0001.wav", dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
from math import gcd
divisor = gcd(rate, 16000)
audio = resample_poly(audio, 16000 // divisor, rate // divisor)
t = time.perf_counter()
recognized = engine.transcribe(audio)
asr_time = time.perf_counter() - t
print("Recognized:", recognized, flush=True)
assert recognized.strip(), "No recognized speech"
text = "Привет! Это проверка моего нового голоса."
t = time.perf_counter()
result, sr = engine.synthesize(text, steps=16)
synth_time = time.perf_counter() - t
assert result.size > sr * 0.3 and np.max(np.abs(result)) > 0.001
output = ROOT / "output"
output.mkdir(exist_ok=True)
sf.write(output / "dj-arbuz-test.wav", result, sr)
report = {"load_seconds": loaded, "input_text": recognized, "asr_seconds": asr_time,
          "synth_text": text, "synth_seconds": synth_time, "audio_seconds": len(result)/sr,
          "sample_rate": sr, "finite": bool(np.isfinite(result).all())}
(ROOT / "docs/smoke-result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
print(report, flush=True)
