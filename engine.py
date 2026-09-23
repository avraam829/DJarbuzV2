"""Local Russian ASR + the user's fine-tuned F5-TTS checkpoint."""
import os
import sys
from pathlib import Path


class VoiceEngine:
    def __init__(self, root: Path, status=print):
        self.root = Path(root).resolve()
        self.status = status
        self.tts = self.asr = None
        cache = self.root / "cache"
        cache.mkdir(exist_ok=True)
        os.environ["HF_HOME"] = str(cache / "huggingface")
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["MPLCONFIGDIR"] = str(cache / "matplotlib")
        sys.path.insert(0, str(self.root / "vendor"))

    def load(self):
        import torch
        from f5_tts.api import F5TTS
        from faster_whisper import WhisperModel

        torch.set_num_threads(4)
        if self.tts is None:
            self.status("Загрузка обученного DJ Arbuz…")
            # EMA-only export avoids loading multi-GB optimizer state into GPU memory.
            checkpoint = self.root / "assets/dj-arbuz/voice.safetensors"
            if not checkpoint.is_file():
                raise FileNotFoundError("Нет voice.safetensors. Выполните setup.ps1.")
            self.tts = F5TTS(
                model="F5TTS_Base", ckpt_file=str(checkpoint),
                vocab_file=str(self.root / "assets/training/prepared/vocab.txt"),
                vocoder_local_path=str(self.root / "assets/vocos"),
                device="cuda" if torch.cuda.is_available() else "cpu",
                hf_cache_dir=str(self.root / "cache/huggingface"),
            )
        if self.asr is None:
            self.status("Загрузка распознавания русской речи (при первом запуске скачивание)…")
            # Leave the 6 GB GPU to F5; int8 ASR stays on CPU.
            self.asr = WhisperModel("small", device="cpu", compute_type="int8", cpu_threads=4,
                                    download_root=str(self.root / "assets/asr"))
        metadata = self.root / "assets/training/dataset/metadata.csv"
        lines = metadata.read_text(encoding="utf-8-sig").splitlines()
        refs = dict(line.split("|", 1) for line in lines if "|" in line)
        self.ref_audio = self.root / "assets/training/dataset/wavs/0002.wav"
        self.ref_text = refs["0002"]
        self.status("Модель готова" + (" · CUDA" if torch.cuda.is_available() else " · CPU (медленнее)"))

    def transcribe(self, audio):
        import numpy as np
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim != 1 or not np.isfinite(audio).all():
            raise ValueError("Ожидался конечный моносигнал 16 кГц")
        segments, _ = self.asr.transcribe(audio, language="ru", beam_size=3,
                                          condition_on_previous_text=False, vad_filter=True)
        return " ".join(s.text.strip() for s in segments).strip()

    def synthesize(self, text, steps=24):
        import numpy as np
        if not text.strip():
            raise ValueError("Пустая фраза")
        if len(text) > 500:
            raise ValueError("Фраза слишком длинная: максимум 500 символов")
        wav, sr, _ = self.tts.infer(
            ref_file=str(self.ref_audio), ref_text=self.ref_text, gen_text=text,
            nfe_step=int(steps), show_info=lambda *_: None, seed=42,
        )
        wav = np.asarray(wav, dtype=np.float32)
        if not wav.size or not np.isfinite(wav).all():
            raise RuntimeError("Модель вернула некорректный звук")
        return np.clip(wav, -1, 1), sr
