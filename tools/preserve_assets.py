"""Copy irreplaceable voice assets; verify every copy with SHA-256. Never delete."""
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "DJArbuz"


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    mappings = {
        "models/F5-TTS-Russian": "assets/f5-russian-base",
        "F5-TTS/ckpts/dj_arbuz/model_last.pt": "assets/dj-arbuz/model_last.pt",
        "F5-TTS/ckpts/dj_arbuz/setting.json": "assets/dj-arbuz/setting.json",
        "F5-TTS/data/dj_arbuz_char": "assets/training/prepared",
        "dataset": "assets/training/dataset",
        "raw": "assets/training/raw",
        "prepare_dataset.py": "assets/training/prepare_dataset.py",
        "F5-TTS/src/f5_tts/configs": "assets/training/model-configs",
    }
    records = []
    for old, new in mappings.items():
        src, dst = SOURCE / old, ROOT / new
        if not src.exists():
            raise FileNotFoundError(src)
        files = [src] if src.is_file() else sorted(p for p in src.rglob("*") if p.is_file() and ".cache" not in p.parts)
        for file in files:
            target = dst if src.is_file() else dst / file.relative_to(src)
            target.parent.mkdir(parents=True, exist_ok=True)
            source_hash = digest(file)
            if not target.exists() or digest(target) != source_hash:
                shutil.copy2(file, target)
            if digest(target) != source_hash:
                raise RuntimeError(f"Copy verification failed: {target}")
            records.append({"source": str(file), "destination": str(target.relative_to(ROOT)), "bytes": target.stat().st_size, "sha256": source_hash})
            print(target.relative_to(ROOT), flush=True)
    manifest = ROOT / "docs/preserved-assets.json"
    manifest.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Verified {len(records)} files; {sum(r['bytes'] for r in records):,} bytes", flush=True)


if __name__ == "__main__":
    main()
