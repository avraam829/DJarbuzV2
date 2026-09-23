"""Export exact EMA weights, without training optimizer state. Keep original intact."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors.torch import save_file, load_file

root = Path(__file__).resolve().parents[1]
src = root / "assets/dj-arbuz/model_last.pt"
dst = src.with_name("voice.safetensors")
checkpoint = torch.load(src, map_location="cpu", weights_only=True, mmap=True)
state = checkpoint["ema_model_state_dict"]
tensors = {k: v.contiguous() for k, v in state.items()}
save_file(tensors, str(dst))
restored = load_file(str(dst))
assert state.keys() == restored.keys()
assert all(torch.equal(v, restored[k]) for k, v in state.items())
report = {"source": str(src.relative_to(root)), "output": str(dst.relative_to(root)),
          "tensors": len(state), "step": str(checkpoint.get("step", "unknown")),
          "exact_tensor_equality": True, "bytes": dst.stat().st_size}
(root / "docs/voice-export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(report)
