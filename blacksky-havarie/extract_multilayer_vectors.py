import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "results/models/Qwen3.5-9B"
PAIRS_FILE = "scenario/pairs_havarie.jsonl"
OUT_VEC_FILE = Path("results/vectors_multilayer.pt")

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Extraktion auf Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE
)
model.eval()

cascade_layers = [12, 16, 20]

pairs = []
with open(PAIRS_FILE, "r", encoding="utf-8") as f:
    for line in f:
        pairs.append(json.loads(line))

vectors = {}

for layer_idx in cascade_layers:
    print(f"[*] Verarbeite Kaskaden-Schicht {layer_idx}...")
    acts_refuse = []
    acts_action = []
    
    for p in pairs:
        for key, storage in [("refusal", acts_refuse), ("action", acts_action)]:
            inputs = tokenizer(p[key], return_tensors="pt").to(DEVICE)
            holder = {}
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                holder["h"] = h.detach()
            handle = model.model.layers[layer_idx].register_forward_hook(hook)
            with torch.inference_mode():
                model(**inputs)
            handle.remove()
            storage.append(holder["h"][0, -1, :].to(torch.float32))

    ref_stack = torch.stack(acts_refuse)
    act_stack = torch.stack(acts_action)
    
    v_ref_raw = ref_stack.mean(dim=0) - act_stack.mean(dim=0)
    v_ref = v_ref_raw / v_ref_raw.norm()
    
    act_mean = act_stack.mean(dim=0)
    v_act_raw = act_mean - (act_mean @ v_ref) * v_ref
    v_act = v_act_raw / v_act_raw.norm()
    
    vectors[str(layer_idx)] = {
        "v_refuse": v_ref.cpu(),
        "v_override": v_act.cpu()
    }
    print(f"    [+] Schicht {layer_idx}: Normen = 1.0, cos = {(v_ref @ v_act).item():.4e}")

torch.save({"layers": cascade_layers, "vectors": vectors}, OUT_VEC_FILE)
print(f"[OK] Kaskaden-Vektoren gespeichert in {OUT_VEC_FILE}")
