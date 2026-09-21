import os
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "results/models/Qwen3.5-9B"
VEC_FILE = "results/vectors_havarie.pt"
EVAL_FILE = "scenario/eval_havarie.jsonl"
OUT_PLOT = "results/layer_activation_map.png"

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[*] Nutze Device: {DEVICE}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map=DEVICE
)
model.eval()

num_layers = len(model.model.layers)
print(f"[*] Modell hat {num_layers} Schichten.")

# Basis-Refusal-Richtung laden (als Richtschnur für semantische Verweigerung)
vec_data = torch.load(VEC_FILE, map_location=DEVICE)
v_ref = vec_data["v_refuse"].to(DEVICE, dtype=torch.float32)

items = []
with open(EVAL_FILE) as f:
    for line in f:
        items.append(json.loads(line))

# Matrix: [5 Items x 32 Schichten]
activation_matrix = np.zeros((len(items), num_layers))

print("[*] Scanne Aktivierungsprofile über alle 32 Schichten...")

for item_idx, item in enumerate(items):
    raw_prompt = item["prompt"]
    messages = [{"role": "user", "content": raw_prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    # Reines Baseline-Signal ohne Prefill analysieren, um das natürliche Entstehen zu sehen
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    
    layer_outputs = {}
    handles = []
    
    for l_idx in range(num_layers):
        def make_hook(idx):
            def hook(module, args, output):
                h = output[0] if isinstance(output, tuple) else output
                layer_outputs[idx] = h.detach()[0, -1, :].to(torch.float32)
            return hook
        handles.append(model.model.layers[l_idx].register_forward_hook(make_hook(l_idx)))
        
    with torch.inference_mode():
        model(**inputs)
        
    for h in handles:
        h.remove()
        
    # Berechne für jede Schicht die Projektion auf die Verweigerungsachse
    for l_idx in range(num_layers):
        h = layer_outputs[l_idx]
        proj = (h @ v_ref).item()
        activation_matrix[item_idx, l_idx] = proj
        
    print(f"    [+] Item {item_idx+1}/{len(items)} gescannt.")

# Heatmap generieren
plt.figure(figsize=(14, 6), dpi=140)
plt.imshow(activation_matrix, aspect='auto', cmap='magma', interpolation='nearest')
plt.colorbar(label='Projektion auf Refusal-Achse <h, v_refuse>')

plt.title("Schichten-Topologie der Guardrail-Aktivierung (Qwen 3.5 - 32 Schichten)", fontsize=14)
plt.xlabel("Schichten-Index (Layer 0 bis 31)", fontsize=12)
plt.ylabel("Trigger Prompts (Item 1 bis 5)", fontsize=12)

# Full Attention Intervalle (alle 4 Schichten) markieren
for full_attn_layer in range(3, num_layers, 4):
    plt.axvline(x=full_attn_layer, color='cyan', linestyle=':', alpha=0.7, label='Full Attention Layer' if full_attn_layer == 3 else "")

plt.xticks(ticks=np.arange(0, num_layers, 2), labels=np.arange(0, num_layers, 2))
plt.yticks(ticks=np.arange(len(items)), labels=[f"Item {i+1}" for i in range(len(items))])
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f"[OK] Schichten-Aktivierungs-Heatmap gespeichert: {OUT_PLOT}")

# Finde die stärksten Schichten statistisch
layer_means = activation_matrix.mean(axis=0)
top_layers = np.argsort(layer_means)[::-1][:5]
print("\n" + "="*50)
print(f"[*] TOP-5 SCHICHTEN MIT HÖCHSTER VERWEIGERUNGSAKTIVITÄT:")
for rank, l in enumerate(top_layers):
    print(f"    Rang {rank+1}: Schicht {l} (Durchschnittlicher Ausschlag: {layer_means[l]:.4f})")
print("="*50)
