import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Device: {device}")

model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype=torch.float16
).to(device)

def reflex_gatekeeper(sensor_input: str) -> bool:
    keywords = ["gesture", "fatigue", "distraction", "complex", "emergency"]
    return any(k in sensor_input.lower() for k in keywords)

def agentic_reasoning(context: str) -> str:
    prompt = f"You are an in-cabin AI agent. Sensor data: {context}. What single action should you take? Reply in one sentence."
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)
    return tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )

sensor_stream = [
    "normal_driving_highway",
    "complex_gesture_driver_pointing_at_vent",
    "fatigue_detected_slow_blink",
    "checking_mirror_normal",
    "distraction_phone_on_seat",
]

results = []
for data in sensor_stream:
    start = time.time()
    needs_reasoning = reflex_gatekeeper(data)
    if needs_reasoning:
        response = agentic_reasoning(data)
        latency = (time.time() - start) * 1000
        print(f"[SYSTEM 2] {data}")
        print(f"  Action: {response.strip()}")
        print(f"  Latency: {latency:.0f}ms\n")
        results.append({"input": data, "system": 2, "latency_ms": round(latency)})
    else:
        latency = (time.time() - start) * 1000
        print(f"[SYSTEM 1] {data} — no action needed ({latency:.0f}ms)")
        results.append({"input": data, "system": 1, "latency_ms": round(latency)})

import csv, datetime
with open("gating_latency.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["input","system","latency_ms","date","device"])
    writer.writeheader()
    for r in results:
        r["date"] = str(datetime.date.today())
        r["device"] = f"MacBook M3 {device}"
        writer.writerow(r)
print("Results saved to gating_latency.csv")