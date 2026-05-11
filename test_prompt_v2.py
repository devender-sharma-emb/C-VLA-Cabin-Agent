import torch
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("mps")
model_id = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float16
).to(device)

SYSTEM_PROMPT = """You are an automotive cabin AI agent. 
Your job is to take ONE specific action based on driver observations.
Always respond with exactly one sentence starting with an action verb.
Never repeat the same action twice in a row.

Examples:
- Observation: driver yawning repeatedly -> Action: Activate ventilation and suggest 5-minute rest stop ahead.
- Observation: driver looking at phone -> Action: Send hands-free notification and audibly alert driver to focus on road.
- Observation: driver pointing at dashboard -> Action: Display contextual dashboard help overlay for indicated control.
- Observation: driver slouching in seat -> Action: Adjust lumbar support and suggest ergonomic repositioning.
- Observation: driver showing fatigue signs -> Action: Lower cabin temperature by 2 degrees to increase alertness."""

test_cases = [
    "driver yawning at 11pm on highway",
    "driver reaching toward passenger seat",
    "driver pointing at AC vent area",
    "driver head drooping slowly",
    "driver looking away from road repeatedly",
]

for obs in test_cases:
    prompt = f"{SYSTEM_PROMPT}\n\nObservation: {obs}\nAction:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            temperature=1.0,
        )
    latency = (time.time() - start) * 1000
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    ).strip()
    print(f"Obs: {obs}")
    print(f"Action: {response}")
    print(f"Latency: {latency:.0f}ms\n")