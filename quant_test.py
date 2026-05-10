import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "meta-llama/Llama-3.2-1B-Instruct"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.float16,
    device_map="mps"
)

ram = psutil.virtual_memory()
used_gb = round((ram.total - ram.available) / 1e9, 1)
total_gb = round(ram.total / 1e9, 1)
print(f"RAM used: {used_gb}GB / {total_gb}GB")

prompt = "Driver is yawning at 11pm. What should the cabin agent do?"
inputs = tokenizer(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=50)
print("Response:", tokenizer.decode(outputs[0], skip_special_tokens=True))
print("4-bit test complete.")
