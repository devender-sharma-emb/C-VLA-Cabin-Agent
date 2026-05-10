import torch
import timm
import time
import cv2
import csv
import datetime
import numpy as np
from PIL import Image
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')

# --- SYSTEM 1: ViT-tiny gatekeeper ---
print('Loading System 1 (ViT-tiny gatekeeper)...')
vit = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=2)
lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=['qkv'], lora_dropout=0.1, bias='none')
vit = get_peft_model(vit, lora_config)
vit.load_state_dict(torch.load('models/vit_gatekeeper.pth', map_location=device))
vit = vit.to(device)
vit.eval()
print('System 1 ready')

# --- SYSTEM 2: Llama reasoning brain ---
print('Loading System 2 (Llama-3.2-1B reasoning)...')
model_id = 'meta-llama/Llama-3.2-1B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_id)
llm = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float16).to(device)
print('System 2 ready')

def preprocess_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224))
    tensor = torch.tensor(np.array(img)).permute(2,0,1).float() / 255.0
    return tensor.unsqueeze(0).to(device)

def system1_gate(frame):
    with torch.no_grad():
        tensor = preprocess_frame(frame)
        logits = vit(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1).item()
        confidence = probs[0][pred].item()
    return pred == 1, confidence

def system2_reason(context):
    prompt = f'You are an in-cabin automotive AI agent. Driver observation: {context}. What single safety or comfort action should you take? Reply in one sentence.'
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        outputs = llm.generate(**inputs, max_new_tokens=60, do_sample=False)
    return tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
print('Starting C-VLA cabin agent — processing 20 frames...')
print('-' * 60)

results = []
frame_count = 0
s2_triggers = 0

while frame_count < 20:
    ret, frame = cap.read()
    if not ret:
        break
    start = time.time()
    needs_reasoning, confidence = system1_gate(frame)
    s1_latency = (time.time() - start) * 1000
    if needs_reasoning:
        s2_triggers += 1
        response = system2_reason('complex driver behavior detected by visual classifier')
        total_latency = (time.time() - start) * 1000
        print(f'[S2] Frame {frame_count:02d} | conf={confidence:.2f} | {response.strip()[:80]} | {total_latency:.0f}ms')
        results.append({'frame': frame_count, 'system': 2, 's1_latency_ms': round(s1_latency), 'total_latency_ms': round(total_latency), 'confidence': round(confidence, 3)})
    else:
        print(f'[S1] Frame {frame_count:02d} | conf={confidence:.2f} | normal driving | {s1_latency:.0f}ms')
        results.append({'frame': frame_count, 'system': 1, 's1_latency_ms': round(s1_latency), 'total_latency_ms': round(s1_latency), 'confidence': round(confidence, 3)})
    frame_count += 1

cap.release()

# --- SUMMARY ---
trigger_rate = s2_triggers / frame_count * 100
s1_latencies = [r['s1_latency_ms'] for r in results]
s2_latencies = [r['total_latency_ms'] for r in results if r['system'] == 2]
print('-' * 60)
print(f'Frames processed: {frame_count}')
print(f'System 2 trigger rate: {trigger_rate:.1f}%')
print(f'System 1 mean latency: {sum(s1_latencies)/len(s1_latencies):.0f}ms')
if s2_latencies:
    print(f'System 2 mean latency: {sum(s2_latencies)/len(s2_latencies):.0f}ms')
print(f'Energy saving estimate: {100-trigger_rate:.1f}% of frames avoided LLM')

with open('results/gating_v4_latency.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['frame','system','s1_latency_ms','total_latency_ms','confidence','date','device'])
    writer.writeheader()
    for r in results:
        r['date'] = str(datetime.date.today())
        r['device'] = f'MacBook M3 {device}'
        writer.writerow(r)
print('Saved to results/gating_v4_latency.csv')
