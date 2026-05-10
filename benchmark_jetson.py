import torch, torch.nn as nn, time, csv, datetime, numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256*4*4,512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512,2)
        )
    def forward(self, x): return self.classifier(self.features(x))

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_gatekeeper_jetson.pth", map_location=device, weights_only=False))
model.eval()
print("Warming up...")
for _ in range(10):
    with torch.no_grad(): model(torch.randn(1,3,224,224).to(device))
print("Benchmarking 100 frames...")
latencies = []
for i in range(100):
    x = torch.randn(1,3,224,224).to(device)
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad(): model(x)
    torch.cuda.synchronize()
    l = (time.time()-start)*1000
    latencies.append(l)
    if i%20==0: print(f"Frame {i:03d} | {l:.2f}ms")
mean=sum(latencies)/len(latencies)
p95=sorted(latencies)[95]
p99=sorted(latencies)[99]
print(f"Mean: {mean:.2f}ms | P95: {p95:.2f}ms | P99: {p99:.2f}ms | FPS: {1000/mean:.1f}")
import os; os.makedirs("results", exist_ok=True)
with open("results/jetson_latency.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["frame","latency_ms","date","device"])
    w.writeheader()
    for i,l in enumerate(latencies):
        w.writerow({"frame":i,"latency_ms":round(l,2),"date":str(datetime.date.today()),"device":"Jetson Orin Nano Super CUDA"})
print("Saved results/jetson_latency.csv")