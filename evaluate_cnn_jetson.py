import torch, torch.nn as nn, json, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import csv, datetime

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class DMDDataset(Dataset):
    def __init__(self, path, split):
        with open(path) as f:
            raw = [json.loads(l) for l in f if f'"{split}"' in l]
        self.samples = [{"file": s["file"].replace("data/dmd","/home/devvender/dmd"), "label": s["label"]} for s in raw]
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        img = Image.open(s["file"]).convert("RGB").resize((224,224))
        t = torch.tensor(np.array(img)).permute(2,0,1).float()/255.0
        return t, 0 if s["label"]=="simple" else 1

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
            nn.Flatten(),
            nn.Linear(256*4*4, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 2)
        )
    def forward(self, x): return self.classifier(self.features(x))

test_ds = DMDDataset("dmd_labels.jsonl", "test")
test_dl = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Test samples: {len(test_ds)}")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load("cnn_gatekeeper_jetson.pth", map_location=device))
model.eval()

correct = total = tp = tn = fp = fn = 0
with torch.no_grad():
    for x, y in test_dl:
        x, y = x.to(device), y.to(device)
        o = model(x)
        preds = o.argmax(1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        tp += ((preds==1)&(y==1)).sum().item()
        tn += ((preds==0)&(y==0)).sum().item()
        fp += ((preds==1)&(y==0)).sum().item()
        fn += ((preds==0)&(y==1)).sum().item()

acc = correct/total*100
precision = tp/(tp+fp+1e-8)*100
recall = tp/(tp+fn+1e-8)*100
f1 = 2*precision*recall/(precision+recall+1e-8)

print(f"Test Accuracy:  {acc:.1f}%")
print(f"Precision:      {precision:.1f}%")
print(f"Recall:         {recall:.1f}%")
print(f"F1 Score:       {f1:.1f}%")
print(f"TP:{tp} TN:{tn} FP:{fp} FN:{fn}")
print(f"Device: {device}")
