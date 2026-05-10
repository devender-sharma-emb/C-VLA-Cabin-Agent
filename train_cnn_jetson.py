import torch, torch.nn as nn, json, numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")

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

train_ds = DMDDataset("dmd_labels.jsonl", "train")
valid_ds = DMDDataset("dmd_labels.jsonl", "valid")
train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
valid_dl = DataLoader(valid_ds, batch_size=32, shuffle=False, num_workers=0)
print(f"Train: {len(train_ds)}, Valid: {len(valid_ds)}")

model = SimpleCNN().to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=20)
crit = nn.CrossEntropyLoss()

best = 0
for ep in range(20):
    model.train(); ls=cor=tot=0
    for x,y in train_dl:
        x,y = x.to(device),y.to(device)
        opt.zero_grad(); o=model(x); l=crit(o,y); l.backward(); opt.step()
        ls+=l.item(); cor+=(o.argmax(1)==y).sum().item(); tot+=y.size(0)
    scheduler.step()
    model.eval(); vc=vt=0
    with torch.no_grad():
        for x,y in valid_dl:
            x,y=x.to(device),y.to(device); o=model(x)
            vc+=(o.argmax(1)==y).sum().item(); vt+=y.size(0)
    va=vc/vt*100
    print(f"Epoch {ep+1}/20 | Loss: {ls/len(train_dl):.3f} | Train: {cor/tot*100:.1f}% | Val: {va:.1f}%")
    if va>best: best=va; torch.save(model.state_dict(),"cnn_gatekeeper_jetson.pth"); print(f"  -> Saved ({va:.1f}%)")
print(f"Done. Best: {best:.1f}%")
