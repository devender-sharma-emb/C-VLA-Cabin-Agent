import os, json
SIMPLE = {3}
splits = {'train': 'data/dmd/train', 'test': 'data/dmd/test', 'valid': 'data/dmd/valid'}
records = []
for split, path in splits.items():
    img_dir = os.path.join(path, 'images')
    lbl_dir = os.path.join(path, 'labels')
    for img_file in os.listdir(img_dir):
        if not img_file.endswith('.jpg'): continue
        lbl_path = os.path.join(lbl_dir, img_file.replace('.jpg', '.txt'))
        if not os.path.exists(lbl_path): continue
        lines = open(lbl_path).readlines()
        if not lines: continue
        class_id = int(lines[0].split()[0])
        label = 'simple' if class_id == 3 else 'complex_intent'
        records.append({'clip_id': img_file, 'file': os.path.join(img_dir, img_file), 'label': label, 'split': split, 'source': 'DMD'})
with open('data/dmd_labels.jsonl', 'w') as f:
    for r in records:
        f.write(json.dumps(r) + chr(10))
simple = sum(1 for r in records if r['label'] == 'simple')
print(f'Total: {len(records)}, Simple: {simple}, Complex: {len(records)-simple}')
print('Saved to data/dmd_labels.jsonl')
