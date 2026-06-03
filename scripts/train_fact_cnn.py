"""训练事实域判断 ChuchuCNN（第 6 个 CNN）。

任务：输入两条记忆，判断是否属于同一事实域（二分类）。
数据：从 training_labeled.jsonl 自动生成正负样本对。
输出：app/brain/model_fact/chuchu_cnn.pt（~500KB）

用法: python scripts/train_fact_cnn.py
"""
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.brain.chuchu_tok import ChuchuTok
from app.brain.chuchu_model import ChuchuCNN

# ── 配置 ────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "training_labeled.jsonl"
SAVE_DIR = PROJECT_ROOT / "app" / "brain" / "model_fact"
SAVE_PATH = SAVE_DIR / "chuchu_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 8
BATCH_SIZE = 32
LR = 1e-3
SEP = " [SEP] "  # 两条文本的分隔符

print(f"设备: {DEVICE}")

# ── 加载数据 ─────────────────────────────────────────────────────
lines = [l.strip() for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
items = [json.loads(l) for l in lines]

# 按簇分组
clusters: dict[int, list[str]] = {}
for it in items:
    lid = int(it["label"])
    clusters.setdefault(lid, []).append(it["text"])

print(f"簇数: {len(clusters)}")
print(f"总样本: {len(items)}")

# ── 生成训练对 ─────────────────────────────────────────────────
random.seed(42)
POS_PER_CLUSTER = 15   # 每簇正样本对数
NEG_PER_CLUSTER = 8    # 每簇负样本对数
MIN_CLUSTER_SIZE = 5   # 太小的簇跳过

data: list[tuple[str, int]] = []  # (text, 0=不同/1=相同)

cluster_ids = list(clusters.keys())

# 正样本：同簇内的两个不同文本
for lid, texts in clusters.items():
    if len(texts) < MIN_CLUSTER_SIZE:
        continue
    for _ in range(POS_PER_CLUSTER):
        a, b = random.sample(texts, 2)
        data.append((a + SEP + b, 1))

# 负样本：不同簇的两个文本
all_keys = [k for k in cluster_ids if len(clusters[k]) >= MIN_CLUSTER_SIZE]
for _ in range(NEG_PER_CLUSTER * len(all_keys)):
    k1, k2 = random.sample(all_keys, 2)
    a = random.choice(clusters[k1])
    b = random.choice(clusters[k2])
    data.append((a + SEP + b, 0))

random.shuffle(data)
texts, labels = zip(*data)

labels_list = ["different", "same"]
label2id = {"different": 0, "same": 1}
num_classes = 2

print(f"训练对总数: {len(data)}")
same_count = sum(1 for _, l in data if l == 1)
diff_count = sum(1 for _, l in data if l == 0)
print(f"  same: {same_count}, different: {diff_count}")

# ── Tokenizer ──────────────────────────────────────────────────
tok = ChuchuTok.load(str(PROJECT_ROOT / "app" / "brain" / "chuchu_tok.json"))

# ── 数据集 ─────────────────────────────────────────────────────
MAX_LEN = 128  # 两条文本拼接需要更长

class PairDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = [tok.encode(t, MAX_LEN) for t in texts]
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.encodings[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

split = int(len(texts) * 0.85)
train_ds = PairDataset(texts[:split], labels[:split])
val_ds = PairDataset(texts[split:], labels[split:])
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ── 训练 ─────────────────────────────────────────────────────
model = ChuchuCNN(vocab_size=tok.vocab_size, num_classes=num_classes, max_len=MAX_LEN).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"\n开始训练: {EPOCHS} epochs, max_len={MAX_LEN}...")
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct = total = 0
    val_loss = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            val_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"  Epoch {epoch+1:>2}/{EPOCHS}  "
          f"train_loss={train_loss/len(train_loader):.4f}  "
          f"val_loss={val_loss/len(val_loader):.4f}  "
          f"val_acc={acc:.4f}")

# ── 分类报告 ─────────────────────────────────────────────────
model.eval()
all_preds, all_gold = [], []
with torch.no_grad():
    for x, y in val_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        preds = model(x).argmax(dim=1)
        all_preds.extend(preds.cpu().tolist())
        all_gold.extend(y.cpu().tolist())

from sklearn.metrics import classification_report
report = classification_report(all_gold, all_preds, target_names=labels_list, zero_division=0)
print(f"\n分类报告:")
print(report)

# ── 保存 ─────────────────────────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": tok.vocab_size,
    "num_classes": num_classes,
    "label2id": label2id,
    "id2label": {0: "different", 1: "same"},
    "embed_dim": 64,
    "num_filters": 128,
    "kernel_sizes": (3, 4, 5),
    "dropout": 0.5,
    "max_len": MAX_LEN,
    "name": "fact_domain",
}, SAVE_PATH)
print(f"\n模型保存: {SAVE_PATH}")
print(f"大小: {SAVE_PATH.stat().st_size / 1024:.1f}KB")
