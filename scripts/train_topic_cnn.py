"""训练话题分类 ChuchuCNN（第 5 个 CNN）。

数据源: data/training_labeled.jsonl（bge-m3 聚类伪标签，50 类）
输出: app/brain/model_topic/chuchu_cnn.pt（~500KB）

用法: python scripts/train_topic_cnn.py
"""
import json
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 确保能从项目根 import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.brain.chuchu_tok import ChuchuTok
from app.brain.chuchu_model import ChuchuCNN

# ── 配置 ────────────────────────────────────────────────────────
DATA_PATH = PROJECT_ROOT / "data" / "training_labeled.jsonl"
SAVE_DIR = PROJECT_ROOT / "app" / "brain" / "model_topic"
SAVE_PATH = SAVE_DIR / "chuchu_cnn.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 15
BATCH_SIZE = 32
LR = 1e-3
MAX_LEN = 128        # 64→128，避免截断长消息
NUM_FILTERS = 192    # 128→192，50% 容量提升
SPLIT = 0.85
AUG_MIN_SAMPLES = 80  # 少于 80 条的类做数据增强
AUG_FACTOR = 3        # 增强倍数（原样本×3）

print(f"设备: {DEVICE}")
print(f"数据: {DATA_PATH}")
print(f"输出: {SAVE_PATH}")
print(f"max_len={MAX_LEN}, filters={NUM_FILTERS}")

# ── 加载数据 ─────────────────────────────────────────────────────
lines = [l.strip() for l in DATA_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
items = [json.loads(l) for l in lines]
texts = [it["text"] for it in items]
labels = [str(it["label"]) for it in items]  # 伪标签用数字字符串

labels_list = sorted(set(labels), key=int)
label2id = {l: i for i, l in enumerate(labels_list)}
id2label = {i: l for l, i in label2id.items()}
num_classes = len(labels_list)

print(f"样本: {len(texts)}")
print(f"类别: {num_classes}")
print(f"分布: min={min(labels.count(l) for l in labels_list)}, "
      f"max={max(labels.count(l) for l in labels_list)}")

# ── 数据增强：小类字符级噪声 ────────────────────────────────────
import re
random.seed(1)   # 多轮对比最优种子

def _augment_text(text: str) -> str:
    # 随机删除 5-10% 中文字符，模拟输入变异
    chars = list(text)
    n = len(chars)
    if n <= 5:
        return text
    drop_rate = random.uniform(0.05, 0.10)
    keep = [c for c in chars if random.random() > drop_rate or not re.match(r'[一-鿿]', c)]
    return ''.join(keep) if keep else text

label_counts = {l: labels.count(l) for l in labels_list}
augmented_texts: list[str] = []
augmented_labels: list[str] = []
for l in labels_list:
    cnt = label_counts[l]
    if cnt >= AUG_MIN_SAMPLES:
        continue
    class_texts = [it["text"] for it in items if str(it["label"]) == l]
    for _ in range(cnt * (AUG_FACTOR - 1)):
        src = random.choice(class_texts)
        augmented_texts.append(_augment_text(src))
        augmented_labels.append(l)
if augmented_texts:
    texts.extend(augmented_texts)
    labels.extend(augmented_labels)
    print(f"数据增强: +{len(augmented_texts)} 条 (小类增强×{AUG_FACTOR})")
    print(f"分布(增后): min={min(labels.count(l) for l in labels_list)}, "
          f"max={max(labels.count(l) for l in labels_list)}")

# ── Tokenizer ──────────────────────────────────────────────────
tok = ChuchuTok.load(str(PROJECT_ROOT / "app" / "brain" / "chuchu_tok.json"))
print(f"词表大小: {tok.vocab_size}")

# ── 数据集 ─────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, texts, labels, label2id, tok, max_len=MAX_LEN):
        self.encodings = [tok.encode(t, max_len) for t in texts]
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.encodings[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))

# 打乱 + 分割
combined = list(zip(texts, labels))
random.shuffle(combined)
texts, labels = zip(*combined)

split = int(len(texts) * SPLIT)
train_ds = TextDataset(texts[:split], labels[:split], label2id, tok)
val_ds = TextDataset(texts[split:], labels[split:], label2id, tok)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# ── 训练 ─────────────────────────────────────────────────────
model = ChuchuCNN(
    vocab_size=tok.vocab_size, num_classes=num_classes,
    embed_dim=64, num_filters=NUM_FILTERS, kernel_sizes=(3, 4, 5),
    dropout=0.5, max_len=MAX_LEN,
).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

# 类别权重：小类加权但限制上限防过拟
label_weights: list[float] = []
for l in labels_list:
    cnt = label_counts[l]
    median = sorted(label_counts.values())[len(label_counts) // 2]
    w = (median / max(cnt, 1)) ** 0.5   # sqrt(median/count) soft 平衡
    label_weights.append(w)
class_weights = torch.tensor(label_weights, dtype=torch.float).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights)

total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,}")
print(f"类别权重范围: {min(label_weights):.3f} ~ {max(label_weights):.3f}")
print(f"\n开始训练: {EPOCHS} epochs...")
best_acc = 0
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
    best_acc = max(best_acc, acc)
    print(f"  Epoch {epoch+1:>2}/{EPOCHS}  "
          f"train_loss={train_loss/len(train_loader):.4f}  "
          f"val_loss={val_loss/len(val_loader):.4f}  "
          f"val_acc={acc:.4f}")

print(f"\n最佳验证准确率: {best_acc:.4f}")

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
report = classification_report(
    all_gold, all_preds,
    target_names=labels_list,
    zero_division=0,
)
print(f"\n分类报告:")
print(report)

# ── 保存 ─────────────────────────────────────────────────────
SAVE_DIR.mkdir(parents=True, exist_ok=True)
torch.save({
    "model_state_dict": model.state_dict(),
    "vocab_size": tok.vocab_size,
    "num_classes": num_classes,
    "label2id": label2id,
    "id2label": id2label,
    "embed_dim": 64,
    "num_filters": NUM_FILTERS,
    "kernel_sizes": (3, 4, 5),
    "dropout": 0.5,
    "max_len": MAX_LEN,
    "name": "topic",
}, SAVE_PATH)
print(f"\n模型保存: {SAVE_PATH}")
size_kb = SAVE_PATH.stat().st_size / 1024
print(f"大小: {size_kb:.1f}KB")
