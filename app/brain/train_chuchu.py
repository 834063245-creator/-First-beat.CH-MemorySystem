"""ChuchuCNN 训练脚本 — 自动造数据 + 训练意图/情绪两个模型。

用法:
    python app/brain/train_chuchu.py              # 训两个模型
    python app/brain/train_chuchu.py --intent      # 只训意图
    python app/brain/train_chuchu.py --emotion     # 只训情绪
"""

import argparse
import json
import os
import random
import sys
from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 确保能从项目根 import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.brain.keywords import INTENT_KEYWORDS as _SHARED_INTENT, EMOTION_KEYWORDS as _SHARED_EMOTION
from app.brain.chuchu_tok import ChuchuTok
from app.brain.chuchu_model import ChuchuCNN


# ── 设备 ──────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {DEVICE}")
HERE = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════
# 1. 合成训练数据生成
# ═══════════════════════════════════════════════════════

# 意图关键词（和 models.py / circuit.py 保持一致）
_INTENT_KEYWORDS = {**_SHARED_INTENT, **{
    "request": ["帮我", "请你", "需要你", "帮我查", "帮我找", "帮我写",
                "帮我改", "帮我看看", "能不能帮我",
                "帮我写一个", "帮我做个", "帮我查一下",
                "帮我看看这个", "帮我改一下"],
    "request_verbs": ["写", "查", "找", "改", "看看", "查一下", "做个", "处理"],
}}
_REQUEST_VERBS = _INTENT_KEYWORDS["request_verbs"]
_EMOTION_KEYWORDS = _SHARED_EMOTION
# 随机前缀/后缀/连接词 — 增加多样性
_PREFIXES = ["", "我", "我今天", "最近", "刚刚", "刚才", "突然", "真的", "其实", "我有点"]
_SUFFIXES = ["", "啊", "呀", "呢", "哦", "了", "啊", "吧", "呢", "了", "哈", "吗"]
_ENDINGS = ["", "，怎么办", "，真的", "，好烦", "，无语", "，你说呢", "，你知道吗",
            "，咋整", "，受不了了", "，哎"]

_INTENT_TEMPLATES = {
    "recall": [
        "{kw}",
        "你还{kw吗}",
        "我{kww}",
        "我们之前{kww}",
        "上次我们{kww}",
        "你还{kw}那件事吗",
    ],
    "emotional_sharing": [
        "{kw}",
        "我{kww}",
        "我好{kw}",
        "最近{kw}",
        "今天{kw}",
        "我感觉{kw}",
        "我现在{kw}",
        "最近{kww}，真的好累",
        "我今天{kww}",
        "最近{kww}，感觉好难受",
        "{kw}，我快撑不住了",
        "{kw}死了",
        "有点{kw}",
        "我有点{kw}",
        "真的有点{kw}",
    ],
    "conflict": [
        "{kw}",
        "我觉得{kw}",
        "我说{kw}",
        "你{kw}",
        "我都说了{kw}",
    ],
    "ask_fact": [
        "{kw}",
        "我想问{kw}",
        "请问{kw}",
        "你知道{kw}吗",
        "{kw}是什么意思",
        "{kw}怎么做",
    ],
    "request": [
        "{kw}",
        "你帮我{verb}一下",
        "帮我{verb}一个",
        "帮我{verb}",
        "帮我{verb}一下这个",
        "请帮我{verb}",
        "能不能帮我{verb}一下",
        "帮我{verb}这个",
        "需要你帮我{verb}一下",
    ],
    "meta": [
        "{kw}",
        "我想知道{kww}",
        "请问{kww}",
    ],
    "casual": [
        "你好",
        "早上好",
        "晚安",
        "今天天气不错",
        "吃了吗",
        "在吗",
        "没事了",
        "好的",
        "嗯嗯",
        "哈哈哈",
        "再见",
        "拜拜",
        "谢谢",
        "好的好的",
        "行",
        "知道了",
        "没什么",
        "随便",
        "都可以",
        "好吧",
        "没事",
        "好的谢谢",
        "可以",
        "不行",
        "对的",
        "没错",
        "嗯",
        "好",
        "哦",
        "没事了",
        "博客写好了",
        "中午想吃什么",
        "今天星期几",
    ],
}

_EMOTION_TEMPLATES = {
    "intimate": [
        "{kw}",
        "我好{kww}",
        "我{kw}你",
        "我想{kww}",
        "我真的好{kw}你",
    ],
    "positive": [
        "{kw}",
        "今天好{kw}",
        "太{kw}了",
        "真的好{kw}",
        "我很{kw}",
        "非常{kw}",
    ],
    "negative": [
        "{kw}",
        "我好{kw}",
        "最近很{kw}",
        "今天好{kw}",
        "我真的好{kw}",
        "感觉好{kw}",
        "非常{kw}",
    ],
    "frustrated": [
        "{kw}",
        "我真的{kw}",
        "我{kw}了",
        "实在{kw}",
        "真是{kw}",
    ],
    "neutral": [
        "今天天气不错",
        "帮我查一下资料",
        "什么是Python",
        "你好",
        "晚安",
        "吃了吗",
        "在吗",
        "好的",
        "谢谢",
        "嗯嗯",
        "代码报错了",
        "这个bug修好了",
        "项目部署完了",
        "数据库连不上",
        "接口返回500",
        "服务器重启了",
        "版本更新了",
        "这个功能实现了",
        "代码合并了",
    ],
}


def _fill(template: str, kw: str) -> str:
    """填充模板，随机加前后缀."""
    text = template.replace("{kw}", kw).replace("{kww}", kw)
    if "{kw吗}" in template:
        text = text.replace("{kw吗}", kw + "吗")
    # 随机加前缀
    prefix = random.choice(_PREFIXES)
    suffix = random.choice(_SUFFIXES)
    ending = random.choice(_ENDINGS)
    text = prefix + text + suffix + ending
    return text.strip()


def generate_intent_data(samples_per_class: int = 500) -> list[tuple[str, str]]:
    """生成意图训练数据。"""
    data = []
    for label, kws in _INTENT_KEYWORDS.items():
        templates = _INTENT_TEMPLATES[label]
        for _ in range(samples_per_class):
            kw = random.choice(kws)
            template = random.choice(templates)
            text = _fill(template, kw)
            data.append((text, label))

    # Clean request sentences — verb + object patterns
    _VERBS = ["写", "查", "找", "改", "看看", "做", "处理"]
    _OBJECTS = ["代码", "文件", "配置", "bug", "接口", "脚本", "数据",
                 "文档", "功能", "页面", "数据库", "项目", "日志"]
    for _ in range(samples_per_class * 2):
        verb = random.choice(_VERBS)
        obj = random.choice(_OBJECTS)
        patterns = [
            f"帮我{verb}{obj}",
            f"帮我{verb}一下{obj}",
            f"帮我{verb}一个{obj}",
            f"帮我{verb}一下这个{obj}",
            f"请帮我{verb}{obj}",
            f"能不能帮我{verb}一下{obj}",
            f"帮我看看{obj}",
            f"帮我{verb}这个{obj}",
            f"需要你帮我{verb}{obj}",
        ]
        text = random.choice(patterns)
        suffix = random.choice(_SUFFIXES)
        data.append((text + suffix, "request"))

    # Casual 类
    for _ in range(samples_per_class):
        text = random.choice(_INTENT_TEMPLATES["casual"])
        data.append((text, "casual"))

    # 稀释 conflict 中的 "对" 字 — 给 casual 加含 "对" 的样本
    for _ in range(samples_per_class // 2):
        text = random.choice(["你说得对", "说得对", "对的", "没错", "对呀", "对", "就是这样"])
        data.append((text, "casual"))

    # 稀释 meta 中的 "你" 字 — 给 emotional_sharing 加含 "你" 的样本
    for _ in range(samples_per_class // 3):
        text = random.choice(["我想你", "好想你", "我在想你",
                               "你真好", "有你真好", "想你了"])
        data.append((text, "emotional_sharing"))

    # 加一些混搭
    es_kws = _INTENT_KEYWORDS["emotional_sharing"]
    for _ in range(samples_per_class // 3):
        kw = random.choice(es_kws)
        text = f"{kw}{random.choice(_SUFFIXES)}"
        data.append((text, "emotional_sharing"))

    print(f"意图数据生成: {len(data)} 条")
    for label, count in Counter(l for _, l in data).most_common():
        print(f"  {label}: {count}")
    return data


def generate_emotion_data(samples_per_class: int = 500) -> list[tuple[str, str]]:
    """生成情绪训练数据。"""
    data = []
    for label, kws in _EMOTION_KEYWORDS.items():
        templates = _EMOTION_TEMPLATES[label]
        for _ in range(samples_per_class):
            kw = random.choice(kws)
            template = random.choice(templates)
            text = _fill(template, kw)
            data.append((text, label))

    # Neutral 类
    for _ in range(samples_per_class):
        text = random.choice(_EMOTION_TEMPLATES["neutral"])
        data.append((text, "neutral"))

    # 稀释 negative 中的 "好" 字偏差 — 给 positive 加含 "心情" 的样本
    for _ in range(samples_per_class // 3):
        text = random.choice(["今天心情真好", "心情不错", "心情愉快",
                               "心情好", "心情特别好", "心情美美的"])
        data.append((text, "positive"))

    # 让 intimate 类更丰富 — 加入更多含 "你" 的亲密表达
    for _ in range(samples_per_class // 4):
        text = random.choice(["我想你", "好想你", "想你了",
                               "你真好", "有你真好"])
        data.append((text, "intimate"))

    print(f"情绪数据生成: {len(data)} 条")
    for label, count in Counter(l for _, l in data).most_common():
        print(f"  {label}: {count}")
    return data


# ═══════════════════════════════════════════════════════
# 2. 训练
# ═══════════════════════════════════════════════════════

class TextDataset(Dataset):
    def __init__(self, texts, labels, label2id, tok, max_len=64):
        self.encodings = [tok.encode(t, max_len) for t in texts]
        self.labels = [label2id[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.encodings[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


def train_model(name: str, data: list[tuple[str, str]], tok: ChuchuTok,
                save_path: str, epochs: int = 8, batch_size: int = 32,
                lr: float = 1e-3):
    """训练一个 ChuchuCNN 模型。"""
    random.shuffle(data)
    texts, labels = zip(*data)
    labels_list = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for l, i in label2id.items()}

    print(f"\n{'='*50}")
    print(f"训练: {name}")
    print(f"  标签: {labels_list}")
    print(f"  样本: {len(texts)}")
    print(f"  词表大小: {tok.vocab_size}")
    print(f"{'='*50}")

    # 分 85/15 训练验证
    split = int(len(texts) * 0.85)
    train_ds = TextDataset(texts[:split], labels[:split], label2id, tok)
    val_ds = TextDataset(texts[split:], labels[split:], label2id, tok)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    model = ChuchuCNN(vocab_size=tok.vocab_size,
                       num_classes=len(labels_list)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # 验证
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
        print(f"  Epoch {epoch+1}/{epochs}  "
              f"train_loss={train_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"val_acc={acc:.4f}")

    # 最终测试
    model.eval()
    all_preds = []
    all_gold = []
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
    print(f"\n分类报告 ({name}):")
    print(report)

    # 保存
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tok.vocab_size,
        "num_classes": len(labels_list),
        "label2id": label2id,
        "id2label": id2label,
        "embed_dim": 64,
        "num_filters": 128,
        "kernel_sizes": (3, 4, 5),
        "dropout": 0.5,
        "name": name,
    }, save_path)
    print(f"模型保存: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", action="store_true", help="只训意图")
    parser.add_argument("--emotion", action="store_true", help="只训情绪")
    args = parser.parse_args()

    do_intent = args.intent or not args.emotion
    do_emotion = args.emotion or not args.intent

    random.seed(42)
    torch.manual_seed(42)

    # 初始化 Tokenizer
    tok = ChuchuTok()

    # 训练意图模型
    if do_intent:
        data = generate_intent_data(samples_per_class=600)
        train_model(
            name="intent",
            data=data,
            tok=tok,
            save_path=os.path.join(HERE, "model_intent", "chuchu_cnn.pt"),
        )

    # 训练情绪模型
    if do_emotion:
        data = generate_emotion_data(samples_per_class=600)
        train_model(
            name="emotion",
            data=data,
            tok=tok,
            save_path=os.path.join(HERE, "model_emotion", "chuchu_cnn.pt"),
        )

    # 保存 tokenizer
    tok.save(os.path.join(HERE, "chuchu_tok.json"))
    print(f"\nTokenizer 已保存")


if __name__ == "__main__":
    main()
