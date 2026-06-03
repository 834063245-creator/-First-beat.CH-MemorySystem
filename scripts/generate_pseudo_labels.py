"""伪标签生成：bge-m3 embedding → KMeans 聚类 50 组 → 输出带标签训练集。

依赖: Ollama (bge-m3), scikit-learn
输出: data/training_labeled.jsonl — 每行 {"text": "...", "label": 0-49, "label_name": "..."}
      data/topic_labels.json — {"0": "候选名1", "1": "候选名2", ...} 待人工修正
"""
import json
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import requests
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ── 配置 ────────────────────────────────────────────────────────
INPUT = Path(__file__).parent.parent / "data" / "training_raw.jsonl"
OUTPUT = Path(__file__).parent.parent / "data" / "training_labeled.jsonl"
LABELS_OUT = Path(__file__).parent.parent / "data" / "topic_labels.json"
N_CLUSTERS = 50
OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "bge-m3"
BATCH = 20  # 批量 embed，加速

# ── 加载消息 ─────────────────────────────────────────────────────
print(f"Loading: {INPUT}")
lines = [l.strip() for l in INPUT.read_text(encoding="utf-8").splitlines() if l.strip()]
texts = [json.loads(l)["user_message"] for l in lines]
print(f"  {len(texts)} messages")

# ── Embedding (批量) ─────────────────────────────────────────────
print(f"Embedding with {MODEL}...")
embeddings = []
for i in range(0, len(texts), BATCH):
    batch = texts[i:i + BATCH]
    batch_emb = []
    for text in batch:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL, "prompt": text,
        }, timeout=30)
        batch_emb.append(resp.json()["embedding"])
    embeddings.extend(batch_emb)
    if (i + BATCH) % 200 == 0:
        print(f"  {i + len(batch)}/{len(texts)}")

embeddings = np.array(embeddings)
print(f"  Done: {embeddings.shape}")

# ── 聚类 ─────────────────────────────────────────────────────────
print(f"Clustering into {N_CLUSTERS} groups...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)
print(f"  Done")

# ── 为每个簇生成候选名称 ────────────────────────────────────────
print("Generating cluster names...")
# 取每个簇的 top-3 样例 + 最近邻词
import jieba.analyse
label_names = {}
for cluster_id in range(N_CLUSTERS):
    mask = labels == cluster_id
    cluster_texts = [texts[i] for i in range(len(texts)) if mask[i]]
    if not cluster_texts:
        label_names[str(cluster_id)] = f"cluster_{cluster_id}"
        continue

    # 取最接近簇中心的前 3 条做代表
    center = kmeans.cluster_centers_[cluster_id]
    cluster_embs = embeddings[mask]
    # 将 cluster_texts 转为 list 用本地索引
    local_idx = np.argsort(cosine_similarity([center], cluster_embs)[0])[-3:][::-1]
    representative = [cluster_texts[i][:40] for i in local_idx]

    # 用 jieba TF-IDF 提取关键词
    all_text = " ".join(cluster_texts)
    keywords = jieba.analyse.extract_tags(all_text, topK=5)
    name = "、".join(keywords[:3]) if keywords else f"cluster_{cluster_id}"
    label_names[str(cluster_id)] = {
        "name": name,
        "size": len(cluster_texts),
        "representative": representative,
        "keywords": keywords,
    }

# ── 输出 ─────────────────────────────────────────────────────────
# 标签映射文件
LABELS_OUT.write_text(json.dumps(label_names, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"Labels: {LABELS_OUT}")
print(f"  Review this file to assign human-readable names to each cluster.")

# 训练数据
with open(OUTPUT, "w", encoding="utf-8") as f:
    for i, text in enumerate(texts):
        lid = int(labels[i])
        f.write(json.dumps({
            "text": text,
            "label": lid,
            "label_name": label_names[str(lid)]["name"] if isinstance(label_names[str(lid)], dict) else label_names[str(lid)],
        }, ensure_ascii=False) + "\n")

print(f"Training data: {OUTPUT}")
print(f"  {len(texts)} messages, {N_CLUSTERS} classes")

# 分布统计
dist = Counter(int(l) for l in labels)
print(f"  Distribution: min={min(dist.values())}, max={max(dist.values())}, avg={len(texts)//N_CLUSTERS}")
