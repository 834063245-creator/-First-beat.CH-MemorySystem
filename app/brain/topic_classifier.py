"""话题分类器 — 第 5 个 ChuchuCNN，替代 jieba 标签提取。

使用方式:
    from app.brain.topic_classifier import get_topic_classifier
    tc = get_topic_classifier()
    tags = tc.predict("最近压力好大，代码写不完了")  # → ["调试排查", "情绪表达"]
    tag_id = tc.classify("用户消息")                # → 话题ID (0-49)
"""
import json
import logging
import os
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from app.brain.chuchu_tok import ChuchuTok
from app.brain.chuchu_model import ChuchuCNN

logger = logging.getLogger(__name__)

# ── 话题标签名（从聚类伪标签映射） ──────────────────────────────────
# 候选名来自 bge-m3 聚类的 jieba 关键词，覆盖主要话题域
_TOPIC_NAMES: dict[int, str] = {}


def _init_topic_names():
    """从 topic_labels.json 加载话题名。"""
    global _TOPIC_NAMES
    if _TOPIC_NAMES:
        return
    labels_path = Path(__file__).parent.parent.parent / "data" / "topic_labels.json"
    if labels_path.exists():
        data = json.loads(labels_path.read_text(encoding="utf-8"))
        for k, v in data.items():
            name = v.get("name", f"topic_{k}") if isinstance(v, dict) else v
            _TOPIC_NAMES[int(k)] = name


class TopicClassifier:
    """话题分类 CNN 封装。"""

    def __init__(self, model_path: str = None):
        self._lock = threading.Lock()

        if model_path is None:
            model_path = str(
                Path(__file__).parent / "model_topic" / "chuchu_cnn.pt"
            )

        if not os.path.exists(model_path):
            self._model = None
            self._id2label = {}
            logger.warning("话题模型未找到: %s，降级为 jieba", model_path)
            return

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self._model = ChuchuCNN(
            vocab_size=ckpt["vocab_size"],
            num_classes=ckpt["num_classes"],
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        self._id2label = {
            int(k): v for k, v in ckpt["id2label"].items()
        }
        self._tok = ChuchuTok.load(
            str(Path(__file__).parent / "chuchu_tok.json")
        )
        _init_topic_names()
        logger.info("话题分类器加载完成: %d 类", ckpt["num_classes"])

    def classify(self, text: str) -> int:
        """返回话题 ID (0-49)，模型不可用返回 -1。"""
        if not self._model or not text:
            return -1
        with self._lock:
            ids = self._tok.encode(text, max_len=64)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1)
                confidence, idx = probs.max(dim=-1)
                # 低置信度降级
                if confidence.item() < 0.15:
                    return -1
                return int(idx.item())

    def predict(self, text: str, top_k: int = 3) -> list[str]:
        """返回 top-K 话题标签名（降级时返回空列表）。"""
        if not self._model or not text:
            return []

        with self._lock:
            ids = self._tok.encode(text, max_len=64)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1)[0]

        # 取 top-K
        top_idx = probs.argsort(descending=True)[:top_k]
        tags = []
        for idx in top_idx:
            lid = int(idx.item())
            confidence = float(probs[idx].item())
            if confidence < 0.1:
                continue
            name = _TOPIC_NAMES.get(lid, f"topic_{lid}")
            tags.append(name)
        return tags

    def get_topic_name(self, topic_id: int) -> str:
        return _TOPIC_NAMES.get(topic_id, f"topic_{topic_id}")

    @property
    def available(self) -> bool:
        return self._model is not None


# ── 全局单例 ──────────────────────────────────────────────────
_topic_classifier: TopicClassifier | None = None
_tc_lock = threading.Lock()


def get_topic_classifier() -> TopicClassifier:
    """获取话题分类器单例（惰性加载）。"""
    global _topic_classifier
    if _topic_classifier is None:
        with _tc_lock:
            if _topic_classifier is None:
                _topic_classifier = TopicClassifier()
    return _topic_classifier
