"""事实域判断器 — 第 6 个 ChuchuCNN，判断两条记忆是否同一事实域。

替代 consolidation.py 中的话题树分支交集（第一层过滤）。

使用方式:
    from app.brain.fact_classifier import get_fact_classifier
    fc = get_fact_classifier()
    is_same = fc.is_same_domain("喜欢吃辣", "不爱吃辣了")  # → True
"""
import logging
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

from app.brain.chuchu_tok import ChuchuTok
from app.brain.chuchu_model import ChuchuCNN

logger = logging.getLogger(__name__)
SEP = " [SEP] "


class FactDomainClassifier:
    """事实域二分类 CNN 封装。"""

    def __init__(self, model_path: str = None):
        self._lock = threading.Lock()

        if model_path is None:
            model_path = str(
                Path(__file__).parent / "model_fact" / "chuchu_cnn.pt"
            )

        if not Path(model_path).exists():
            self._model = None
            logger.warning("事实域模型未找到: %s", model_path)
            return

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        max_len = ckpt.get("max_len", 128)
        self._model = ChuchuCNN(
            vocab_size=ckpt["vocab_size"],
            num_classes=ckpt["num_classes"],
            max_len=max_len,
        )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        self._max_len = max_len
        self._tok = ChuchuTok.load(
            str(Path(__file__).parent / "chuchu_tok.json")
        )
        logger.info("事实域分���器加载完成")

    def is_same_domain(self, text_a: str, text_b: str, min_confidence: float = 0.4) -> bool:
        """判断两条文本是否属于同一事实域。"""
        if not self._model:
            return True  # 模型不可用时保守处理（允许通过）

        combined = text_a + SEP + text_b
        with self._lock:
            ids = self._tok.encode(combined, max_len=self._max_len)
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self._model(x)
                probs = F.softmax(logits, dim=-1)[0]

        # probs[0] = different, probs[1] = same
        same_prob = float(probs[1].item())
        return same_prob >= min_confidence

    @property
    def available(self) -> bool:
        return self._model is not None


_fact_classifier: FactDomainClassifier | None = None
_fc_lock = threading.Lock()


def get_fact_classifier() -> FactDomainClassifier:
    global _fact_classifier
    if _fact_classifier is None:
        with _fc_lock:
            if _fact_classifier is None:
                _fact_classifier = FactDomainClassifier()
    return _fact_classifier
