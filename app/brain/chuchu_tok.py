"""Chuchu Tokenizer — 基于汉字的词表，零依赖。

把文本转成汉字 ID 序列，只保留中文字符 + 常见标点。
纯 Python，不需要 transformers/huggingface。
"""

import re
import os
import json

# 常用汉字 + 常见标点 + 英文字母（覆盖代码相关查询）
_COMMON_CHARS = set(
    "的一是不了人我在有他这为之大以来个中上学到说时会就用"
    "要地也出得你下子看过可们没去自多小如作还对又好生发"
    "能而于其心所面前法后开天长那分从还她成只当见知事把"
    "很想看让进着没样都给向道但种些然与因力点里它几间"
    "回又方身去被已起明期头实正经关如至些定各月公无"
    "水第一第次新条些口或手本比活两高女很问什给现什"
    "别那这两都感开她它吧什况做很谁今将已起两带安爱"

    "帮请问记得上次想情难过开心烦恼累困疲惫焦虑担心感动"
    "温暖梦失眠心疼好好好快快乐幸感谢棒厉害"

    "什怎为如能否请你查找我写改看看什么怎么为什么如何"
    "能不能请问啥是不是"

    "你对不不是错说别了他乱听懂样"

    "谁做你会功能名称吗吗"

    "想心温柔抱陪亲在乎爱"

    "高太爽快喜谢棒厉"

    "烦压力生讨厌失望痛苦崩孤独郁闷燥"

    "受不了无语气够算吧"

    "今昨明前天早晚午时候分秒年月日星期"
    "呢吗嘛哈呜嘿唉嗯哦哇呀咯呵哼靠靠"

    "的了吧吗呢啊哦啦哈哇呀嗯呗哟呵"
    "，。！？、；：""''（）【】《》…—·"
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "bugPRGit代码调试修改部署重构调度架构引擎项目进度需求排期"
    "接口数据库缓存线程内存模型训练推理嵌入量化加速版本"
)


# 必须保留的关键字符 — 从关键词列表收集
_KEYWORD_CHARS = set()
_INTENT_KEYWORDS = {
    "conflict": ["不对", "不是", "你错了", "别说了", "你搞错了", "乱说",
                 "你没听懂", "不是这样"],
    "emotional_sharing": ["想", "觉得", "感觉", "心情", "难过", "开心", "烦",
                          "累", "困", "疲惫", "焦虑", "担心", "感动", "温暖",
                          "梦到", "失眠", "心疼", "好烦", "好累", "好开心", "好难过"],
    "recall": ["记得", "之前", "上次", "以前", "曾经", "想起", "是不是说过",
               "那时候", "那会儿", "还记得", "记不记得"],
    "request": ["帮我", "请你", "需要你", "帮我查", "帮我找", "帮我写",
                "帮我改", "帮我看看", "能不能帮我"],
    "ask_fact": ["什么", "怎么", "为什么", "如何", "能不能", "请问", "啥", "是不是"],
    "meta": ["你是谁", "你能做什么", "你会什么", "你叫什么", "你有什么功能", "你是吗"],
}
for words in _INTENT_KEYWORDS.values():
    for w in words:
        _KEYWORD_CHARS.update(w)

_EMOTION_KEYWORDS = {
    "intimate": ["想你", "爱", "心疼", "抱", "陪", "温暖", "梦到", "亲", "在乎", "爱你", "抱抱"],
    "positive": ["开心", "高兴", "好", "棒", "喜欢", "感动", "幸福", "感谢", "太棒", "太好了", "不错", "厉害"],
    "negative": ["难过", "烦", "累", "焦虑", "担心", "生气", "讨厌", "失望", "痛苦", "崩溃", "孤独", "压力", "郁闷", "烦躁"],
    "frustrated": ["烦死了", "受不了", "无语", "气死", "崩溃", "不想说了", "够了", "算了吧"],
}
for words in _EMOTION_KEYWORDS.values():
    for w in words:
        _KEYWORD_CHARS.update(w)

_ALL_CHARS = _COMMON_CHARS | _KEYWORD_CHARS
_ALL_CHARS = sorted(_ALL_CHARS - {"", " "})

# 特殊 token
PAD = 0
UNK = 1


class ChuchuTok:
    """汉字级 Tokenizer。

    Usage:
        tok = ChuchuTok()
        ids = tok.encode("最近压力好大")
        # → [2, 15, 87, 128, 15, 33, ...]
        text = tok.decode(ids)
    """

    def __init__(self, chars: list[str] | None = None):
        if chars is None:
            chars = _ALL_CHARS
        self._char2id = {c: i + 2 for i, c in enumerate(chars)}  # 0=PAD, 1=UNK
        self._id2char = {i + 2: c for i, c in enumerate(chars)}
        self._id2char[PAD] = "<PAD>"
        self._id2char[UNK] = "<UNK>"

    @property
    def vocab_size(self) -> int:
        return len(self._char2id) + 2

    def encode(self, text: str, max_len: int = 64) -> list[int]:
        """文本转 ID 序列，自动截断/补零到 max_len。"""
        ids = []
        for ch in text:
            ids.append(self._char2id.get(ch, UNK))
        # 截断
        if len(ids) > max_len:
            ids = ids[:max_len]
        # 补零
        ids = ids + [PAD] * (max_len - len(ids))
        return ids

    def decode(self, ids: list[int]) -> str:
        """ID 序列转回文本（跳过 PAD）。"""
        chars = []
        for i in ids:
            if i == PAD:
                continue
            chars.append(self._id2char.get(i, "<UNK>"))
        return "".join(chars)

    def save(self, path: str):
        """保存词表到 JSON。"""
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "char2id": self._char2id,
                "id2char": {str(k): v for k, v in self._id2char.items()},
            }, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "ChuchuTok":
        """从 JSON 加载词表。"""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok._char2id = data["char2id"]
        tok._id2char = {int(k): v for k, v in data["id2char"].items()}
        return tok
