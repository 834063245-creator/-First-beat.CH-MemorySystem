"""PortraitManager — PORTRAIT.md 生命周期管理。

职责:
  1. 加载/解析 PORTRAIT.md（YAML frontmatter + Markdown 正文）
  2. 提取/索引条目（entry ID → text/tags/confidence 映射）
  3. 写入 PORTRAIT.md（原子写 via temp + rename）
  4. 条目 CRUD 操作（按 entry ID）
"""

import json
import logging
import os
import re
import threading
from datetime import datetime
from typing import Any, Optional

from app.portrait.state import PortraitEntry, EntryStatus, EntryStateMachine

logger = logging.getLogger(__name__)

# ── 解析正则 ────────────────────────────────────────────

# 维度标记: <!-- dim:usr1 核心特征 -->
_RE_DIM = re.compile(r"<!--\s*dim:(\w+)\s*(.*?)-->")

# 条目标记: <!-- entry:usr1-001 -->
_RE_ENTRY = re.compile(r"<!--\s*entry:(\w+)-(\d{3})\s*-->")

# YAML frontmatter 边界
_RE_FRONTMATTER = re.compile(r"^---\s*$(.+?)^---\s*$", re.MULTILINE | re.DOTALL)

# 行尾元数据: `text with backticks`
_RE_BACKTICK_META = re.compile(r"`[^`]*`")

# 画像维度代码全集
ALL_DIMS = [
    "usr1", "usr2", "usr3", "usr4", "usr5", "usr6",
    "ai1", "ai2", "ai3", "ai4", "ai5", "ai6",
]

DIM_LABELS: dict[str, str] = {
    "usr1": "核心特征",
    "usr2": "当前状态",
    "usr3": "行为节律",
    "usr4": "关系快照",
    "usr5": "兴趣图谱",
    "usr6": "情绪图谱",
    "ai1": "核心表达特征",
    "ai2": "当前状态",
    "ai3": "行为节律",
    "ai4": "关系快照",
    "ai5": "兴趣/知识图谱",
    "ai6": "情绪/表达图谱",
}

# 每个维度的条目上限
DIM_CAPS: dict[str, int] = {
    "usr1": 10, "usr2": 8, "usr3": 15, "usr4": 6, "usr5": 20, "usr6": 15,
    "ai1": 10, "ai2": 8, "ai3": 15, "ai4": 6, "ai5": 20, "ai6": 15,
}

# ── 默认画像模板 ────────────────────────────────────────

def _default_frontmatter(version: int = 1) -> dict:
    return {
        "version": version,
        "last_updated": datetime.now().isoformat(),
        "dimensions": {
            "user": ["core_traits", "current_state", "behavioral_rhythm",
                     "relationship", "interests", "emotion_landscape"],
            "ai": ["core_traits", "current_state", "behavioral_rhythm",
                   "relationship", "interests", "emotion_expressiveness"],
        },
    }


def _default_portrait_md() -> str:
    """生成空画像的默认模板。"""
    frontmatter = json.dumps(_default_frontmatter(1), ensure_ascii=False, indent=2)
    lines = [
        "---",
        frontmatter,
        "---",
        "",
        "# 认知画像",
        "",
        "## 用户画像",
        "",
        "<!-- dim:usr1 核心特征 -->",
        "### 1. 核心特征",
        "<!-- 深巩固(24h)更新 — 画像积累不足时可能为空 -->",
        "",
        "<!-- dim:usr2 当前状态 -->",
        "### 2. 当前状态",
        "<!-- 实时更新，待浅巩固确认 -->",
        "",
        "<!-- dim:usr3 行为节律 -->",
        "### 3. 行为节律",
        "<!-- 浅巩固(4h)更新 -->",
        "",
        "<!-- dim:usr4 关系快照 -->",
        "### 4. 关系快照",
        "<!-- 实时更新 -->",
        "",
        "<!-- dim:usr5 兴趣图谱 -->",
        "### 5. 兴趣图谱",
        "<!-- 浅巩固(4h)更新 -->",
        "",
        "<!-- dim:usr6 情绪图谱 -->",
        "### 6. 情绪图谱",
        "<!-- 浅巩固(4h)更新 -->",
        "",
        "## AI 画像",
        "",
        "<!-- dim:ai1 核心表达特征 -->",
        "### 1. 核心表达特征",
        "<!-- 深巩固(24h)更新 -->",
        "",
        "<!-- dim:ai2 当前状态 -->",
        "### 2. 当前状态",
        "<!-- 实时更新 -->",
        "",
        "<!-- dim:ai3 行为节律 -->",
        "### 3. 行为节律",
        "<!-- 浅巩固(4h)更新 -->",
        "",
        "<!-- dim:ai4 关系快照 -->",
        "### 4. 关系快照",
        "<!-- 实时更新 -->",
        "",
        "<!-- dim:ai5 兴趣/知识图谱 -->",
        "### 5. 兴趣/知识图谱",
        "<!-- 浅巩固(4h)更新 -->",
        "",
        "<!-- dim:ai6 情绪/表达图谱 -->",
        "### 6. 情绪/表达图谱",
        "<!-- 浅巩固(4h)更新 -->",
        "",
    ]
    return "\n".join(lines)


# ── PortraitManager ─────────────────────────────────────

class PortraitManager:
    """PORTRAIT.md 文件的管理者。

    线程安全（threading.Lock 保护读写）。
    """

    def __init__(self, file_path: str):
        self._path = file_path
        self._lock = threading.Lock()
        self._entries: dict[str, PortraitEntry] = {}  # entry_id → PortraitEntry
        self._version: int = 0
        self._last_updated: str = ""

        # 初始化：文件不存在则创建空画像
        if os.path.exists(file_path):
            self._load()
        else:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self._save_raw(_default_portrait_md())
            self._load()

    # ── 加载/保存 ───────────────────────────────────────

    def _load(self):
        """从磁盘加载 PORTRAIT.md 并解析条目。"""
        with self._lock:
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    raw = f.read()
                self._parse(raw)
            except (OSError, json.JSONDecodeError) as exc:
                logger.warning("PORTRAIT.md 加载失败: %s", exc)
                # 损坏时用空白重建
                self._entries = {}
                self._version = 0

    def _parse(self, raw: str):
        """解析 Markdown 内容，提取条目索引。"""
        # 解析 frontmatter
        fm_match = _RE_FRONTMATTER.search(raw)
        if fm_match:
            try:
                fm = json.loads(fm_match.group(1).strip())
                self._version = fm.get("version", 0)
                self._last_updated = fm.get("last_updated", "")
            except json.JSONDecodeError:
                self._version = 0

        # 解析条目
        self._entries = {}
        # 先找维度标记确定当前维度
        current_dim = None
        lines = raw.split("\n")

        for i, line in enumerate(lines):
            dim_match = _RE_DIM.search(line)
            if dim_match:
                current_dim = dim_match.group(1)
                continue

            entry_match = _RE_ENTRY.search(line)
            if entry_match and current_dim:
                dim = entry_match.group(1)
                seq = entry_match.group(2)
                entry_id = f"{dim}-{seq}"

                # 提取条目文本（可能在注释同行，也可能在下一行）
                text = line[entry_match.end():].strip()
                source_line = line  # 用于提取 backtick 元数据
                if not text and i + 1 < len(lines):
                    # 条目内容在下一行（如 "- **情绪**: positive  `...`"）
                    source_line = lines[i + 1]
                    text = source_line.strip()
                if text.startswith("- "):
                    text = text[2:]  # 去掉列表标记

                # 解析行尾 backtick 元数据（在解析前提取，解析后从 text 剥离）
                tags, confidence, evidence_count = self._parse_backtick_meta(source_line)
                # 从 text 中剥离 backtick 区域
                text = _RE_BACKTICK_META.sub("", text).strip()

                # 尝试从文本推断状态
                status = EntryStatus.ACTIVE
                if "待验证" in text:
                    status = EntryStatus.PENDING

                self._entries[entry_id] = PortraitEntry(
                    id=entry_id,
                    dim=dim,
                    text=text,
                    tags=tags,
                    confidence=confidence,
                    status=status,
                    evidence_count=evidence_count,
                    last_observed=self._last_updated or None,
                )

    @staticmethod
    def _parse_backtick_meta(line: str) -> tuple[list[str], float, int]:
        """从行尾 backtick 区域解析 tags/confidence/evidence_count。

        Returns:
            (tags, confidence, evidence_count)
        """
        tags: list[str] = []
        confidence: float = 1.0
        evidence_count: int = 0

        # 提取最后一个 backtick 区域
        bt_matches = list(_RE_BACKTICK_META.finditer(line))
        if bt_matches:
            meta_str = bt_matches[-1].group(0).strip("`")

            # confidence: "高"/"中"/"低" or 数值
            if "高" in meta_str:
                confidence = 0.90
            elif "中" in meta_str:
                confidence = 0.60
            elif "低" in meta_str:
                confidence = 0.35

            # evidence_count: "N条证据"
            ec_match = re.search(r"(\d+)条证据", meta_str)
            if ec_match:
                evidence_count = int(ec_match.group(1))

            # tags: "tags:xxx yyy"
            tag_match = re.search(r"tags:(.+)", meta_str)
            if tag_match:
                tags = tag_match.group(1).strip().split()

        return tags, confidence, evidence_count

    def reload(self):
        """重新从磁盘加载（供外部同步）。"""
        self._load()

    def save(self):
        """保存当前画像到磁盘（原子写）。"""
        raw = self._render_markdown()
        self._save_raw(raw)

    def _save_raw(self, content: str):
        """原子写入磁盘。"""
        tmp = self._path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, self._path)  # atomic on Windows & POSIX

    def _render_markdown(self) -> str:
        """将当前条目渲染为完整 Markdown 文档。"""
        # 更新 frontmatter
        fm = _default_frontmatter(self._version + 1)
        fm["last_updated"] = datetime.now().isoformat()

        lines = [
            "---",
            json.dumps(fm, ensure_ascii=False, indent=2),
            "---",
            "",
            "# 认知画像",
            "",
        ]

        # 渲染用户画像
        user_dims = [("usr1", "核心特征"), ("usr2", "当前状态"), ("usr3", "行为节律"),
                     ("usr4", "关系快照"), ("usr5", "兴趣图谱"), ("usr6", "情绪图谱")]
        lines.append("## 用户画像")
        lines.append("")
        for dim, label in user_dims:
            lines.append(f"<!-- dim:{dim} {label} -->")
            lines.append(f"### {user_dims.index((dim, label)) + 1}. {label}")
            lines.append("")
            dim_entries = [e for e in self._entries.values()
                          if e.dim == dim and e.status != EntryStatus.DECAYED]
            for entry in sorted(dim_entries, key=lambda e: e.id):
                lines.append(self._render_entry(entry))
            if not dim_entries:
                lines.append("<!-- 暂无数据 -->")
                lines.append("")
            lines.append("")

        # 渲染 AI 画像
        ai_dims = [("ai1", "核心表达特征"), ("ai2", "当前状态"), ("ai3", "行为节律"),
                   ("ai4", "关系快照"), ("ai5", "兴趣/知识图谱"), ("ai6", "情绪/表达图谱")]
        lines.append("## AI 画像")
        lines.append("")
        for dim, label in ai_dims:
            lines.append(f"<!-- dim:{dim} {label} -->")
            lines.append(f"### {ai_dims.index((dim, label)) + 1}. {label}")
            lines.append("")
            dim_entries = [e for e in self._entries.values()
                          if e.dim == dim and e.status != EntryStatus.DECAYED]
            for entry in sorted(dim_entries, key=lambda e: e.id):
                lines.append(self._render_entry(entry))
            if not dim_entries:
                lines.append("<!-- 暂无数据 -->")
                lines.append("")
            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_entry(entry: PortraitEntry) -> str:
        """渲染单个条目为 Markdown 行。"""
        status_tags = []
        if entry.status == EntryStatus.PENDING:
            status_tags.append("待验证")
        elif entry.status == EntryStatus.COOLING:
            status_tags.append("cooling")

        meta_parts = []
        if entry.confidence >= 0.80:
            meta_parts.append("高")
        elif entry.confidence >= 0.50:
            meta_parts.append("中")
        elif entry.confidence >= 0.30:
            meta_parts.append("低")
        if entry.evidence_count > 0:
            meta_parts.append(f"{entry.evidence_count}条证据")
        if entry.tags:
            meta_parts.append(f"tags:{' '.join(entry.tags)}")

        meta_str = " · ".join(meta_parts)
        status_str = f"（{' · '.join(status_tags)}）" if status_tags else ""

        line = f"<!-- entry:{entry.id} -->\n- {entry.text}{status_str}  `{meta_str}`"
        return line

    # ── 条目 CRUD ────────────────────────────────────────

    def get_entry(self, entry_id: str) -> Optional[PortraitEntry]:
        """按 ID 获取条目。"""
        return self._entries.get(entry_id)

    def set_entry(self, entry_id: str, text: str, **kwargs):
        """设置/更新条目文本。"""
        with self._lock:
            if entry_id in self._entries:
                entry = self._entries[entry_id]
                entry.text = text
                entry.last_observed = datetime.now().isoformat()
                for k, v in kwargs.items():
                    if hasattr(entry, k):
                        setattr(entry, k, v)
            else:
                # 新条目
                dim = entry_id.rsplit("-", 1)[0]
                allowed_fields = {"tags", "confidence", "evidence_count", "first_observed",
                                 "last_observed", "status"}
                extra = {k: v for k, v in kwargs.items() if k in allowed_fields}
                entry = PortraitEntry(
                    id=entry_id,
                    dim=dim,
                    text=text,
                    status=EntryStatus.PENDING,
                    first_observed=datetime.now().isoformat(),
                    last_observed=datetime.now().isoformat(),
                )
                for k, v in extra.items():
                    setattr(entry, k, v)
                self._entries[entry_id] = entry

    def delete_entry(self, entry_id: str):
        """删除条目。"""
        with self._lock:
            self._entries.pop(entry_id, None)

    def get_dim_entries(self, dim: str) -> list[PortraitEntry]:
        """获取某维度的所有活跃条目。"""
        return [e for e in self._entries.values()
                if e.dim == dim and e.status != EntryStatus.DECAYED]

    def get_all_active(self) -> dict[str, list[PortraitEntry]]:
        """获取所有维度的活跃条目（按 dim 分组）。"""
        result: dict[str, list[PortraitEntry]] = {}
        for dim in ALL_DIMS:
            entries = [e for e in self._entries.values()
                      if e.dim == dim and e.should_inject]
            if entries:
                result[dim] = sorted(entries, key=lambda e: e.id)
        return result

    def next_seq(self, dim: str) -> int:
        """获取某维度的下一个可用序号。"""
        existing = [int(e.id.rsplit("-", 1)[1])
                   for e in self._entries.values() if e.dim == dim]
        return max(existing, default=0) + 1

    def apply_state_machine(self):
        """对所有条目应用状态机转换。"""
        with self._lock:
            for entry_id, entry in list(self._entries.items()):
                EntryStateMachine.transition(entry)
                if entry.status == EntryStatus.DECAYED:
                    del self._entries[entry_id]

    # ── 查询接口 ─────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def last_updated(self) -> str:
        return self._last_updated

    @property
    def is_empty(self) -> bool:
        """画像是否为空（没有任何活跃条目）。"""
        return len(self._entries) == 0

    def get_dimension_summary(self) -> dict:
        """获取各维度的条目计数摘要。"""
        summary = {}
        for dim in ALL_DIMS:
            active = sum(1 for e in self._entries.values()
                        if e.dim == dim and e.should_inject)
            total = sum(1 for e in self._entries.values() if e.dim == dim)
            summary[dim] = {"active": active, "total": total, "label": DIM_LABELS.get(dim, dim)}
        return summary

    def extract_tags_for_dim(self, dim: str) -> list[str]:
        """提取某个维度所有条目的标签集合。"""
        tags: list[str] = []
        for e in self._entries.values():
            if e.dim == dim and e.status != EntryStatus.DECAYED:
                tags.extend(e.tags)
        return list(dict.fromkeys(tags))

    def extract_hot_topics(self) -> list[str]:
        """提取当前热点话题（dim usr5 中 status=ACTIVE 的标签）。"""
        tags = set()
        for e in self._entries.values():
            if e.dim == "usr5" and e.should_inject:
                tags.update(e.tags)
        return list(tags)

    def extract_negative_triggers(self) -> list[str]:
        """提取负向触发话题（dim usr6 中的负向标签）。"""
        tags = set()
        for e in self._entries.values():
            if e.dim == "usr6" and e.should_inject:
                negative_keywords = {"项目受阻", "不被理解", "压力", "焦虑", "重复性工作"}
                if any(kw in e.text for kw in negative_keywords):
                    tags.update(e.tags)
        return list(tags)

    def extract_focus_keywords(self) -> list[str]:
        """提取当前关注焦点关键词（dim usr2）。"""
        tags = set()
        for e in self._entries.values():
            if e.dim == "usr2" and e.should_inject and "关注焦点" in e.text:
                tags.update(e.tags)
        return list(tags)

    def compute_portrait_boost_map(self) -> dict[str, float]:
        """计算画像 → tag 的 boost 映射。

        供检索精排阶段使用。预计算避免每次 rerank 重复查表。

        Returns:
            {tag: boost_value} — boost 区间 [-0.2, +0.3]
        """
        boost_map: dict[str, float] = {}

        # hot topics (usr5): +0.2
        for e in self._entries.values():
            if e.dim == "usr5" and e.should_inject and e.days_since_last_observed <= 3:
                for tag in e.tags:
                    boost_map[tag] = max(boost_map.get(tag, 0.0), 0.2)

        # warm topics (usr5): +0.1
        for e in self._entries.values():
            if e.dim == "usr5" and e.should_inject and 3 < e.days_since_last_observed <= 7:
                for tag in e.tags:
                    if tag not in boost_map:
                        boost_map[tag] = 0.1

        # focus keywords (usr2): +0.1
        for tag in self.extract_focus_keywords():
            if tag not in boost_map:
                boost_map[tag] = 0.1

        # negative triggers (usr6): -0.2
        for tag in self.extract_negative_triggers():
            boost_map[tag] = min(boost_map.get(tag, 0.0), -0.2)

        # AI active domains (ai5): +0.1
        for e in self._entries.values():
            if e.dim == "ai5" and e.should_inject and e.days_since_last_observed <= 7:
                for tag in e.tags:
                    if tag not in boost_map:
                        boost_map[tag] = 0.1

        return boost_map
