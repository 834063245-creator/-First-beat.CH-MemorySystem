"""话题树 — 从标签亲和图自动聚类，检索时沿树扩展相关标签。

用法
----
    tree = TopicTree(data_dir)
    tree.rebuild(affinity_matrix)
    tree.expand(["格式"])          → ["列表", "标题", ...]
    tree.get_branch("记忆")        → ["记忆", "本体", "初痕", ...]
"""

import json
import logging
import os
import threading
from collections import defaultdict

from app.tools.atomic import atomic_write

logger = logging.getLogger(__name__)


def _find_root(tag: str, parent_of: dict[str, tuple[str, int]]) -> str:
    """并查集查根（路径压缩，不含秩）。"""
    while tag in parent_of:
        tag = parent_of[tag][0]
    return tag


class TopicTree:
    """从标签亲和图自动长出一棵话题树。

    聚类策略（Kruskal 式）：
      1. 所有共现边按强度降序排列
      2. 逐条合并，并查集确保无环
      3. 总频更高的标签倾向作父节点
    建树后整枝以 DFS 递归收集，不遗漏孙代。
    """

    MIN_STRENGTH = 3        # 有效共现最低次数
    MAX_SIBLINGS = 5        # expand 最多返回的扩展标签数
    CACHE_FILE = "topic_tree.json"

    def __init__(self, data_dir: str):
        self._path = os.path.join(data_dir, self.CACHE_FILE)
        self._tree: dict = {"name": "root", "children": []}
        self._tag_to_branch: dict[str, list[str]] = {}  # tag → 所在枝的全部标签
        self._lock = threading.Lock()
        self._load()

    # ── 公开接口 ──────────────────────────────────────────────

    def rebuild(self, matrix: dict[str, dict[str, int]]):
        """从亲和矩阵重建话题树。"""
        with self._lock:
            self._rebuild_impl(matrix)
            self._save()

    def _rebuild_impl(self, matrix: dict[str, dict[str, int]]):
        # Step 1: 过滤噪音边（只取 tag_a < tag_b 避免重复）
        edges: list[tuple[str, str, int]] = []
        for tag_a, rels in matrix.items():
            for tag_b, cnt in rels.items():
                if tag_a < tag_b and cnt >= self.MIN_STRENGTH:
                    edges.append((tag_a, tag_b, cnt))

        if not edges:
            self._tree = {"name": "root", "children": []}
            self._tag_to_branch = {}
            self._save()
            return

        # Step 2: Kruskal 式聚类 — 并查集去环
        tag_total = {t: sum(v.values()) for t, v in matrix.items()}
        parent_of: dict[str, tuple[str, int]] = {}
        children_map: dict[str, list[str]] = defaultdict(list)

        for tag_a, tag_b, cnt in sorted(edges, key=lambda x: -x[2]):
            root_a = _find_root(tag_a, parent_of)
            root_b = _find_root(tag_b, parent_of)
            if root_a == root_b:
                continue  # 已在同树 → 跳过（去环）

            # 总频更高的作父节点
            if tag_total.get(tag_a, 0) >= tag_total.get(tag_b, 0):
                parent_of[tag_b] = (tag_a, cnt)
                children_map[tag_a].append(tag_b)
            else:
                parent_of[tag_a] = (tag_b, cnt)
                children_map[tag_b].append(tag_a)

        # Step 3: DFS 递归收集整枝，建树
        def _dfs(tag: str, visited: set) -> list[str]:
            """递归收集一个枝上的全部标签。"""
            if tag in visited:
                return []
            visited.add(tag)
            result = [tag]
            for child in children_map.get(tag, []):
                result.extend(_dfs(child, visited))
            return result

        tree = {"name": "root", "children": []}
        tag_to_branch: dict[str, list[str]] = {}
        has_parent = set(parent_of.keys())

        # 递归构建树节点（含孙代）
        def _build_node(tag: str, visited_tree: set) -> dict:
            if tag in visited_tree:
                return {"name": tag}
            visited_tree.add(tag)
            kids = children_map.get(tag, [])
            if kids:
                return {
                    "name": tag,
                    "children": [_build_node(c, visited_tree) for c in kids],
                }
            return {"name": tag}

        visited_tree: set = set()
        for tag in matrix:
            if tag not in has_parent:
                branch = _dfs(tag, set())
                tag_to_branch[tag] = branch
                for bt in branch:
                    tag_to_branch[bt] = branch
                tree["children"].append(_build_node(tag, visited_tree))

        self._tree = tree
        self._tag_to_branch = tag_to_branch
        self._save()
        logger.info(
            "话题树重建完成: %d 枝, %d 标签已聚类",
            len(tree["children"]),
            len(tag_to_branch),
        )

    def expand(self, tags: list[str]) -> list[str]:
        """给定标签，返回同枝的兄弟标签（去重，排除自身，最多 MAX_SIBLINGS 个）。"""
        with self._lock:
            expanded = []
            seen = set(tags)
            for t in tags:
                branch = self._tag_to_branch.get(t, [])
                for bt in branch:
                    if bt not in seen:
                        seen.add(bt)
                        expanded.append(bt)
            return expanded[: self.MAX_SIBLINGS]

    def get_branch(self, tag: str) -> list[str]:
        """返回标签所在的整枝。"""
        with self._lock:
            return self._tag_to_branch.get(tag, [tag])

    # ── 持久化 ────────────────────────────────────────────────

    def _save(self):
        try:
            atomic_write(
                self._path,
                {
                    "version": 1,
                    "tree": self._tree,
                    "tag_to_branch": self._tag_to_branch,
                },
            )
        except Exception:
            logger.exception("话题树写入失败")

    def _load(self):
        try:
            if os.path.exists(self._path):
                with open(self._path, encoding="utf-8") as f:
                    data = json.load(f)
                self._tree = data.get("tree", {"name": "root", "children": []})
                self._tag_to_branch = data.get("tag_to_branch", {})
        except Exception:
            self._tree = {"name": "root", "children": []}
            self._tag_to_branch = {}
