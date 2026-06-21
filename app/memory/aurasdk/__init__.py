# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
"""
AuraSDK — 认知记忆引擎 (Rust 核心 + Python 绑定)

零 embedding 事实召回：SDR 稀疏哈希 + MinHash n-gram + 倒排索引。
确定性编码，<1ms 召回，~3MB 内存。

用法:
    from app.memory.aurasdk import Aura, Level

    brain = Aura("./data/aura")
    brain.store("用户喜欢暗色主题", level=Level.Identity, tags=["preference"])
    results = brain.recall_structured("主题偏好", top_k=5)
"""

# ── 动态加载同目录下的 _core.pyd ──
import importlib.util
from pathlib import Path

_CORE_DIR = Path(__file__).resolve().parent
_PYD_PATH = _CORE_DIR / "_core.cp314-win_amd64.pyd"

if not _PYD_PATH.exists():
    raise ImportError(
        f"AuraSDK native module not found at {_PYD_PATH}. "
        f"Ensure the .pyd file was copied from auraSDK."
    )

# 加载 _core.pyd
_spec = importlib.util.spec_from_file_location(
    "app.memory.aurasdk._core",
    str(_PYD_PATH),
)
_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_core)

# ── 公开 API ──
Aura = _core.Aura
Level = _core.Level
Record = _core.Record

# 可选：画像 & 信任配置（引擎用）
TagTaxonomy = _core.TagTaxonomy
TrustConfig = _core.TrustConfig
AgentPersona = _core.AgentPersona
PersonaTraits = _core.PersonaTraits

# 后台维护（对应 ConsolidationEngine 的 Rust 版本）
MaintenanceConfig = _core.MaintenanceConfig
MaintenanceReport = _core.MaintenanceReport
ArchivalRule = _core.ArchivalRule
DecayReport = _core.DecayReport
ReflectReport = _core.ReflectReport
ConsolidationReport = _core.ConsolidationReport
CircuitBreakerConfig = _core.CircuitBreakerConfig

__version__ = "1.5.4"

__all__ = [
    "Aura",
    "Level",
    "Record",
    "TagTaxonomy",
    "TrustConfig",
    "MaintenanceConfig",
    "MaintenanceReport",
    "ArchivalRule",
    "DecayReport",
    "ReflectReport",
    "ConsolidationReport",
    "AgentPersona",
    "PersonaTraits",
    "CircuitBreakerConfig",
]
