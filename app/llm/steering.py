# Copyright (c) 2026 初痕 (Chuchen)
# SPDX-License-Identifier: MIT
# wm: a1b2c3d4

"""
残差分层注入引擎 — 16 模块产出短文本 → embed → CVEC 注入指定层。

本地推理模式（LOCAL_LLM_MODE=true）下替代 deepseek.py 的消息拼装。
引擎 UtteranceSpec → build_steering_segments() → SteeringInjector.generate()

原理:
  引擎 16 个模块 → 各产 1-2 句短中文 → qwen_embed → mean-pooled 向量
  → llama_set_adapter_cvec(h_L += r × α) → 本地生成

依赖:
  llama-cpp-python + libllama.dll (MinGW bin dir 需在 PATH 或 MINGW_BIN_DIR 配置)
"""

import ctypes
import logging
import os
import threading
import time as _time
from pathlib import Path
from typing import Optional

import numpy as np

from app.config.settings import (
    QWEN_GGUF_PATH,
    STEERING_ENABLED,
    STEERING_STRENGTH,
    MINGW_BIN_DIR,
)

logger = logging.getLogger(__name__)

# ── Windows: MinGW DLL 运行时路径 ───────────────────────────
if MINGW_BIN_DIR and os.path.isdir(MINGW_BIN_DIR):
    try:
        os.add_dll_directory(MINGW_BIN_DIR)
    except Exception:
        pass  # 非 Windows 或权限不足，忽略


# ═══════════════════════════════════════════════════════════════
# 模块→层映射 (1-indexed, qwen2.5 28层)
# ═══════════════════════════════════════════════════════════════
# 每个条目: (模块名, 注入层, α幅度, output_key)
# 同一层多个模块的向量累加。α 值来自 Phase 8 实验标定。

MODULE_LAYER_MAP = [
    # ── 浅层: 身份语义 (L3-5) ──
    ("portrait_identity",      3,  0.08, "user_identity"),
    ("portrait_identity_ai",   3,  0.05, "ai_identity"),
    # ── 语义层: 相关记忆 (L5-10) ──
    ("relevant_memory_1",      5,  0.06, "memory_1"),
    ("relevant_memory_2",      6,  0.06, "memory_2"),
    ("relevant_memory_3",      7,  0.06, "memory_3"),
    ("relevant_memory_4",      8,  0.06, "memory_4"),
    ("relevant_memory_5",      9,  0.06, "memory_5"),
    # ── 中层: 情绪+冲动 (L10-15) ──
    ("portrait_emotion",      12,  0.10, "emotion_context"),
    ("impulse_signal",        12,  0.05, "impulse"),
    # ── 中深层: 偏移率+兴趣 (L15-20) ──
    ("drift_context",         15,  0.07, "drift"),
    ("portrait_interest",     16,  0.07, "interest"),
    # ── 深层: 关系+镜像 (L18-25) ──
    ("relationship_state",    20,  0.08, "relationship"),
    ("self_mirror",           22,  0.05, "mirror"),
    # ── 近输出层: 门控+预测 (L25+) ──
    ("gate_tone",             25,  0.12, "tone"),
    ("behavior_predictor",    26,  0.05, "predictor"),
]


# ═══════════════════════════════════════════════════════════════
# build_steering_segments — 从 UtteranceSpec 提取模块短文本
# ═══════════════════════════════════════════════════════════════

def build_steering_segments(utterance_spec) -> dict[str, str]:
    """从 UtteranceSpec 提取 16 个模块各产出的短文本。

    每段 1-2 句中文，10-60 token。空字符串表示该模块无产出，注入时跳过。

    Returns:
        dict[output_key, text_string] — key 对应对 MODULE_LAYER_MAP 的 output_key
    """
    m = {}
    spec = utterance_spec

    # ── 画像身份 (usr1 + ai1) ──
    portrait_stable = getattr(spec, "portrait_stable", "") or ""
    m["user_identity"] = portrait_stable.strip()[:150] if isinstance(portrait_stable, str) and portrait_stable.strip() else ""

    notes_ai = getattr(spec, "personality_notes_ai", []) or []
    ai_parts = []
    for note in notes_ai[:2]:
        if isinstance(note, dict):
            ai_parts.append(str(note.get("content", ""))[:80])
        elif isinstance(note, str):
            ai_parts.append(note[:80])
    m["ai_identity"] = "AI助手，" + "；".join(ai_parts) if ai_parts else ""

    # ── 相关记忆 (最多 5 条) ──
    memories = getattr(spec, "memories", []) or []
    for i in range(5):
        key = f"memory_{i+1}"
        if i < len(memories):
            mem = memories[i]
            if isinstance(mem, dict):
                summary = mem.get("summary", "") or mem.get("document", "") or ""
            else:
                summary = getattr(mem, "summary", "") or getattr(mem, "document", "") or ""
            m[key] = str(summary)[:150]
        else:
            m[key] = ""

    # ── 情绪上下文 ──
    emotion_parts = []
    user = getattr(spec, "user", None)
    if user:
        uemotion = getattr(user, "emotion", "")
        if uemotion:
            emotion_parts.append(f"用户情绪：{uemotion}")
    portrait_dynamic = getattr(spec, "portrait_dynamic", "") or ""
    if isinstance(portrait_dynamic, str) and portrait_dynamic.strip():
        emotion_parts.append(portrait_dynamic.strip()[:150])
    m["emotion_context"] = "。".join(emotion_parts) if emotion_parts else ""

    # ── 冲动信号 ──
    impulses = getattr(spec, "impulses", []) or []
    if impulses:
        imp_parts = []
        for imp in impulses[:2]:
            target = ""
            if isinstance(imp, dict):
                target = imp.get("target_concept", "")
            else:
                target = getattr(imp, "target_concept", "")
            if target:
                imp_parts.append(f"想到：{target}")
        m["impulse"] = "；".join(imp_parts) if imp_parts else ""
    else:
        m["impulse"] = ""

    # ── 偏移率 ──
    drift_text = getattr(spec, "drift_text", "") or ""
    m["drift"] = str(drift_text).strip()[:120] if drift_text else ""

    # ── 兴趣图谱 ──
    personality_notes = getattr(spec, "personality_notes", []) or []
    interest_parts = []
    for note in personality_notes[:3]:
        if isinstance(note, dict):
            interest_parts.append(str(note.get("content", ""))[:80])
        elif isinstance(note, str):
            interest_parts.append(note[:80])
    m["interest"] = "用户关注：" + "；".join(interest_parts) if interest_parts else ""

    # ── 关系状态 ──
    rs = getattr(spec, "relationship", None)
    if rs:
        trust = getattr(rs, "trust", 0.5)
        mode = getattr(rs, "interaction_mode", "casual")
        m["relationship"] = f"信任度{trust:.0%}，{mode}关系模式"
    else:
        m["relationship"] = ""

    # ── 自我镜像 ──
    self_mirror_text = getattr(spec, "self_mirror_text", "") or ""
    m["mirror"] = str(self_mirror_text).strip()[:150] if self_mirror_text else ""

    # ── 门控语气 ──
    gate = getattr(spec, "gate", None)
    if gate:
        tone = getattr(gate, "tone", "warm")
        tone_map = {
            "soft": "温柔共情语气，先理解再回应",
            "caring": "关怀语气，先共情再给建议",
            "warm": "温暖友好语气，像朋友聊天",
            "direct": "直接简洁语气",
            "neutral": "中性理性语气",
        }
        m["tone"] = tone_map.get(tone, "自然回应语气")
    else:
        m["tone"] = ""

    # ── 行为预测 ──
    mp = getattr(spec, "mirror_prediction", None) or {}
    if mp:
        nexts = mp.get("next_intents") or [mp.get("next_intent", "")]
        if nexts and nexts[0]:
            m["predictor"] = f"预测用户可能：{' → '.join(nexts[:2])}"
        else:
            m["predictor"] = ""
    else:
        m["predictor"] = ""

    return m


# ═══════════════════════════════════════════════════════════════
# SteeringInjector — CVEC 注入 + 本地推理
# ═══════════════════════════════════════════════════════════════

class SteeringInjector:
    """CVEC 残差注入 + 本地 llama.cpp 推理。

    单例模式：一个进程只有一个 llama 模型实例。
    线程安全：generate() / generate_stream() 用 Lock 串行化 CVEC 操作。

    用法:
        injector = get_steering_injector()
        result = injector.generate(user_message, utterance_spec)
    """

    _instance: Optional["SteeringInjector"] = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_init_done"):
            return
        self._init_done = True
        self._model = None
        self._n_embd: int = 0
        self._n_layer: int = 28
        self._generate_lock = threading.Lock()
        self._loaded: bool = False

    # ── 生命周期 ─────────────────────────────────────────

    def load(self) -> bool:
        """加载 qwen2.5 模型。幂等：已加载则跳过。

        Returns:
            True if model loaded successfully (or already loaded).
        """
        if self._loaded:
            return True

        from llama_cpp import Llama

        gguf_path = QWEN_GGUF_PATH
        if not os.path.exists(gguf_path):
            logger.error("SteeringInjector: GGUF not found: %s", gguf_path)
            return False

        try:
            _t0 = _time.perf_counter()
            self._model = Llama(
                model_path=gguf_path,
                n_ctx=2048,
                n_gpu_layers=0,
                embedding=False,
                verbose=False,
            )
            self._n_embd = self._model.n_embd()

            from llama_cpp import llama_cpp as llc
            model_ptr = llc.llama_get_model(self._model.ctx)
            self._n_layer = llc.llama_model_n_layer(model_ptr)

            self._loaded = True
            elapsed = _time.perf_counter() - _t0
            logger.info(
                "SteeringInjector loaded: n_embd=%d n_layer=%d (%.1fs)",
                self._n_embd, self._n_layer, elapsed,
            )
            return True
        except Exception as exc:
            logger.error("SteeringInjector load failed: %s", exc, exc_info=True)
            return False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def unload(self):
        """释放模型内存。"""
        with self._generate_lock:
            if self._model is not None:
                try:
                    self._model.close()
                except Exception:
                    pass
                self._model = None
            self._loaded = False

    # ── 公开接口 ─────────────────────────────────────────

    def generate(
        self,
        user_message: str,
        utterance_spec,
        max_tokens: int = 4096,
        temperature: float = 0.7,
    ) -> dict:
        """生成回复 — CVEC steering + 本地推理。

        Args:
            user_message: 用户消息
            utterance_spec: 引擎产出的 UtteranceSpec
            max_tokens: 最大生成 token 数
            temperature: 采样温度

        Returns:
            {"content": str, "tool_calls": []} 兼容 LLMClient.generate() 接口
        """
        if not self._loaded:
            return {"content": "[错误：本地模型未加载，请检查 QWEN_GGUF_PATH 配置]", "tool_calls": []}

        if STEERING_ENABLED:
            segments = build_steering_segments(utterance_spec)
            active = sum(1 for v in segments.values() if v and v.strip())
            logger.debug("steering segments: %d/%d active", active, len(segments))
            content = self._generate_with_cvec(user_message, segments, max_tokens, temperature)
        else:
            content = self._generate_plain(user_message, max_tokens, temperature)

        return {"content": content, "tool_calls": []}

    def generate_stream(self, user_message: str, utterance_spec,
                        max_tokens: int = 4096, temperature: float = 0.7):
        """流式生成 — 同步生成器。

        Yields: ("content", token_text) tuples
        与 LLMClient.generate_stream() 接口兼容。
        """
        if not self._loaded:
            yield ("content", "[错误：本地模型未加载]")
            return

        if STEERING_ENABLED:
            segments = build_steering_segments(utterance_spec)
            yield from self._generate_with_cvec_stream(
                user_message, segments, max_tokens, temperature)
        else:
            yield from self._generate_plain_stream(
                user_message, max_tokens, temperature)

    # ── CVEC 操作 ────────────────────────────────────────

    def _setup_cvec(self, segments: dict[str, str]) -> bool:
        """embed 各模块短句 → 注入指定层的 CVEC buffer。

        同一层多个模块的向量累加。空字符串模块跳过。

        Returns:
            True if CVEC was set successfully (at least one module active).
        """
        from llama_cpp import llama_cpp as llc
        from app.llm.qwen_embed import get_qwen_embedder

        embedder = get_qwen_embedder()
        n_embd = self._n_embd
        n_layer = self._n_layer

        buf = np.zeros(n_layer * n_embd, dtype=np.float32)
        active_count = 0

        for _mod_name, layer, alpha, output_key in MODULE_LAYER_MAP:
            text = segments.get(output_key, "")
            if not text or not text.strip():
                continue

            vec = embedder.embed(text).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm == 0:
                continue

            # 归一化 → α 缩放 → 全局强度倍率
            vec = vec / norm * alpha * STEERING_STRENGTH

            layer_idx = layer - 1  # 0-indexed
            buf[layer_idx * n_embd : (layer_idx + 1) * n_embd] += vec
            active_count += 1

        if active_count == 0:
            return False

        data = (ctypes.c_float * (n_layer * n_embd))(*buf.tolist())
        ret = llc.llama_set_adapter_cvec(
            self._model.ctx, data, n_layer * n_embd, n_embd, 1, n_layer)

        if ret != 0:
            logger.warning("llama_set_adapter_cvec returned %d (non-zero)", ret)
        return ret == 0

    def _clear_cvec(self):
        """清除 CVEC buffer。"""
        from llama_cpp import llama_cpp as llc
        llc.llama_set_adapter_cvec(
            self._model.ctx, None, 0, self._n_embd, 0, 0)

    # ── 生成（内部）───────────────────────────────────────

    def _generate_with_cvec(self, user_message: str, segments: dict,
                            max_tokens: int, temperature: float) -> str:
        prompt = f"用户消息: {user_message}\n回复:"
        with self._generate_lock:
            cvec_ok = self._setup_cvec(segments)
            try:
                result = self._model.create_completion(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    echo=False, stream=False)
            finally:
                if cvec_ok:
                    self._clear_cvec()
        return result["choices"][0].get("text", "")

    def _generate_plain(self, user_message: str,
                        max_tokens: int, temperature: float) -> str:
        prompt = f"用户消息: {user_message}\n回复:"
        with self._generate_lock:
            result = self._model.create_completion(
                prompt, max_tokens=max_tokens, temperature=temperature,
                echo=False, stream=False)
        return result["choices"][0].get("text", "")

    def _generate_with_cvec_stream(self, user_message: str, segments: dict,
                                   max_tokens: int, temperature: float):
        prompt = f"用户消息: {user_message}\n回复:"
        with self._generate_lock:
            cvec_ok = self._setup_cvec(segments)
            try:
                stream = self._model.create_completion(
                    prompt, max_tokens=max_tokens, temperature=temperature,
                    echo=False, stream=True)
                for chunk in stream:
                    token = chunk["choices"][0].get("text", "")
                    if token:
                        yield ("content", token)
            finally:
                if cvec_ok:
                    self._clear_cvec()

    def _generate_plain_stream(self, user_message: str,
                               max_tokens: int, temperature: float):
        prompt = f"用户消息: {user_message}\n回复:"
        with self._generate_lock:
            stream = self._model.create_completion(
                prompt, max_tokens=max_tokens, temperature=temperature,
                echo=False, stream=True)
            for chunk in stream:
                token = chunk["choices"][0].get("text", "")
                if token:
                    yield ("content", token)


# ═══════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════

def get_steering_injector() -> SteeringInjector:
    """获取 SteeringInjector 单例，自动触发模型加载。"""
    injector = SteeringInjector()
    if not injector.is_loaded:
        injector.load()
    return injector
