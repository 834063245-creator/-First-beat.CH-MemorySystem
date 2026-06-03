"""导出训练数据：合并 Claude Code + WorkBuddy + amazing3 三源。

输出格式（每行一条用户消息）：
  {"user_message": "用户说的内容", "source": "claude_code|workbuddy|amazing3", "timestamp": "YYYY-MM-DD HH:MM:SS"}

过滤规则：
  - 空消息 / 纯空白
  - 长度 < 4 字符（中文短句也过滤，太短没训练价值）
  - IDE 系统消息（opened file / selected range / etc）
  - 纯代码块（无中文）
"""
import json
import os
import sys
from pathlib import Path

OUTPUT = Path(__file__).parent.parent / "data" / "training_raw.jsonl"
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
SEEN = set()
COUNT = 0


def _dedup(text: str) -> bool:
    """检查是否重复，重复返回 True。"""
    key = text[:60].strip()
    if key in SEEN:
        return True
    SEEN.add(key)
    return False


def _is_system_msg(text: str) -> bool:
    """过滤 IDE 系统消息。"""
    noise = [
        "<ide_opened_file>", "<ide_selected_range>",
        "The user opened the file", "The user selected",
        "isSidechain", "queue-operation", "file-history-snapshot",
    ]
    return any(n in text for n in noise)


def _is_pure_code(text: str) -> bool:
    """纯代码/无中文 → 对中文话题分类无训练价值。"""
    has_chinese = any('\u4e00' <= c <= '\u9fff' for c in text)
    return not has_chinese


def _clean(text: str) -> str:
    """清理文本。"""
    return text.strip().replace('\n', ' ').replace('\r', '')


def _write(msg: str, source: str, ts: str = ""):
    global COUNT
    msg = _clean(msg)
    if len(msg) < 4 or not msg:
        return
    if _dedup(msg) or _is_system_msg(msg) or _is_pure_code(msg):
        return
    with open(OUTPUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "user_message": msg,
            "source": source,
            "timestamp": ts,
        }, ensure_ascii=False) + "\n")
    COUNT += 1


# ═══ 源1: Claude Code projects ═══════════════════════════════════
print("Source 1: Claude Code projects...")
cc_base = Path(os.path.expanduser("~/.claude/projects"))
if cc_base.exists():
    jsonl_files = list(cc_base.glob("*/*.jsonl"))
    print(f"  Found {len(jsonl_files)} jsonl files")
    for f in sorted(jsonl_files):
        fsize = f.stat().st_size
        if fsize < 1000:
            continue
        try:
            lines = [l.strip() for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
            for line in lines:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") != "user":
                    continue
                msg = obj.get("message", {})
                content = msg.get("content", [])
                if not content:
                    continue
                # 提取所有 text 块
                texts = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        texts.append(block.get("text", ""))
                full = " ".join(texts)
                if full:
                    ts = obj.get("timestamp", "")
                    _write(full, "claude_code", ts)
        except Exception as e:
            print(f"  skip {f.name}: {e}")

# ═══ 源2: WorkBuddy (Reasonix) ═══════════════════════════════════
print("Source 2: WorkBuddy...")
rx_root = Path(os.path.expanduser("~/AppData/Roaming/reasonix/sessions"))
if rx_root.exists():
    for f in sorted(rx_root.glob("*.jsonl")):
        if f.stat().st_size < 100:
            continue
        try:
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                role = obj.get("role", "")
                if role != "user":
                    continue
                content = obj.get("content", "")
                ts = obj.get("timestamp", "")
                _write(content, "workbuddy", ts)
        except Exception as e:
            print(f"  skip {f.name}: {e}")

# ═══ 源3: amazing3 chat_history ══════════════════════════════════
print("Source 3: amazing3...")
am3 = Path(r"D:\amazing3\instances\predecessor\data\chat_history.jsonl")
if am3.exists():
    for line in am3.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = obj.get("user_message", "")
        ts = obj.get("timestamp", "")
        _write(msg, "amazing3", ts)

print(f"\nDone: {COUNT} messages → {OUTPUT}")
print(f"File size: {OUTPUT.stat().st_size / 1024:.0f}KB")
