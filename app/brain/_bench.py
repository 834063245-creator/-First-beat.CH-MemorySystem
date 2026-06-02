"""测试 ChuchenBrain 实际响应延迟"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.brain.models import ChuchenBrain

brain = ChuchenBrain(model_name="qwen2.5:3b")
brain.load_all()

messages = [
    "最近压力好大，项目快崩了",
    "帮我看看这段代码为什么报错",
    "我今天心情特别好",
    "你还记得上次我们聊的那个架构方案吗",
    "不对，你说的不是这样的",
]

print("=" * 60)
print("ChuchenBrain 延迟实测 (qwen2.5:3b via Ollama)")
print("=" * 60)

intent_times = []
emotion_times = []

for msg in messages:
    t0 = time.perf_counter()
    r = brain.classify_intent(msg)
    t1 = time.perf_counter()
    intent_ms = (t1 - t0) * 1000
    intent_times.append(intent_ms)

    t0 = time.perf_counter()
    r2 = brain.analyze_emotion(msg)
    t1 = time.perf_counter()
    emotion_ms = (t1 - t0) * 1000
    emotion_times.append(emotion_ms)

    print(f" \"{msg[:25]}...\"  意图:{intent_ms:.0f}ms  情绪:{emotion_ms:.0f}ms")

avg_intent = sum(intent_times)/len(intent_times)
avg_emotion = sum(emotion_times)/len(emotion_times)
combined = sum(intent_times) + sum(emotion_times)

print(f"\n 意图平均: {avg_intent:.0f}ms")
print(f" 情绪平均: {avg_emotion:.0f}ms")
print(f" 两个加起来: {avg_intent + avg_emotion:.0f}ms")
print(f" 对比 DeepSeek API 通常 1-3s => 只占约 {int((avg_intent+avg_emotion)/1500*100)}%")
