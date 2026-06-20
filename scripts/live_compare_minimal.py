#!/usr/bin/env python3
"""M@q 记忆场 live 对比 — 精简单进程版。"""
import os, sys, time, json, numpy as np
_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJ_ROOT)

os.environ["LOCAL_LLM_MODE"] = "true"
os.environ["STEERING_ENABLED"] = "true"
os.environ["STEERING_DIRECT"] = "true"

from app.core.state import UtteranceSpec, UserMessageAnalysis

SCENARIOS = [
    ("emotional", "最近工作压力好大，感觉自己什么都不太行，想放弃了"),
    ("tech", "Python 异步编程好难，asyncio 和 await 到底什么关系？"),
    ("casual", "今天天气真好，去公园散步看到很多花开了"),
]

def main():
    print("="*60)
    print("M@q Live Compare (single-process)")
    print("="*60)

    # Build M once
    from app.llm.steering_direct import _rebuild_m_matrix, _load_m_matrix
    ok = _rebuild_m_matrix()
    M = _load_m_matrix()
    print(f"M matrix: {M.shape}, rank~{np.linalg.matrix_rank(M)}, ok={ok}")

    # Load model
    print("Loading model...")
    t0 = time.time()
    from app.llm.steering import get_steering_injector
    injector = get_steering_injector()
    print(f"Model loaded in {time.time()-t0:.1f}s, loaded={injector.is_loaded}")

    if not injector.is_loaded:
        print("FAIL: model not loaded")
        return

    from app.llm.steering_direct import MODULE_DIRECT_CONFIG
    from app.config import settings

    all_results = {}

    for name, msg in SCENARIOS:
        spec = UtteranceSpec(user=UserMessageAnalysis(raw_text=msg))
        print(f"\n{'='*60}")
        print(f"Scenario: {name}")
        print(f"User: {msg}")
        print(f"{'='*60}")

        results = {}
        old_enabled = settings.STEERING_ENABLED
        old_direct = settings.STEERING_DIRECT

        try:
            # 1: no steering
            settings.STEERING_ENABLED = False
            settings.STEERING_DIRECT = False
            t0 = time.time()
            r = injector.generate(msg, spec, max_tokens=150, temperature=0.7)
            results["none"] = {"t": time.time()-t0, "text": r["content"].strip()}
            print(f"\n[NONE] {results['none']['t']:.0f}s")
            for line in results["none"]["text"].split("\n")[:10]:
                print(f"  {line}")

            # 2: direct with M@q
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            r = injector.generate(msg, spec, max_tokens=150, temperature=0.7)
            results["mfield"] = {"t": time.time()-t0, "text": r["content"].strip()}
            print(f"\n[MFIELD] {results['mfield']['t']:.0f}s")
            for line in results["mfield"]["text"].split("\n")[:10]:
                print(f"  {line}")

            # 3: direct without M@q
            saved = list(MODULE_DIRECT_CONFIG)
            MODULE_DIRECT_CONFIG.clear()
            for cfg in saved:
                if cfg.extractor != "memory_field":
                    MODULE_DIRECT_CONFIG.append(cfg)
            settings.STEERING_ENABLED = True
            settings.STEERING_DIRECT = True
            t0 = time.time()
            r = injector.generate(msg, spec, max_tokens=150, temperature=0.7)
            results["nomfield"] = {"t": time.time()-t0, "text": r["content"].strip()}
            MODULE_DIRECT_CONFIG.clear()
            MODULE_DIRECT_CONFIG.extend(saved)
            print(f"\n[NOMFIELD] {results['nomfield']['t']:.0f}s")
            for line in results["nomfield"]["text"].split("\n")[:10]:
                print(f"  {line}")

        finally:
            settings.STEERING_ENABLED = old_enabled
            settings.STEERING_DIRECT = old_direct

        all_results[name] = results

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for name, results in all_results.items():
        print(f"\n{name}:")
        for path, data in results.items():
            wc = len(data["text"])
            print(f"  {path:10s}: {wc:4d} chars, {data['t']:.0f}s")

    # Save
    with open("data/live_compare_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to data/live_compare_results.json")

if __name__ == "__main__":
    main()
