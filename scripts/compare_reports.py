#!/usr/bin/env python3
"""对比两份审计报告的分数变化。

用法:
  python scripts/compare_reports.py audit/report_A.json audit/report_B.json
"""
import sys
import json


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compare(old_path: str, new_path: str) -> None:
    old = load(old_path)
    new = load(new_path)

    fmt = "{:25s} | {:>10s} | {:>10s} | {:>8s}"
    print(fmt.format("Category", "Old", "New", "Change"))
    print("-" * 62)

    all_cats = sorted(set(
        list(old.get("scores", {}).keys()) + list(new.get("scores", {}).keys())
    ))

    for cat_name in all_cats:
        old_s = old.get("scores", {}).get(cat_name, {})
        new_s = new.get("scores", {}).get(cat_name, {})
        old_score = old_s.get("score", 0)
        new_score = new_s.get("score", 0)

        try:
            diff = float(new_score) - float(old_score)
        except (TypeError, ValueError):
            diff_str = "-"
        else:
            if diff > 0.005:
                diff_str = f"+{diff:+.0%}"
            elif diff < -0.005:
                diff_str = f"{diff:+.0%}"
            else:
                diff_str = "="

        try:
            os_display = f"{float(old_score):.0%}" if old_score is not None else "-"
            ns_display = f"{float(new_score):.0%}" if new_score is not None else "-"
        except (TypeError, ValueError):
            os_display = str(old_score)[:10]
            ns_display = str(new_score)[:10]

        print(f"{cat_name:25s} | {os_display:>10s} | {ns_display:>10s} | {diff_str:>8s}")

    print("-" * 62)
    ow = old.get("weighted_total", 0)
    nw = new.get("weighted_total", 0)
    diff_total = nw - ow
    print("{:25s} | {:>10.0%} | {:>10.0%} | +{:.0%}".format(
        "Weighted Total", ow, nw, diff_total
    ))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python scripts/compare_reports.py <old_report.json> <new_report.json>")
        sys.exit(1)
    compare(sys.argv[1], sys.argv[2])
