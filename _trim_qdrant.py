"""Trim CoOccurrenceStore and HyperEdgeStore from qdrant.py."""
path = r"D:\First Beat CH Memory System\app\memory\qdrant.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Find the cut point: comment block before CoOccurrenceStore
cut = None
for i, line in enumerate(lines):
    if "CoOccurrenceStore" in line and "class CoOccurrenceStore" in line:
        # Go back to find the section header comment
        for j in range(i, max(i-5, 0), -1):
            if "CoOccurrenceStore" in lines[j] and "===" in lines[j]:
                cut = j
                break
        if cut is None:
            cut = i
        break

if cut is None:
    print("ERROR: CoOccurrenceStore not found")
    exit(1)

print(f"Cut at line {cut+1}: {lines[cut].rstrip()}")
new_lines = lines[:cut]
new_lines.append("\n")
new_lines.append("# CoOccurrenceStore 和 HyperEdgeStore 已拆分至独立文件：\n")
new_lines.append("#   from app.memory.qdrant_cooccur import CoOccurrenceStore\n")
new_lines.append("#   from app.memory.qdrant_hyperedge import HyperEdgeStore\n")

with open(path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Done. Original: {len(lines)} lines, New: {len(new_lines)} lines")
