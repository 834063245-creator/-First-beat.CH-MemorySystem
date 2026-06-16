import json, os

for label, path in [
    ("FirstBeat", r"D:\First Beat CH Memory System\hologram_graph.json"),
    ("HoloGram", r"D:\HoloGramHG\hologram_graph.json"),
]:
    size_mb = os.path.getsize(path) / (1024*1024)
    with open(path) as f:
        data = json.load(f)
    nc = data.get("node_count", len(data.get("nodes", [])))
    ec = data.get("edge_count", len(data.get("edges", [])))
    
    # Check if nodes array exists
    nodes = data.get("nodes", [])
    edges = data.get("edges", [])
    
    print(f"{label}: {size_mb:.1f}MB, node_count={nc}, edge_count={ec}")
    print(f"  nodes array: {len(nodes)} items")
    print(f"  edges array: {len(edges)} items")
    
    # Print top-level keys
    keys = [k for k in data.keys() if k not in ("nodes", "edges")]
    print(f"  top keys: {keys[:15]}")
    print()
