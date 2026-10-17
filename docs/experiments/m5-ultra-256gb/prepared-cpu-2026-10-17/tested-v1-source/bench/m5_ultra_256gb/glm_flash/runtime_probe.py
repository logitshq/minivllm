"""CPU-only GLM load inventory; this command never imports MLX or loads tensors.

Requires download.py's completed SHA receipt. Reads safetensors headers and file
stats only; does not repeat the 190 GiB content hash or establish a runtime peak.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import struct

HERE = Path(__file__).resolve().parent
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
               "U16": 2, "I16": 2, "F16": 2, "BF16": 2,
               "U32": 4, "I32": 4, "F32": 4, "U64": 8, "I64": 8, "F64": 8}
EXPERT = re.compile(r"model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
                    r"(gate_proj|up_proj|down_proj)\.(weight|scales|biases)$")
LAYER = re.compile(r"model\.language_model\.layers\.(\d+)\.")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_header(path):
    """Validate bounded header/offset arithmetic without reading tensor payload."""
    path = Path(path)
    before = path.stat()
    with path.open("rb") as source:
        prefix = source.read(8)
        require(len(prefix) == 8, "short safetensors prefix")
        length = struct.unpack("<Q", prefix)[0]
        require(0 < length <= 32 * 1024**2 and 8 + length <= before.st_size,
                "invalid or oversized safetensors header")
        raw = source.read(length)
    header = json.loads(raw, object_pairs_hook=unique_object)
    entries, spans = {}, []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        shape, dtype, offsets = item["shape"], item["dtype"], item["data_offsets"]
        require(dtype in DTYPE_BYTES, f"unsupported dtype: {dtype}")
        require(isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape),
                f"bad shape: {name}")
        require(isinstance(offsets, list) and len(offsets) == 2 and
                all(type(d) is int for d in offsets), f"bad offsets: {name}")
        begin, end = offsets
        size = math.prod(shape) * DTYPE_BYTES[dtype]
        require(0 <= begin <= end <= before.st_size - 8 - length and end - begin == size,
                f"size/offset mismatch: {name}")
        entries[name] = {"dtype": dtype, "shape": shape, "bytes": size}
        spans.append((begin, end))
    cursor = 0
    for begin, end in sorted(spans):
        require(begin == cursor, "payload has overlap or a gap")
        cursor = end
    require(cursor == before.st_size - 8 - length, "unaccounted payload bytes")
    after = path.stat()
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            "file changed during header inspection")
    return entries, {"name": path.name, "file_bytes": before.st_size,
                     "header_bytes": length + 8, "payload_bytes": cursor,
                     "header_sha256": hashlib.sha256(prefix + raw).hexdigest(),
                     "tensor_count": len(entries)}


def inventory(receipt_path):
    manifest_path = HERE / "metadata-v1/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    receipt_path = Path(receipt_path).resolve()
    receipt = json.loads(receipt_path.read_text())
    require(receipt["status"] == "PASS_PINNED_DOWNLOAD_AND_SHA_ONLY", "download incomplete")
    require(all(receipt[k] == manifest[k] for k in ("repo", "revision", "weight_bytes")),
            "receipt differs from pinned model")
    expected = {s["name"]: s for s in manifest["shards"]}
    verified = {s["name"]: s for s in receipt["shards"]}
    require(len(verified) == len(receipt["shards"]) and verified.keys() == expected.keys(),
            "receipt shard set differs")
    for name, item in expected.items():
        require(verified[name].get("sha256_verified") is True and
                all(verified[name][k] == item[k] for k in ("bytes", "sha256")),
                f"unverified shard: {name}")
    snapshot = Path(receipt["snapshot"]).resolve()
    require(snapshot.name == manifest["revision"], "wrong snapshot revision")
    require(not (snapshot / "offload_index.json").exists(), "offload snapshot is not this baseline")
    for name, item in manifest["metadata"].items():
        data = (snapshot / name).read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"],
                f"metadata changed: {name}")
    weight_map = json.loads((snapshot / "model.safetensors.index.json").read_text())["weight_map"]
    config = json.loads((snapshot / "config.json").read_text())["text_config"]
    n_layers, n_experts = config["num_hidden_layers"], config["n_routed_experts"]
    seen, shards = set(), []
    categories, dtypes = defaultdict(int), defaultdict(int)
    layers, groups = defaultdict(lambda: defaultdict(int)), defaultdict(set)
    for name in sorted(expected):
        path = snapshot / name
        require(path.stat().st_size == expected[name]["bytes"], f"file size changed: {name}")
        entries, info = read_header(path)
        shards.append(info)
        for key, value in entries.items():
            require(key not in seen and weight_map.get(key) == name, f"index mismatch: {key}")
            seen.add(key)
            size = value["bytes"]
            dtypes[value["dtype"]] += size
            layer = LAYER.match(key)
            dropped = bool(layer and int(layer[1]) >= n_layers)
            if dropped:
                category = "next_token_layers_removed_by_sanitize"
            elif key.startswith(("model.visual.", "visual.", "vision_tower.")):
                category = "vision_retained_for_official_vlm"
            else:
                category = "other_retained_checkpoint_tensors"
            expert = EXPERT.fullmatch(key)
            if expert and not dropped:
                layer_id, expert_id, projection, field = expert.groups()
                category = "backbone_routed_experts"
                layers[layer_id][projection] += size
                groups[(layer_id, projection, field)].add(int(expert_id))
            categories[category] += size
    require(seen == set(weight_map), "missing indexed tensor")
    for key, experts in groups.items():
        require(experts == set(range(n_experts)), f"incomplete expert group: {key}")
    for layer_id, projections in layers.items():
        require(set(projections) == {"gate_proj", "up_proj", "down_proj"},
                f"incomplete routed layer: {layer_id}")
        for projection in projections:
            require(all((layer_id, projection, field) in groups
                        for field in ("weight", "scales", "biases")),
                    f"incomplete packed fields: {layer_id}/{projection}")
    payload = sum(x["payload_bytes"] for x in shards)
    require(sum(categories.values()) == payload, "category arithmetic mismatch")
    return {"status": "PASS_HEADER_INVENTORY_ONLY", "GPU_work": False,
            "runtime_load_executed": False, "peak_memory_measured": False,
            "approval_to_load": False,
            "repo": manifest["repo"], "revision": manifest["revision"],
            "snapshot": str(snapshot), "download_receipt": str(receipt_path),
            "download_receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "weight_content_SHA_rechecked": False,
            "weight_content_SHA_provenance": "trusted completed download receipt; stat/header rechecked only",
            "file_bytes": sum(x["file_bytes"] for x in shards), "payload_bytes": payload,
            "tensor_count": len(seen), "bytes_by_category": dict(categories),
            "bytes_by_dtype": dict(dtypes), "backbone_expert_layer_bytes": dict(layers),
            "max_one_layer_gate_up_new_payload_bytes": max(
                (v["gate_proj"] + v["up_proj"] for v in layers.values()), default=0),
            "max_one_projection_stack_payload_bytes": max(
                (s for v in layers.values() for s in v.values()), default=0),
            "shards": shards,
            "limits": ["Header byte counts are logical payload, not allocator/physical peak.",
                       "No tensor payload was read by this inventory.",
                       "No initial stack materialization or driver-retention peak is bounded here.",
                       "A completed earlier content hash does not prevent later same-size payload mutation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-receipt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = inventory(args.download_receipt)
    with args.out.open("x") as dest:
        dest.write(json.dumps(result, indent=2) + "\n")
    print(result["status"], "payload_bytes", result["payload_bytes"], "load_approval=False")


if __name__ == "__main__":
    main()
