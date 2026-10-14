"""Read stopping IDs from local model/tokenizer metadata; no tokenization or model execution."""
import json
from pathlib import Path


def read_eos_ids(model_dir: str) -> set[int]:
    root = Path(model_dir)
    ids = set()
    for name in ("generation_config.json", "config.json"):
        path = root / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        for config in (data, data.get("text_config", {})):
            value = config.get("eos_token_id")
            if value is not None:
                ids.update(value if isinstance(value, list) else [value])
    config_path, tokens_path = root / "tokenizer_config.json", root / "tokenizer.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        eos = config.get("eos_token")
        eos = eos.get("content") if isinstance(eos, dict) else eos
        if eos is not None and tokens_path.exists():
            tokens = json.loads(tokens_path.read_text())
            ids.update(t["id"] for t in tokens.get("added_tokens", []) if t.get("content") == eos)
            vocab = tokens.get("model", {}).get("vocab", {})
            if isinstance(vocab, dict) and eos in vocab:
                ids.add(vocab[eos])
    return ids
