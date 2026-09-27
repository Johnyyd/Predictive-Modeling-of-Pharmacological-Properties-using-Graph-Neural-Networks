import os
import json
from pathlib import Path

def load_model_config(config_path: str = "model_config.json") -> dict:
    """
    Load model configuration dictionary from primary path or fallback configs/ folder.
    """
    paths_to_try = [
        Path(config_path),
        Path("configs") / Path(config_path).name,
        Path("model_config.json")
    ]
    for p in paths_to_try:
        if p.exists():
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}
