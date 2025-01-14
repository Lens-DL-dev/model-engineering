# utils/config.py
import yaml
import os

def load_config(config_path: str) -> dict:
    """
    주어진 YAML 파일을 로드하여 dict 형태로 반환
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config
