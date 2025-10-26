import yaml, os

def load_config(path: str = "config.yaml"):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # environment overrides (optional)
    cfg["project"] = cfg.get("project", {})
    return cfg
