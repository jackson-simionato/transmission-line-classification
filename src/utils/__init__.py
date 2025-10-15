import hashlib

from src.config.settings import get_config

config = get_config()


def get_class_mapping(class_label: str) -> int:
    return config.CLASSES.get(class_label, 0)


def deterministic_id(geom):
    # Use WKB (well-known binary) for a stable serialization of geometry
    wkb = geom.wkb
    return hashlib.sha256(wkb).hexdigest()
