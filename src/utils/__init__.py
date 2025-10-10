from src.config.settings import get_config

config = get_config()


def get_class_mapping(class_label: str) -> int:
    return config.CLASSES.get(class_label, 0)
