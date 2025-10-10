class ApplicationConfig:
    CLASSES = {"line": 1, "tower": 2, "grassland": 3, "tall_vegetation": 4, "soil": 5}


def get_config():
    return ApplicationConfig
