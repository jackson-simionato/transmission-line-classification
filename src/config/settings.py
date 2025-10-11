class ApplicationConfig:
    CLASSES = {"line": 1, "tower": 2, "grassland": 3, "tall_vegetation": 4, "soil": 5}
    TRAIN_DATASET_IDS = [
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_15",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_14",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_18",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_17",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_16",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_22",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_20",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_19",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_17",
    ]

    VAL_DATASET_IDS = [
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_15",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_21",
    ]

    TEST_DATASET_IDS = [
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_18",
        "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_16",
    ]


def get_config():
    return ApplicationConfig
