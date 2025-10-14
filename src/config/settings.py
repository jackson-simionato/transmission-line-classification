class ApplicationConfig:
    def __init__(self):
        self.CLASSES = {
            "line": 1,
            "tower": 2,
            "grassland": 3,
            "tall_vegetation": 4,
            "soil": 5,
        }
        self.NUM_CLASSES = len(self.CLASSES)
        self.CLASS_NAMES = list(self.CLASSES.keys())

        self.TRAIN_DATASET_IDS = [
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

        self.VAL_DATASET_IDS = [
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_15",
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_21",
        ]

        self.TEST_DATASET_IDS = [
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_18",
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_16",
        ]

        # Feature extraction configuration
        self.FEATURE_EXTRACTION = {
            "include_spectral": True,
            "include_geometric": True,
            "include_texture": True,
            "include_cnn": False,  # Disabled by default (memory intensive)
            "cnn_model": "mobilenet_v3_small",  # or "efficientnet_b0"
            "glcm_distances": [1],
            "glcm_angles": [0],  # degrees
            "glcm_levels": 256,
            "output_format": "csv",  # or "parquet"
        }


def get_config():
    return ApplicationConfig()
