class ApplicationConfig:
    def __init__(self):
        self.CLASSES = {
            "line": 0,
            "tower": 1,
            "grassland": 2,
            "tall_vegetation": 3,
            "soil": 4,
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
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_2_21",
        ]

        self.TEST_DATASET_IDS = [
            "OXAPAMPA-VILLA_RICA_ortofoto_tile_1_15",
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

        # ML Training configuration
        self.ML_TRAINING = {
            "default_models": ["random_forest", "xgboost", "svm", "lightgbm"],
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 4,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "svm": {"kernel": "rbf", "random_state": 42, "C": 10, "gamma": 0.01},
            "lightgbm": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 4,
                "num_leaves": 31,
                "min_child_samples": 20,
                "random_state": 42,
                "verbose": -1,  # Suppress LightGBM warnings
            },
            "preprocessing": {
                "normalize_features": True,
                "handle_class_imbalance": True,
                "cv_folds": 3,
            },
            "grid_search": {
                "enable": True,
                "cv_folds": 3,
                "scoring": "f1_weighted",
                "n_jobs": -1,
                "random_forest": {
                    "n_estimators": [50, 100, 200],
                    "max_depth": [5, 10, 15, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                },
                "xgboost": {
                    "max_depth": [3, 4, 5],
                    "min_child_weight": [5, 10, 20],
                    "subsample": [0.6, 0.8],
                    "colsample_bytree": [0.5, 0.8],
                    "lambda": [1, 2, 5],
                    "alpha": [0, 0.1, 0.5],
                    "eta": [0.05, 0.1]
                },
                "svm": {
                    "C": [0.1, 1, 10, 100, 1000],  # Wider range
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1, 10],  # More gamma values
                    "kernel": ["rbf", "poly", "sigmoid"],  # Test other kernels
                    "degree": [2, 3, 4, 5],  # For poly kernel
                    "coef0": [0.0, 0.1, 0.5, 1.0]  # For poly/sigmoid kernels
                },
                "lightgbm": {
                    "num_leaves": [31, 63],
                    "max_depth": [4, 6],
                    "learning_rate": [0.1, 0.2],
                    "n_estimators": [100, 200],
                    "min_child_samples": [20, 30],
                },
            },
            "feature_selection": {
                "enable": False,
                "method": "mutual_info",  # mutual_info, f_score, rfe
                "n_features": 50,
                "methods_available": ["mutual_info", "f_score", "rfe"],
            },
        }


def get_config():
    return ApplicationConfig()
