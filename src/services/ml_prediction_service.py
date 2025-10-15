"""
ML Prediction Service for Transmission Line Classification
Handles model loading and prediction for trained ML models
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config.settings import get_config

logger = logging.getLogger(__name__)


class MLPredictionService:
    """
    Service for loading trained models and making predictions
    """

    def __init__(self, model_dir: str = "models"):
        """
        Initialize prediction service

        Args:
            model_dir: Path to directory containing trained models
        """
        self.model_dir = Path(model_dir)
        self.config = get_config()
        self.models = {}
        self.scaler = None
        self.feature_selector = None
        self.preprocessing_info = None
        self.available_models = []

        # Load preprocessing artifacts
        self._load_preprocessing_artifacts()

        # Discover available models
        self._discover_available_models()

    def _load_preprocessing_artifacts(self):
        """Load scaler and feature selector from preprocessing directory"""
        preprocessing_dir = self.model_dir / "preprocessing"

        # Load scaler
        scaler_path = preprocessing_dir / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")

        # Load feature selector
        feature_selector_path = preprocessing_dir / "feature_selector.joblib"
        if feature_selector_path.exists():
            self.feature_selector = joblib.load(feature_selector_path)
            logger.info(f"Loaded feature selector from {feature_selector_path}")
        else:
            logger.info("No feature selector found - using all features")

        # Load preprocessing info
        evaluation_dir = self.model_dir / "evaluation"
        preprocessing_info_path = evaluation_dir / "preprocessing_info.json"
        if preprocessing_info_path.exists():
            import json
            with open(preprocessing_info_path, "r") as f:
                self.preprocessing_info = json.load(f)
            logger.info(f"Loaded preprocessing info from {preprocessing_info_path}")

    def _discover_available_models(self):
        """Discover available trained models"""
        trained_models_dir = self.model_dir / "trained_models"
        
        if not trained_models_dir.exists():
            logger.warning(f"Trained models directory not found: {trained_models_dir}")
            return

        # Look for model files
        model_files = list(trained_models_dir.glob("*_model.joblib"))
        
        for model_file in model_files:
            model_name = model_file.stem.replace("_model", "")
            self.available_models.append(model_name)

        logger.info(f"Found available models: {self.available_models}")

    def load_model(self, model_name: str) -> Any:
        """
        Load a specific trained model

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded model object
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {self.available_models}")

        if model_name in self.models:
            return self.models[model_name]

        model_path = self.model_dir / "trained_models" / f"{model_name}_model.joblib"
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        self.models[model_name] = model
        logger.info(f"Loaded model: {model_name}")
        
        return model

    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess features using the same pipeline as training

        Args:
            features_df: DataFrame with features

        Returns:
            Preprocessed feature array
        """
        if self.scaler is None:
            raise ValueError("Scaler not loaded. Cannot preprocess features.")

        # Handle NaN values (fill with median for each column)
        logger.info(f"Data shape before NaN handling: {features_df.shape}")
        logger.info(f"NaN values per column: {features_df.isnull().sum().sum()}")
        features_clean = features_df.fillna(features_df.median())
        logger.info(f"Data shape after NaN handling: {features_clean.shape}")

        # Scale the features using the same scaler used during training
        features_scaled = self.scaler.transform(features_clean)

        # Apply feature selection if it was used during training
        if self.feature_selector is not None:
            features_scaled = self.feature_selector.transform(features_scaled)
            logger.info(f"Applied feature selection: {features_scaled.shape[1]} features selected")

        return features_scaled

    def predict(self, features_df: pd.DataFrame, model_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using a specific model

        Args:
            features_df: DataFrame with features
            model_name: Name of the model to use

        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        # Load model if not already loaded
        model = self.load_model(model_name)

        # Preprocess features
        features_processed = self.preprocess_features(features_df)

        # Make predictions
        predictions = model.predict(features_processed)

        # Get prediction probabilities if available
        try:
            prediction_probs = model.predict_proba(features_processed)
        except AttributeError:
            # Some models don't have predict_proba
            prediction_probs = None
            logger.warning(f"Model {model_name} does not support probability predictions")

        return predictions, prediction_probs

    def predict_all_models(self, features_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Make predictions using all available models

        Args:
            features_df: DataFrame with features

        Returns:
            Dictionary with predictions for each model
        """
        results = {}

        for model_name in self.available_models:
            try:
                predictions, prediction_probs = self.predict(features_df, model_name)
                
                # Convert predictions to class names
                prediction_classes = [self.config.CLASS_NAMES[pred] for pred in predictions]
                
                results[model_name] = {
                    "predictions": predictions,
                    "prediction_classes": prediction_classes,
                    "prediction_probabilities": prediction_probs,
                    "success": True
                }
                
                logger.info(f"Successfully predicted with {model_name}: {len(predictions)} samples")
                
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {str(e)}")
                results[model_name] = {
                    "error": str(e),
                    "success": False
                }

        return results

    def get_prediction_summary(self, prediction_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate summary statistics for predictions

        Args:
            prediction_results: Results from predict_all_models

        Returns:
            Summary statistics
        """
        summary = {
            "total_samples": 0,
            "models": {},
            "overall_success_rate": 0
        }

        successful_models = 0
        total_models = len(prediction_results)

        for model_name, results in prediction_results.items():
            if results.get("success", False):
                predictions = results["prediction_classes"]
                summary["total_samples"] = len(predictions)
                
                # Count predictions per class
                class_counts = {}
                for class_name in self.config.CLASS_NAMES:
                    class_counts[class_name] = predictions.count(class_name)
                
                summary["models"][model_name] = {
                    "class_distribution": class_counts,
                    "total_predictions": len(predictions)
                }
                successful_models += 1
            else:
                summary["models"][model_name] = {
                    "error": results.get("error", "Unknown error"),
                    "success": False
                }

        summary["overall_success_rate"] = successful_models / total_models if total_models > 0 else 0

        return summary

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models and preprocessing"""
        return {
            "available_models": self.available_models,
            "scaler_loaded": self.scaler is not None,
            "feature_selector_loaded": self.feature_selector is not None,
            "preprocessing_info_loaded": self.preprocessing_info is not None,
            "loaded_models": list(self.models.keys())
        }
