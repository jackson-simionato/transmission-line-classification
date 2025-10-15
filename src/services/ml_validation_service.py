"""
ML Validation Service for Transmission Line Classification
Handles validation of model predictions against ground truth data
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config.settings import get_config

logger = logging.getLogger(__name__)


class MLValidationService:
    """
    Service for validating model predictions against ground truth data
    """

    def __init__(self, ground_truth_path: str, predictions_dir: str):
        """
        Initialize validation service

        Args:
            ground_truth_path: Path to ground truth GeoJSON file
            predictions_dir: Directory containing prediction GeoJSON files
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.predictions_dir = Path(predictions_dir)
        self.config = get_config()
        self.ground_truth = None
        self.predictions = {}
        self.validation_results = {}
        self.available_models = []

        # Load ground truth data
        self._load_ground_truth()

        # Discover available prediction files
        self._discover_prediction_files()

    def _load_ground_truth(self):
        """Load ground truth GeoJSON data"""
        if not self.ground_truth_path.exists():
            raise FileNotFoundError(f"Ground truth file not found: {self.ground_truth_path}")

        self.ground_truth = gpd.read_file(self.ground_truth_path)
        logger.info(f"Loaded ground truth data: {self.ground_truth.shape}")
        logger.info(f"Ground truth columns: {list(self.ground_truth.columns)}")

    def _discover_prediction_files(self):
        """Discover available prediction files"""
        if not self.predictions_dir.exists():
            logger.warning(f"Predictions directory not found: {self.predictions_dir}")
            return

        # Look for prediction files
        prediction_files = list(self.predictions_dir.glob("*_predictions.geojson"))
        
        for pred_file in prediction_files:
            model_name = pred_file.stem.replace("_predictions", "")
            self.available_models.append(model_name)

        logger.info(f"Found prediction files for models: {self.available_models}")

    def load_predictions(self, model_name: str) -> gpd.GeoDataFrame:
        """
        Load predictions for a specific model

        Args:
            model_name: Name of the model

        Returns:
            GeoDataFrame with predictions
        """
        if model_name not in self.available_models:
            raise ValueError(f"Model '{model_name}' not available. Available models: {self.available_models}")

        if model_name in self.predictions:
            return self.predictions[model_name]

        pred_file = self.predictions_dir / f"{model_name}_predictions.geojson"
        
        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        predictions = gpd.read_file(pred_file)
        self.predictions[model_name] = predictions
        logger.info(f"Loaded predictions for {model_name}: {predictions.shape}")
        
        return predictions

    def validate_model(self, model_name: str) -> Dict[str, Any]:
        """
        Validate a specific model against ground truth

        Args:
            model_name: Name of the model to validate

        Returns:
            Dictionary with validation results
        """
        logger.info(f"Validating model: {model_name}")

        # Load predictions
        predictions = self.load_predictions(model_name)

        # Perform spatial join
        sjoin = gpd.sjoin(
            self.ground_truth, 
            predictions[['predicted_class', 'geometry']], 
            how='left', 
            predicate='intersects'
        )

        # Check for successful matches
        matched_count = sjoin['predicted_class'].notna().sum()
        total_truth = len(sjoin)
        match_rate = matched_count / total_truth if total_truth > 0 else 0

        logger.info(f"Spatial join results: {matched_count}/{total_truth} matched ({match_rate:.2%})")

        # Filter to only matched predictions
        matched_data = sjoin.dropna(subset=['predicted_class'])

        if len(matched_data) == 0:
            logger.warning(f"No matched predictions for {model_name}")
            return {
                "model_name": model_name,
                "success": False,
                "error": "No matched predictions found",
                "match_rate": 0,
                "total_ground_truth": total_truth,
                "matched_predictions": 0
            }

        # Get true and predicted labels
        y_true = matched_data['class_label']
        y_pred = matched_data['predicted_class']

        # Calculate success/failure
        success_mask = y_true == y_pred
        success_count = success_mask.sum()
        failure_count = len(success_mask) - success_count

        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Calculate precision, recall, f1-score
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        # Calculate macro and weighted averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )

        # Get unique classes present
        unique_classes = sorted(list(set(y_true.unique()) | set(y_pred.unique())))
        
        # Create per-class metrics
        class_metrics = {}
        for i, class_name in enumerate(unique_classes):
            if i < len(precision):
                class_metrics[class_name] = {
                    "precision": float(precision[i]),
                    "recall": float(recall[i]),
                    "f1_score": float(f1[i]),
                    "support": int(support[i])
                }

        # Create confusion matrix
        conf_matrix = confusion_matrix(y_true, y_pred, labels=unique_classes)
        confusion_matrix_df = pd.DataFrame(
            conf_matrix,
            index=[f"true_{cls}" for cls in unique_classes],
            columns=[f"pred_{cls}" for cls in unique_classes]
        )

        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            labels=unique_classes,
            target_names=unique_classes,
            output_dict=True,
            zero_division=0
        )

        # Create results dictionary
        results = {
            "model_name": model_name,
            "success": True,
            "match_rate": float(match_rate),
            "total_ground_truth": int(total_truth),
            "matched_predictions": int(matched_count),
            "success_count": int(success_count),
            "failure_count": int(failure_count),
            "accuracy": float(accuracy),
            "precision_macro": float(precision_macro),
            "recall_macro": float(recall_macro),
            "f1_macro": float(f1_macro),
            "precision_weighted": float(precision_weighted),
            "recall_weighted": float(recall_weighted),
            "f1_weighted": float(f1_weighted),
            "class_metrics": class_metrics,
            "confusion_matrix": conf_matrix.tolist(),
            "confusion_matrix_df": confusion_matrix_df.to_dict(),
            "classification_report": class_report,
            "unique_classes": unique_classes,
            "class_names": unique_classes
        }

        logger.info(f"Validation completed for {model_name}: Accuracy = {accuracy:.4f}")

        return results

    def validate_all_models(self) -> Dict[str, Dict[str, Any]]:
        """
        Validate all available models

        Returns:
            Dictionary with validation results for each model
        """
        logger.info("Validating all models...")
        results = {}

        for model_name in self.available_models:
            try:
                results[model_name] = self.validate_model(model_name)
            except Exception as e:
                logger.error(f"Failed to validate {model_name}: {str(e)}")
                results[model_name] = {
                    "model_name": model_name,
                    "success": False,
                    "error": str(e)
                }

        self.validation_results = results
        return results

    def generate_comparison_report(self) -> Dict[str, Any]:
        """
        Generate comparison report across all models

        Returns:
            Dictionary with comparison statistics
        """
        if not self.validation_results:
            logger.warning("No validation results available. Run validate_all_models() first.")
            return {}

        # Filter successful validations
        successful_results = {
            name: results for name, results in self.validation_results.items()
            if results.get("success", False)
        }

        if not successful_results:
            logger.warning("No successful validations found")
            return {"error": "No successful validations found"}

        # Create comparison DataFrame
        comparison_data = []
        for model_name, results in successful_results.items():
            comparison_data.append({
                "model": model_name,
                "accuracy": results["accuracy"],
                "precision_macro": results["precision_macro"],
                "recall_macro": results["recall_macro"],
                "f1_macro": results["f1_macro"],
                "precision_weighted": results["precision_weighted"],
                "recall_weighted": results["recall_weighted"],
                "f1_weighted": results["f1_weighted"],
                "match_rate": results["match_rate"],
                "matched_predictions": results["matched_predictions"]
            })

        comparison_df = pd.DataFrame(comparison_data).sort_values("accuracy", ascending=False)

        # Find best models
        best_overall = comparison_df.iloc[0]
        best_accuracy = comparison_df.loc[comparison_df["accuracy"].idxmax()]
        best_f1_macro = comparison_df.loc[comparison_df["f1_macro"].idxmax()]
        best_f1_weighted = comparison_df.loc[comparison_df["f1_weighted"].idxmax()]

        # Class-specific analysis
        class_analysis = {}
        for class_name in self.config.CLASS_NAMES:
            class_best_model = None
            class_best_f1 = 0
            
            for model_name, results in successful_results.items():
                if class_name in results["class_metrics"]:
                    f1_score = results["class_metrics"][class_name]["f1_score"]
                    if f1_score > class_best_f1:
                        class_best_f1 = f1_score
                        class_best_model = model_name
            
            class_analysis[class_name] = {
                "best_model": class_best_model,
                "best_f1": class_best_f1
            }

        # Model agreement analysis
        agreement_analysis = self._analyze_model_agreement(successful_results)

        comparison_report = {
            "summary": {
                "total_models": len(self.validation_results),
                "successful_models": len(successful_results),
                "failed_models": len(self.validation_results) - len(successful_results)
            },
            "best_models": {
                "overall": best_overall["model"],
                "accuracy": best_accuracy["model"],
                "f1_macro": best_f1_macro["model"],
                "f1_weighted": best_f1_weighted["model"]
            },
            "comparison_table": comparison_df.to_dict("records"),
            "class_analysis": class_analysis,
            "agreement_analysis": agreement_analysis,
            "statistics": {
                "accuracy_mean": comparison_df["accuracy"].mean(),
                "accuracy_std": comparison_df["accuracy"].std(),
                "f1_macro_mean": comparison_df["f1_macro"].mean(),
                "f1_macro_std": comparison_df["f1_macro"].std(),
                "f1_weighted_mean": comparison_df["f1_weighted"].mean(),
                "f1_weighted_std": comparison_df["f1_weighted"].std()
            }
        }

        return comparison_report

    def _analyze_model_agreement(self, successful_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze agreement between models"""
        if len(successful_results) < 2:
            return {"error": "Need at least 2 models for agreement analysis"}

        # This would require loading the actual predictions and comparing them
        # For now, return a placeholder
        return {
            "note": "Model agreement analysis requires loading prediction data",
            "models_compared": list(successful_results.keys())
        }

    def export_results(self, output_dir: str) -> Dict[str, str]:
        """
        Export validation results to files

        Args:
            output_dir: Directory to save results

        Returns:
            Dictionary with paths to exported files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        exported_files = {}

        # Export individual model results
        for model_name, results in self.validation_results.items():
            if results.get("success", False):
                # Export detailed results
                model_file = output_path / f"{model_name}_validation.json"
                with open(model_file, "w") as f:
                    json.dump(results, f, indent=2, default=str)
                exported_files[f"{model_name}_validation"] = str(model_file)

                # Export confusion matrix
                if "confusion_matrix_df" in results:
                    conf_matrix_file = output_path / f"{model_name}_confusion_matrix.csv"
                    confusion_df = pd.DataFrame(results["confusion_matrix_df"])
                    confusion_df.to_csv(conf_matrix_file, index=True)
                    exported_files[f"{model_name}_confusion_matrix"] = str(conf_matrix_file)

        # Export comparison report
        comparison_report = self.generate_comparison_report()
        if comparison_report:
            # Export comparison summary
            summary_file = output_path / "validation_summary.json"
            with open(summary_file, "w") as f:
                json.dump(comparison_report, f, indent=2, default=str)
            exported_files["validation_summary"] = str(summary_file)

            # Export comparison table
            if "comparison_table" in comparison_report:
                comparison_file = output_path / "model_comparison.csv"
                comparison_df = pd.DataFrame(comparison_report["comparison_table"])
                comparison_df.to_csv(comparison_file, index=False)
                exported_files["model_comparison"] = str(comparison_file)

        logger.info(f"Exported validation results to {output_path}")
        return exported_files

    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        return self.available_models.copy()

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of validation results"""
        if not self.validation_results:
            return {"error": "No validation results available"}

        successful_count = sum(1 for r in self.validation_results.values() if r.get("success", False))
        
        return {
            "total_models": len(self.validation_results),
            "successful_validations": successful_count,
            "failed_validations": len(self.validation_results) - successful_count,
            "available_models": self.available_models
        }
