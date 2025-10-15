"""
CLI route for predicting on GeoJSON files using trained ML models
"""

import argparse
import json
import logging
from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config.settings import get_config
from src.services.ml_prediction_service import MLPredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function for GeoJSON prediction"""
    parser = argparse.ArgumentParser(
        description="Predict transmission line classes on GeoJSON files using trained ML models"
    )

    parser.add_argument(
        "--features",
        type=str,
        default="data/features/features_all.csv",
        help="Path to features CSV file (default: data/features/features_all.csv)",
    )

    parser.add_argument(
        "--geojson",
        type=str,
        default="data/labels/all_labels.geojson",
        help="Path to input GeoJSON file (default: data/labels/all_labels.geojson)",
    )

    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Path to model directory (default: models)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "svm", "lightgbm"],
        default=None,
        help="Specific models to use (default: all available)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/predictions",
        help="Output directory for predictions (default: data/predictions)",
    )

    parser.add_argument(
        "--id-column",
        type=str,
        default="id",
        help="Column name for joining features and GeoJSON (default: id)",
    )

    args = parser.parse_args()

    # Validate input files
    features_path = Path(args.features)
    geojson_path = Path(args.geojson)
    model_dir = Path(args.model_dir)

    if not features_path.exists():
        logger.error(f"Features file not found: {features_path}")
        return 1

    if not geojson_path.exists():
        logger.error(f"GeoJSON file not found: {geojson_path}")
        return 1

    if not model_dir.exists():
        logger.error(f"Model directory not found: {model_dir}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        config = get_config()

        # Initialize prediction service
        logger.info("Initializing ML Prediction Service...")
        prediction_service = MLPredictionService(model_dir=str(model_dir))

        # Get available models
        available_models = prediction_service.get_available_models()
        if not available_models:
            logger.error("No trained models found in the model directory")
            return 1

        # Filter models if specific ones requested
        if args.models:
            models_to_use = [m for m in args.models if m in available_models]
            if not models_to_use:
                logger.error(f"None of the requested models are available. Available: {available_models}")
                return 1
        else:
            models_to_use = available_models

        logger.info(f"Using models: {models_to_use}")

        # Load features
        logger.info(f"Loading features from {features_path}")
        features_df = pd.read_csv(features_path)

        # Prepare features (exclude non-feature columns)
        feature_columns = [
            col for col in features_df.columns 
            if col not in ["class_label", "segment_id", "tile_id", args.id_column]
        ]
        X = features_df[feature_columns]

        logger.info(f"Features shape: {X.shape}")
        logger.info(f"Using {len(feature_columns)} features")

        # Load GeoJSON
        logger.info(f"Loading GeoJSON from {geojson_path}")
        gdf = gpd.read_file(geojson_path)

        logger.info(f"GeoJSON shape: {gdf.shape}")
        logger.info(f"GeoJSON columns: {list(gdf.columns)}")

        # Make predictions for each model
        all_results = {}
        prediction_summary = {}

        for model_name in models_to_use:
            logger.info(f"Making predictions with {model_name}...")
            
            try:
                # Make predictions
                predictions, prediction_probs = prediction_service.predict(X, model_name)
                
                # Convert predictions to class names
                prediction_classes = [config.CLASS_NAMES[pred] for pred in predictions]
                
                # Add predictions to features dataframe
                features_with_predictions = features_df.copy()
                features_with_predictions[f"{model_name}_predicted_class_num"] = predictions
                features_with_predictions[f"{model_name}_predicted_class"] = prediction_classes
                
                # Merge with GeoJSON
                gdf_with_predictions = pd.merge(
                    gdf, 
                    features_with_predictions[[args.id_column, f"{model_name}_predicted_class"]], 
                    on=args.id_column, 
                    how="left"
                )
                
                # Save GeoJSON with predictions
                output_file = output_dir / f"{model_name}_predictions.geojson"
                gdf_with_predictions.to_file(output_file, driver="GeoJSON")
                logger.info(f"Saved {model_name} predictions to {output_file}")
                
                # Store results for summary
                all_results[model_name] = {
                    "predictions": predictions,
                    "prediction_classes": prediction_classes,
                    "prediction_probabilities": prediction_probs,
                    "output_file": str(output_file),
                    "success": True
                }
                
                # Print prediction distribution
                print(f"\n{model_name.upper()} Prediction Distribution:")
                class_counts = pd.Series(prediction_classes).value_counts()
                for class_name, count in class_counts.items():
                    print(f"  {class_name}: {count}")
                
            except Exception as e:
                logger.error(f"Failed to predict with {model_name}: {str(e)}")
                all_results[model_name] = {
                    "error": str(e),
                    "success": False
                }

        # Generate prediction summary
        prediction_summary = prediction_service.get_prediction_summary(all_results)
        
        # Add metadata
        prediction_summary["metadata"] = {
            "features_file": str(features_path),
            "geojson_file": str(geojson_path),
            "model_directory": str(model_dir),
            "output_directory": str(output_dir),
            "id_column": args.id_column,
            "total_features": len(feature_columns),
            "total_samples": len(features_df)
        }

        # Save prediction summary
        summary_path = output_dir / "prediction_summary.json"
        with open(summary_path, "w") as f:
            json.dump(prediction_summary, f, indent=2, default=str)
        logger.info(f"Saved prediction summary to {summary_path}")

        # Print overall summary
        print("\n" + "=" * 80)
        print("PREDICTION SUMMARY")
        print("=" * 80)
        print(f"Total samples: {prediction_summary['total_samples']}")
        print(f"Successful models: {prediction_summary['overall_success_rate']:.1%}")
        print(f"Output directory: {output_dir}")
        
        print("\nModel Results:")
        for model_name, results in all_results.items():
            if results.get("success", False):
                print(f"  ✅ {model_name}: {len(results['prediction_classes'])} predictions")
            else:
                print(f"  ❌ {model_name}: {results.get('error', 'Unknown error')}")

        logger.info("Prediction completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
