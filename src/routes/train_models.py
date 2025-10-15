"""
CLI route for training ML models for transmission line classification
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.config.settings import get_config
from src.services.ml_training_service import MLTrainingService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function for ML model training"""
    parser = argparse.ArgumentParser(
        description="Train ML models for transmission line classification"
    )

    parser.add_argument(
        "--train-data",
        type=str,
        default="data/features/features_train.csv",
        help="Path to training data CSV file (default: data/features/features_train.csv)",
    )

    parser.add_argument(
        "--test-data",
        type=str,
        default="data/features/features_test.csv",
        help="Path to test data CSV file (default: data/features/features_test.csv)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Output directory for trained models and results (default: models)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "svm"],
        default=["random_forest", "xgboost", "svm"],
        help="Models to train (default: all)",
    )

    parser.add_argument(
        "--target-column",
        type=str,
        default="class_label",
        help="Name of target column (default: class_label)",
    )

    parser.add_argument(
        "--save-models", action="store_true", help="Save trained models to disk"
    )

    parser.add_argument(
        "--save-results", action="store_true", help="Save evaluation results to JSON"
    )

    parser.add_argument(
        "--grid-search",
        action="store_true",
        help="Enable grid search hyperparameter optimization",
    )

    args = parser.parse_args()

    # Validate input files
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)

    if not train_path.exists():
        logger.error(f"Training data file not found: {train_path}")
        return 1

    if not test_path.exists():
        logger.error(f"Test data file not found: {test_path}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        config = get_config()

        # Load data
        logger.info(f"Loading training data from {train_path}")
        train_df = pd.read_csv(train_path)

        logger.info(f"Loading test data from {test_path}")
        test_df = pd.read_csv(test_path)

        # Filter data to only include valid classes
        valid_classes = config.CLASS_NAMES
        train_df = train_df[train_df[args.target_column].isin(valid_classes)]
        test_df = test_df[test_df[args.target_column].isin(valid_classes)]

        logger.info(f"Training data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")

        # Convert class labels to numeric
        train_df[args.target_column] = train_df[args.target_column].map(config.CLASSES)
        test_df[args.target_column] = test_df[args.target_column].map(config.CLASSES)

        # Initialize training service
        logger.info("Initializing ML Training Service...")
        training_service = MLTrainingService(
            config=config.ML_TRAINING, enable_grid_search=args.grid_search
        )

        # Preprocess data
        logger.info("Preprocessing data...")
        X_train, X_test, y_train, y_test, preprocessing_info = (
            training_service.preprocess_data(train_df, test_df, args.target_column)
        )

        # Train models
        logger.info(f"Training models: {args.models}")
        trained_models = training_service.train_models(
            X_train, y_train, X_test, y_test, preprocessing_info, args.models
        )

        # Evaluate models
        logger.info("Evaluating models...")
        evaluation_results = training_service.evaluate_models(
            trained_models, X_test, y_test, config.CLASS_NAMES
        )

        # Print results
        print("\n" + "=" * 80)
        print("MODEL TRAINING RESULTS")
        print("=" * 80)

        # Grid search results
        if args.grid_search:
            grid_results_df = training_service.get_best_parameters()
            if not grid_results_df.empty:
                print("\nGrid Search - Best Parameters:")
                print(grid_results_df.to_string(index=False, float_format="%.4f"))

        # Model comparison
        comparison_df = training_service.get_model_comparison()
        if not comparison_df.empty:
            print("\nModel Comparison:")
            print(comparison_df.to_string(index=False, float_format="%.3f"))

        # Detailed results for each model
        for model_name, results in evaluation_results.items():
            if "error" in results:
                print(f"\n{model_name.upper()}: ERROR - {results['error']}")
                continue

            print(f"\n{model_name.upper()} Results:")
            print(f"  Accuracy: {results['accuracy']:.3f}")
            print(f"  CV Score: {results['cv_mean']:.3f} ± {results['cv_std']:.3f}")

            # Feature importance
            if results.get("feature_importance"):
                feature_names = preprocessing_info["feature_columns"]
                importance_df = training_service.get_feature_importance(
                    model_name, feature_names
                )
                if not importance_df.empty:
                    print("  Top 5 Features:")
                    for _, row in importance_df.head(5).iterrows():
                        print(f"    {row['feature']}: {row['importance']:.4f}")

        # Save models if requested
        if args.save_models:
            logger.info("Saving trained models...")
            import joblib

            for model_name, model in trained_models.items():
                model_path = output_dir / f"{model_name}_model.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} to {model_path}")

            # Save scaler
            scaler_path = output_dir / "scaler.joblib"
            joblib.dump(training_service.scaler, scaler_path)
            logger.info(f"Saved scaler to {scaler_path}")

        # Save results if requested
        if args.save_results:
            logger.info("Saving evaluation results...")

            # Save evaluation results
            results_path = output_dir / "evaluation_results.json"
            with open(results_path, "w") as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            logger.info(f"Saved evaluation results to {results_path}")

            # Save preprocessing info
            preprocessing_path = output_dir / "preprocessing_info.json"
            with open(preprocessing_path, "w") as f:
                json.dump(preprocessing_info, f, indent=2, default=str)
            logger.info(f"Saved preprocessing info to {preprocessing_path}")

            # Save model comparison
            comparison_path = output_dir / "model_comparison.csv"
            comparison_df.to_csv(comparison_path, index=False)
            logger.info(f"Saved model comparison to {comparison_path}")

            # Save grid search results if available
            if args.grid_search:
                grid_results = training_service.get_grid_search_results()
                if grid_results:
                    grid_results_path = output_dir / "grid_search_results.json"
                    with open(grid_results_path, "w") as f:
                        json.dump(grid_results, f, indent=2, default=str)
                    logger.info(f"Saved grid search results to {grid_results_path}")

                    best_params_df = training_service.get_best_parameters()
                    if not best_params_df.empty:
                        best_params_path = output_dir / "best_parameters.csv"
                        best_params_df.to_csv(best_params_path, index=False)
                        logger.info(f"Saved best parameters to {best_params_path}")

        logger.info("Training completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
