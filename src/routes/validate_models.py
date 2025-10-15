"""
CLI route for validating model predictions against ground truth data
"""

import argparse
import json
import logging
from pathlib import Path

from src.config.settings import get_config
from src.services.ml_validation_service import MLValidationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function for model validation"""
    parser = argparse.ArgumentParser(
        description="Validate model predictions against ground truth data"
    )

    parser.add_argument(
        "--ground-truth",
        type=str,
        default="data/ground_truth/all_ground_truth_points.geojson",
        help="Path to ground truth GeoJSON file (default: data/ground_truth/all_ground_truth_points.geojson)",
    )

    parser.add_argument(
        "--predictions-dir",
        type=str,
        default="data/predictions",
        help="Directory containing prediction GeoJSON files (default: data/predictions)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=["random_forest", "xgboost", "svm", "lightgbm"],
        default=None,
        help="Specific models to validate (default: all available)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/validation",
        help="Output directory for validation results (default: data/validation)",
    )

    parser.add_argument(
        "--export-format",
        type=str,
        choices=["json", "csv", "html"],
        default="json",
        help="Export format for results (default: json)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with detailed statistics",
    )

    args = parser.parse_args()

    # Validate input files
    ground_truth_path = Path(args.ground_truth)
    predictions_dir = Path(args.predictions_dir)

    if not ground_truth_path.exists():
        logger.error(f"Ground truth file not found: {ground_truth_path}")
        return 1

    if not predictions_dir.exists():
        logger.error(f"Predictions directory not found: {predictions_dir}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Load configuration
        config = get_config()

        # Initialize validation service
        logger.info("Initializing ML Validation Service...")
        validation_service = MLValidationService(
            ground_truth_path=str(ground_truth_path),
            predictions_dir=str(predictions_dir)
        )

        # Get available models
        available_models = validation_service.get_available_models()
        if not available_models:
            logger.error("No prediction files found in the predictions directory")
            return 1

        # Filter models if specific ones requested
        if args.models:
            models_to_validate = [m for m in args.models if m in available_models]
            if not models_to_validate:
                logger.error(f"None of the requested models are available. Available: {available_models}")
                return 1
        else:
            models_to_validate = available_models

        logger.info(f"Validating models: {models_to_validate}")

        # Validate all models
        logger.info("Running validation...")
        validation_results = validation_service.validate_all_models()

        # Filter to requested models if specified
        if args.models:
            validation_results = {
                name: results for name, results in validation_results.items()
                if name in models_to_validate
            }

        # Print validation results
        print("\n" + "=" * 80)
        print("MODEL VALIDATION RESULTS")
        print("=" * 80)

        # Print individual model results
        for model_name, results in validation_results.items():
            if results.get("success", False):
                print(f"\n{model_name.upper()} Results:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  F1 Macro: {results['f1_macro']:.4f}")
                print(f"  F1 Weighted: {results['f1_weighted']:.4f}")
                print(f"  Match Rate: {results['match_rate']:.2%}")
                print(f"  Matched Predictions: {results['matched_predictions']}/{results['total_ground_truth']}")

                if args.verbose:
                    print(f"  Precision Macro: {results['precision_macro']:.4f}")
                    print(f"  Recall Macro: {results['recall_macro']:.4f}")
                    print(f"  Success Count: {results['success_count']}")
                    print(f"  Failure Count: {results['failure_count']}")

                    # Print per-class metrics
                    print("  Per-Class Metrics:")
                    for class_name, metrics in results['class_metrics'].items():
                        print(f"    {class_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")

            else:
                print(f"\n{model_name.upper()}: ERROR - {results.get('error', 'Unknown error')}")

        # Generate and print comparison report
        logger.info("Generating comparison report...")
        comparison_report = validation_service.generate_comparison_report()

        if comparison_report and "error" not in comparison_report:
            print(f"\n" + "=" * 80)
            print("MODEL COMPARISON")
            print("=" * 80)

            # Print best models
            best_models = comparison_report.get("best_models", {})
            print(f"\nBest Models:")
            print(f"  Overall: {best_models.get('overall', 'N/A')}")
            print(f"  Accuracy: {best_models.get('accuracy', 'N/A')}")
            print(f"  F1 Macro: {best_models.get('f1_macro', 'N/A')}")
            print(f"  F1 Weighted: {best_models.get('f1_weighted', 'N/A')}")

            # Print statistics
            stats = comparison_report.get("statistics", {})
            print(f"\nOverall Statistics:")
            print(f"  Mean Accuracy: {stats.get('accuracy_mean', 0):.4f} ± {stats.get('accuracy_std', 0):.4f}")
            print(f"  Mean F1 Macro: {stats.get('f1_macro_mean', 0):.4f} ± {stats.get('f1_macro_std', 0):.4f}")
            print(f"  Mean F1 Weighted: {stats.get('f1_weighted_mean', 0):.4f} ± {stats.get('f1_weighted_std', 0):.4f}")

            # Print class analysis
            class_analysis = comparison_report.get("class_analysis", {})
            if class_analysis:
                print(f"\nBest Model per Class:")
                for class_name, analysis in class_analysis.items():
                    print(f"  {class_name}: {analysis.get('best_model', 'N/A')} (F1={analysis.get('best_f1', 0):.3f})")

            # Print comparison table
            comparison_table = comparison_report.get("comparison_table", [])
            if comparison_table:
                print(f"\nDetailed Comparison:")
                print(f"{'Model':<15} {'Accuracy':<10} {'F1 Macro':<10} {'F1 Weighted':<12} {'Match Rate':<10}")
                print("-" * 70)
                for row in comparison_table:
                    print(f"{row['model']:<15} {row['accuracy']:<10.4f} {row['f1_macro']:<10.4f} {row['f1_weighted']:<12.4f} {row['match_rate']:<10.2%}")

        # Export results
        logger.info("Exporting validation results...")
        exported_files = validation_service.export_results(str(output_dir))

        print(f"\n" + "=" * 80)
        print("EXPORTED FILES")
        print("=" * 80)
        for file_type, file_path in exported_files.items():
            print(f"  {file_type}: {file_path}")

        # Print summary
        summary = validation_service.get_validation_summary()
        print(f"\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Total models: {summary['total_models']}")
        print(f"Successful validations: {summary['successful_validations']}")
        print(f"Failed validations: {summary['failed_validations']}")
        print(f"Output directory: {output_dir}")

        logger.info("Validation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit(main())
