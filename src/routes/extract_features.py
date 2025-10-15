"""
CLI route for extracting features from labeled polygon segments
"""

import argparse
import logging
from pathlib import Path

from src.services.feature_extraction_service import FeatureExtractionService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function for feature extraction"""
    parser = argparse.ArgumentParser(
        description="Extract features from labeled polygon segments"
    )

    parser.add_argument(
        "--labels-dir",
        type=str,
        default="data/labels",
        help="Directory with labeled GeoJSON files (default: data/labels)",
    )

    parser.add_argument(
        "--images-dir",
        type=str,
        default="data/ortofoto/raw",
        help="Directory with ortofoto .tif images (default: data/ortofoto/raw)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/features",
        help="Output directory for feature files (default: data/features)",
    )

    parser.add_argument(
        "--include-cnn",
        action="store_true",
        help="Include CNN embeddings (memory intensive)",
    )

    parser.add_argument(
        "--cnn-model",
        type=str,
        choices=["mobilenet_v3_small", "efficientnet_b0"],
        default="mobilenet_v3_small",
        help="CNN model for embeddings (default: mobilenet_v3_small)",
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test", "all"],
        default="all",
        help="Process specific split (default: all)",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        default=None,
        help="Device for CNN inference (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # Convert paths
    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    output_dir = Path(args.output_dir)

    # Validate input directories
    if not labels_dir.exists():
        logger.error(f"Labels directory not found: {labels_dir}")
        return 1

    if not images_dir.exists():
        logger.error(f"Images directory not found: {images_dir}")
        return 1

    # Initialize feature extraction service
    try:
        service = FeatureExtractionService(
            include_cnn=args.include_cnn, cnn_model=args.cnn_model, device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize feature extraction service: {e}")
        return 1

    # Process dataset
    try:
        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Labels directory: {labels_dir}")
        logger.info(f"Images directory: {images_dir}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Split: {args.split}")
        logger.info(f"CNN embeddings: {'enabled' if args.include_cnn else 'disabled'}")
        if args.include_cnn:
            logger.info(f"CNN model: {args.cnn_model}")
        logger.info("=" * 60)

        stats = service.process_dataset(
            labels_dir=labels_dir,
            images_dir=images_dir,
            output_dir=output_dir,
            split=args.split if args.split != "all" else None,
        )

        logger.info("=" * 60)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Processed tiles: {stats['processed_tiles']}")
        logger.info(f"Total polygons: {stats['total_polygons']}")
        if "output_file" in stats:
            logger.info(f"Output file: {stats['output_file']}")
        logger.info("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during feature extraction: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
