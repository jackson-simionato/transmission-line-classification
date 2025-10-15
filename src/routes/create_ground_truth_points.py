"""
CLI script for generating ground truth points across all processed tiles
Creates ground truth points for manual labeling in QGIS
"""

import argparse
import logging
from pathlib import Path

from src.config.settings import get_config
from src.services.ground_truth_service import GroundTruthService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description="Generate spatially distributed ground truth points for ground truth labeling"
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/sam_segments/processed"),
        help="Directory with processed segments (default: data/sam_segments/processed)",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/ground_truth"),
        help="Directory for ground truth points (default: data/ground_truth)",
    )

    parser.add_argument(
        "--points-per-tile",
        type=int,
        default=15,
        help="Number of points per tile (default: 15)",
    )

    parser.add_argument(
        "--min-distance",
        type=float,
        default=10.0,
        help="Minimum distance between points in meters (default: 10.0)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate input directory
    if not args.input_dir.exists():
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return 1

    # Get configuration
    config = get_config()

    # Initialize ground truth service
    sampling_service = GroundTruthService(
        min_distance_meters=args.min_distance, random_seed=args.seed
    )

    logger.info("=" * 60)
    logger.info("GROUND TRUTH POINTS GENERATION")
    logger.info("=" * 60)
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Points per tile: {args.points_per_tile}")
    logger.info(f"Minimum distance: {args.min_distance}m")
    logger.info(f"Random seed: {args.seed}")
    logger.info("=" * 60)

    try:
        # Generate ground truth points
        summary = sampling_service.generate_all_tiles_samples(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            points_per_tile=args.points_per_tile,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("GROUND TRUTH POINTS GENERATION COMPLETE")
        print("=" * 60)
        print(
            f"Tiles processed: {summary['successful_tiles']}/{summary['total_tiles']}"
        )
        print(f"Total points generated: {summary['total_points']}")
        print(
            f"Average points per tile: {summary['total_points'] / max(summary['successful_tiles'], 1):.1f}"
        )
        print(f"Output directory: {args.output_dir}")
        print("\nNext steps:")
        print("1. Open 'all_ground_truth_points.geojson' in QGIS")
        print("2. Load orthophoto tiles as background")
        print("3. Manually label each point using the 'class_label' attribute")
        print("4. Save labeled points for training")
        print("=" * 60)

        return 0

    except Exception as e:
        logger.error(f"Error during ground truth points generation: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
