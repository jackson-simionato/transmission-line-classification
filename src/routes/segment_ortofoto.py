"""
Route for segmenting orthophoto tiles using SAM
Specific logic for transmission line classification project
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rasterio
from tqdm import tqdm

try:
    import geopandas as gpd
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import shape

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from src.services.preprocessing_service import ImagePreProcessingService
from src.services.sam_segmentation_service import SAMSegmentationService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OrtofotoSegmentationPipeline:
    """
    Pipeline for segmenting orthophoto tiles and exporting georeferenced polygons
    Optimized for 1666x1666 pixel tiles
    """

    def __init__(
        self,
        sam_service: SAMSegmentationService,
        output_base_dir: str = "data/sam_segments",
        enable_preprocessing: bool = True,
        max_area: Optional[int] = None,
        force: bool = False,
    ):
        """
        Initialize segmentation pipeline

        Args:
            sam_service: Initialized SAMSegmentationService
            output_base_dir: Base directory for outputs
            enable_preprocessing: Whether to apply image preprocessing
            max_area: Maximum segment area in pixels (filters huge segments)
            force: Whether to recreate existing segmentation results
        """
        self.sam_service = sam_service
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        self.max_area = max_area
        self.force = force

        # Initialize preprocessing service
        self.enable_preprocessing = enable_preprocessing
        self.clahe_clip_limit = 2.0  # Default, can be overridden
        if self.enable_preprocessing:
            self.preprocessor = ImagePreProcessingService(
                enable_denoising=False,  # Disable - original images are clean
                enable_clahe=True,  # Keep - this provides QGIS-like contrast
                enable_sharpening=False,  # Disable - not needed for contrast
                clahe_clip_limit=self.clahe_clip_limit,
                clahe_tile_size=(8, 8),  # Standard tile size
            )
            logger.info(
                f"QGIS-style contrast enhancement enabled (clip_limit={self.clahe_clip_limit})"
            )
        else:
            self.preprocessor = None
            logger.info("Image preprocessing disabled")

        if self.max_area:
            logger.info(f"Maximum segment area filter: {self.max_area} pixels")

        logger.info("Ortofoto segmentation pipeline initialized")
        logger.info(f"Output directory: {self.output_base_dir}")
        if self.force:
            logger.info("Force mode enabled - will recreate existing segmentations")
        else:
            logger.info("Skip mode enabled - will skip existing segmentations")

    def _tile_exists(self, tile_name: str) -> bool:
        """
        Check if segmentation results already exist for a tile

        Args:
            tile_name: Name of the tile (without extension)

        Returns:
            True if segmentation results exist, False otherwise
        """
        tile_output_dir = self.output_base_dir / tile_name

        # Check if the tile directory exists and has the required files
        if not tile_output_dir.exists():
            return False

        # Check for key output files
        required_files = ["segments.geojson", "metadata.json", "visualization.png"]

        return all((tile_output_dir / file).exists() for file in required_files)

    def segment_tile(
        self, tile_path: Path, save_outputs: bool = True, return_segments: bool = False
    ) -> tuple[Optional[List[Dict]], Dict]:
        """
        Segment a single ortofoto tile (memory optimized)

        Args:
            tile_path: Path to GeoTIFF tile
            save_outputs: Whether to save visualization and metadata
            return_segments: Whether to return segments (uses more memory)

        Returns:
            (segments or None, metadata) tuple
        """
        tile_path = Path(tile_path)

        if not tile_path.exists():
            raise FileNotFoundError(f"Tile not found: {tile_path}")

        tile_name = tile_path.stem

        # Check if tile already exists and skip if not forced
        if not self.force and self._tile_exists(tile_name):
            logger.info(f"Skipping existing tile: {tile_path.name}")
            # Return dummy metadata for consistency
            return None, {
                "tile_name": tile_name,
                "skipped": True,
                "reason": "already_exists",
            }

        logger.info(f"Processing tile: {tile_path.name}")

        # Read tile with rasterio to preserve georeference
        with rasterio.open(tile_path) as src:
            # Read RGB bands
            image = src.read([1, 2, 3])
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

            # Store georef metadata (avoid storing full transform object)
            metadata = {
                "transform": src.transform,
                "crs": src.crs,
                "width": src.width,
                "height": src.height,
                "bounds": src.bounds,
                "tile_name": tile_path.stem,
            }

        logger.info(f"  Image shape: {image.shape}, CRS: {metadata['crs']}")

        # Apply preprocessing if enabled
        preprocessing_stats = None
        if self.enable_preprocessing and self.preprocessor is not None:
            logger.info("  Applying image preprocessing...")
            image, preprocessing_stats = self.preprocessor.preprocess(
                image, return_stats=True
            )

            # Log preprocessing improvements
            if preprocessing_stats:
                improvements = preprocessing_stats["improvements"]
                logger.info(
                    f"  QGIS-style contrast enhancement: "
                    f"contrast_ratio={improvements['contrast_ratio']:.2f}x, "
                    f"brightness_change={improvements['brightness_change']:.1f}, "
                    f"std_change={improvements['std_change']:.1f}"
                )

        # Segment with SAM
        logger.info("  Running SAM segmentation...")
        max_area = getattr(self, "max_area", None)
        segments = self.sam_service.segment_image(image, max_area=max_area)
        logger.info(f"  Generated {len(segments)} segments")

        # Free image memory immediately after segmentation
        del image

        # Add tile metadata to segments
        for i, seg in enumerate(segments):
            seg["segment_id"] = i
            seg["tile_name"] = tile_path.stem

        # Save outputs
        if save_outputs:
            tile_output_dir = self.output_base_dir / tile_path.stem
            tile_output_dir.mkdir(parents=True, exist_ok=True)

            # Need to reload image only for visualization
            with rasterio.open(tile_path) as src:
                image_for_vis = src.read([1, 2, 3])
                image_for_vis = np.transpose(image_for_vis, (1, 2, 0))

            # Apply same preprocessing to visualization image
            if self.enable_preprocessing and self.preprocessor is not None:
                image_for_vis = self.preprocessor.preprocess(
                    image_for_vis, return_stats=False
                )

            # Visualization
            vis_path = tile_output_dir / "visualization.png"
            self._save_visualization(image_for_vis, segments, vis_path)
            logger.info(f"  Saved visualization: {vis_path.name}")

            # Free visualization image
            del image_for_vis

            # Metadata JSON (without storing large mask arrays)
            meta_path = tile_output_dir / "metadata.json"
            self._save_metadata(metadata, segments, meta_path, preprocessing_stats)
            logger.info(f"  Saved metadata: {meta_path.name}")

            # Export to GeoJSON if available
            if GEOPANDAS_AVAILABLE:
                geojson_path = tile_output_dir / "segments.geojson"
                self.export_to_geojson(segments, metadata, geojson_path)
                logger.info(f"  Exported GeoJSON: {geojson_path.name}")

        # Return segments only if requested, otherwise free memory
        if return_segments:
            return segments, metadata
        else:
            # Extract only essential stats before freeing segments
            stats = {
                "num_segments": len(segments),
                "mean_area": float(np.mean([s["area"] for s in segments])),
                "mean_iou": float(np.mean([s["predicted_iou"] for s in segments])),
                "mean_stability": float(
                    np.mean([s["stability_score"] for s in segments])
                ),
            }
            del segments
            return None, {**metadata, "stats": stats}

    def process_directory(
        self,
        tiles_dir: Path,
        pattern: str = "*.tif",
        create_summary: bool = True,
        batch_size: int = 1,
    ) -> Dict:
        """
        Process all tiles in a directory (memory optimized)

        Args:
            tiles_dir: Directory containing ortofoto tiles
            pattern: File pattern to match
            create_summary: Create summary statistics
            batch_size: Number of tiles to process before clearing memory (1=safest)

        Returns:
            Summary statistics dictionary
        """
        import gc  # For explicit garbage collection

        tiles_dir = Path(tiles_dir)
        tile_files = sorted(list(tiles_dir.glob(pattern)))

        if not tile_files:
            raise ValueError(f"No tiles found in {tiles_dir} with pattern '{pattern}'")

        logger.info("=" * 60)
        logger.info("ORTOFOTO SEGMENTATION PIPELINE (MEMORY OPTIMIZED)")
        logger.info("=" * 60)
        logger.info(f"Input: {tiles_dir}")
        logger.info(f"Output: {self.output_base_dir}")
        logger.info(f"Tiles found: {len(tile_files)}")
        logger.info(f"SAM model: {self.sam_service.model_type}")
        logger.info(f"Batch size: {batch_size}")
        logger.info("=" * 60)

        # Process each tile WITHOUT storing all segments in memory
        tile_stats = []
        processed_count = 0
        skipped_count = 0

        for idx, tile_path in enumerate(tqdm(tile_files, desc="Processing tiles"), 1):
            try:
                # Process tile without keeping segments in memory
                segments, metadata = self.segment_tile(
                    tile_path,
                    save_outputs=True,
                    return_segments=False,  # Don't return segments to save memory
                )

                # Check if tile was skipped
                if metadata.get("skipped", False):
                    skipped_count += 1
                    continue

                # Extract stats from metadata
                stats = metadata.get("stats", {})
                tile_stats.append({"tile_name": tile_path.stem, **stats})

                processed_count += 1

                # Explicit garbage collection every batch_size tiles
                if idx % batch_size == 0:
                    gc.collect()
                    logger.debug(f"Memory cleared after {idx} tiles")

            except Exception as e:
                logger.error(f"Error processing {tile_path.name}: {e}", exc_info=True)
                continue

        # Final garbage collection
        gc.collect()

        # Create summary
        summary = {
            "total_tiles_processed": processed_count,
            "total_tiles_skipped": skipped_count,
            "total_tiles_found": len(tile_files),
            "total_segments": sum(s["num_segments"] for s in tile_stats),
            "tiles": tile_stats,
        }

        if tile_stats:
            summary["global_stats"] = {
                "mean_segments_per_tile": float(
                    np.mean([s["num_segments"] for s in tile_stats])
                ),
                "mean_segment_area": float(
                    np.mean([s["mean_area"] for s in tile_stats])
                ),
                "mean_iou": float(np.mean([s["mean_iou"] for s in tile_stats])),
                "mean_stability": float(
                    np.mean([s["mean_stability"] for s in tile_stats])
                ),
            }

        # Save summary
        if create_summary:
            summary_dir = self.output_base_dir / "summary"
            summary_dir.mkdir(exist_ok=True)

            summary_path = summary_dir / "statistics.json"
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

            logger.info(f"Summary saved: {summary_path}")

        # Log final stats
        logger.info("=" * 60)
        logger.info("SEGMENTATION COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Processed: {summary['total_tiles_processed']}/{summary['total_tiles_found']} tiles"
        )
        if skipped_count > 0:
            logger.info(f"Skipped: {skipped_count} tiles (already exist)")
        logger.info(f"Total segments: {summary['total_segments']}")
        if "global_stats" in summary:
            logger.info(
                f"Avg segments/tile: {summary['global_stats']['mean_segments_per_tile']:.1f}"
            )
            logger.info(
                f"Avg segment area: {summary['global_stats']['mean_segment_area']:.1f} pixels"
            )
            logger.info(f"Avg IoU: {summary['global_stats']['mean_iou']:.3f}")
        logger.info("=" * 60)

        return summary

    def export_to_geojson(
        self, segments: List[Dict], metadata: Dict, output_path: Path
    ):
        """
        Export segments as georeferenced GeoJSON

        Args:
            segments: List of segment dictionaries from SAM
            metadata: Tile metadata with CRS and transform
            output_path: Path to save GeoJSON
        """
        if not GEOPANDAS_AVAILABLE:
            logger.warning("geopandas not available, skipping GeoJSON export")
            return

        polygons = []

        for segment in segments:
            # Convert binary mask to polygon
            mask = segment["segmentation"].astype(np.uint8)

            # Use rasterio to convert mask to georeferenced polygon
            for geom, value in rasterio_shapes(mask, transform=metadata["transform"]):
                if value == 1:  # Segment polygon
                    polygons.append(
                        {
                            "geometry": shape(geom),
                            "segment_id": segment["segment_id"],
                            "area_pixels": int(segment["area"]),
                            "predicted_iou": float(segment["predicted_iou"]),
                            "stability_score": float(segment["stability_score"]),
                            "bbox_x": int(segment["bbox"][0]),
                            "bbox_y": int(segment["bbox"][1]),
                            "bbox_width": int(segment["bbox"][2]),
                            "bbox_height": int(segment["bbox"][3]),
                            "tile_name": metadata["tile_name"],
                        }
                    )

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(polygons, crs=metadata["crs"])

        # Save
        output_path = Path(output_path)
        gdf.to_file(output_path, driver="GeoJSON")

        logger.debug(f"Exported {len(polygons)} polygons to GeoJSON")

    def _save_visualization(
        self,
        image: np.ndarray,
        segments: List[Dict],
        output_path: Path,
        max_segments: int = 5000,  # Increased limit
    ):
        """
        Save visualization of segments (memory optimized)

        Args:
            image: Input image
            segments: List of segments
            output_path: Output path
            max_segments: Maximum segments to visualize (prevents memory issues)
        """
        # If too many segments, sample uniformly across image instead of just taking first N
        if len(segments) > max_segments:
            logger.warning(
                f"Too many segments ({len(segments)}), sampling {max_segments} uniformly"
            )
            # Sort by area (descending) to keep larger, more important segments
            segments_sorted = sorted(segments, key=lambda s: s["area"], reverse=True)
            segments = segments_sorted[:max_segments]

        vis = image.copy()

        # Create overlay more efficiently
        overlay = np.zeros_like(image, dtype=np.uint8)

        # Color each segment
        for seg in segments:
            color = np.random.randint(
                50, 255, 3, dtype=np.uint8
            )  # Avoid very dark colors
            mask = seg["segmentation"]
            overlay[mask] = color

        # Blend
        vis = cv2.addWeighted(vis.astype(np.uint8), 0.6, overlay, 0.4, 0)

        # Draw boundaries for better visibility
        for seg in segments:
            contours, _ = cv2.findContours(
                seg["segmentation"].astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            cv2.drawContours(vis, contours, -1, (255, 255, 255), 1)

        # Save with compression
        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_PNG_COMPRESSION, 9],
        )

        # Clean up
        del vis, overlay

    def _save_metadata(
        self,
        metadata: Dict,
        segments: List[Dict],
        output_path: Path,
        preprocessing_stats: Optional[Dict] = None,
    ):
        """Save metadata as JSON"""
        meta_clean = {
            "tile_name": metadata["tile_name"],
            "crs": str(metadata["crs"]),
            "width": metadata["width"],
            "height": metadata["height"],
            "bounds": list(metadata["bounds"]),
            "num_segments": len(segments),
            "segment_stats": {
                "mean_area": float(np.mean([s["area"] for s in segments])),
                "mean_iou": float(np.mean([s["predicted_iou"] for s in segments])),
                "mean_stability": float(
                    np.mean([s["stability_score"] for s in segments])
                ),
            },
        }

        # Add preprocessing statistics if available
        if preprocessing_stats is not None:
            meta_clean["preprocessing"] = {
                "enabled": True,
                "config": self.preprocessor.get_config() if self.preprocessor else None,
                "improvements": preprocessing_stats["improvements"],
            }
        else:
            meta_clean["preprocessing"] = {"enabled": False}

        with open(output_path, "w") as f:
            json.dump(meta_clean, f, indent=2)


# CLI usage example
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Segment orthophoto tiles using SAM")
    parser.add_argument("--input-dir", required=True, help="Input directory with tiles")
    parser.add_argument(
        "--output-dir", default="data/sam_segments", help="Output directory"
    )
    parser.add_argument(
        "--model",
        default="vit_h",
        choices=["vit_h", "vit_l", "vit_b"],
        help="SAM model type",
    )
    parser.add_argument("--checkpoint", default=None, help="Path to SAM checkpoint")
    parser.add_argument(
        "--points-per-side",
        type=int,
        default=32,
        help="Grid density for SAM sampling (higher=more segments, 16-64 recommended)",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=100,
        help="Minimum segment area in pixels (smaller=more detail, 50-500 recommended)",
    )
    parser.add_argument(
        "--max-area",
        type=int,
        default=None,
        help="Maximum segment area in pixels (rejects huge overlapping segments, e.g., 500000)",
    )
    parser.add_argument(
        "--pred-iou-thresh",
        type=float,
        default=0.88,
        help="Predicted IoU threshold (lower=more segments, 0.7-0.95 recommended)",
    )
    parser.add_argument(
        "--stability-score-thresh",
        type=float,
        default=0.90,
        help="Stability score threshold (lower=more segments, 0.8-0.95 recommended)",
    )
    parser.add_argument(
        "--box-nms-thresh",
        type=float,
        default=0.7,
        help="NMS threshold for removing overlaps (lower=fewer overlaps, 0.5-0.9 recommended)",
    )
    parser.add_argument(
        "--crop-layers",
        type=int,
        default=1,
        help="Number of crop layers for multi-scale (0-2, higher=more detail but slower)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Tiles to process before memory cleanup (1=safest)",
    )
    parser.add_argument(
        "--device", default="cuda", choices=["cuda", "cpu"], help="Device for inference"
    )
    parser.add_argument(
        "--no-preprocessing", action="store_true", help="Disable image preprocessing"
    )
    parser.add_argument(
        "--clahe-clip-limit",
        type=float,
        default=2.0,
        help="CLAHE contrast enhancement strength (1.0-4.0, higher=more contrast)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recreation of existing segmentation results (default: skip existing)",
    )

    args = parser.parse_args()

    # Memory usage tip
    if args.model == "vit_h":
        logger.warning(
            "⚠️  Using vit_h (largest model). Consider vit_l or vit_b if running out of memory."
        )

    # Initialize SAM
    logger.info(f"Initializing SAM model: {args.model} on {args.device}")
    sam_service = SAMSegmentationService(
        model_type=args.model, checkpoint_path=args.checkpoint
    )

    # Configure for ortofoto tiles (1666x1666) with user-specified parameters
    logger.info("SAM configuration:")
    logger.info(f"  points_per_side: {args.points_per_side}")
    logger.info(f"  min_area: {args.min_area}")
    logger.info(f"  pred_iou_thresh: {args.pred_iou_thresh}")
    logger.info(f"  stability_score_thresh: {args.stability_score_thresh}")
    logger.info(f"  box_nms_thresh: {args.box_nms_thresh}")
    logger.info(f"  crop_layers: {args.crop_layers}")

    sam_service.configure_mask_generator(
        points_per_side=args.points_per_side,
        min_mask_region_area=args.min_area,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_layers,
        box_nms_thresh=args.box_nms_thresh,
    )

    # Run pipeline
    pipeline = OrtofotoSegmentationPipeline(
        sam_service,
        args.output_dir,
        enable_preprocessing=not args.no_preprocessing,
        max_area=args.max_area,
        force=args.force,
    )
    summary = pipeline.process_directory(
        Path(args.input_dir), batch_size=args.batch_size
    )

    logger.info("Done!")
