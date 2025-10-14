"""
Route for post-processing SAM segmentation outputs
Fills gaps, smooths boundaries, and merges small segments
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import rasterio
from tqdm import tqdm

try:
    import geopandas as gpd
    import pandas as pd
    from rasterio.features import rasterize
    from rasterio.features import shapes as rasterio_shapes
    from shapely.geometry import box, shape

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False

from src.services.postprocessing_service import SegmentPostprocessingService

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SegmentPostprocessingPipeline:
    """
    Pipeline for post-processing SAM segmentation outputs
    Processes GeoJSON segments to fill gaps and improve quality
    """

    def __init__(
        self,
        output_dir: Path,
        original_images_dir: Optional[Path] = None,
        config: Optional[Dict] = None,
        force: bool = False
    ):
        """
        Initialize post-processing pipeline

        Args:
            output_dir: Directory for processed outputs
            original_images_dir: Directory with original .tif images (for superpixel method)
            config: Configuration dictionary with processing parameters
            force: Whether to overwrite existing outputs (default: False)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.original_images_dir = (
            Path(original_images_dir) if original_images_dir else None
        )
        self.config = config or {}
        self.force = force

        if not GEOPANDAS_AVAILABLE:
            logger.error("geopandas not available. Install with: pip install geopandas")
            raise ImportError("geopandas is required for post-processing")

        logger.info("Post-processing pipeline initialized")
        logger.info(f"Output directory: {self.output_dir}")
        if self.original_images_dir:
            logger.info(f"Original images directory: {self.original_images_dir}")

    def load_tile_info(self, tile_dir: Path) -> Dict:
        """
        Load tile metadata from SAM output

        Args:
            tile_dir: Directory containing SAM outputs for a tile

        Returns:
            Dictionary with tile metadata
        """
        metadata_path = tile_dir / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata

    def load_segments_from_geojson(
        self, geojson_path: Path, metadata: Dict
    ) -> List[Dict]:
        """
        Load segments from GeoJSON and convert to SAM format

        Args:
            geojson_path: Path to GeoJSON file
            metadata: Tile metadata with transform and dimensions

        Returns:
            List of segment dictionaries in SAM format
        """
        logger.info(f"Loading segments from {geojson_path.name}")

        gdf = gpd.read_file(geojson_path)
        segments = []

        height = metadata["height"]
        width = metadata["width"]

        # Calculate transform from bounds if not provided
        if "transform" in metadata:
            transform = rasterio.Affine(*metadata["transform"])
        elif "bounds" in metadata:
            bounds = metadata["bounds"]
            # bounds = [minx, miny, maxx, maxy]
            pixel_width = (bounds[2] - bounds[0]) / width
            pixel_height = (bounds[3] - bounds[1]) / height
            transform = rasterio.Affine(
                pixel_width, 0.0, bounds[0], 0.0, -pixel_height, bounds[3]
            )
        else:
            logger.warning(
                "No transform or bounds in metadata, using identity transform"
            )
            transform = rasterio.Affine(1, 0, 0, 0, 1, 0)

        for idx, row in gdf.iterrows():
            try:
                # Rasterize polygon to binary mask
                mask = rasterize(
                    [(row.geometry, 1)],
                    out_shape=(height, width),
                    transform=transform,
                    fill=0,
                    dtype=np.uint8,
                ).astype(bool)

                segment = {
                    "segmentation": mask,
                    "area": int(row.get("area_pixels", mask.sum())),
                    "bbox": [
                        int(row.get("bbox_x", 0)),
                        int(row.get("bbox_y", 0)),
                        int(row.get("bbox_width", 0)),
                        int(row.get("bbox_height", 0)),
                    ],
                    "predicted_iou": float(row.get("predicted_iou", 0.5)),
                    "stability_score": float(row.get("stability_score", 0.5)),
                    "segment_id": int(row.get("segment_id", idx)),
                }
                segments.append(segment)
            except Exception as e:
                logger.warning(f"Error loading segment {idx}: {e}")
                continue

        logger.info(f"Loaded {len(segments)} segments from GeoJSON")

        return segments, transform

    def load_original_image(self, tile_name: str) -> Optional[np.ndarray]:
        """
        Load original .tif image for superpixel processing

        Args:
            tile_name: Name of the tile

        Returns:
            RGB image as numpy array or None if not found
        """
        if not self.original_images_dir:
            return None

        # Try to find the image file
        image_path = self.original_images_dir / f"{tile_name}.tif"

        if not image_path.exists():
            logger.warning(f"Original image not found: {image_path}")
            return None

        try:
            with rasterio.open(image_path) as src:
                # Read RGB bands
                image = src.read([1, 2, 3])
                image = np.transpose(image, (1, 2, 0))  # CHW -> HWC

                # Convert to uint8 if needed
                if image.dtype == np.uint16:
                    image = (image / 256).astype(np.uint8)
                elif image.dtype != np.uint8:
                    image = image.astype(np.uint8)

                return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            return None

    def save_processed_geojson(
        self,
        segments: List[Dict],
        metadata: Dict,
        output_path: Path,
        transform: rasterio.Affine,
        close_boundaries: bool = True,
        nodata_mask: Optional[np.ndarray] = None,
    ):
        """
        Export processed segments to GeoJSON with optional boundary closing

        Args:
            segments: List of processed segments
            metadata: Tile metadata with CRS and transform
            output_path: Path to save GeoJSON
            transform: Rasterio transform for coordinate conversion
            close_boundaries: Whether to close tile boundaries by filling gaps
            nodata_mask: Optional nodata mask to exclude from boundary closing
        """
        logger.info(f"Saving {len(segments)} segments to GeoJSON")

        polygons = []
        for segment in segments:
            mask = segment["segmentation"].astype(np.uint8)

            # Convert mask to polygons
            for geom, value in rasterio_shapes(mask, transform=transform):
                if value == 1:
                    polygons.append(
                        {
                            "geometry": shape(geom),
                            "segment_id": segment["segment_id"],
                            "area_pixels": segment["area"],
                            "predicted_iou": segment["predicted_iou"],
                            "stability_score": segment["stability_score"],
                            "is_gap_fill": segment.get("is_gap_fill", False),
                            "is_superpixel_fill": segment.get(
                                "is_superpixel_fill", False
                            ),
                            "is_border_fill": segment.get("is_border_fill", False),
                            "suggested_class": segment.get(
                                "suggested_class", "unknown"
                            ),
                        }
                    )
                    break  # Only take first polygon per segment

        # Create GeoDataFrame with proper CRS
        crs = metadata.get("crs", "EPSG:4326")
        if hasattr(crs, 'to_string'):
            crs = crs.to_string()
        gdf = gpd.GeoDataFrame(polygons, crs=crs)

        # Close boundaries if requested
        if close_boundaries and len(gdf) > 0:
            logger.info("Closing tile boundaries using bbox difference")

            try:
                # Create tile bbox from metadata bounds
                if "bounds" in metadata:
                    bounds = metadata["bounds"]  # [minx, miny, maxx, maxy]
                    tile_bbox = box(bounds[0], bounds[1], bounds[2], bounds[3])
                else:
                    # Fallback: use bounds of existing geometries
                    total_bounds = gdf.total_bounds
                    tile_bbox = box(total_bounds[0], total_bounds[1], total_bounds[2], total_bounds[3])

                # Union all existing geometries
                existing_union = gdf.geometry.unary_union

                # Find difference (gaps)
                gaps = tile_bbox.difference(existing_union)

                if not gaps.is_empty and gaps.area > 0:
                    # Convert gaps to individual polygons
                    if gaps.geom_type == "Polygon":
                        gap_polygons = [gaps]
                    elif gaps.geom_type == "MultiPolygon":
                        gap_polygons = list(gaps.geoms)
                    else:
                        gap_polygons = []

                    # Add gap polygons to GeoDataFrame
                    if gap_polygons:
                        gap_data = []
                        segment_id_offset = (
                            max(gdf["segment_id"]) + 1 if len(gdf) > 0 else 0
                        )

                        # Note: Nodata filtering is now done after boundary closing via direct clipping

                        for i, gap_geom in enumerate(gap_polygons):
                            # Calculate pixel area more accurately
                            pixel_area = int(gap_geom.area / (abs(transform.a) * abs(transform.e)))
                            
                            gap_data.append(
                                {
                                    "geometry": gap_geom,
                                    "segment_id": segment_id_offset + i,
                                    "area_pixels": pixel_area,
                                    "predicted_iou": 0.5,
                                    "stability_score": 0.5,
                                    "is_gap_fill": False,
                                    "is_superpixel_fill": False,
                                    "is_border_fill": True,
                                    "suggested_class": "grassland",
                                }
                            )

                        # Concatenate gap polygons
                        gap_gdf = gpd.GeoDataFrame(gap_data, crs=gdf.crs)
                        gdf = pd.concat([gdf, gap_gdf], ignore_index=True)

                        logger.info(
                            f"Added {len(gap_polygons)} gap polygons to close tile boundaries"
                        )
                    else:
                        logger.info("No gap polygons to add")
                else:
                    logger.info("No gaps found in tile boundaries")
            except Exception as e:
                logger.warning(f"Error in boundary closing: {e}")
                logger.info("Continuing without boundary closing")

        # Clip polygons against nodata mask if provided
        if nodata_mask is not None and len(gdf) > 0:
            logger.info("Clipping final polygons against nodata mask")
            gdf = self._clip_polygons_with_nodata(gdf, nodata_mask, transform)
            logger.info(f"After nodata clipping: {len(gdf)} polygons remain")

        # Save to file
        gdf.to_file(output_path, driver="GeoJSON")
        logger.info(f"Saved {len(gdf)} polygons to {output_path}")

    def _clip_polygons_with_nodata(self, gdf: gpd.GeoDataFrame, nodata_mask: np.ndarray, transform: rasterio.Affine) -> gpd.GeoDataFrame:
        """
        Clip polygons against nodata mask by removing parts that overlap with nodata areas
        
        Args:
            gdf: GeoDataFrame with polygons to clip
            nodata_mask: Boolean mask where True indicates nodata pixels
            transform: Rasterio transform for coordinate conversion
            
        Returns:
            GeoDataFrame with clipped polygons
        """
        try:
            # Convert nodata mask to polygon
            nodata_polygons = self._mask_to_polygons(nodata_mask, transform)
            
            if not nodata_polygons:
                logger.info("No nodata polygons found, returning original GeoDataFrame")
                return gdf
            
            # Create a single nodata geometry (union of all nodata polygons)
            nodata_union = gpd.GeoSeries(nodata_polygons).unary_union
            
            if nodata_union.is_empty:
                logger.info("Nodata union is empty, returning original GeoDataFrame")
                return gdf
            
            # Clip each polygon against nodata areas
            clipped_geometries = []
            clipped_data = []
            
            for idx, row in gdf.iterrows():
                geom = row.geometry
                
                # Check if polygon overlaps with nodata
                if geom.intersects(nodata_union):
                    # Clip polygon to remove nodata areas
                    clipped_geom = geom.difference(nodata_union)
                    
                    # Handle different geometry types
                    if clipped_geom.is_empty:
                        # Polygon completely in nodata, skip it
                        logger.debug(f"Removed polygon {idx} (completely in nodata)")
                        continue
                    elif clipped_geom.geom_type == 'MultiPolygon':
                        # Keep the largest part
                        largest_poly = max(clipped_geom.geoms, key=lambda p: p.area)
                        clipped_geometries.append(largest_poly)
                        clipped_data.append(row.drop('geometry').to_dict())
                    else:
                        # Single polygon
                        clipped_geometries.append(clipped_geom)
                        clipped_data.append(row.drop('geometry').to_dict())
                else:
                    # No overlap with nodata, keep original
                    clipped_geometries.append(geom)
                    clipped_data.append(row.drop('geometry').to_dict())
            
            if not clipped_geometries:
                logger.info("All polygons were in nodata areas, returning empty GeoDataFrame")
                return gpd.GeoDataFrame(columns=gdf.columns, crs=gdf.crs)
            
            # Create new GeoDataFrame
            clipped_gdf = gpd.GeoDataFrame(clipped_data, geometry=clipped_geometries, crs=gdf.crs)
            logger.info(f"Clipped {len(gdf)} polygons to {len(clipped_gdf)} polygons")
            
            return clipped_gdf
            
        except Exception as e:
            logger.error(f"Error clipping polygons with nodata: {e}")
            logger.info("Returning original GeoDataFrame")
            return gdf

    def _mask_to_polygons(self, mask: np.ndarray, transform: rasterio.Affine) -> List:
        """
        Convert binary mask to list of polygons
        
        Args:
            mask: Boolean mask where True indicates nodata pixels
            transform: Rasterio transform for coordinate conversion
            
        Returns:
            List of Shapely polygons
        """
        try:
            # Convert mask to uint8 for rasterio
            mask_uint8 = mask.astype(np.uint8)
            
            # Extract shapes from mask
            shapes = list(rasterio_shapes(mask_uint8, transform=transform))
            
            polygons = []
            for geom, value in shapes:
                if value == 1:  # nodata pixels
                    polygons.append(shape(geom))
            
            logger.debug(f"Converted mask to {len(polygons)} nodata polygons")
            return polygons
            
        except Exception as e:
            logger.error(f"Error converting mask to polygons: {e}")
            return []

    def save_metadata(
        self,
        metadata: Dict,
        original_segments: int,
        processed_segments: int,
        coverage_before: float,
        coverage_after: float,
        output_path: Path,
    ):
        """
        Save metadata with post-processing information

        Args:
            metadata: Original tile metadata
            original_segments: Number of original segments
            processed_segments: Number of processed segments
            coverage_before: Coverage percentage before processing
            coverage_after: Coverage percentage after processing
            output_path: Path to save metadata
        """
        metadata_out = {
            "tile_name": metadata["tile_name"],
            "crs": str(metadata.get("crs", "EPSG:4326")),
            "width": metadata["width"],
            "height": metadata["height"],
            "original_segments": original_segments,
            "processed_segments": processed_segments,
            "coverage_before": coverage_before,
            "coverage_after": coverage_after,
            "postprocessing": self.config,
        }

        with open(output_path, "w") as f:
            json.dump(metadata_out, f, indent=2)

        logger.info(f"Saved metadata: {output_path}")

    def postprocess_tile(self, tile_dir: Path) -> Dict:
        """
        Post-process a single tile's segments

        Args:
            tile_dir: Directory containing SAM outputs for the tile

        Returns:
            Dictionary with processing statistics
        """
        tile_name = tile_dir.name
        logger.info(f"Processing tile: {tile_name}")

        # Check if output already exists and force is False
        if not self.force:
            tile_output_dir = self.output_dir / tile_name
            geojson_output = tile_output_dir / "segments_processed.geojson"
            if geojson_output.exists():
                logger.info("  Output already exists, skipping (use --force to overwrite)")
                return {
                    "tile_name": tile_name,
                    "coverage_before": 0.0,
                    "coverage_after": 0.0,
                    "segments_before": 0,
                    "segments_after": 0,
                    "skipped": True
                }

        # Load metadata
        metadata = self.load_tile_info(tile_dir)

        # Load segments from GeoJSON
        geojson_path = tile_dir / "segments.geojson"
        if not geojson_path.exists():
            logger.error(f"GeoJSON not found: {geojson_path}")
            return {}

        segments, transform = self.load_segments_from_geojson(geojson_path, metadata)

        if not segments:
            logger.warning(f"No segments loaded for {tile_name}")
            return {}

        # Initialize post-processor
        postprocessor = SegmentPostprocessingService(
            image_height=metadata["height"], image_width=metadata["width"]
        )

        # Calculate initial coverage
        initial_stats = postprocessor.calculate_coverage_stats(segments)

        # Load original image if needed for superpixels or nodata detection
        original_image = None
        nodata_mask = None
        if self.config.get("use_superpixels", False) or self.config.get("close_boundaries", False):
            original_image = self.load_original_image(tile_name)
            if original_image is not None:
                # Create nodata mask (pixels where any band = 0)
                nodata_mask = np.any(original_image == 0, axis=2)

        # Apply post-processing
        processed_segments = postprocessor.process_complete(
            segments, image=original_image, **self.config
        )

        # Calculate final coverage
        final_stats = postprocessor.calculate_coverage_stats(processed_segments)

        # Create output directory
        tile_output_dir = self.output_dir / tile_name
        tile_output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed GeoJSON
        geojson_output = tile_output_dir / "segments_processed.geojson"
        self.save_processed_geojson(
            processed_segments,
            metadata,
            geojson_output,
            transform,
            config["close_boundaries"],
            nodata_mask,
        )

        # Save metadata
        metadata_output = tile_output_dir / "metadata_processed.json"
        self.save_metadata(
            metadata,
            len(segments),
            len(processed_segments),
            initial_stats["coverage_percent"],
            final_stats["coverage_percent"],
            metadata_output,
        )

        # Save coverage statistics
        stats_output = tile_output_dir / "stats.json"
        with open(stats_output, "w") as f:
            json.dump({"before": initial_stats, "after": final_stats}, f, indent=2)

        logger.info(
            f"✓ Completed {tile_name}: {initial_stats['coverage_percent']:.1f}% → {final_stats['coverage_percent']:.1f}% coverage"
        )

        return {
            "tile_name": tile_name,
            "coverage_before": initial_stats["coverage_percent"],
            "coverage_after": final_stats["coverage_percent"],
            "segments_before": len(segments),
            "segments_after": len(processed_segments),
        }

    def process_directory(self, input_dir: Path) -> Dict:
        """
        Process all tiles in a directory

        Args:
            input_dir: Directory containing SAM outputs

        Returns:
            Summary statistics dictionary
        """
        input_dir = Path(input_dir)

        # Find all tile directories
        tile_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and (d / "segments.geojson").exists()
        ]

        if not tile_dirs:
            logger.error(f"No tile directories with GeoJSON found in {input_dir}")
            return {}

        logger.info("=" * 60)
        logger.info("SEGMENT POST-PROCESSING PIPELINE")
        logger.info("=" * 60)
        logger.info(f"Input: {input_dir}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"Tiles found: {len(tile_dirs)}")
        logger.info(f"Configuration: {self.config}")
        logger.info("=" * 60)

        # Process each tile
        results = []

        for tile_dir in tqdm(tile_dirs, desc="Processing tiles"):
            try:
                result = self.postprocess_tile(tile_dir)
                if result:
                    results.append(result)
            except Exception as e:
                logger.error(f"Error processing {tile_dir.name}: {e}", exc_info=True)
                continue

        # Create summary
        if results:
            # Separate processed and skipped tiles
            processed_results = [r for r in results if not r.get("skipped", False)]
            skipped_results = [r for r in results if r.get("skipped", False)]
            
            summary = {
                "total_tiles": len(tile_dirs),
                "processed_tiles": len(processed_results),
                "skipped_tiles": len(skipped_results),
                "average_coverage_before": np.mean(
                    [r["coverage_before"] for r in processed_results]
                ) if processed_results else 0.0,
                "average_coverage_after": np.mean(
                    [r["coverage_after"] for r in processed_results]
                ) if processed_results else 0.0,
                "tiles": results,
            }
        else:
            summary = {"total_tiles": len(tile_dirs), "processed_tiles": 0, "skipped_tiles": 0}

        # Save summary
        summary_path = self.output_dir / "processing_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("=" * 60)
        logger.info("POST-PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(
            f"Processed: {summary['processed_tiles']}/{summary['total_tiles']} tiles"
        )
        if summary.get('skipped_tiles', 0) > 0:
            logger.info(f"Skipped: {summary['skipped_tiles']} tiles (outputs already exist)")
        if processed_results:
            logger.info(
                f"Avg coverage before: {summary['average_coverage_before']:.1f}%"
            )
            logger.info(f"Avg coverage after: {summary['average_coverage_after']:.1f}%")
        logger.info("=" * 60)

        return summary


# CLI usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Post-process SAM segmentation outputs"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory with SAM segment outputs"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for processed outputs (default: input_dir + '_processed')",
    )
    parser.add_argument(
        "--original-images-dir",
        default=None,
        help="Directory with original .tif images (required for superpixel method)",
    )
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        default=True,
        help="Fill unsegmented gaps (default: True)",
    )
    parser.add_argument(
        "--no-fill-gaps",
        dest="fill_gaps",
        action="store_false",
        help="Disable gap filling",
    )
    parser.add_argument(
        "--use-superpixels",
        action="store_true",
        help="Use superpixel method for gap filling (requires --original-images-dir)",
    )
    parser.add_argument(
        "--smooth-boundaries", action="store_true", help="Smooth segment boundaries"
    )
    parser.add_argument(
        "--remove-overlaps",
        action="store_true",
        default=True,
        help="Remove overlaps between gap-fills and originals (default: True)",
    )
    parser.add_argument(
        "--no-remove-overlaps",
        dest="remove_overlaps",
        action="store_false",
        help="Disable overlap removal between gap-fills and originals",
    )
    parser.add_argument(
        "--remove-large-overlapping",
        action="store_true",
        default=True,
        help="Remove large segments that overlap many others (default: True)",
    )
    parser.add_argument(
        "--no-remove-large-overlapping",
        dest="remove_large_overlapping",
        action="store_false",
        help="Disable removal of large segments that overlap many others",
    )
    parser.add_argument(
        "--close-boundaries",
        action="store_true",
        default=True,
        help="Close tile boundaries by filling gaps (default: True)",
    )
    parser.add_argument(
        "--no-close-boundaries",
        dest="close_boundaries",
        action="store_false",
        help="Disable tile boundary closing",
    )
    parser.add_argument(
        "--min-gap-size",
        type=int,
        default=500,
        help="Minimum gap size in pixels to fill (default: 500)",
    )
    parser.add_argument(
        "--smoothing-kernel-size",
        type=int,
        default=5,
        help="Kernel size for boundary smoothing (default: 5)",
    )
    parser.add_argument(
        "--n-superpixels",
        type=int,
        default=200,
        help="Number of superpixels for gap filling (default: 200)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs (default: skip if output exists)",
    )

    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = str(
            Path(args.input_dir).parent / (Path(args.input_dir).name + "_processed")
        )

    # Build configuration
    config = {
        "fill_gaps": args.fill_gaps,
        "use_superpixels": args.use_superpixels,
        "smooth_boundaries": args.smooth_boundaries,
        "remove_overlaps": args.remove_overlaps,
        "remove_large_overlapping": args.remove_large_overlapping,
        "close_boundaries": args.close_boundaries,
        "min_gap_size": args.min_gap_size,
        "smoothing_kernel_size": args.smoothing_kernel_size,
        "n_superpixels": args.n_superpixels,
    }

    # Initialize pipeline
    pipeline = SegmentPostprocessingPipeline(
        output_dir=Path(args.output_dir),
        original_images_dir=Path(args.original_images_dir)
        if args.original_images_dir
        else None,
        config=config,
        force=args.force
    )

    # Process directory
    summary = pipeline.process_directory(Path(args.input_dir))

    logger.info("Done!")
