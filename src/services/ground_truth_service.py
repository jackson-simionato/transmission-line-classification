"""
Ground Truth Point Generation Service
Generates spatially distributed sampling points across segmented tiles for manual labeling
"""

import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

logger = logging.getLogger(__name__)


class GroundTruthService:
    """
    Service for generating spatially distributed sampling points
    """

    def __init__(self, min_distance_meters: float = 10.0, random_seed: int = 42):
        """
        Initialize ground truth point generation service

        Args:
            min_distance_meters: Minimum distance between points in meters
            random_seed: Random seed for reproducibility
        """
        self.min_distance = min_distance_meters
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)

        logger.info(
            f"GroundTruthService initialized: min_distance={min_distance_meters}m, seed={random_seed}"
        )

    @staticmethod
    def load_segments_geojson(geojson_path: Path) -> gpd.GeoDataFrame:
        """
        Load and validate GeoDataFrame from file

        Args:
            geojson_path: Path to GeoJSON file

        Returns:
            GeoDataFrame with segments

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

        try:
            gdf = gpd.read_file(geojson_path)
            if gdf.empty:
                raise ValueError(f"Empty GeoDataFrame loaded from {geojson_path}")

            logger.info(f"Loaded {len(gdf)} segments from {geojson_path.name}")
            return gdf

        except Exception as e:
            raise ValueError(f"Error loading GeoJSON {geojson_path}: {e}")

    @staticmethod
    def calculate_point_weights(segments: gpd.GeoDataFrame) -> np.ndarray:
        """
        Calculate sampling probability based on segment area

        Args:
            segments: GeoDataFrame with segment polygons

        Returns:
            Array of weights for each segment (proportional to area)
        """
        if "area_pixels" in segments.columns:
            areas = segments["area_pixels"].values
        else:
            # Calculate area if not available
            areas = segments.geometry.area.values

        # Add small constant to avoid zero weights
        weights = areas + 1
        # Normalize to probabilities
        weights = weights / weights.sum()

        logger.debug(
            f"Calculated weights for {len(segments)} segments (min: {weights.min():.4f}, max: {weights.max():.4f})"
        )
        return weights

    @staticmethod
    def generate_random_point_in_polygon(
        polygon, max_attempts: int = 100
    ) -> Optional[Point]:
        """
        Generate single random point within polygon bounds

        Args:
            polygon: Shapely polygon geometry
            max_attempts: Maximum attempts to find valid point

        Returns:
            Point geometry or None if failed
        """
        bounds = polygon.bounds  # (minx, miny, maxx, maxy)

        for _ in range(max_attempts):
            # Generate random point within bounding box
            x = random.uniform(bounds[0], bounds[2])
            y = random.uniform(bounds[1], bounds[3])
            point = Point(x, y)

            # Check if point is within polygon
            if polygon.contains(point):
                return point

        logger.warning(
            f"Failed to generate point in polygon after {max_attempts} attempts"
        )
        return None

    @staticmethod
    def is_valid_distance(
        point: Point, existing_points: List[Point], min_distance: float
    ) -> bool:
        """
        Check if point satisfies minimum distance constraint

        Args:
            point: Point to validate
            existing_points: List of existing points
            min_distance: Minimum distance in meters

        Returns:
            True if point is valid (far enough from others)
        """
        if not existing_points:
            return True

        for existing_point in existing_points:
            if point.distance(existing_point) < min_distance:
                return False

        return True

    @staticmethod
    def create_point_attributes(
        point: Point, segment: Dict, sample_id: str, tile_name: str
    ) -> Dict:
        """
        Build attribute dictionary for a ground truth point

        Args:
            point: Point geometry
            segment: Segment data dictionary
            sample_id: Unique sample identifier
            tile_name: Name of source tile

        Returns:
            Dictionary with point attributes
        """
        return {
            "sample_id": sample_id,
            "tile_name": tile_name,
            "segment_id": segment.get("segment_id", "unknown"),
            "class_label": "",  # Empty for manual labeling
            "x_coord": float(point.x),
            "y_coord": float(point.y),
        }

    def generate_tile_samples(
        self, segments_gdf: gpd.GeoDataFrame, num_points: int = 15
    ) -> gpd.GeoDataFrame:
        """
        Generate well-distributed points for one tile

        Args:
            segments_gdf: GeoDataFrame with segment polygons
            num_points: Number of points to generate

        Returns:
            GeoDataFrame with ground truth points
        """
        if segments_gdf.empty:
            logger.warning("Empty segments GeoDataFrame provided")
            return gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)

        # Calculate sampling weights based on area
        weights = self.calculate_point_weights(segments_gdf)

        # Generate points using weighted random sampling
        points = []
        point_geometries = []
        sample_counter = 0

        # Convert to list for easier indexing
        segments_list = segments_gdf.to_dict("records")

        logger.info(
            f"Generating {num_points} points from {len(segments_list)} segments"
        )

        attempts = 0
        max_attempts = num_points * 10  # Prevent infinite loops

        while len(points) < num_points and attempts < max_attempts:
            attempts += 1

            # Select segment based on weights
            segment_idx = np.random.choice(len(segments_list), p=weights)
            segment = segments_list[segment_idx]
            polygon = segment["geometry"]

            # Generate random point in polygon
            point = self.generate_random_point_in_polygon(polygon)
            if point is None:
                continue

            # Check distance constraint
            if self.is_valid_distance(point, point_geometries, self.min_distance):
                # Create attributes
                sample_id = f"sample_{sample_counter:04d}"
                tile_name = segment.get("tile_name", "unknown")

                attributes = self.create_point_attributes(
                    point, segment, sample_id, tile_name
                )
                points.append(attributes)
                point_geometries.append(point)
                sample_counter += 1

                logger.debug(f"Generated point {sample_counter}/{num_points}")

        if len(points) < num_points:
            logger.warning(
                f"Only generated {len(points)}/{num_points} points (attempts: {attempts})"
            )

        # Create GeoDataFrame
        if points:
            points_gdf = gpd.GeoDataFrame(
                points, geometry=point_geometries, crs=segments_gdf.crs
            )
            logger.info(f"Generated {len(points_gdf)} ground truth points")
        else:
            points_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=segments_gdf.crs)
            logger.warning("No points generated")

        return points_gdf

    def export_points(self, points_gdf: gpd.GeoDataFrame, output_path: Path) -> None:
        """
        Save points as GeoJSON

        Args:
            points_gdf: GeoDataFrame with ground truth points
            output_path: Path to save GeoJSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if points_gdf.empty:
            logger.warning("Empty points GeoDataFrame, creating empty file")
            # Create empty GeoJSON
            empty_gdf = gpd.GeoDataFrame(columns=["geometry"], crs=points_gdf.crs)
            empty_gdf.to_file(output_path, driver="GeoJSON")
        else:
            points_gdf.to_file(output_path, driver="GeoJSON")
            logger.info(f"Exported {len(points_gdf)} points to {output_path}")

    def generate_all_tiles_samples(
        self, input_dir: Path, output_dir: Path, points_per_tile: int = 15
    ) -> Dict:
        """
        Generate ground truth points for all tiles in directory

        Args:
            input_dir: Directory with processed segments
            output_dir: Directory to save ground truth points
            points_per_tile: Number of points per tile

        Returns:
            Dictionary with processing summary
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all processed tile directories
        tile_dirs = [
            d
            for d in input_dir.iterdir()
            if d.is_dir() and (d / "segments_processed.geojson").exists()
        ]

        if not tile_dirs:
            raise ValueError(f"No processed segments found in {input_dir}")

        logger.info(f"Processing {len(tile_dirs)} tiles")

        all_points = []
        tile_summaries = []

        # Process each tile
        for tile_dir in tile_dirs:
            tile_name = tile_dir.name
            logger.info(f"Processing tile: {tile_name}")

            # Load segments
            segments_path = tile_dir / "segments_processed.geojson"
            try:
                segments_gdf = self.load_segments_geojson(segments_path)
            except Exception as e:
                logger.error(f"Error loading segments for {tile_name}: {e}")
                continue

            # Generate ground truth points
            points_gdf = self.generate_tile_samples(segments_gdf, points_per_tile)

            if not points_gdf.empty:
                # Save individual tile points
                tile_output_path = (
                    output_dir / "by_tile" / f"{tile_name}_ground_truth_points.geojson"
                )
                tile_output_path.parent.mkdir(exist_ok=True)
                self.export_points(points_gdf, tile_output_path)

                # Add to combined dataset
                all_points.append(points_gdf)

                tile_summaries.append(
                    {
                        "tile_name": tile_name,
                        "segments_count": len(segments_gdf),
                        "points_generated": len(points_gdf),
                        "success": True,
                    }
                )
            else:
                tile_summaries.append(
                    {
                        "tile_name": tile_name,
                        "segments_count": len(segments_gdf),
                        "points_generated": 0,
                        "success": False,
                    }
                )

        # Combine all points
        if all_points:
            combined_points = gpd.pd.concat(all_points, ignore_index=True)
            combined_output_path = output_dir / "all_ground_truth_points.geojson"
            self.export_points(combined_points, combined_output_path)

            total_points = len(combined_points)
            logger.info(f"Generated {total_points} total ground truth points")
        else:
            total_points = 0
            logger.warning("No points generated for any tile")

        # Create summary
        summary = {
            "total_tiles": len(tile_dirs),
            "successful_tiles": sum(1 for t in tile_summaries if t["success"]),
            "total_points": total_points,
            "points_per_tile": points_per_tile,
            "min_distance_meters": self.min_distance,
            "random_seed": self.random_seed,
            "tiles": tile_summaries,
        }

        # Save summary
        summary_path = output_dir / "ground_truth_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"Sampling complete: {summary['successful_tiles']}/{summary['total_tiles']} tiles, {total_points} total points"
        )

        return summary
