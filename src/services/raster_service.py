from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window
from rasterio.features import rasterize
import geopandas as gpd

from src.utils import get_class_mapping


class RasterService:
    def __init__(self):
        """Initialize the RasterService."""
        pass

    def read_raster(self, raster_path: Union[str, Path]) -> Tuple[np.ndarray, Dict]:
        """
        Read a raster file and return its data array and metadata.

        Args:
            raster_path: Path to the raster file

        Returns:
            Tuple of (data array, metadata dictionary)
        """
        raster_path = Path(raster_path)
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            data = src.read()
            metadata = src.meta.copy()

        return data, metadata

    def get_raster_info(self, raster_path: Union[str, Path]) -> Dict:
        """
        Get basic information and metadata about a raster file.

        Args:
            raster_path: Path to the raster file

        Returns:
            Dictionary containing raster information
        """
        raster_path = Path(raster_path)
        if not raster_path.exists():
            raise FileNotFoundError(f"Raster file not found: {raster_path}")

        with rasterio.open(raster_path) as src:
            info = {
                "filename": raster_path.name,
                "driver": src.driver,
                "width": src.width,
                "height": src.height,
                "count": src.count,  # number of bands
                "dtype": str(src.dtypes[0]),
                "crs": str(src.crs) if src.crs else None,
                "transform": list(src.transform),
                "bounds": {
                    "left": src.bounds.left,
                    "bottom": src.bounds.bottom,
                    "right": src.bounds.right,
                    "top": src.bounds.top,
                },
                "resolution": (src.res[0], src.res[1]),
                "nodata": src.nodata,
                "compression": src.compression.value if src.compression else None,
                "interleave": src.interleaving.value if src.interleaving else None,
            }

            # Add band-specific information
            info["bands"] = []
            for i in range(1, src.count + 1):
                band_info = {
                    "band_number": i,
                    "dtype": str(src.dtypes[i - 1]),
                    "nodata": src.nodatavals[i - 1],
                }
                info["bands"].append(band_info)

        return info

    def merge_rasters(
        self,
        raster_paths: List[Union[str, Path]],
        output_path: Union[str, Path],
        method: str = "first",
        nodata: Optional[float] = None,
    ) -> Dict:
        """
        Merge multiple raster files into a single raster.

        Args:
            raster_paths: List of paths to raster files to merge
            output_path: Path where the merged raster will be saved
            method: Merge method ('first', 'last', 'min', 'max', 'sum', 'mean')
            nodata: NoData value for the output raster

        Returns:
            Dictionary with information about the merged raster
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open all raster files
        src_files = []
        try:
            for path in raster_paths:
                src = rasterio.open(path)
                src_files.append(src)

            # Merge rasters
            mosaic, transform = merge(src_files, method=method, nodata=nodata)

            # Get metadata from the first raster
            out_meta = src_files[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": transform,
                    "compress": "lzw",
                }
            )

            if nodata is not None:
                out_meta.update({"nodata": nodata})

            # Write the merged raster
            with rasterio.open(output_path, "w", **out_meta) as dest:
                dest.write(mosaic)

            return {
                "output_path": str(output_path),
                "width": mosaic.shape[2],
                "height": mosaic.shape[1],
                "bands": mosaic.shape[0],
                "input_files": len(raster_paths),
            }

        finally:
            # Close all opened files
            for src in src_files:
                src.close()

    def create_raster_patches(
        self,
        raster_path: Union[str, Path],
        output_dir: Union[str, Path],
        patch_size: Tuple[int, int] = (512, 512),
        overlap_percent: float = 0.25,
        prefix: str = "patch",
        min_patch_percent: float = 0.75,
    ) -> List[Dict]:
        """
        Subdivide a raster into smaller patches for deep learning segmentation.

        Args:
            raster_path: Path to the input raster file
            output_dir: Directory where patches will be saved
            patch_size: Tuple of (width, height) for each patch in pixels
            overlap_percent: Percentage of overlap between patches (0.0-1.0).
                           Default 0.25 (25%) is recommended for segmentation tasks
            prefix: Prefix for patch filenames
            min_patch_percent: Minimum acceptable patch size as percentage of full patch_size.
                              Patches smaller than this will be skipped. Default 0.75 (75%)

        Returns:
            List of dictionaries with information about each patch
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        patches_info = []

        with rasterio.open(raster_path) as src:
            width, height = src.width, src.height
            patch_width, patch_height = patch_size

            # Calculate overlap in pixels from percentage
            overlap_x = int(patch_width * overlap_percent)
            overlap_y = int(patch_height * overlap_percent)

            # Calculate stride (step size)
            stride_x = patch_width - overlap_x
            stride_y = patch_height - overlap_y

            patch_idx = 0

            # Iterate over the raster in patches
            for y in range(0, height, stride_y):
                for x in range(0, width, stride_x):
                    # Calculate window size (may be smaller at edges)
                    window_width = min(patch_width, width - x)
                    window_height = min(patch_height, height - y)

                    # Skip if patch is too small based on min_patch_percent threshold
                    if (
                        window_width < patch_width * min_patch_percent
                        or window_height < patch_height * min_patch_percent
                    ):
                        continue

                    # Create window
                    window = Window(x, y, window_width, window_height)

                    # Read data from window
                    patch_data = src.read(window=window)

                    # Calculate transform for the patch
                    patch_transform = src.window_transform(window)

                    # Update metadata for patch
                    patch_meta = src.meta.copy()
                    patch_meta.update(
                        {
                            "driver": "GTiff",
                            "height": window_height,
                            "width": window_width,
                            "transform": patch_transform,
                            "compress": "lzw",
                        }
                    )

                    # Create output filename
                    output_filename = f"{prefix}_{patch_idx:04d}_r{y}_c{x}.tif"
                    output_path = output_dir / output_filename

                    # Write patch
                    with rasterio.open(output_path, "w", **patch_meta) as dest:
                        dest.write(patch_data)

                    # Store patch information
                    patch_info = {
                        "patch_id": patch_idx,
                        "filename": output_filename,
                        "path": str(output_path),
                        "window": {
                            "col_off": x,
                            "row_off": y,
                            "width": window_width,
                            "height": window_height,
                        },
                        "bounds": {
                            "left": patch_transform.c,
                            "top": patch_transform.f,
                            "right": patch_transform.c
                            + window_width * patch_transform.a,
                            "bottom": patch_transform.f
                            + window_height * patch_transform.e,
                        },
                    }
                    patches_info.append(patch_info)

                    patch_idx += 1

        return patches_info

    def save_patches_info(
        self, patches_info: List[Dict], output_path: Union[str, Path]
    ) -> None:
        """
        Save patches information to a JSON file.

        Args:
            patches_info: List of patch information dictionaries
            output_path: Path to save the JSON file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(patches_info, f, indent=2)

    # Helper methods for rasterization
    def _reproject_if_needed(
        self, gdf: gpd.GeoDataFrame, target_crs
    ) -> gpd.GeoDataFrame:
        """
        Reproject GeoDataFrame to target CRS if needed.

        Args:
            gdf: GeoDataFrame to reproject
            target_crs: Target CRS

        Returns:
            Reprojected GeoDataFrame (or original if CRS matches)
        """
        if gdf.crs != target_crs:
            return gdf.to_crs(target_crs)
        return gdf

    def _prepare_shapes_for_rasterization(
        self, gdf: gpd.GeoDataFrame, label_column: str
    ) -> List[Tuple]:
        """
        Prepare geometry-value pairs for rasterization.

        Args:
            gdf: GeoDataFrame with geometries and labels
            label_column: Column name containing class labels

        Returns:
            List of (geometry, value) tuples for valid geometries
        """
        shapes = [
            (geom, int(label))
            for geom, label in zip(gdf.geometry, gdf[label_column])
            if geom is not None and geom.is_valid
        ]
        return shapes

    def _write_mask_to_file(
        self,
        mask: np.ndarray,
        output_path: Path,
        width: int,
        height: int,
        transform,
        crs,
        dtype: str,
    ) -> None:
        """
        Write a mask array to a GeoTIFF file.

        Args:
            mask: Mask array to write
            output_path: Path to output file
            width: Raster width
            height: Raster height
            transform: Affine transform
            crs: Coordinate reference system
            dtype: Data type
        """
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=dtype,
            crs=crs,
            transform=transform,
            compress="lzw",
            nodata=None,
        ) as dst:
            dst.write(mask, 1)

    def rasterize_polygons(
        self,
        gdf: gpd.GeoDataFrame,
        reference_raster_path: Union[str, Path],
        output_dir: Union[str, Path],
        tile_id_column: str = "tile_id",
        label_column: str = "class",
        background_value: int = 0,
        dtype: str = "uint8",
    ) -> Dict[str, str]:
        """
        Create raster masks from polygon geometries in a shapefile.

        Creates one raster mask per unique tile_id, where each pixel value
        corresponds to the class label of the polygon it intersects.

        Args:
            gdf: GeoDataFrame containing polygon geometries and attributes
            reference_raster_path: Path to a reference raster to match CRS, resolution, and bounds
            output_dir: Directory where raster masks will be saved
            tile_id_column: Column name containing tile identifiers
            label_column: Column name containing class labels
            background_value: Pixel value for areas with no polygon (default: 0)
            dtype: Data type for output raster (default: 'uint8')

        Returns:
            Dictionary mapping tile_id to output raster path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get reference raster metadata with proper transform
        with rasterio.open(reference_raster_path) as ref:
            ref_transform = ref.transform
            ref_width = ref.width
            ref_height = ref.height
            ref_crs = ref.crs

        print(f"📐 Reference raster: {ref_width}x{ref_height}, CRS: {ref_crs}")

        # Reproject shapefile to match reference raster CRS if needed
        gdf = self._reproject_if_needed(gdf, ref_crs)

        # Get unique tile ID and labels
        tile_id = gdf[tile_id_column].unique()[0]
        unique_labels = sorted(gdf[label_column].unique())
        print(f"🏷️  Class labels: {unique_labels}")

        gdf["class_value"] = gdf[label_column].apply(get_class_mapping)

        # Prepare shapes for rasterization
        shapes = self._prepare_shapes_for_rasterization(gdf, "class_value")

        if not shapes:
            print(f"   ⚠️  No valid geometries found for tile {tile_id}, skipping...")
            return None

        # Rasterize the polygons
        mask = rasterize(
            shapes=shapes,
            out_shape=(ref_height, ref_width),
            transform=ref_transform,
            fill=background_value,
            dtype=dtype,
            all_touched=True,
        )

        # Create output path and write mask
        output_filename = f"mask_{tile_id}.tif"
        output_path = output_dir / output_filename

        self._write_mask_to_file(
            mask=mask,
            output_path=output_path,
            width=ref_width,
            height=ref_height,
            transform=ref_transform,
            crs=ref_crs,
            dtype=dtype,
        )

        print(f"   ✅ Saved: {output_filename}")

        return {tile_id: str(output_path)}
