from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import Window


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
