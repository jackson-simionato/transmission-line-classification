"""
Script to create raster masks from polygon shapefiles for deep learning segmentation.
Creates one mask per tile matching the corresponding raster file.
"""

import argparse
from pathlib import Path

import geopandas as gpd

from src.services.raster_service import RasterService


def main(
    input_shapefile: str,
    rasters_dir: str,
    output_dir: str = "data/labels/mask",
    tile_id_column: str = "tile_id",
    label_column: str = "class",
    background_value: int = 0,
    dtype: str = "uint8",
):
    """
    Create raster masks from a shapefile for each corresponding raster tile.

    Args:
        input_shapefile: Path to the input shapefile with polygon geometries
        rasters_dir: Directory containing reference raster files
        output_dir: Directory where raster masks will be saved
        tile_id_column: Column name containing tile identifiers
        label_column: Column name containing class labels
        background_value: Pixel value for areas with no polygon (default: 0)
        dtype: Data type for output raster (default: 'uint8')
    """
    print("=" * 80)
    print("CREATING RASTER MASKS FROM POLYGONS")
    print("=" * 80)

    # Initialize service
    raster_service = RasterService()

    # Convert paths
    input_shapefile = Path(input_shapefile)
    rasters_dir = Path(rasters_dir)
    output_dir = Path(output_dir)

    # Validate input shapefile
    if not input_shapefile.exists():
        print(f"❌ Error: Shapefile not found: {input_shapefile}")
        return

    if not rasters_dir.exists():
        print(f"❌ Error: Rasters directory not found: {rasters_dir}")
        return

    # Read shapefile
    print(f"\n📁 Reading shapefile: {input_shapefile.name}")
    gdf = gpd.read_file(input_shapefile)

    print(f"   Total polygons: {len(gdf)}")
    print(f"   CRS: {gdf.crs}")

    # Get unique tile IDs
    unique_tiles = gdf[tile_id_column].unique()
    print(f"\n🗂️  Found {len(unique_tiles)} unique tile IDs: {sorted(unique_tiles)}")

    # Get unique labels
    unique_labels = sorted(gdf[label_column].unique())
    print(f"🏷️  Class labels: {unique_labels}")

    # Configuration
    print("\n⚙️  Configuration:")
    print(f"   Output directory: {output_dir}")
    print(f"   Background value: {background_value}")
    print(f"   Data type: {dtype}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each tile
    created_masks = {}
    skipped_tiles = []

    for idx, tile_id in enumerate(sorted(unique_tiles), 1):
        print(f"\n{'=' * 80}")
        print(f"Processing {idx}/{len(unique_tiles)}: Tile {tile_id}")
        print(f"{'=' * 80}")

        # Filter geometries for this tile
        tile_gdf = gdf[gdf[tile_id_column] == tile_id]
        print(f"   Polygons for this tile: {len(tile_gdf)}")

        # Find corresponding raster file
        # Try common patterns: tile_id.tif, *tile_id*.tif, etc.
        raster_candidates = list(rasters_dir.glob(f"*{tile_id}*.tif"))

        if not raster_candidates:
            print(f"   ⚠️  No matching raster file found for tile '{tile_id}'")
            print(f"      Searched in: {rasters_dir}")
            skipped_tiles.append(tile_id)
            continue

        if len(raster_candidates) > 1:
            print(f"   ⚠️  Multiple raster files match tile '{tile_id}':")
            for candidate in raster_candidates:
                print(f"      - {candidate.name}")
            print(f"   Using first match: {raster_candidates[0].name}")

        reference_raster = raster_candidates[0]
        print(f"   Reference raster: {reference_raster.name}")

        try:
            # Get raster info
            raster_info = raster_service.get_raster_info(reference_raster)
            print(
                f"   Raster dimensions: {raster_info['width']}x{raster_info['height']} pixels"
            )

            # Create mask for this tile
            result = raster_service.rasterize_polygons(
                gdf=tile_gdf,
                reference_raster_path=reference_raster,
                output_dir=output_dir,
                tile_id_column=tile_id_column,
                label_column=label_column,
                background_value=background_value,
                dtype=dtype,
            )

            if result:
                created_masks.update(result)
                print("   ✅ Mask created successfully")

        except Exception as e:
            print(f"   ❌ Error processing tile {tile_id}: {e}")
            skipped_tiles.append(tile_id)
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("MASK CREATION COMPLETE")
    print("=" * 80)
    print(f"✅ Masks created: {len(created_masks)}")
    print(f"⚠️  Tiles skipped: {len(skipped_tiles)}")
    if skipped_tiles:
        print(f"   Skipped tile IDs: {skipped_tiles}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)

    # Print created masks
    if created_masks:
        print("\n📄 Created mask files:")
        for tile_id, mask_path in created_masks.items():
            print(f"   {tile_id}: {Path(mask_path).name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create raster masks from polygon shapefiles for deep learning segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python -m src.routes.create_masks --input-shapefile data/labels.shp --rasters-dir data/raw

  # Custom output directory
  python -m src.routes.create_masks --input-shapefile data/labels.shp \\
      --rasters-dir data/raw --output-dir data/processed/masks

  # Custom column names
  python -m src.routes.create_masks --input-shapefile data/labels.shp \\
      --rasters-dir data/raw --tile-id-column "tile" --label-column "class_id"

  # Custom background value and data type
  python -m src.routes.create_masks --input-shapefile data/labels.shp \\
      --rasters-dir data/raw --background-value 255 --dtype uint16
        """,
    )

    parser.add_argument(
        "--input-shapefile",
        type=str,
        required=True,
        help="Path to input shapefile with polygon geometries and labels",
    )

    parser.add_argument(
        "--rasters-dir",
        type=str,
        required=True,
        help="Directory containing reference raster files (one per tile)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/masks",
        help="Output directory for raster masks (default: data/masks)",
    )

    parser.add_argument(
        "--tile-id-column",
        type=str,
        default="tile_id",
        help="Column name containing tile identifiers (default: tile_id)",
    )

    parser.add_argument(
        "--label-column",
        type=str,
        default="class",
        help="Column name containing class labels (default: class)",
    )

    parser.add_argument(
        "--background-value",
        type=int,
        default=0,
        help="Pixel value for areas with no polygon (default: 0)",
    )

    parser.add_argument(
        "--dtype",
        type=str,
        default="uint8",
        choices=["uint8", "uint16", "int32", "float32"],
        help="Data type for output raster (default: uint8)",
    )

    args = parser.parse_args()

    main(
        input_shapefile=args.input_shapefile,
        rasters_dir=args.rasters_dir,
        output_dir=args.output_dir,
        tile_id_column=args.tile_id_column,
        label_column=args.label_column,
        background_value=args.background_value,
        dtype=args.dtype,
    )
