"""
Script to create patches from raster files for deep learning segmentation.
Batch processes all TIF files in a directory.
"""

import argparse
from pathlib import Path

from src.services.raster_service import RasterService


def main(
    input_dir: str = "data/ortofoto/raw",
    output_dir: str = "data/ortofoto/patches",
    patch_size: int = 256,
    overlap_percent: float = 0.25,
    pattern: str = "*.tif",
    prefix: str = "patch",
):
    """
    Create patches from all raster files in a directory.

    Args:
        input_dir: Directory containing input raster files
        output_dir: Base directory for outputs (subdirs created per file)
        patch_size: Size of each patch (width and height in pixels)
        overlap_percent: Percentage of overlap between patches (0.0-1.0)
        pattern: File pattern to match (e.g., '*.tif', '*.tiff')
        prefix: Prefix for patch filenames
    """
    print("=" * 80)
    print("BATCH PROCESSING: CREATING PATCHES FOR DEEP LEARNING")
    print("=" * 80)

    # Initialize service
    raster_service = RasterService()

    # Convert paths
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # Find all raster files
    raster_files = sorted(input_dir.glob(pattern))

    if not raster_files:
        print(f"❌ Error: No files matching pattern '{pattern}' found in {input_dir}")
        return

    print(f"\n📁 Found {len(raster_files)} raster files in {input_dir}")

    # Calculate overlap in pixels
    overlap_pixels = int(patch_size * overlap_percent)
    print("\n⚙️  Configuration:")
    print(f"   Patch size: {patch_size}x{patch_size} pixels")
    print(f"   Overlap: {overlap_percent * 100:.0f}% ({overlap_pixels} pixels)")
    print(f"   Output directory: {output_dir}")

    # Process each file
    all_patches = []
    for idx, raster_file in enumerate(raster_files, 1):
        print(f"\n{'=' * 80}")
        print(f"Processing {idx}/{len(raster_files)}: {raster_file.name}")
        print(f"{'=' * 80}")

        try:
            # Get raster info
            info = raster_service.get_raster_info(raster_file)
            print(f"   Dimensions: {info['width']} x {info['height']} pixels")
            print(f"   Bands: {info['count']}")

            # Create output directory for this raster
            file_output_dir = output_dir / raster_file.stem
            file_output_dir.mkdir(parents=True, exist_ok=True)

            # Create patches
            print("   Creating patches...")
            patches_info = raster_service.create_raster_patches(
                raster_path=raster_file,
                output_dir=file_output_dir,
                patch_size=(patch_size, patch_size),
                overlap_percent=overlap_percent,
                prefix=f"{prefix}_{raster_file.stem}",
                min_patch_percent=0.75,
            )

            print(f"   ✅ Created {len(patches_info)} patches")
            print(f"   📁 Saved to: {file_output_dir}")

            all_patches.extend(patches_info)

        except Exception as e:
            print(f"   ❌ Error: {e}")
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE")
    print("=" * 80)
    print(f"✅ Total patches created: {len(all_patches)}")
    print(f"📂 Files processed: {len(raster_files)}")
    print(f"📁 Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create patches from raster files for deep learning segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all TIF files with defaults
  python -m src.routes.create_patches

  # Custom input/output directories
  python -m src.routes.create_patches --input-dir data/raw --output-dir data/patches

  # Custom patch size and overlap
  python -m src.routes.create_patches --patch-size 256 --overlap 0.5

  # Process specific pattern
  python -m src.routes.create_patches --pattern "OXAPAMPA*.tif"
        """,
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Directory containing input raster files (default: data/raw)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/patches",
        help="Output directory for patches (default: data/patches)",
    )

    parser.add_argument(
        "--patch-size",
        type=int,
        default=512,
        help="Size of each patch in pixels (default: 512)",
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.25,
        help="Overlap percentage between patches, 0.0-1.0 (default: 0.25 = 25%%)",
    )

    parser.add_argument(
        "--pattern",
        type=str,
        default="*.tif",
        help="File pattern to match (default: *.tif)",
    )

    parser.add_argument(
        "--prefix",
        type=str,
        default="patch",
        help="Prefix for patch filenames (default: patch)",
    )

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        patch_size=args.patch_size,
        overlap_percent=args.overlap,
        pattern=args.pattern,
        prefix=args.prefix,
    )
