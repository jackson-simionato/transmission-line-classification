"""
Feature Extraction Service for Transmission Line Classification
Extracts spectral, geometric, and texture features from polygon segments
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import rasterio
import torch
import torchvision.transforms as transforms
from PIL import Image
from rasterio.features import rasterize
from shapely.geometry import Polygon
from skimage.color import rgb2hsv
from skimage.feature import graycomatrix, graycoprops

try:
    import timm

    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureExtractionService:
    """
    Service for extracting features from polygon segments in ortofoto images
    """

    def __init__(
        self,
        include_cnn: bool = False,
        cnn_model: str = "mobilenet_v3_small",
        device: Optional[str] = None,
    ):
        """
        Initialize feature extraction service

        Args:
            include_cnn: Whether to extract CNN embeddings
            cnn_model: CNN model to use ('mobilenet_v3_small' or 'efficientnet_b0')
            device: Device for CNN inference ('cuda' or 'cpu')
        """
        self.include_cnn = include_cnn
        self.cnn_model_name = cnn_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize CNN model if requested
        if self.include_cnn:
            self._initialize_cnn_model()

        logger.info("Feature extraction service initialized")
        logger.info(f"CNN embeddings: {'enabled' if include_cnn else 'disabled'}")
        if include_cnn:
            logger.info(f"CNN model: {cnn_model} on {self.device}")

    def _get_default_features(
        self, segment_id: int, class_label: Optional[str]
    ) -> Dict[str, Union[float, int, str]]:
        """Return default zero features for invalid polygons"""
        return {
            "segment_id": segment_id,
            "class_label": class_label or "unlabeled",
            "rgb_mean_r": 0.0,
            "rgb_mean_g": 0.0,
            "rgb_mean_b": 0.0,
            "rgb_std_r": 0.0,
            "rgb_std_g": 0.0,
            "rgb_std_b": 0.0,
            "hsv_mean_h": 0.0,
            "hsv_mean_s": 0.0,
            "hsv_mean_v": 0.0,
            "hsv_std_h": 0.0,
            "hsv_std_s": 0.0,
            "hsv_std_v": 0.0,
            "compactness": 0.0,
            "solidity": 0.0,
            "aspect_ratio": 0.0,
            "elongation": 0.0,
            "rectangularity": 0.0,
            "convexity": 0.0,
            "circularity": 0.0,
            "glcm_contrast": 0.0,
            "glcm_dissimilarity": 0.0,
            "glcm_homogeneity": 0.0,
            "glcm_energy": 0.0,
            "glcm_correlation": 0.0,
            "glcm_asm": 0.0,
        }

    def _initialize_cnn_model(self):
        """Initialize CNN model for feature extraction"""
        if not TIMM_AVAILABLE:
            raise ImportError("timm not available. Install with: pip install timm")

        try:
            if self.cnn_model_name == "mobilenet_v3_small":
                self.cnn_model = timm.create_model(
                    "mobilenetv3_small_100", pretrained=True
                )
                self.cnn_model.eval()
                self.cnn_model.to(self.device)
                # Remove classification head to get embeddings
                self.cnn_model.classifier = torch.nn.Identity()
                self.embedding_dim = 576
            elif self.cnn_model_name == "efficientnet_b0":
                self.cnn_model = timm.create_model("efficientnet_b0", pretrained=True)
                self.cnn_model.eval()
                self.cnn_model.to(self.device)
                # Remove classification head to get embeddings
                self.cnn_model.classifier = torch.nn.Identity()
                self.embedding_dim = 1280
            else:
                raise ValueError(f"Unsupported CNN model: {self.cnn_model_name}")

            # Image preprocessing for CNN
            self.cnn_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

            logger.info(f"CNN model {self.cnn_model_name} loaded successfully")

        except Exception as e:
            logger.error(f"Failed to initialize CNN model: {e}")
            raise

    def extract_spectral_features(
        self, image_patch: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract spectral features from image patch

        Args:
            image_patch: RGB image patch (H, W, 3)
            mask: Binary mask for the polygon (H, W)

        Returns:
            Dictionary of spectral features
        """
        # Apply mask to image
        masked_image = image_patch[mask > 0]

        if len(masked_image) == 0:
            # Return zeros if no valid pixels
            return {
                "rgb_mean_r": 0.0,
                "rgb_mean_g": 0.0,
                "rgb_mean_b": 0.0,
                "rgb_std_r": 0.0,
                "rgb_std_g": 0.0,
                "rgb_std_b": 0.0,
                "hsv_mean_h": 0.0,
                "hsv_mean_s": 0.0,
                "hsv_mean_v": 0.0,
                "hsv_std_h": 0.0,
                "hsv_std_s": 0.0,
                "hsv_std_v": 0.0,
            }

        # RGB features
        rgb_mean = np.mean(masked_image, axis=0)
        rgb_std = np.std(masked_image, axis=0)

        # Convert to HSV
        hsv_image = rgb2hsv(image_patch)
        masked_hsv = hsv_image[mask > 0]
        hsv_mean = np.mean(masked_hsv, axis=0)
        hsv_std = np.std(masked_hsv, axis=0)

        return {
            "rgb_mean_r": float(rgb_mean[0]),
            "rgb_mean_g": float(rgb_mean[1]),
            "rgb_mean_b": float(rgb_mean[2]),
            "rgb_std_r": float(rgb_std[0]),
            "rgb_std_g": float(rgb_std[1]),
            "rgb_std_b": float(rgb_std[2]),
            "hsv_mean_h": float(hsv_mean[0]),
            "hsv_mean_s": float(hsv_mean[1]),
            "hsv_mean_v": float(hsv_mean[2]),
            "hsv_std_h": float(hsv_std[0]),
            "hsv_std_s": float(hsv_std[1]),
            "hsv_std_v": float(hsv_std[2]),
        }

    def extract_geometric_features(self, polygon: Polygon) -> Dict[str, float]:
        """
        Extract geometric features from polygon

        Args:
            polygon: Shapely polygon geometry

        Returns:
            Dictionary of geometric features
        """
        if polygon.is_empty:
            return {
                "compactness": 0.0,
                "solidity": 0.0,
                "aspect_ratio": 0.0,
                "elongation": 0.0,
                "rectangularity": 0.0,
                "convexity": 0.0,
                "circularity": 0.0,
            }

        # Basic geometric properties
        area = polygon.area
        perimeter = polygon.length

        # Compactness (4π * area / perimeter²)
        compactness = (4 * np.pi * area) / (perimeter**2) if perimeter > 0 else 0.0

        # Solidity (area / convex_hull_area)
        convex_hull = polygon.convex_hull
        solidity = area / convex_hull.area if convex_hull.area > 0 else 0.0

        # Bounding box properties
        minx, miny, maxx, maxy = polygon.bounds
        width = maxx - minx
        height = maxy - miny

        # Aspect ratio
        aspect_ratio = width / height if height > 0 else 0.0

        # Elongation (1 - height/width for width > height)
        elongation = 1 - (height / width) if width > height else 1 - (width / height)

        # Rectangularity (area / (width * height))
        rectangularity = area / (width * height) if width * height > 0 else 0.0

        # Convexity (perimeter / convex_hull_perimeter)
        convexity = perimeter / convex_hull.length if convex_hull.length > 0 else 0.0

        # Circularity (4π * area / perimeter²) - same as compactness
        circularity = compactness

        return {
            "compactness": float(compactness),
            "solidity": float(solidity),
            "aspect_ratio": float(aspect_ratio),
            "elongation": float(elongation),
            "rectangularity": float(rectangularity),
            "convexity": float(convexity),
            "circularity": float(circularity),
        }

    def extract_texture_features(
        self, image_patch: np.ndarray, mask: np.ndarray
    ) -> Dict[str, float]:
        """
        Extract texture features using GLCM

        Args:
            image_patch: RGB image patch (H, W, 3)
            mask: Binary mask for the polygon (H, W)

        Returns:
            Dictionary of texture features
        """
        # Convert to grayscale
        gray_image = np.mean(image_patch, axis=2).astype(np.uint8)

        # Apply mask
        masked_gray = gray_image.copy()
        masked_gray[mask == 0] = 0

        if np.sum(mask) < 4:  # Need at least 4 pixels for GLCM
            return {
                "glcm_contrast": 0.0,
                "glcm_dissimilarity": 0.0,
                "glcm_homogeneity": 0.0,
                "glcm_energy": 0.0,
                "glcm_correlation": 0.0,
                "glcm_asm": 0.0,
            }

        try:
            # Compute GLCM for horizontal direction (0 degrees)
            glcm = graycomatrix(
                masked_gray,
                distances=[1],
                angles=[0],
                levels=256,
                symmetric=True,
                normed=True,
            )

            # Extract texture properties
            contrast = graycoprops(glcm, "contrast")[0, 0]
            dissimilarity = graycoprops(glcm, "dissimilarity")[0, 0]
            homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
            energy = graycoprops(glcm, "energy")[0, 0]
            correlation = graycoprops(glcm, "correlation")[0, 0]
            asm = graycoprops(glcm, "ASM")[0, 0]

            return {
                "glcm_contrast": float(contrast),
                "glcm_dissimilarity": float(dissimilarity),
                "glcm_homogeneity": float(homogeneity),
                "glcm_energy": float(energy),
                "glcm_correlation": float(correlation),
                "glcm_asm": float(asm),
            }

        except Exception as e:
            logger.warning(f"Error computing GLCM features: {e}")
            return {
                "glcm_contrast": 0.0,
                "glcm_dissimilarity": 0.0,
                "glcm_homogeneity": 0.0,
                "glcm_energy": 0.0,
                "glcm_correlation": 0.0,
                "glcm_asm": 0.0,
            }

    def extract_cnn_embeddings(self, image_patch: np.ndarray) -> np.ndarray:
        """
        Extract CNN embeddings from image patch

        Args:
            image_patch: RGB image patch (H, W, 3)

        Returns:
            CNN embedding vector
        """
        if not self.include_cnn:
            return np.array([])

        try:
            # Convert to PIL Image
            pil_image = Image.fromarray(image_patch.astype(np.uint8))

            # Apply transforms
            input_tensor = self.cnn_transform(pil_image).unsqueeze(0).to(self.device)

            # Extract features
            with torch.no_grad():
                embeddings = self.cnn_model(input_tensor)
                embeddings = embeddings.cpu().numpy().flatten()

            return embeddings

        except Exception as e:
            logger.warning(f"Error extracting CNN embeddings: {e}")
            return np.zeros(self.embedding_dim)

    def extract_features_from_polygon(
        self,
        polygon: Polygon,
        tile_image: np.ndarray,
        transform: rasterio.Affine,
        segment_id: int,
        class_label: Optional[str] = None,
    ) -> Dict[str, Union[float, int, str, np.ndarray]]:
        """
        Extract all features from a single polygon

        Args:
            polygon: Shapely polygon geometry
            tile_image: Full tile image (H, W, 3)
            transform: Rasterio transform for coordinate conversion
            segment_id: Unique segment identifier
            class_label: Class label (if available)

        Returns:
            Dictionary containing all extracted features
        """
        # Create mask for this polygon
        h, w = tile_image.shape[:2]
        full_mask = rasterize(
            [(polygon, 1)],
            out_shape=(h, w),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        # Find actual polygon bounds in the full mask
        rows, cols = np.where(full_mask > 0)
        if len(rows) == 0:
            logger.warning(
                f"No polygon pixels found in full mask for segment {segment_id}"
            )
            return self._get_default_features(segment_id, class_label)

        # Get actual bounds of the polygon in the image
        actual_min_row, actual_max_row = rows.min(), rows.max() + 1
        actual_min_col, actual_max_col = cols.min(), cols.max() + 1

        # Add padding to ensure we capture the full polygon
        padding = 5
        actual_min_row = max(0, actual_min_row - padding)
        actual_min_col = max(0, actual_min_col - padding)
        actual_max_row = min(h, actual_max_row + padding)
        actual_max_col = min(w, actual_max_col + padding)

        # Extract the correct image patch and mask
        image_patch = tile_image[
            actual_min_row:actual_max_row, actual_min_col:actual_max_col
        ]
        mask = full_mask[actual_min_row:actual_max_row, actual_min_col:actual_max_col]

        # Extract features
        features = {"segment_id": segment_id, "class_label": class_label or "unlabeled"}

        # Spectral features
        spectral_features = self.extract_spectral_features(image_patch, mask)
        features.update(spectral_features)

        # Geometric features
        geometric_features = self.extract_geometric_features(polygon)
        features.update(geometric_features)

        # Texture features
        texture_features = self.extract_texture_features(image_patch, mask)
        features.update(texture_features)

        # CNN embeddings (optional)
        if self.include_cnn:
            cnn_embeddings = self.extract_cnn_embeddings(image_patch)
            for i, emb in enumerate(cnn_embeddings):
                features[f"cnn_emb_{i}"] = float(emb)

        return features

    def process_tile(
        self, tile_name: str, geojson_path: Path, image_path: Path
    ) -> List[Dict[str, Union[float, int, str, np.ndarray]]]:
        """
        Process all polygons in a tile

        Args:
            tile_name: Name of the tile
            geojson_path: Path to GeoJSON file with polygons
            image_path: Path to ortofoto image

        Returns:
            List of feature dictionaries for all polygons
        """
        logger.info(f"Processing tile: {tile_name}")

        # Load image
        with rasterio.open(image_path) as src:
            image = src.read([1, 2, 3])  # RGB bands
            image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
            transform = src.transform

            # Convert to uint8 if needed
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            elif image.dtype != np.uint8:
                image = image.astype(np.uint8)

        # Load polygons
        import geopandas as gpd

        gdf = gpd.read_file(geojson_path)

        features_list = []

        for idx, row in gdf.iterrows():
            try:
                polygon = row.geometry
                id = row.get("id", idx)
                segment_id = row.get("segment_id", idx)
                class_label = row.get("class", None)

                features = self.extract_features_from_polygon(
                    polygon, image, transform, segment_id, class_label
                )
                features["tile_id"] = tile_name
                features["id"] = id
                features_list.append(features)

            except Exception as e:
                logger.warning(f"Error processing polygon {idx} in {tile_name}: {e}")
                continue

        logger.info(
            f"Extracted features for {len(features_list)} polygons in {tile_name}"
        )
        return features_list

    def process_dataset(
        self,
        labels_dir: Path,
        images_dir: Path,
        output_dir: Path,
        split: Optional[str] = None,
    ) -> Dict[str, int]:
        """
        Process entire dataset and save features

        Args:
            labels_dir: Directory with labeled GeoJSON files
            images_dir: Directory with ortofoto images
            output_dir: Output directory for feature files
            split: Specific split to process ('train', 'val', 'test', or None for all)

        Returns:
            Dictionary with processing statistics
        """
        from src.config.settings import get_config

        config = get_config()

        # Determine which tiles to process based on split
        if split == "train":
            tile_ids = config.TRAIN_DATASET_IDS
        elif split == "val":
            tile_ids = config.VAL_DATASET_IDS
        elif split == "test":
            tile_ids = config.TEST_DATASET_IDS
        else:
            # Process all tiles
            tile_ids = (
                config.TRAIN_DATASET_IDS
                + config.TEST_DATASET_IDS
            )

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        all_features = []
        processed_tiles = 0

        for tile_id in tile_ids:
            # Find corresponding files
            geojson_path = labels_dir / "by_tile" / tile_id / "labels.geojson"
            image_path = images_dir / f"{tile_id}.tif"

            if not geojson_path.exists():
                logger.warning(f"GeoJSON not found: {geojson_path}")
                continue

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            try:
                features = self.process_tile(tile_id, geojson_path, image_path)
                all_features.extend(features)
                processed_tiles += 1

            except Exception as e:
                logger.error(f"Error processing tile {tile_id}: {e}")
                continue

        if not all_features:
            logger.warning("No features extracted")
            return {"processed_tiles": 0, "total_polygons": 0}

        # Convert to DataFrame
        df = pd.DataFrame(all_features)

        # Save features
        if split:
            output_file = output_dir / f"features_{split}.csv"
        else:
            output_file = output_dir / "features_all.csv"

        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} feature vectors to {output_file}")

        return {
            "processed_tiles": processed_tiles,
            "total_polygons": len(df),
            "output_file": str(output_file),
        }
