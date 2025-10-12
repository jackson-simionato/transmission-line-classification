"""
Image Preprocessing Service for Transmission Line Classification
Specialized preprocessing to enhance differences between: trees, grassland, soil/road, towers, and power lines
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class ImagePreProcessingService:
    """
    Specialized preprocessing service for image classification
    Enhances visual differences between: trees, grassland, soil/road, towers, and power lines
    """

    def __init__(
        self,
        enable_denoising: bool = True,
        enable_clahe: bool = True,
        enable_texture_enhancement: bool = True,
        enable_vegetation_separation: bool = True,
        enable_metallic_enhancement: bool = True,
        enable_soil_enhancement: bool = True,
        clahe_clip_limit: float = 2.5,
        clahe_tile_size: Tuple[int, int] = (8, 8),
        denoise_strength: float = 12.0,
        texture_strength: float = 1.5,
        vegetation_separation_strength: float = 1.2,
        metallic_enhancement_strength: float = 1.6,
        soil_enhancement_strength: float = 1.3,
    ):
        """
        Initialize image preprocessing service

        Args:
            enable_denoising: Whether to apply denoising
            enable_clahe: Whether to apply CLAHE contrast enhancement
            enable_texture_enhancement: Whether to enhance texture differences
            enable_vegetation_separation: Whether to separate trees from grassland
            enable_metallic_enhancement: Whether to enhance metallic structures
            enable_soil_enhancement: Whether to enhance soil/road features
            clahe_clip_limit: CLAHE clip limit
            clahe_tile_size: CLAHE tile grid size
            denoise_strength: Denoising strength
            texture_strength: Texture enhancement strength
            vegetation_separation_strength: How much to separate vegetation types
            metallic_enhancement_strength: How much to enhance metallic features
            soil_enhancement_strength: How much to enhance soil/road features
        """
        self.enable_denoising = enable_denoising
        self.enable_clahe = enable_clahe
        self.enable_texture_enhancement = enable_texture_enhancement
        self.enable_vegetation_separation = enable_vegetation_separation
        self.enable_metallic_enhancement = enable_metallic_enhancement
        self.enable_soil_enhancement = enable_soil_enhancement

        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size
        self.denoise_strength = denoise_strength
        self.texture_strength = texture_strength
        self.vegetation_separation_strength = vegetation_separation_strength
        self.metallic_enhancement_strength = metallic_enhancement_strength
        self.soil_enhancement_strength = soil_enhancement_strength

        # Initialize CLAHE
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size
            )

        # Initialize texture enhancement kernel (Laplacian)
        if self.enable_texture_enhancement:
            self.texture_kernel = np.array(
                [[0, -1, 0], [-1, 4 + self.texture_strength, -1], [0, -1, 0]]
            )

        logger.info(
            f"Image preprocessing initialized: "
            f"denoise={enable_denoising}, clahe={enable_clahe}, "
            f"texture_enhance={enable_texture_enhancement}, "
            f"veg_separation={enable_vegetation_separation}, "
            f"metallic_enhance={enable_metallic_enhancement}, "
            f"soil_enhance={enable_soil_enhancement}"
        )

    def preprocess(
        self, image: np.ndarray, return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Apply image specific preprocessing pipeline

        Args:
            image: Input image as numpy array (H, W, C)
            return_stats: Whether to return preprocessing statistics

        Returns:
            Preprocessed image, optionally with statistics
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is None or empty")

        # Store original for statistics
        original = image.copy()
        original_stats = self._calculate_image_stats(original)

        # Ensure image is uint8
        if image.dtype != np.uint8:
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Apply specialized preprocessing pipeline
        processed = image.copy()

        # 1. Denoising (remove noise first)
        if self.enable_denoising:
            processed = self._apply_denoising(processed)

        # 2. Vegetation separation (distinguish trees from grassland)
        if self.enable_vegetation_separation:
            processed = self._separate_vegetation_types(processed)

        # 3. Soil/road enhancement
        if self.enable_soil_enhancement:
            processed = self._enhance_soil_roads(processed)

        # 4. Metallic structure enhancement (towers and lines)
        if self.enable_metallic_enhancement:
            processed = self._enhance_metallic_structures(processed)

        # 5. CLAHE contrast enhancement
        if self.enable_clahe:
            processed = self._apply_clahe(processed)

        # 6. Texture enhancement (helps distinguish different surface types)
        if self.enable_texture_enhancement:
            processed = self._apply_texture_enhancement(processed)

        # Ensure output is uint8
        processed = np.clip(processed, 0, 255).astype(np.uint8)

        # Calculate final statistics
        final_stats = self._calculate_image_stats(processed)

        if return_stats:
            stats = {
                "original": original_stats,
                "processed": final_stats,
                "improvements": {
                    "contrast_ratio": final_stats["contrast_ratio"]
                    / original_stats["contrast_ratio"]
                    if original_stats["contrast_ratio"] > 0
                    else 1.0,
                    "brightness_change": final_stats["mean"] - original_stats["mean"],
                    "std_change": final_stats["std"] - original_stats["std"],
                    "texture_enhancement": self._calculate_texture_strength(processed)
                    / self._calculate_texture_strength(original)
                    if self._calculate_texture_strength(original) > 0
                    else 1.0,
                },
            }
            return processed, stats

        return processed

    def _apply_denoising(self, image: np.ndarray) -> np.ndarray:
        """Apply bilateral denoising to preserve edges while removing noise"""
        if len(image.shape) == 3:
            denoised = np.zeros_like(image)
            for i in range(image.shape[2]):
                denoised[:, :, i] = cv2.bilateralFilter(
                    image[:, :, i],
                    d=9,
                    sigmaColor=self.denoise_strength,
                    sigmaSpace=self.denoise_strength,
                )
            return denoised
        else:
            return cv2.bilateralFilter(
                image,
                d=9,
                sigmaColor=self.denoise_strength,
                sigmaSpace=self.denoise_strength,
            )

    def _separate_vegetation_types(self, image: np.ndarray) -> np.ndarray:
        """
        Separate trees from grassland by enhancing height/texture differences
        Trees typically have more texture variation and darker shadows
        """
        if len(image.shape) != 3:
            return image

        # Convert to float for calculations
        img_float = image.astype(np.float32)

        # Calculate vegetation index (green dominance)
        green_channel = img_float[:, :, 1]
        red_channel = img_float[:, :, 0]
        blue_channel = img_float[:, :, 2]

        # Vegetation index: (G - R) / (G + R + B)
        veg_index = (green_channel - red_channel) / (
            green_channel + red_channel + blue_channel + 1e-6
        )

        # Calculate local texture (standard deviation in small windows)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        texture_map = cv2.Laplacian(gray, cv2.CV_64F)
        texture_map = np.abs(texture_map)

        # Normalize texture map
        texture_map = texture_map / (np.max(texture_map) + 1e-6)

        # Trees: high vegetation index + high texture (shadows, canopy variation)
        # Grassland: high vegetation index + low texture (uniform surface)
        tree_mask = ((veg_index > 0.1) & (texture_map > 0.3)).astype(np.float32)
        grassland_mask = ((veg_index > 0.1) & (texture_map <= 0.3)).astype(np.float32)

        # Enhance trees (more contrast, darker shadows)
        tree_enhancement = 1.0 + self.vegetation_separation_strength * tree_mask
        img_float[:, :, 1] = (
            img_float[:, :, 1] * tree_enhancement
        )  # Enhance green for trees

        # Slightly suppress grassland (make it more uniform)
        grassland_suppression = 1.0 - 0.2 * grassland_mask
        img_float[:, :, 1] = img_float[:, :, 1] * grassland_suppression

        # Ensure no negative values
        img_float = np.clip(img_float, 0, 255)

        return img_float.astype(np.uint8)

    def _enhance_soil_roads(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance soil and road features
        These typically have low vegetation index and specific color characteristics
        """
        if len(image.shape) != 3:
            return image

        img_float = image.astype(np.float32)

        # Calculate vegetation index
        green_channel = img_float[:, :, 1]
        red_channel = img_float[:, :, 0]
        blue_channel = img_float[:, :, 2]

        veg_index = (green_channel - red_channel) / (
            green_channel + red_channel + blue_channel + 1e-6
        )

        # Soil/road mask: low vegetation index
        soil_mask = (veg_index < 0.05).astype(np.float32)

        # Enhance soil/road features
        enhancement_factor = 1.0 + self.soil_enhancement_strength * soil_mask

        # Apply enhancement to all channels
        for i in range(3):
            img_float[:, :, i] = img_float[:, :, i] * enhancement_factor

        # Clip to valid range
        img_float = np.clip(img_float, 0, 255)

        return img_float.astype(np.uint8)

    def _enhance_metallic_structures(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance metallic structures (towers and power lines)
        These appear as bright, low-saturation areas with specific geometric patterns
        """
        if len(image.shape) != 3:
            return image

        # Convert to HSV for saturation analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Extract V (value/brightness) and S (saturation) channels
        value_channel = hsv[:, :, 2].astype(np.float32)
        saturation_channel = hsv[:, :, 1].astype(np.float32)

        # Metallic mask: bright areas with low saturation
        brightness_threshold = np.percentile(value_channel, 75)  # Top 25% brightest
        saturation_threshold = np.percentile(
            saturation_channel, 25
        )  # Bottom 25% saturation

        metallic_mask = (
            (value_channel > brightness_threshold)
            & (saturation_channel < saturation_threshold)
        ).astype(np.float32)

        # Enhance metallic areas
        enhancement_factor = 1.0 + self.metallic_enhancement_strength * metallic_mask

        # Apply enhancement to RGB channels
        enhanced = image.astype(np.float32)
        for i in range(3):
            enhanced[:, :, i] = enhanced[:, :, i] * enhancement_factor

        # Clip to valid range
        enhanced = np.clip(enhanced, 0, 255)

        return enhanced.astype(np.uint8)

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Apply CLAHE to L channel only
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])

            # Convert back to RGB
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(image)

    def _apply_texture_enhancement(self, image: np.ndarray) -> np.ndarray:
        """Apply texture enhancement to make surface differences more visible"""
        if len(image.shape) == 3:
            # Apply texture enhancement to each channel
            enhanced = np.zeros_like(image)
            for i in range(image.shape[2]):
                enhanced[:, :, i] = cv2.filter2D(
                    image[:, :, i], -1, self.texture_kernel
                )
            return np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            enhanced = cv2.filter2D(image, -1, self.texture_kernel)
            return np.clip(enhanced, 0, 255).astype(np.uint8)

    def _calculate_texture_strength(self, image: np.ndarray) -> float:
        """Calculate texture strength using Laplacian operator"""
        if len(image.shape) == 3:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)

        # Calculate variance of Laplacian (texture measure)
        texture_strength = np.var(laplacian)

        return float(texture_strength)

    def _calculate_image_stats(self, image: np.ndarray) -> Dict:
        """Calculate image statistics for monitoring"""
        if len(image.shape) == 3:
            stats = {
                "mean": float(np.mean(image)),
                "std": float(np.std(image)),
                "min": int(np.min(image)),
                "max": int(np.max(image)),
                "contrast_ratio": float(np.max(image) - np.min(image)) / 255.0,
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
            }

            # Add per-channel stats
            stats["channel_stats"] = []
            for i in range(image.shape[2]):
                channel = image[:, :, i]
                stats["channel_stats"].append(
                    {
                        "channel": i,
                        "mean": float(np.mean(channel)),
                        "std": float(np.std(channel)),
                        "min": int(np.min(channel)),
                        "max": int(np.max(channel)),
                    }
                )
        else:
            stats = {
                "mean": float(np.mean(image)),
                "std": float(np.std(image)),
                "min": int(np.min(image)),
                "max": int(np.max(image)),
                "contrast_ratio": float(np.max(image) - np.min(image)) / 255.0,
                "channels": 1,
            }

        return stats

    def get_config(self) -> Dict:
        """Get current configuration"""
        return {
            "enable_denoising": self.enable_denoising,
            "enable_clahe": self.enable_clahe,
            "enable_texture_enhancement": self.enable_texture_enhancement,
            "enable_vegetation_separation": self.enable_vegetation_separation,
            "enable_metallic_enhancement": self.enable_metallic_enhancement,
            "enable_soil_enhancement": self.enable_soil_enhancement,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
            "denoise_strength": self.denoise_strength,
            "texture_strength": self.texture_strength,
            "vegetation_separation_strength": self.vegetation_separation_strength,
            "metallic_enhancement_strength": self.metallic_enhancement_strength,
            "soil_enhancement_strength": self.soil_enhancement_strength,
        }


# Convenience function for quick image preprocessing
def preprocess_image(
    image: np.ndarray,
    enable_denoising: bool = True,
    enable_clahe: bool = True,
    enable_texture_enhancement: bool = True,
    enable_vegetation_separation: bool = True,
    enable_metallic_enhancement: bool = True,
    enable_soil_enhancement: bool = True,
    return_stats: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Quick preprocessing function for image classification

    Args:
        image: Input image
        enable_denoising: Whether to apply denoising
        enable_clahe: Whether to apply CLAHE
        enable_texture_enhancement: Whether to enhance texture
        enable_vegetation_separation: Whether to separate vegetation types
        enable_metallic_enhancement: Whether to enhance metallic structures
        enable_soil_enhancement: Whether to enhance soil/road features
        return_stats: Whether to return statistics

    Returns:
        Preprocessed image, optionally with statistics
    """
    preprocessor = ImagePreProcessingService(
        enable_denoising=enable_denoising,
        enable_clahe=enable_clahe,
        enable_texture_enhancement=enable_texture_enhancement,
        enable_vegetation_separation=enable_vegetation_separation,
        enable_metallic_enhancement=enable_metallic_enhancement,
        enable_soil_enhancement=enable_soil_enhancement,
    )

    return preprocessor.preprocess(image, return_stats=return_stats)
