"""
Image Preprocessing Service for Transmission Line Classification
Simple QGIS-style contrast enhancement for SAM segmentation
"""

import logging
from typing import Dict, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class ImagePreProcessingService:
    """
    Simple preprocessing service for SAM segmentation
    Focuses on QGIS-style contrast enhancement only
    """

    def __init__(
        self,
        enable_denoising: bool = False,  # Disabled - original images are clean
        enable_clahe: bool = True,  # Main preprocessing step
        enable_sharpening: bool = False,  # Disabled - creates artifacts
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: Tuple[int, int] = (8, 8),
    ):
        """
        Initialize simple preprocessing service

        Args:
            enable_denoising: Whether to apply denoising (default: False)
            enable_clahe: Whether to apply CLAHE contrast enhancement (default: True)
            enable_sharpening: Whether to apply sharpening (default: False)
            clahe_clip_limit: CLAHE clip limit (1.0-4.0, higher=more contrast)
            clahe_tile_size: CLAHE tile grid size
        """
        self.enable_denoising = enable_denoising
        self.enable_clahe = enable_clahe
        self.enable_sharpening = enable_sharpening
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = clahe_tile_size

        # Initialize CLAHE
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_size
            )

        logger.info(
            f"Simple preprocessing initialized: "
            f"denoise={enable_denoising}, clahe={enable_clahe}, "
            f"sharpen={enable_sharpening}, clahe_clip_limit={clahe_clip_limit}"
        )

    def preprocess(
        self, image: np.ndarray, return_stats: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
        """
        Apply simple preprocessing pipeline (QGIS-style contrast enhancement)

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

        # Apply simple preprocessing pipeline
        processed = image.copy()

        # 1. Denoising (optional, usually disabled)
        if self.enable_denoising:
            processed = self._apply_denoising(processed)

        # 2. CLAHE contrast enhancement (main preprocessing step)
        if self.enable_clahe:
            processed = self._apply_clahe(processed)

        # 3. Sharpening (optional, usually disabled)
        if self.enable_sharpening:
            processed = self._apply_sharpening(processed)

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
                    sigmaColor=12.0,
                    sigmaSpace=12.0,
                )
            return denoised
        else:
            return cv2.bilateralFilter(
                image,
                d=9,
                sigmaColor=12.0,
                sigmaSpace=12.0,
            )

    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """Apply CLAHE contrast enhancement (QGIS-style)"""
        if len(image.shape) == 3:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

            # Apply CLAHE to L channel only
            lab[:, :, 0] = self.clahe.apply(lab[:, :, 0])

            # Convert back to RGB
            return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            return self.clahe.apply(image)

    def _apply_sharpening(self, image: np.ndarray) -> np.ndarray:
        """Apply sharpening filter (usually disabled to avoid artifacts)"""
        # Sharpening kernel
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

        if len(image.shape) == 3:
            # Apply sharpening to each channel
            sharpened = np.zeros_like(image)
            for i in range(image.shape[2]):
                sharpened[:, :, i] = cv2.filter2D(image[:, :, i], -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = cv2.filter2D(image, -1, kernel)
            return np.clip(sharpened, 0, 255).astype(np.uint8)

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
            "enable_sharpening": self.enable_sharpening,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_size": self.clahe_tile_size,
        }


# Convenience function for quick image preprocessing
def preprocess_image(
    image: np.ndarray,
    enable_denoising: bool = False,
    enable_clahe: bool = True,
    enable_sharpening: bool = False,
    clahe_clip_limit: float = 2.0,
    return_stats: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, Dict]]:
    """
    Quick preprocessing function for SAM segmentation

    Args:
        image: Input image
        enable_denoising: Whether to apply denoising (default: False)
        enable_clahe: Whether to apply CLAHE (default: True)
        enable_sharpening: Whether to apply sharpening (default: False)
        clahe_clip_limit: CLAHE clip limit (default: 2.0)
        return_stats: Whether to return statistics

    Returns:
        Preprocessed image, optionally with statistics
    """
    preprocessor = ImagePreProcessingService(
        enable_denoising=enable_denoising,
        enable_clahe=enable_clahe,
        enable_sharpening=enable_sharpening,
        clahe_clip_limit=clahe_clip_limit,
    )

    return preprocessor.preprocess(image, return_stats=return_stats)
