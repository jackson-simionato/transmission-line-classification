"""
Generic SAM (Segment Anything Model) Segmentation Service
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False

logger = logging.getLogger(__name__)


class SAMSegmentationService:
    """
    Generic service for automatic image segmentation using SAM
    """

    def __init__(
        self,
        model_type: str = "vit_h",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize SAM segmentation service

        Args:
            model_type: SAM model type ('vit_h', 'vit_l', or 'vit_b')
            checkpoint_path: Path to SAM checkpoint file
            device: 'cuda' or 'cpu' (auto-detected if None)
        """
        if not SAM_AVAILABLE:
            raise ImportError(
                "segment-anything not installed. "
                "Install with: pip install segment-anything"
            )

        self.model_type = model_type
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Initializing SAM model '{model_type}' on {self.device}")

        if checkpoint_path is None:
            checkpoint_path = f"checkpoints/sam_{model_type}.pth"

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            logger.error(f"Checkpoint not found: {self.checkpoint_path}")
            raise FileNotFoundError(f"SAM checkpoint not found: {self.checkpoint_path}")

        # Load SAM model (MEMORY INTENSIVE: ~3-8 GB depending on model)
        logger.info(
            f"Loading checkpoint: {self.checkpoint_path.name} (~{self._get_checkpoint_size_mb():.0f} MB)"
        )
        self.sam = sam_model_registry[model_type](checkpoint=str(self.checkpoint_path))

        # Move to device (copies model to GPU if cuda)
        logger.info(f"Moving model to {self.device}...")
        self.sam.to(device=self.device)

        # Enable evaluation mode to save memory (disables dropout, batchnorm updates)
        self.sam.eval()

        # Enable gradient checkpointing for further memory reduction (if available)
        # This trades compute for memory by not storing intermediate activations
        if (
            hasattr(self.sam.image_encoder, "gradient_checkpointing")
            and self.device == "cuda"
        ):
            try:
                self.sam.image_encoder.gradient_checkpointing = True
                logger.info("Enabled gradient checkpointing for memory optimization")
            except Exception as e:
                logger.debug(f"Could not enable gradient checkpointing: {e}")

        logger.info(f"✓ SAM model loaded successfully on {self.device}")

        self.mask_generator = None

    def _get_checkpoint_size_mb(self) -> float:
        """Get checkpoint file size in MB"""
        if self.checkpoint_path.exists():
            return self.checkpoint_path.stat().st_size / (1024 * 1024)
        return 0.0

    def configure_mask_generator(
        self,
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.92,
        min_mask_region_area: int = 100,
        crop_n_layers: int = 1,
        box_nms_thresh: float = 0.7,
        **kwargs,
    ):
        """
        Configure automatic mask generator with custom parameters

        Args:
            points_per_side: Grid density for point sampling
            pred_iou_thresh: Minimum predicted IoU threshold
            stability_score_thresh: Minimum stability score threshold
            min_mask_region_area: Minimum segment area in pixels
            crop_n_layers: Number of crop layers for multi-scale
            box_nms_thresh: NMS threshold for removing overlapping boxes
            **kwargs: Additional parameters for SamAutomaticMaskGenerator
        """
        config = {
            "model": self.sam,
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "crop_n_layers": crop_n_layers,
            "crop_n_points_downscale_factor": 2,
            "min_mask_region_area": min_mask_region_area,
            "box_nms_thresh": box_nms_thresh,
            **kwargs,
        }

        self.mask_generator = SamAutomaticMaskGenerator(**config)

        logger.info(
            f"Mask generator configured: "
            f"points_per_side={points_per_side}, "
            f"min_area={min_mask_region_area}, "
            f"iou_thresh={pred_iou_thresh}"
        )

    def segment_image(
        self,
        image: np.ndarray,
        auto_configure: bool = True,
        max_area: Optional[int] = None,
        ignore_nodata: bool = True,
        nodata_mask: Optional[np.ndarray] = None,
    ) -> List[Dict]:
        """
        Segment a single image using SAM

        Args:
            image: RGB image as numpy array (H, W, 3)
            auto_configure: Auto-configure mask generator if not configured
            max_area: Maximum segment area in pixels (filters out huge segments)
            ignore_nodata: Whether to mask nodata pixels before SAM processing
            nodata_mask: Pre-computed nodata mask (if None, will be computed from image)

        Returns:
            List of segment dictionaries with masks and properties
        """
        if self.mask_generator is None:
            if auto_configure:
                logger.warning("Mask generator not configured, using default settings")
                self.configure_mask_generator()
            else:
                raise ValueError(
                    "Mask generator not configured. Call configure_mask_generator() first"
                )

        # Ensure uint8 format
        if image.dtype != np.uint8:
            if image.dtype == np.uint16:
                image = (image / 256).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        logger.debug(f"Segmenting image of shape {image.shape}")

        # Handle nodata pixels if requested
        if ignore_nodata:
            # Use pre-computed mask if provided, otherwise compute from image
            if nodata_mask is None:
                # Create mask where ANY band is 0 (nodata)
                nodata_mask = np.any(image == 0, axis=2)
                nodata_pixels = nodata_mask.sum()
                total_pixels = image.size // 3  # RGB image
                logger.info(f"Nodata pixels detected: {nodata_pixels}/{total_pixels} ({nodata_pixels/total_pixels*100:.1f}%)")
            else:
                nodata_pixels = nodata_mask.sum()
                total_pixels = image.size // 3  # RGB image
                logger.info(f"Using pre-computed nodata mask: {nodata_pixels}/{total_pixels} ({nodata_pixels/total_pixels*100:.1f}%)")
            
            if nodata_pixels > 0:
                # Mask out nodata pixels by setting them to a neutral value
                # This prevents SAM from segmenting these areas
                masked_image = image.copy()
                masked_image[nodata_mask] = [128, 128, 128]  # Neutral gray
                logger.info("Masked nodata pixels with neutral gray before SAM processing")
            else:
                masked_image = image
        else:
            masked_image = image

        # Generate masks on the masked image
        masks = self.mask_generator.generate(masked_image)

        logger.info(f"Generated {len(masks)} segments")

        # Filter by maximum area if specified
        if max_area is not None:
            original_count = len(masks)
            masks = [m for m in masks if m["area"] <= max_area]
            if len(masks) < original_count:
                logger.info(
                    f"Filtered out {original_count - len(masks)} segments larger than {max_area} pixels"
                )

        # Note: Nodata filtering is now handled by masking the input image
        # before SAM processing, so no post-processing filtering is needed

        return masks

    def clear_cache(self):
        """Clear GPU/CPU cache to free memory"""
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

    def unload_model(self):
        """Unload model from memory (useful for freeing memory between batches)"""
        if hasattr(self, "sam"):
            del self.sam
        if hasattr(self, "mask_generator"):
            del self.mask_generator
            self.mask_generator = None

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Model unloaded from memory")

    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        return {
            "model_type": self.model_type,
            "checkpoint_path": str(self.checkpoint_path),
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "configured": self.mask_generator is not None,
        }
