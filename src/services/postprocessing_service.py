"""
Segment Post-Processing Service for SAM Segmentation Results
Fills gaps, smooths boundaries, and merges small segments for complete coverage
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional
import logging
from scipy import ndimage

logger = logging.getLogger(__name__)


class SegmentPostprocessingService:
    """
    Service for post-processing SAM segmentation results
    Handles gap filling, boundary smoothing, and segment merging
    """
    
    def __init__(self, image_height: int, image_width: int):
        """
        Initialize post-processing service
        
        Args:
            image_height: Height of image in pixels
            image_width: Width of image in pixels
        """
        self.height = image_height
        self.width = image_width
        logger.info(f"Post-processing service initialized for {image_width}x{image_height} images")
    
    def fill_gaps_with_background_class(
        self,
        segments: List[Dict],
        min_gap_size: int = 500
    ) -> List[Dict]:
        """
        Fill unsegmented gaps with background segments
        
        Args:
            segments: List of segment dictionaries from SAM
            min_gap_size: Minimum gap size in pixels to fill
            
        Returns:
            List of segments including new gap-fill segments
        """
        # Create coverage mask
        coverage_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for segment in segments:
            coverage_mask[segment['segmentation']] = 1
        
        # Find gaps (unsegmented areas)
        gaps_mask = (coverage_mask == 0).astype(np.uint8)
        
        # Find connected components in gaps
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            gaps_mask, connectivity=8
        )
        
        new_segments = []
        segment_id_offset = max([s['segment_id'] for s in segments]) + 1 if segments else 0
        
        # Process each gap
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            
            if area >= min_gap_size:
                # Create segment for this gap
                gap_mask = (labels == i).astype(bool)
                
                # Get bounding box
                x, y, w, h = (
                    stats[i, cv2.CC_STAT_LEFT],
                    stats[i, cv2.CC_STAT_TOP],
                    stats[i, cv2.CC_STAT_WIDTH],
                    stats[i, cv2.CC_STAT_HEIGHT],
                )
                
                new_segment = {
                    'segmentation': gap_mask,
                    'area': int(area),
                    'bbox': [x, y, w, h],
                    'predicted_iou': 0.5,  # Assign moderate confidence
                    'stability_score': 0.5,
                    'segment_id': segment_id_offset + len(new_segments),
                    'is_gap_fill': True,  # Mark as gap-fill
                    'suggested_class': 'grassland'  # Suggest grassland for gaps
                }
                
                new_segments.append(new_segment)
        
        logger.info(f"Created {len(new_segments)} gap-fill segments")
        
        return segments + new_segments
    
    def fill_gaps_with_superpixels(
        self,
        segments: List[Dict],
        image: np.ndarray,
        n_segments: int = 200
    ) -> List[Dict]:
        """
        Fill unsegmented gaps using superpixel segmentation (SLIC)
        Better for preserving structure in uniform areas
        
        Args:
            segments: List of segment dictionaries from SAM
            image: Original RGB image
            n_segments: Number of superpixels to create in gaps
            
        Returns:
            List of segments including superpixel gap fills
        """
        try:
            from skimage.segmentation import slic
        except ImportError:
            logger.error("scikit-image not available. Install with: pip install scikit-image")
            logger.info("Falling back to simple gap filling")
            return self.fill_gaps_with_background_class(segments)
        
        # Create coverage mask
        coverage_mask = np.zeros((self.height, self.width), dtype=np.uint8)
        
        for segment in segments:
            coverage_mask[segment['segmentation']] = 1
        
        # Find gaps
        gaps_mask = (coverage_mask == 0)
        
        if gaps_mask.sum() == 0:
            logger.info("No gaps to fill")
            return segments
        
        logger.info(f"Filling {gaps_mask.sum()} gap pixels with superpixels")
        
        # Create superpixels only in gap regions
        gap_image = image.copy()
        gap_image[~gaps_mask] = 0  # Mask out already segmented areas
        
        superpixels = slic(gap_image, n_segments=n_segments, compactness=10.0, mask=gaps_mask)
        
        # Convert superpixels to segments
        new_segments = []
        segment_id_offset = max([s['segment_id'] for s in segments]) + 1 if segments else 0
        
        for sp_id in np.unique(superpixels):
            if sp_id == 0:  # Skip background
                continue
            
            sp_mask = (superpixels == sp_id)
            area = sp_mask.sum()
            
            if area > 50:  # Minimum size threshold
                # Get bounding box
                coords = np.argwhere(sp_mask)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                new_segment = {
                    'segmentation': sp_mask,
                    'area': int(area),
                    'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    'predicted_iou': 0.6,
                    'stability_score': 0.6,
                    'segment_id': segment_id_offset + len(new_segments),
                    'is_superpixel_fill': True,
                    'suggested_class': 'grassland'
                }
                
                new_segments.append(new_segment)
        
        logger.info(f"Created {len(new_segments)} superpixel gap-fill segments")
        
        return segments + new_segments
    
    def smooth_segment_boundaries(
        self,
        segments: List[Dict],
        kernel_size: int = 5,
        iterations: int = 1
    ) -> List[Dict]:
        """
        Smooth segment boundaries using morphological operations
        
        Args:
            segments: List of segment dictionaries
            kernel_size: Size of morphological kernel
            iterations: Number of smoothing iterations
            
        Returns:
            List of segments with smoothed boundaries
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        smoothed_segments = []
        
        for segment in segments:
            mask = segment['segmentation'].astype(np.uint8)
            
            # Close small holes
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
            
            # Smooth boundaries
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=iterations)
            
            # Update segment
            segment_copy = segment.copy()
            segment_copy['segmentation'] = mask.astype(bool)
            segment_copy['area'] = int(mask.sum())
            
            smoothed_segments.append(segment_copy)
        
        logger.info(f"Smoothed {len(smoothed_segments)} segment boundaries")
        
        return smoothed_segments
    
    def remove_large_overlapping_segments(
        self,
        segments: List[Dict],
        max_segment_area_ratio: float = 0.25,
        min_overlapped_segments: int = 3
    ) -> List[Dict]:
        """
        Remove large segments that overlap many other segments.
        Uses two criteria:
        1. Segments covering >25% of image that overlap ANY other segments
        2. Segments overlapping 3+ other segments significantly
        
        Args:
            segments: List of segment dictionaries
            max_segment_area_ratio: Max ratio of segment area to total image area (default: 0.5 = 50%)
            min_overlapped_segments: Minimum number of segments that must be overlapped
            
        Returns:
            List of segments with problematic large overlapping segments removed
        """
        total_pixels = self.height * self.width
        logger.info(f"Checking for large overlapping segments")
        logger.info(f"  Criteria 1: Segments > {max_segment_area_ratio*100}% of image ({int(total_pixels*max_segment_area_ratio)} pixels)")
        logger.info(f"  Criteria 2: Segments overlapping {min_overlapped_segments}+ others significantly")
        
        segments_to_remove = []
        
        for i, seg_i in enumerate(segments):
            overlaps = []
            
            # Check overlap with all other segments
            for j, seg_j in enumerate(segments):
                if i == j:
                    continue
                
                # Calculate overlap
                overlap = np.logical_and(seg_i['segmentation'], seg_j['segmentation']).sum()
                
                if overlap > 0:
                    # Calculate overlap ratio relative to the smaller segment
                    overlap_ratio_to_smaller = overlap / min(seg_i['area'], seg_j['area'])
                    # Also calculate ratio relative to the current segment
                    overlap_ratio_to_self = overlap / seg_i['area']
                    overlaps.append({
                        'index': j,
                        'overlap_pixels': overlap,
                        'overlap_ratio': overlap_ratio_to_smaller,
                        'overlap_ratio_self': overlap_ratio_to_self
                    })
            
            # Criteria 1: Very large segments (>50% of image) that overlap anything
            if seg_i['area'] > total_pixels * max_segment_area_ratio:
                if len(overlaps) > 0:
                    logger.info(f"  Segment {seg_i['segment_id']}: VERY LARGE ({seg_i['area']} pixels = " +
                               f"{seg_i['area']/total_pixels*100:.1f}% of image) overlaps {len(overlaps)} segments - marking for removal")
                    segments_to_remove.append(i)
                    continue
            
            # Criteria 2: Segments overlapping multiple others significantly
            # Count overlaps where >10% of the smaller segment overlaps
            significant_overlaps = [o for o in overlaps if o['overlap_ratio'] > 0.1]
            
            if len(significant_overlaps) >= min_overlapped_segments:
                # Calculate what % of THIS segment's area is overlapping
                total_overlap_pixels = sum(o['overlap_pixels'] for o in significant_overlaps)
                # But avoid double-counting (union, not sum)
                overlap_mask = np.zeros((self.height, self.width), dtype=bool)
                for o in significant_overlaps:
                    j = o['index']
                    overlap_with_j = np.logical_and(seg_i['segmentation'], segments[j]['segmentation'])
                    overlap_mask = np.logical_or(overlap_mask, overlap_with_j)
                
                unique_overlap_pixels = overlap_mask.sum()
                overlap_ratio_self = unique_overlap_pixels / seg_i['area']
                
                # SAFEGUARD: Don't remove thin elongated segments (likely transmission lines/cables)
                # Calculate elongation ratio using bounding box
                bbox = seg_i.get('bbox', [0, 0, 0, 0])
                bbox_width = bbox[2]
                bbox_height = bbox[3]
                
                if bbox_width > 0 and bbox_height > 0:
                    aspect_ratio = max(bbox_width, bbox_height) / min(bbox_width, bbox_height)
                    # If aspect ratio > 4 and segment is small-to-medium, it's likely a line/cable
                    # Lowered from 10 to 4 to catch transmission lines (typical aspect ratio ~5-6)
                    if aspect_ratio > 4 and seg_i['area'] < 50000:
                        logger.info(f"  Segment {seg_i['segment_id']} has high aspect ratio ({aspect_ratio:.1f}) " +
                                   f"- likely a line/cable, PRESERVING despite overlaps")
                        continue
                
                logger.info(f"  Segment {seg_i['segment_id']} ({seg_i['area']} pixels) overlaps {len(significant_overlaps)} segments " +
                           f"({overlap_ratio_self*100:.1f}% of its area overlaps) - marking for removal")
                segments_to_remove.append(i)
        
        # Remove problematic segments
        cleaned_segments = [seg for i, seg in enumerate(segments) if i not in segments_to_remove]
        
        logger.info(f"Removed {len(segments_to_remove)} large overlapping segments")
        logger.info(f"Remaining segments: {len(cleaned_segments)}")
        
        return cleaned_segments
    
    def remove_overlaps(self, segments: List[Dict]) -> List[Dict]:
        """
        Remove overlaps between segments by giving priority to original segments
        over gap-filled segments. Gap-filled segments will have overlapping areas removed.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segments with overlaps removed
        """
        # Separate original and gap-fill segments
        original_segments = [s for s in segments if not s.get('is_gap_fill', False) and not s.get('is_superpixel_fill', False)]
        gap_segments = [s for s in segments if s.get('is_gap_fill', False) or s.get('is_superpixel_fill', False)]
        
        if not gap_segments:
            logger.info("No gap-fill segments to process for overlaps")
            return segments
        
        logger.info(f"Removing overlaps: {len(original_segments)} original, {len(gap_segments)} gap-fill segments")
        
        # Create a combined mask of all original segments
        original_mask = np.zeros((self.height, self.width), dtype=bool)
        for segment in original_segments:
            original_mask = np.logical_or(original_mask, segment['segmentation'])
        
        # Remove overlapping areas from gap-fill segments
        cleaned_gap_segments = []
        for gap_seg in gap_segments:
            # Subtract original segments from gap segment
            cleaned_mask = np.logical_and(gap_seg['segmentation'], ~original_mask)
            
            # Only keep if there's still area left
            area = cleaned_mask.sum()
            if area > 0:
                gap_seg_copy = gap_seg.copy()
                gap_seg_copy['segmentation'] = cleaned_mask
                gap_seg_copy['area'] = int(area)
                cleaned_gap_segments.append(gap_seg_copy)
        
        logger.info(f"After overlap removal: kept {len(cleaned_gap_segments)}/{len(gap_segments)} gap-fill segments")
        
        return original_segments + cleaned_gap_segments
    
    def close_tile_boundaries(self, segments: List[Dict]) -> List[Dict]:
        """
        Fill gaps at tile boundaries by creating a square based on bbox
        and finding the difference with existing geometries.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segments including border segments
        """
        logger.info("Closing tile boundaries (bbox difference approach)")
        
        # Create a mask of all existing segments
        coverage_mask = np.zeros((self.height, self.width), dtype=bool)
        for segment in segments:
            coverage_mask = np.logical_or(coverage_mask, segment['segmentation'])
        
        # Find uncovered pixels
        uncovered_mask = ~coverage_mask
        
        if uncovered_mask.sum() == 0:
            logger.info("Tile already fully covered")
            return segments
        
        # Create segments for uncovered areas
        from scipy import ndimage
        
        # Label connected components in uncovered areas
        labeled_gaps, num_gaps = ndimage.label(uncovered_mask)
        
        if num_gaps == 0:
            logger.info("No gaps to fill")
            return segments
        
        logger.info(f"Found {num_gaps} gap components")
        
        # Create segments for gaps
        new_segments = []
        segment_id_offset = max([s['segment_id'] for s in segments]) + 1 if segments else 0
        
        for gap_label in range(1, num_gaps + 1):
            gap_mask = (labeled_gaps == gap_label)
            area = gap_mask.sum()
            
            if area > 0:
                # Get bounding box
                coords = np.argwhere(gap_mask)
                if len(coords) == 0:
                    continue
                y_min, x_min = coords.min(axis=0)
                y_max, x_max = coords.max(axis=0)
                
                new_segment = {
                    'segmentation': gap_mask,
                    'area': int(area),
                    'bbox': [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)],
                    'predicted_iou': 0.5,
                    'stability_score': 0.5,
                    'segment_id': segment_id_offset + len(new_segments),
                    'is_border_fill': True,
                    'suggested_class': 'grassland'
                }
                
                new_segments.append(new_segment)
        
        logger.info(f"Created {len(new_segments)} segments to close tile")
        
        return segments + new_segments
    
    def calculate_coverage_stats(self, segments: List[Dict]) -> Dict:
        """
        Calculate coverage statistics for segments
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            Dictionary with coverage statistics
        """
        # Create coverage mask
        coverage_mask = np.zeros((self.height, self.width), dtype=bool)
        
        for segment in segments:
            coverage_mask = np.logical_or(coverage_mask, segment['segmentation'])
        
        total_pixels = self.height * self.width
        covered_pixels = coverage_mask.sum()
        coverage_percent = (covered_pixels / total_pixels) * 100
        
        # Count gap-filled segments
        gap_fill_count = sum(1 for s in segments if s.get('is_gap_fill', False))
        superpixel_count = sum(1 for s in segments if s.get('is_superpixel_fill', False))
        
        stats = {
            'total_pixels': total_pixels,
            'covered_pixels': int(covered_pixels),
            'uncovered_pixels': int(total_pixels - covered_pixels),
            'coverage_percent': float(coverage_percent),
            'total_segments': len(segments),
            'gap_fill_segments': gap_fill_count,
            'superpixel_segments': superpixel_count,
            'mean_segment_area': float(np.mean([s['area'] for s in segments])) if segments else 0,
            'median_segment_area': float(np.median([s['area'] for s in segments])) if segments else 0
        }
        
        return stats
    
    def process_complete(
        self,
        segments: List[Dict],
        image: Optional[np.ndarray] = None,
        fill_gaps: bool = True,
        use_superpixels: bool = False,
        smooth_boundaries: bool = False,
        remove_overlaps: bool = True,
        remove_large_overlapping: bool = True,
        close_boundaries: bool = True,
        min_gap_size: int = 500,
        min_segment_size: int = 200,
        smoothing_kernel_size: int = 5,
        n_superpixels: int = 200
    ) -> List[Dict]:
        """
        Complete post-processing pipeline
        
        Args:
            segments: Original SAM segments
            image: Original RGB image (needed for superpixel method)
            fill_gaps: Whether to fill unsegmented gaps
            use_superpixels: Use superpixel method for gap filling
            smooth_boundaries: Whether to smooth segment boundaries
            remove_overlaps: Whether to remove overlaps between segments (default: True)
            remove_large_overlapping: Whether to remove large segments that overlap many others (default: True)
            min_gap_size: Minimum gap size to fill
            min_segment_size: Minimum segment size to keep
            smoothing_kernel_size: Kernel size for smoothing
            n_superpixels: Number of superpixels for gap filling
            
        Returns:
            Post-processed segments
        """
        processed = segments
        
        logger.info("Starting post-processing pipeline")
        logger.info(f"Initial segments: {len(processed)}")
        
        # Calculate initial coverage
        initial_stats = self.calculate_coverage_stats(processed)
        logger.info(f"Initial coverage: {initial_stats['coverage_percent']:.1f}%")
        
        # 1. Remove large overlapping segments (before gap filling)
        if remove_large_overlapping:
            processed = self.remove_large_overlapping_segments(processed)
        
        # 2. Fill gaps
        if fill_gaps:
            if use_superpixels and image is not None:
                processed = self.fill_gaps_with_superpixels(processed, image, n_superpixels)
            else:
                if use_superpixels and image is None:
                    logger.warning("Superpixels requested but no image provided, using simple fill")
                processed = self.fill_gaps_with_background_class(processed, min_gap_size=min_gap_size)
        
        # 3. Remove overlaps (prioritize original segments over gap fills)
        if remove_overlaps and fill_gaps:
            processed = self.remove_overlaps(processed)
        
        # 4. Smooth boundaries
        if smooth_boundaries:
            processed = self.smooth_segment_boundaries(processed, kernel_size=smoothing_kernel_size)
        
        # Note: Tile boundary closing is now handled at GeoDataFrame level in save_processed_geojson
        
        # Calculate final coverage
        final_stats = self.calculate_coverage_stats(processed)
        logger.info(f"Post-processing complete: {len(processed)} final segments")
        logger.info(f"Final coverage: {final_stats['coverage_percent']:.1f}%")
        
        return processed

