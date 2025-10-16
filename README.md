# Orthophoto Imagery Classification

Automated classification of transmission line infrastructure from orthophoto imagery using computer vision and machine learning techniques.

## Overview

This project addresses the challenge of automatically identifying and classifying different land cover types around transmission lines from aerial imagery. The system segments orthophoto tiles into polygons, extracts comprehensive features, and classifies them into 5 categories:

- **Line** (0): Transmission line infrastructure
- **Tower** (1): Transmission line support structures  
- **Grassland** (2): Low vegetation areas
- **Tall Vegetation** (3): Trees and high vegetation
- **Soil** (4): Bare ground and soil areas

## Architecture

### Complete Pipeline
```
Orthophoto Tiles → Preprocessing → SAM Segmentation → Post-processing → Feature Extraction → ML Training → Classification → Validation
```

1. **Image Preprocessing**: QGIS-style contrast enhancement (CLAHE) to improve segmentation quality
2. **Image Segmentation**: Uses Meta's Segment Anything Model (SAM) to generate high-quality polygons with minimal computational overhead
3. **Post-processing**: Spatial operations suchas segment filtering and gap filling to optimize polygon quality
4. **Feature Extraction**: Extracts 1,000+ features from image segments, including spectral, geometric, texture, and CNN embeddings
5. **Ground Truth Creation**: Creation of randomly weighted points for labeling with QGIS
6. **ML Training**: Trains multiple algorithms (Random Forest, XGBoost, SVM, LightGBM) with hyperparameter tuning
7. **Prediction**: Full classification pipeline with GeoJSON output
8. **Validation**: Evaluation of model performance against ground truth points with detailed metrics

## Quick Start

### Installation
```bash
# Clone repository
git clone <repository-url>
cd transmission-line-classification

# Install dependencies
uv sync

# Download SAM ViT-B checkpoint
mkdir -p checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth \
  -O checkpoints/sam_vit_b.pth
```


## Project Structure

```
transmission-line-classification/
├── src/                          # Source code
│   ├── config/                   # Configuration management
│   ├── routes/                   # CLI entry points
│   │   ├── segment_ortofoto.py   # SAM segmentation
│   │   ├── postprocess_segments.py # Segment filtering
│   │   ├── extract_features.py   # Feature extraction
│   │   ├── train_models.py       # ML training
│   │   ├── predict_geojson.py    # Classification
│   │   └── validate_models.py    # Model validation
│   ├── services/                 # Core business logic
│   │   ├── preprocessing_service.py # Image preprocessing
│   │   ├── sam_segmentation_service.py # SAM integration
│   │   ├── postprocessing_service.py # Segment post-processing
│   │   ├── feature_extraction_service.py # Feature engineering
│   │   ├── ml_training_service.py # ML training
│   │   ├── ml_prediction_service.py # Classification
│   │   └── ml_validation_service.py # Model validation
│   └── utils/                    # Utility functions
├── data/                         # Data directories
│   ├── ortofoto/                 # Input orthophoto tiles
│   ├── sam_segments/             # Segmentation results
│   ├── features/                 # Extracted features
│   ├── predictions/              # Classification results
│   └── ground_truth/             # Manual labels
├── models/                       # Trained models
├── qgis_project/                 # QGIS project files
└── checkpoints/                  # Model checkpoints
```

## Configuration

All parameters are centralized in `src/config/settings.py`:
- Class definitions and mappings
- Train/test dataset splits
- Feature extraction settings
- ML model hyperparameters
- Grid search configurations


## Technical Details

### Image Preprocessing
- **CLAHE Enhancement**: QGIS-style contrast enhancement to improve segmentation quality
- **Nodata Handling**: Automatically detects and masks nodata pixels before processing

### Segmentation
- **Model**: ViT-b SAM - Meta's foundation model
- **Output**: Georeferenced polygons with quality metrics

### Post-processing
- **Segment Filtering**: Area constraints and quality thresholds to remove low-quality segments
- **Quality Metrics**: IoU scores, stability scores, and geometric validation
- **Memory Management**: Optimized for large-scale processing with configurable batch sizes

### Feature Engineering
- **Spectral**: RGB statistics, vegetation indices
- **Geometric**: Shape descriptors
- **Texture**: GLCM features, local binary patterns
- **CNN**: MobileNet V3 embeddings (optional)

### Machine Learning
- **Algorithms**: Random Forest, XGBoost, SVM, LightGBM
- **Optimization**: Grid search with cross-validation
- **Preprocessing**: Feature normalization and class weighting
- **Evaluation**: Comprehensive metrics with confusion matrices

### Validation Service
- **Ground Truth Comparison**: Validates predictions against manually labeled points
- **Performance Metrics**: Accuracy, precision, recall, F1-score with per-class analysis
- **Model Comparison**: Side-by-side evaluation of all trained models
- **Export Capabilities**: JSON, CSV, and HTML reports for detailed analysis

## Basic Usage

1. **Segment orthophoto tiles**:
```bash
python -m src.routes.segment_ortofoto \
  --input-dir data/ortofoto/raw \
  --output-dir data/sam_segments \
  --model vit_h
```

2. **Post-process segments**:
```bash
python -m src.routes.postprocess_segments \
  --input-dir data/sam_segments \
  --output-dir data/sam_segments_processed \
  --original-images-dir data/ortofoto/raw \
  --close-boundaries
```

3. **Extract features**:
```bash
python -m src.routes.extract_features \
  --labels-dir data/labels \
  --images-dir data/ortofoto/raw \
  --output-dir data/features
```

4. **Train models**:
```bash
python -m src.routes.train_models \
  --train-data data/features/features_train.csv \
  --test-data data/features/features_test.csv \
  --output-dir models
```

5. **Make predictions**:
```bash
python -m src.routes.predict_geojson \
  --features data/features/features_all.csv \
  --geojson data/labels/all_labels.geojson \
  --model-dir models
```

6. **Validate model performance**:
```bash
python -m src.routes.validate_models \
  --ground-truth data/ground_truth/all_ground_truth_points.geojson \
  --predictions-dir data/predictions \
  --output-dir data/validation
```

## 📊 Results

### Segmentation
- Larger SAM models could not be used due to local computational limitations.
- Segmentation performed well for linear features and towers.
- Segmentation struggled to distinguish grassland areas, likely due to their homogeneity.
- Increasing the grid density may help improve segmentation, but this was limited by available computational resources.


### Segmentation Post-processing
- Large overlapping polygons were eliminated
- Boundaries were smoothed and gaps were closed
- Overall segmentation quality improved significantly
- Some segments still contain mixed classes in the final results

### Feature Engineering
#### Feature Extraction

- Feature extraction generated a comprehensive set of per-segment statistics for all labeled points, including spectral means from RGB channels, shape metrics (e.g., area, perimeter, compactness), and texture descriptors.
- Most geometric features effectively separated towers and linear infrastructure from background segments.
- Color-based features alone were insufficient for discriminating grassland classes, likely due to overlapping spectral profiles.
- The extraction process was robust against most label noise, but small or highly irregular segments produced less reliable features.
- Pseudo-NDVI and other spectral indices provided additional separability for some classes, though the benefit was limited since only RGB bands were available.
- Overall, the engineered features enabled downstream models to learn class distinctions, but performance for vegetation types heavily depended on the quality and diversity of the sampling.

---
