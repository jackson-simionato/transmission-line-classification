"""
ML Training Service for Transmission Line Classification
Handles complete ML training workflow including preprocessing, training, and evaluation
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE

logger = logging.getLogger(__name__)


class MLTrainingService:
    """
    Service for training and evaluating ML models for transmission line classification
    """

    def __init__(self, config: Optional[Dict] = None, enable_grid_search: bool = False):
        """
        Initialize ML training service

        Args:
            config: Optional configuration dictionary
            enable_grid_search: Whether to enable grid search for hyperparameter optimization
        """
        self.config = config or {}
        self.enable_grid_search = enable_grid_search
        self.scaler = StandardScaler()
        self.models = {}
        self.evaluation_results = {}
        self.grid_search_results = {}
        self.feature_selector = None
        self.selected_feature_names = None

        # Model configurations
        self.model_configs = {
            "random_forest": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_split": 10,
                "min_samples_leaf": 2,
                "random_state": 42,
            },
            "xgboost": {
                "n_estimators": 200,
                "learning_rate": 0.2,
                "max_depth": 6,
                "colsample_bytree": 0.8,
                "random_state": 42,
            },
            "svm": {"kernel": "rbf", "random_state": 42},
        }

    def preprocess_data(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str = "class_label",
        feature_columns: Optional[List[str]] = None,
        enable_feature_selection: bool = False,
        feature_selection_method: str = "mutual_info",
        n_features_to_select: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Preprocess training and test data

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            target_column: Name of target column
            feature_columns: List of feature columns (if None, auto-detect)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test, preprocessing_info)
        """
        logger.info("Starting data preprocessing...")

        # Auto-detect feature columns if not provided
        if feature_columns is None:
            feature_columns = [
                col
                for col in train_df.columns
                if col not in [target_column, "id", "segment_id", "tile_id"]
            ]

        logger.info(f"Using {len(feature_columns)} features: {feature_columns}")

        # Extract features and targets
        X_train = train_df[feature_columns].values
        y_train = train_df[target_column].values
        X_test = test_df[feature_columns].values
        y_test = test_df[target_column].values

        # Normalize features
        logger.info("Normalizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Compute class weights for imbalanced data
        logger.info("Computing class weights...")
        unique_classes = np.unique(y_train)
        class_weights = compute_class_weight(
            "balanced", classes=unique_classes, y=y_train
        )

        class_weight_dict = {
            int(cls): float(weight)
            for cls, weight in zip(unique_classes, class_weights)
        }

        # Create sample weights for XGBoost
        sample_weights = np.array([class_weight_dict[label] for label in y_train])

        # Feature selection (if enabled)
        if enable_feature_selection:
            X_train_scaled, X_test_scaled, selected_features = self.select_features(
                X_train_scaled, y_train, X_test_scaled,
                feature_columns, feature_selection_method, n_features_to_select
            )
            preprocessing_info = {
                "feature_columns": feature_columns,
                "selected_features": selected_features,
                "n_features_selected": len(selected_features),
                "class_weights": class_weight_dict,
                "sample_weights": sample_weights,
                "unique_classes": unique_classes,
                "train_shape": X_train_scaled.shape,
                "test_shape": X_test_scaled.shape,
            }
        else:
            preprocessing_info = {
                "feature_columns": feature_columns,
                "selected_features": feature_columns,
                "n_features_selected": len(feature_columns),
                "class_weights": class_weight_dict,
                "sample_weights": sample_weights,
                "unique_classes": unique_classes,
                "train_shape": X_train_scaled.shape,
                "test_shape": X_test_scaled.shape,
            }

        logger.info(
            f"Preprocessing complete. Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}"
        )
        logger.info(f"Class weights: {class_weight_dict}")

        return X_train_scaled, X_test_scaled, y_train, y_test, preprocessing_info

    def select_features(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        feature_names: List[str],
        method: str = "mutual_info",
        n_features: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Select top k features using various methods

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            feature_names: List of feature names
            method: Selection method ('mutual_info', 'f_score', 'rfe')
            n_features: Number of features to select

        Returns:
            Tuple of (X_train_selected, X_test_selected, selected_feature_names)
        """
        logger.info(f"Selecting {n_features} features using {method} method...")

        if method == "mutual_info":
            selector = SelectKBest(mutual_info_classif, k=n_features)
        elif method == "f_score":
            selector = SelectKBest(f_classif, k=n_features)
        elif method == "rfe":
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            selector = RFE(rf, n_features_to_select=n_features)
        else:
            raise ValueError(f"Unknown selection method: {method}")

        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)

        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_names = [feature_names[i] for i in selected_indices]

        self.feature_selector = selector
        self.selected_feature_names = selected_names

        logger.info(f"Selected features: {selected_names}")

        return X_train_selected, X_test_selected, selected_names

    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        preprocessing_info: Dict[str, Any],
        models_to_train: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train multiple ML models

        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            preprocessing_info: Preprocessing metadata
            models_to_train: List of models to train (default: all)

        Returns:
            Dictionary of trained models
        """
        if models_to_train is None:
            models_to_train = ["random_forest", "xgboost", "svm"]

        logger.info(f"Training models: {models_to_train}")
        trained_models = {}

        for model_name in models_to_train:
            logger.info(f"Training {model_name}...")

            try:
                if self.enable_grid_search:
                    model, grid_results = self._train_with_grid_search(
                        model_name, X_train, y_train, preprocessing_info
                    )
                    self.grid_search_results[model_name] = grid_results
                else:
                    if model_name == "random_forest":
                        model = self._train_random_forest(
                            X_train, y_train, preprocessing_info
                        )
                    elif model_name == "xgboost":
                        model = self._train_xgboost(
                            X_train, y_train, preprocessing_info
                        )
                    elif model_name == "svm":
                        model = self._train_svm(X_train, y_train, preprocessing_info)
                    else:
                        logger.warning(f"Unknown model: {model_name}")
                        continue

                trained_models[model_name] = model
                logger.info(f"Successfully trained {model_name}")

            except Exception as e:
                logger.error(f"Failed to train {model_name}: {str(e)}")
                continue

        self.models = trained_models
        return trained_models

    def _train_random_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        preprocessing_info: Dict[str, Any],
    ) -> RandomForestClassifier:
        """Train Random Forest model"""
        config = self.model_configs["random_forest"].copy()
        config["class_weight"] = preprocessing_info["class_weights"]

        model = RandomForestClassifier(**config)
        model.fit(X_train, y_train)
        return model

    def _train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        preprocessing_info: Dict[str, Any],
    ) -> xgb.XGBClassifier:
        """Train XGBoost model"""
        config = self.model_configs["xgboost"].copy()

        model = xgb.XGBClassifier(**config)
        model.fit(X_train, y_train, sample_weight=preprocessing_info["sample_weights"])
        return model

    def _train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        preprocessing_info: Dict[str, Any],
    ) -> SVC:
        """Train SVM model"""
        config = self.model_configs["svm"].copy()
        config["class_weight"] = preprocessing_info["class_weights"]

        model = SVC(**config)
        model.fit(X_train, y_train)
        return model

    def _train_with_grid_search(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        preprocessing_info: Dict[str, Any],
    ) -> Tuple[Any, Dict[str, Any]]:
        """
        Train model with grid search hyperparameter optimization

        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            preprocessing_info: Preprocessing metadata

        Returns:
            Tuple of (best_model, grid_search_results)
        """
        logger.info(f"Starting grid search for {model_name}...")

        # Get grid search parameters from config
        grid_config = self.config.get("grid_search", {})
        param_grid = grid_config.get(model_name, {})

        if not param_grid:
            logger.warning(
                f"No grid search parameters found for {model_name}, using default training"
            )
            if model_name == "random_forest":
                return self._train_random_forest(
                    X_train, y_train, preprocessing_info
                ), {}
            elif model_name == "xgboost":
                return self._train_xgboost(X_train, y_train, preprocessing_info), {}
            elif model_name == "svm":
                return self._train_svm(X_train, y_train, preprocessing_info), {}

        # Create base model
        if model_name == "random_forest":
            base_model = RandomForestClassifier(random_state=42)
            # Add class weights
            param_grid["class_weight"] = [preprocessing_info["class_weights"]]
        elif model_name == "xgboost":
            base_model = xgb.XGBClassifier(random_state=42)
            # XGBoost uses sample_weight instead of class_weight
            fit_params = {"sample_weight": preprocessing_info["sample_weights"]}
        elif model_name == "svm":
            base_model = SVC(random_state=42)
            # Add class weights
            param_grid["class_weight"] = [preprocessing_info["class_weights"]]
        else:
            raise ValueError(f"Unknown model for grid search: {model_name}")

        # Set up grid search
        cv_folds = grid_config.get("cv_folds", 3)
        scoring = grid_config.get("scoring", "f1_weighted")
        n_jobs = grid_config.get("n_jobs", -1)

        # Use stratified CV for imbalanced data
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

        grid_search = GridSearchCV(
            base_model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=1
        )

        # Fit grid search
        if model_name == "xgboost":
            grid_search.fit(X_train, y_train, **fit_params)
        else:
            grid_search.fit(X_train, y_train)

        # Extract results
        grid_results = {
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "best_index": grid_search.best_index_,
            "cv_results": grid_search.cv_results_,
        }

        logger.info(f"Grid search completed for {model_name}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_, grid_results

    def evaluate_models(
        self,
        models: Dict[str, Any],
        X_test: np.ndarray,
        y_test: np.ndarray,
        class_names: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Evaluate trained models

        Args:
            models: Dictionary of trained models
            X_test: Test features
            y_test: Test labels
            class_names: Optional class names for reporting

        Returns:
            Dictionary of evaluation results for each model
        """
        logger.info("Evaluating models...")
        results = {}

        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")

            try:
                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred, average=None, zero_division=0
                )

                # Cross-validation score with stratified CV to handle class imbalance
                try:
                    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(
                        model, X_test, y_test, cv=cv, scoring="accuracy"
                    )
                except ValueError as e:
                    if "Invalid classes" in str(e):
                        logger.warning(
                            f"Skipping CV for {model_name} due to missing classes in folds"
                        )
                        cv_scores = np.array([np.nan, np.nan, np.nan])
                    else:
                        raise

                # Confusion matrix
                cm = confusion_matrix(y_test, y_pred)

                # Classification report
                if class_names:
                    # Get unique classes present in y_test and y_pred
                    unique_test_classes = np.unique(y_test)
                    unique_pred_classes = np.unique(y_pred)
                    all_unique_classes = np.unique(
                        np.concatenate([unique_test_classes, unique_pred_classes])
                    )

                    # Filter class_names to only include present classes
                    present_class_names = [
                        class_names[i]
                        for i in all_unique_classes
                        if i < len(class_names)
                    ]

                    report = classification_report(
                        y_test,
                        y_pred,
                        target_names=present_class_names,
                        labels=all_unique_classes,
                        output_dict=True,
                        zero_division=0,
                    )
                else:
                    report = classification_report(
                        y_test, y_pred, output_dict=True, zero_division=0
                    )

                # Feature importance (if available)
                feature_importance = None
                if hasattr(model, "feature_importances_"):
                    feature_importance = model.feature_importances_.tolist()
                elif hasattr(model, "coef_"):
                    feature_importance = np.abs(model.coef_[0]).tolist()

                results[model_name] = {
                    "accuracy": float(accuracy),
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "f1_score": f1.tolist(),
                    "support": support.tolist(),
                    "cv_mean": float(cv_scores.mean()),
                    "cv_std": float(cv_scores.std()),
                    "confusion_matrix": cm.tolist(),
                    "classification_report": report,
                    "feature_importance": feature_importance,
                }

                logger.info(
                    f"{model_name} - Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {model_name}: {str(e)}")
                results[model_name] = {"error": str(e)}

        self.evaluation_results = results
        return results

    def get_model_comparison(self) -> pd.DataFrame:
        """
        Get model comparison summary

        Returns:
            DataFrame with model comparison metrics
        """
        if not self.evaluation_results:
            logger.warning("No evaluation results available")
            return pd.DataFrame(
                columns=[
                    "model",
                    "accuracy",
                    "cv_mean",
                    "cv_std",
                    "f1_macro",
                    "f1_weighted",
                ]
            )

        comparison_data = []
        for model_name, results in self.evaluation_results.items():
            if "error" in results:
                continue

            comparison_data.append(
                {
                    "model": model_name,
                    "accuracy": results["accuracy"],
                    "cv_mean": results["cv_mean"],
                    "cv_std": results["cv_std"],
                    "f1_macro": results["classification_report"]
                    .get("macro avg", {})
                    .get("f1-score", 0),
                    "f1_weighted": results["classification_report"]
                    .get("weighted avg", {})
                    .get("f1-score", 0),
                }
            )

        if not comparison_data:
            return pd.DataFrame(
                columns=[
                    "model",
                    "accuracy",
                    "cv_mean",
                    "cv_std",
                    "f1_macro",
                    "f1_weighted",
                ]
            )

        return pd.DataFrame(comparison_data).sort_values("accuracy", ascending=False)

    def get_feature_importance(
        self, model_name: str, feature_names: List[str]
    ) -> pd.DataFrame:
        """
        Get feature importance for a specific model

        Args:
            model_name: Name of the model
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        if model_name not in self.evaluation_results:
            logger.warning(f"No results for model: {model_name}")
            return pd.DataFrame()

        results = self.evaluation_results[model_name]
        if "feature_importance" not in results or results["feature_importance"] is None:
            logger.warning(f"No feature importance available for {model_name}")
            return pd.DataFrame()

        # Ensure feature names match the importance array length
        importance_array = results["feature_importance"]
        if len(feature_names) != len(importance_array):
            logger.warning(f"Feature names length ({len(feature_names)}) doesn't match importance array length ({len(importance_array)})")
            # Use selected feature names if available, otherwise truncate
            if hasattr(self, 'selected_feature_names') and self.selected_feature_names:
                feature_names = self.selected_feature_names
            else:
                feature_names = feature_names[:len(importance_array)]

        importance_df = pd.DataFrame(
            {"feature": feature_names, "importance": importance_array}
        ).sort_values("importance", ascending=False)

        return importance_df

    def get_grid_search_results(self) -> Dict[str, Dict[str, Any]]:
        """
        Get grid search results for all models

        Returns:
            Dictionary of grid search results for each model
        """
        return self.grid_search_results

    def get_best_parameters(self) -> pd.DataFrame:
        """
        Get best parameters from grid search as DataFrame

        Returns:
            DataFrame with best parameters for each model, including model_config as JSON
        """
        if not self.grid_search_results:
            return pd.DataFrame()

        results_data = []
        for model_name, results in self.grid_search_results.items():
            if not results:
                continue

            # Create standardized row with consistent schema
            row = {
                "model": model_name,
                "best_cv_score": results["best_score"],
                "model_config": json.dumps(results["best_params"]),
            }
            results_data.append(row)

        if not results_data:
            return pd.DataFrame()

        return pd.DataFrame(results_data)
