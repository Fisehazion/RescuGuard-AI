"""
RescuGuard AI - Memory-Optimized ML Pipeline Trainer
-----------------------------------------------------
Trains a multi-class Random Forest classification pipeline on the US Accidents dataset
with automated median/mode imputation, standard scaling, and one-hot encoding.
Evaluates using multi-class metrics (Accuracy, Macro F1, Precision, Recall, Confusion Matrix)
and exports the serialized pipeline artifact.
"""

import os
import json
import logging
from typing import Optional, Dict, Any
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("rescu_guard.trainer")


class USAccidentsModelTrainer:
    """
    Production-grade, memory-efficient trainer designed for multi-gigabyte tabular datasets.
    Optimizes memory footprint via selective column loading, integer downcasting,
    and stratified sampling.
    """

    def __init__(self, data_path: str = "../data/raw/US_Accidents_March23.csv"):
        self.data_path = Path(data_path)
        
        # Schema mapped to US_Accidents_March23.csv
        self.numeric_features = [
            'Start_Lat', 'Start_Lng', 'Temperature(F)', 
            'Humidity(%)', 'Visibility(mi)'
        ]
        self.boolean_features = ['Bump', 'Crossing', 'Traffic_Signal']
        self.categorical_features = ['Weather_Condition', 'Sunrise_Sunset']
        self.target = 'Severity'
        
        self.all_required_cols = (
            self.numeric_features + 
            self.boolean_features + 
            self.categorical_features + 
            [self.target]
        )

    def load_data(self, sample_size: Optional[int] = 300000) -> pd.DataFrame:
        """
        Reads only required columns from the CSV and performs memory-conscious preprocessing.
        
        Args:
            sample_size: Maximum rows to sample for training. Set to None for full dataset.
            
        Returns:
            Preprocessed pandas DataFrame ready for feature/target split.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Dataset missing at '{self.data_path.resolve()}'. "
                "Ensure 'US_Accidents_March23.csv' is placed inside 'data/raw/'."
            )

        logger.info(f"Reading selected columns from {self.data_path}...")
        
        # Memory optimization: read ONLY designated columns using fast C engine
        df = pd.read_csv(
            self.data_path, 
            usecols=self.all_required_cols,
            engine='c'
        )
        
        # Drop rows with null target
        df = df.dropna(subset=[self.target])
        df[self.target] = df[self.target].astype(int)

        # Cast boolean columns to integers (1 / 0)
        for col in self.boolean_features:
            df[col] = df[col].astype(bool).astype(int)

        # Apply stratified or random sampling for fast local training if specified
        if sample_size and len(df) > sample_size:
            logger.info(
                f"Full dataset size is {len(df):,} records. "
                f"Sampling {sample_size:,} rows for rapid iterative training..."
            )
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            logger.info(f"Training on full dataset containing {len(df):,} records.")

        return df

    def build_pipeline(self) -> Pipeline:
        """
        Constructs an end-to-end scikit-learn Pipeline with median numerical imputers,
        categorical mode/constant imputers, and balanced Random Forest classifier.
        """
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

        cat_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, self.numeric_features + self.boolean_features),
                ('cat', cat_pipeline, self.categorical_features)
            ]
        )

        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', RandomForestClassifier(
                n_estimators=100, 
                max_depth=18, 
                random_state=42, 
                class_weight='balanced',
                n_jobs=-1
            ))
        ])
        return pipeline

    def train_and_save(
        self, 
        sample_size: Optional[int] = 300000, 
        output_path: str = '../data/processed/model_pipeline.joblib'
    ) -> Dict[str, Any]:
        """
        Executes the training workflow, logs multi-class evaluation metrics,
        and saves the serialized model pipeline artifact.
        """
        df = self.load_data(sample_size=sample_size)

        X = df[self.numeric_features + self.boolean_features + self.categorical_features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info("Fitting scikit-learn machine learning pipeline...")
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        # Predictions on holdout test set
        y_pred = pipeline.predict(X_test)

        # Multi-class evaluation metrics
        train_acc = float(pipeline.score(X_train, y_train))
        test_acc = float(accuracy_score(y_test, y_pred))
        macro_f1 = float(f1_score(y_test, y_pred, average='macro'))
        weighted_f1 = float(f1_score(y_test, y_pred, average='weighted'))

        logger.info(f"Training Complete -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
        logger.info(f"Macro F1-Score: {macro_f1:.4f} | Weighted F1-Score: {weighted_f1:.4f}")
        logger.info("\n" + classification_report(y_test, y_pred))

        # Save metrics to JSON alongside model
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)

        metrics = {
            "train_accuracy": round(train_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "sample_size": sample_size or len(df)
        }
        metrics_file = output_path_obj.parent / "metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        # Save model pipeline
        joblib.dump(pipeline, output_path)
        logger.info(f"Model artifact successfully saved to '{output_path}'")
        logger.info(f"Evaluation metrics saved to '{metrics_file}'")

        return metrics


if __name__ == '__main__':
    # Execute training (sample 300k rows for rapid developer iteration)
    trainer = USAccidentsModelTrainer()
    trainer.train_and_save(sample_size=300000)