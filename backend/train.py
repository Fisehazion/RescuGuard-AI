import os
import joblib
import pandas as pd
from typing import Optional
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier


class USAccidentsModelTrainer:
    """
    OOP Trainer optimized for US_Accidents_March23.csv (~3 GB).
    Handles selective column loading, missing data imputation, and configurable sampling.
    """

    def __init__(self, data_path: str = "../data/raw/US_Accidents_March23.csv"):
        self.data_path = data_path
        
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
        """Reads required columns from large CSV and samples rows to optimize RAM usage."""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset missing at '{self.data_path}'. "
                "Ensure 'US_Accidents_March23.csv' is placed inside 'data/raw/'."
            )

        print(f"[INFO] Reading selected columns from {self.data_path}...")
        
        # Memory optimization: read ONLY needed columns
        df = pd.read_csv(
            self.data_path, 
            usecols=self.all_required_cols,
            engine='c'
        )
        
        # Drop rows missing the target variable
        df = df.dropna(subset=[self.target])
        df[self.target] = df[self.target].astype(int)

        # Convert boolean columns (True/False -> 1/0)
        for col in self.boolean_features:
            df[col] = df[col].astype(bool).astype(int)

        # Apply row sampling for fast local training if specified
        if sample_size and len(df) > sample_size:
            print(f"[INFO] Dataset size is {len(df):,} rows. Sampling {sample_size:,} rows for rapid training...")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        else:
            print(f"[INFO] Training on full dataset of {len(df):,} rows.")

        return df

    def build_pipeline(self) -> Pipeline:
        """Constructs an sklearn processing and classification pipeline with inline imputers."""
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
    ) -> None:
        df = self.load_data(sample_size=sample_size)

        X = df[self.numeric_features + self.boolean_features + self.categorical_features]
        y = df[self.target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("[INFO] Fitting machine learning pipeline...")
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        train_acc = pipeline.score(X_train, y_train)
        test_acc = pipeline.score(X_test, y_test)
        print(f"[SUCCESS] Training Complete -> Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        joblib.dump(pipeline, output_path)
        print(f"[SUCCESS] Model artifact saved to '{output_path}'")


if __name__ == '__main__':
    # Set sample_size=None to train on all ~7.7 million rows if running on a high-RAM machine
    trainer = USAccidentsModelTrainer()
    trainer.train_and_save(sample_size=300000)