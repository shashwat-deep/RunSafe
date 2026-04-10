import pandas as pd
import numpy as np
import json
import joblib
import os
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from xgboost import XGBClassifier

class RunSafeDataProcessor:
    """
    Handles data extraction from SQLite, preprocessing, and training 
    the XGBoost foundational model using multi:softprob for downstream N=1 use.
    """
    
    def __init__(self, db_url="sqlite:///run_safe.db"):
        self.engine = create_engine(db_url)
        self.label_encoder = LabelEncoder()
        
    def extract_and_prepare_data(self) -> pd.DataFrame:
        """Extracts data via SQL JOIN and aligns features with the target."""
        
        # Query joining users and daily_logs
        query = """
        SELECT 
            u.age, u.foot_morphology, u.medical_history, u.average_weekly_mileage,
            d.previous_week_mileage, d.current_week_mileage, d.mileage_spike_pct, 
            d.average_rpe, d.max_pain_score, d.injury_risk_level
        FROM daily_logs d
        JOIN users u ON d.user_id = u.id
        WHERE d.injury_risk_level IS NOT NULL
        """
        
        df = pd.read_sql(query, self.engine)
        
        # Drop rows with nulls in vital dynamic columns (first days)
        df = df.dropna(subset=['previous_week_mileage', 'mileage_spike_pct'])
        return df

    def build_pipeline(self) -> Pipeline:
        """Creates the preprocessing and modeling pipeline."""
        
        numeric_features = ['age', 'average_weekly_mileage', 'previous_week_mileage', 
                            'current_week_mileage', 'mileage_spike_pct', 'average_rpe', 'max_pain_score']
        categorical_features = ['foot_morphology', 'medical_history']

        # Preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ])

        # XGBoost Classifier returning full probability distributions
        xgb_model = XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            learning_rate=0.05,
            max_depth=5,
            n_estimators=100,
            eval_metric='mlogloss',
            random_state=42
        )

        return Pipeline(steps=[('preprocessor', preprocessor), ('classifier', xgb_model)])

    def process(self) -> None:
        """Main execution flow."""
        print("Extracting data from SQLite...")
        df = self.extract_and_prepare_data()

        X = df.drop('injury_risk_level', axis=1)
        # Encode string labels (Low, Medium, High) to integers (0, 1, 2)
        y = self.label_encoder.fit_transform(df['injury_risk_level'])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Training Preprocessing Pipeline & XGBoost Foundational Model...")
        pipeline = self.build_pipeline()
        pipeline.fit(X_train, y_train)

        accuracy = pipeline.score(X_test, y_test)
        print(f"Model trained successfully. Test Accuracy: {accuracy:.4f}")

        # Save artifacts
        os.makedirs("models", exist_ok=True)
        joblib.dump(pipeline, "models/xgboost_pipeline.pkl")
        joblib.dump(self.label_encoder, "models/label_encoder.pkl")
        
        # Save Metadata
        with open("models/xgboost_metadata.json", "w") as f:
            metadata = {
                "objective": "multi:softprob",
                "classes": self.label_encoder.classes_.tolist(),
                "encoded_values": [0, 1, 2],
                "test_accuracy": float(accuracy)
            }
            json.dump(metadata, f, indent=4)
            
        print("Model, Pipeline, and LabelEncoder saved to /models.")

if __name__ == "__main__":
    processor = RunSafeDataProcessor()
    processor.process()
