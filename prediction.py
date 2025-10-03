import os, joblib
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "transformation_pipeline.pkl")

def make_prediction(features):
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    # Reshape features for a single prediction
    features_array = np.array([features])

    # --- BUG FIX: Use the pipeline to transform the features ---
    transformed_features = pipeline.transform(features_array)

    prediction = model.predict(transformed_features)[0]
    return prediction

# --- NEW: Function for handling bulk predictions from a CSV ---
def make_bulk_prediction(features_df):
    model = joblib.load(MODEL_PATH)
    pipeline = joblib.load(PIPELINE_PATH)

    # The input is already a DataFrame, so we can transform it directly
    transformed_features = pipeline.transform(features_df)

    predictions = model.predict(transformed_features)
    return predictions
