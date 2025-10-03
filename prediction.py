import os, joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
PIPELINE_PATH = os.path.join(BASE_DIR, "transformation_pipeline.pkl")

def make_prediction(features):
    model = joblib.load(MODEL_PATH)  # Load latest model
    imputer = joblib.load(PIPELINE_PATH)
    features = np.array([features])
    prediction = model.predict(features)[0]
    return prediction
