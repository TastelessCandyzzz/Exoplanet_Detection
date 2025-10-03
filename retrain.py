import os
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

# --- Define Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.npy")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
# NEW: Add the path to your saved pipeline
PIPELINE_PATH = os.path.join(BASE_DIR, "transformation_pipeline.pkl")


def train_model():
    # --- 1. Check if required files exist ---
    if not os.path.exists(DATA_PATH):
        print("No dataset found to retrain.")
        return

    # NEW: Check for the pipeline file
    if not os.path.exists(PIPELINE_PATH):
        print("No pipeline.pkl found. Cannot transform data.")
        return

    if not os.path.exists(MODEL_PATH):
        print("No model.pkl found to retrain.")
        return

    print("Loading data, pipeline, and model...")
    # --- 2. Load all necessary components ---
    data = np.load(DATA_PATH, allow_pickle=True)
    pipeline = joblib.load(PIPELINE_PATH) # Load the preprocessing pipeline
    model = joblib.load(MODEL_PATH)       # Load the existing model

    # --- 3. Prepare and Transform the Data ---
    # Assuming the last column is the target variable
    X = data[:, 1:]  # Features
    y = data[:, 0]   # Target

    print("Transforming features using the pipeline...")
    # NEW: Use the loaded pipeline to transform the features
    # We use .transform() because the pipeline is already fitted
    X_transformed = pipeline.fit_transform(X)

    # --- 4. Retrain the Model ---
    print("Retraining model on the transformed data...")
    # Retrain (refit) the model on the NEWLY TRANSFORMED data
    model.fit(X_transformed, y)

    # --- 5. Save the Updated Model ---
    joblib.dump(model, MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    print("Model has been retrained and saved at", MODEL_PATH)


if __name__ == "__main__":
    train_model()
