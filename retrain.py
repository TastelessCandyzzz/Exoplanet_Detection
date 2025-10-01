import os, joblib
import numpy as np
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.npy")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

def train_model():
    # check if dataset exists
    if not os.path.exists(DATA_PATH):
        print("No dataset found to retrain.")
        return
    
    # load updated dataset
    data = np.load(DATA_PATH, allow_pickle=True)
    
    # assuming last column = target variable
    X = data[:, :-1]
    y = data[:, -1]

    # load old model
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Loaded existing model.")
    else:
        print("No model found, exiting.")
        return
    
    # retrain (refit) on new data
    model.fit(X, y)
    
    # save updated model
    joblib.dump(model, MODEL_PATH)
    print("Model retrained and saved at", MODEL_PATH)

if __name__ == "__main__":
    train_model()
