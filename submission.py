import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.npy")

def add_data(values):
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        data = np.vstack([data, values])
    else:
        data = np.array([values])
    np.save(DATA_PATH, data)
