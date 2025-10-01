import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data.npy")

def add_data(values):
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        data = np.vstack([data, values])
    else:
        data = np.array([values])
    np.save(DATA_PATH, data)


def add_csv(file_path):
    # load CSV into numpy
    df = pd.read_csv(file_path)
    new_data = df.to_numpy()

    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        data = np.vstack([data, new_data])
    else:
        data = new_data

    np.save(DATA_PATH, data)
