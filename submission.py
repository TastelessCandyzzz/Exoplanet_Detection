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
    first_row = pd.read_csv(file_path, nrows=1, header=None).iloc[0]
    def is_numeric(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    is_header = all(not is_numeric(val) for val in first_row)

    if is_header:
        df = pd.read_csv(file_path, header=0)   # first row as header
    else:
        df = pd.read_csv(file_path, header=None)  # no header
    new_data = df.to_numpy()

    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        data = np.vstack([data, new_data])
        print("New data added")
        print(f"New Data = {df.columns}")
        print(f"New Data = {new_data}")
    else:
        data = new_data

    np.save(DATA_PATH, data)
