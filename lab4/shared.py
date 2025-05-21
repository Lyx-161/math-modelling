import numpy as np
import pandas as pd


def load_data(path_dir):
    data = pd.read_excel(path_dir)
    years = data["Year"].values
    population = data["Population"].values
    return years, population

def evaluate_model(true, pred, max):
    mse = np.mean((true - pred) ** 2) / (max * max)
    mae = np.mean(np.abs(true - pred)) / max
    return mse, mae