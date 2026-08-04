import os

import pandas as pd
import matplotlib.pyplot as plt
import time


def create_windowed_dataframe(data: pd.DataFrame, window_size: int, last_skip = True):
    feature_columns = [col for col in data.columns if col != 'class']

    num_windows = data.shape[0] // window_size

    if not last_skip and data.shape[0] % window_size != 0:
        num_windows += 1

    new_rows = []

    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size

        window = data.iloc[start_idx:end_idx]

        new_row = {}

        for col in feature_columns:
            new_row[col] = window[col].values

        new_row['class'] = window['class'].iloc[0]

        new_rows.append(new_row)

    new_df = pd.DataFrame(new_rows)

    return new_df


def get_chronos_cache_embeddings(path:str, dataset:str, context:int):
    dataset = dataset.lower()
    filename = os.path.splitext(os.path.basename(path))[0]
    cacheFilePath = f".cache/{dataset}/{filename}_{context}.cache"

    if not os.path.isdir(".cache"):
        print("Creating cache directory [...]")
        os.mkdir(".cache")
        return None

    if not os.path.isdir(f".cache/{dataset}"):
        print(f"Creating {dataset} cache directory [...]")
        os.mkdir(f".cache/{dataset}")
        return None

    if os.path.exists(cacheFilePath):
        print("Loading data from cache [...]")
        return pd.read_pickle(cacheFilePath)

    return None


def save_chronos_cache_embeddings(data:pd.DataFrame, path:str, dataset: str, context:int):
    dataset = dataset.lower()
    filename = os.path.splitext(os.path.basename(path))[0]
    cacheFilePath = f".cache/{dataset}/{filename}_{context}.cache"

    if not os.path.isdir(".cache"):
        print("Creating cache directory [...]")
        os.mkdir(".cache")

    if not os.path.isdir(f".cache/{dataset}"):
        print(f"Creating {dataset} cache directory [...]")
        os.mkdir(f".cache/{dataset}")

    print("Saving data to cache [...]")
    data.to_pickle(cacheFilePath)
    print("Data saved to cache!")

def save_qualitative_results(test_X:pd.DataFrame, test_y:pd.DataFrame, predictions:list[int],dataset_name:str):
    if not os.path.isdir("qualitative_results"):
        print("Creating qualitative_results directory [...]")
        os.mkdir("qualitative_results")

    if os.path.isdir(f"qualitative_results/{dataset_name}"):
        os.system(f"rm -rf qualitative_results/{dataset_name}")

    if not os.path.isdir(f"qualitative_results/{dataset_name}"):
        print(f"Creating {dataset_name} qualitative_results directory [...]")
        os.mkdir(f"qualitative_results/{dataset_name}")

    for i in range(len(predictions)):
        row = test_X.iloc[i]
        for col in row.keys():
            plt.plot(row[col], color= 'red' if test_y.iloc[i]['class'] == 1 else 'green')
            plt.title(f"Predicted Label: {predictions[i]}")
            plt.savefig(f"qualitative_results/{dataset_name}/{time.time_ns()}_{col}_{test_y.iloc[i]['class']}_{int(predictions[i])}.png")
            plt.close()
