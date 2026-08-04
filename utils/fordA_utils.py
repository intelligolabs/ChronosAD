import pandas as pd
import numpy as np
from utils.utils import create_windowed_dataframe

def fordADatasetGenerator(path: str, context: int):
    dataset = pd.DataFrame()

    test = pd.read_csv(f"{path}/FordA_TEST.tsv", delimiter='\t', header=None)
    train = pd.read_csv(f"{path}/FordA_TRAIN.tsv", delimiter='\t', header=None)

    concat_dataset = pd.concat([test, train], axis=0)

    for idx, row in concat_dataset.iterrows():
        label = row[0]
        data = np.array(row[1:])
        df = pd.DataFrame({"signal":data, "class": [0 if label == -1 else 1 for _ in range(len(data))]})
        windowed = create_windowed_dataframe(df, context)
        dataset = pd.concat([dataset, windowed], axis=0)

    dataset = dataset.reset_index(drop=True)
    yield ("FordA", dataset)