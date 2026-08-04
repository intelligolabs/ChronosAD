import numpy as np
import pandas as pd

from utils.utils import create_windowed_dataframe


def waveForm_dataset_generator(path: str, context: int):
    data = pd.read_csv(f"{path}/waveform.data", header=None)

    dataset = pd.DataFrame([{"signal": x.values[0:20], "class":1 if x.values[-1] == 0 else 0} for idx,x in data.iterrows()])
    dataset = dataset.reset_index(drop=True)
    yield ("WaveForm", dataset)
