import re
import os

import numpy as np
import pandas as pd
import scipy.io as sio

from utils.utils import create_windowed_dataframe


def cwru_get_data(filepath:str , classLabel:int):
    data = sio.loadmat(filepath)

    pattern = re.compile(r'.*_(DE|FE)_.*')
    keys = [key for key in data.keys() if pattern.match(key) and key != 'X217_DE_time']

    data_dict = {key:np.array(data[key]).ravel() for key in keys}
    data_dict['class'] = [classLabel]*len(data[keys[0]])

    patternDE = re.compile(r'.*_DE_.*')
    new_columns = {
        key: ('DE' if patternDE.match(key) else "FE")
        for key in keys
    }
    df = pd.DataFrame(data_dict)

    return df.rename(columns=new_columns)


def cwru_dataset_generator(path: str, context:int):
    print("Loading CWRU the dataset [...]")

    dataset_files = os.listdir(path)

    dfs = {}
    for f in dataset_files:
        dfs[f] = cwru_get_data(os.path.join(path, f), (0 if 'Time_Normal_1_098.mat' == f else 1))

    normal_df = dfs.pop('Time_Normal_1_098.mat')
    normal_df['class'] = [ 0 for _ in range(0, normal_df.shape[0])]

    normal_df = create_windowed_dataframe(normal_df, context)

    for fileName, fault_df in dfs.items():
        fault_df['class'] = [1 for _ in range(0, fault_df.shape[0])]
        fault_df = create_windowed_dataframe(fault_df, context)

        df = pd.concat([fault_df, normal_df],axis=0)
        df = df.reset_index(drop=True)
        yield (fileName, df)
