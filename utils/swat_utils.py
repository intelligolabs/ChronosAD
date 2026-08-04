import numpy as np
import pandas as pd

from imblearn.under_sampling import RandomUnderSampler

from utils.utils import create_windowed_dataframe


def split_by_class(df:pd.DataFrame):
    df_list = []

    start_idx = 0

    for i in range(1, len(df)):
        if df['class'].iloc[i] != df['class'].iloc[start_idx]:
            df_list.append(df.iloc[start_idx:i])
            start_idx = i
    df_list.append(df.iloc[start_idx:])
    
    return df_list


def normalize(df: pd.DataFrame):
    df.columns = df.iloc[0]
    df.drop(0, axis=0, inplace=True,)
    df.reset_index(inplace=True, drop=True)

    df.columns = [c.strip() for c in df.columns] #Normalizing column names

    #Timestamp Normalization
    df['Timestamp'] = df['Timestamp'].str.strip()
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d/%m/%Y %I:%M:%S %p')
    df['Timestamp'] = df['Timestamp'].astype(int) // 10**9
    df = df.set_index("Timestamp")

    #Excuding columns, it is categorical and does not change much. Also class labels to add it at the end
    excluding_columns = ['MV101', 'P101', 'P102', 'MV201', 'P201', 'P202', 'P203', 'P204', 'P205', 'P206', 'MV301',
                           'MV302', 'MV303', 'MV304', 'P301', 'P302', 'P401', 'P402', 'P403', 'P404', 'UV401', 'P501', 'P502',
                           'P601', 'P602', 'P603','Normal/Attack']

    # top_10_features = ['LIT101', 'FIT101', 'P101', 'MV101', 'LIT301', 'AIT201', 'P601', 'FIT501', 'DPIT301', 'PIT501']

    new_df = df[df.columns[~df.columns.isin(excluding_columns)]].copy()
    # new_df = df[df.columns[df.columns.isin(top_10_features)]].copy()

    for column in new_df.columns:
        new_df[column] = new_df[column].astype(float)

    mean = new_df.mean()
    std = new_df.std()
    # mean = 0
    # std = 1

    normalized = (new_df - mean) / std
    normalized['class'] = df['Normal/Attack']

    return normalized


def swat_dataset_generator(path: str, context: int, seed: int, sampling_rate:float = 0.5):
    print("Reading SWaT dataset [...]")
    data = pd.read_csv(f"{path}/SWaT_Dataset_Attack_v0.csv", low_memory=False)
    data.drop(["Unnamed: 0.1"], axis=1, inplace=True)

    normalized = normalize(data)
    splited = split_by_class(normalized)

    data1 = pd.read_csv(f"{path}/SWaT_Dataset_Normal_v1.csv", low_memory=False)
    data1.drop(["Unnamed: 0.1"], axis=1, inplace=True)
    normalized1 = normalize(data1)
    splited1 = split_by_class(normalized1)

    splited = splited + splited1

    dataset = pd.DataFrame(columns=normalized.columns)

    print("Creating window SWaT dataset [...]")

    for d in splited:
        window = create_windowed_dataframe(d, context, last_skip=False)
        window['class'] = [0 if x == 'Normal' else 1 for x in window['class']]

        dataset = pd.concat([dataset, window], axis=0)

    rus = RandomUnderSampler(sampling_strategy=sampling_rate, random_state=seed)
    X_res, y_res = rus.fit_resample(dataset[dataset.columns[:-1]], dataset['class'].astype(int))
    X_res['class'] = y_res

    X_res = X_res.reset_index(drop=True)
    yield ("SWaT", X_res)
