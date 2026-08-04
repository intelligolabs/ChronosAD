import numpy as np
import pandas as pd

from utils.utils import create_windowed_dataframe


def generate_landingGear_dataset(data:np.ndarray, labels:np.array, context:int):
    df = pd.DataFrame()
    columns = ['Time', 'MainActuatorPos', 'MainActuatorPressure', 'MainActuatorVelocity', 'LockActuatorPos', 'LockActuatorPressure', 'LockActuatorVelocity']

    for label in labels:
        ts = data[label]
        ts_df = pd.DataFrame(ts, columns=columns)
        # ts_df = (ts_df - ts_df.mean()) / (ts_df.std())
        ts_df['class'] = [0 if label == 0 else 1 for _ in range(len(ts_df))]
        ts_df_windowed = create_windowed_dataframe(ts_df, context, last_skip=False)
        df  = pd.concat([df, ts_df_windowed], axis=0)

    df.drop(['Time'], axis=1, inplace=True)
    
    return df


def landingGear_dataset_generator(path: str, context: int):
    print("Loading landing gear the dataset [...]")

    data = np.load(f'{path}/data.npy')
    labels = np.load(f'{path}/labels.npy')

    df = generate_landingGear_dataset(data, labels, context)
    df = df.reset_index(drop=True)
    yield ("landingGear", df)
