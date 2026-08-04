import pandas as pd

def mitBIH_dataset_generator(path: str, seed:int):
    print("Loading MIT-BIH dataset [...]")
    
    normal_ratio = 0.5
    abnormal_ratio = 0.5

    df = pd.read_pickle(f"{path}/MIT-BIH.pkl")

    normal = int(df[df["class"] == 0].shape[0] * normal_ratio)
    abnormal = int(df[df["class"] == 1].shape[0] * abnormal_ratio)

    normal_df = df[df["class"] == 0].sample(n=normal, random_state=seed)
    abnormal_df = df[df["class"] == 1].sample(n=abnormal, random_state=seed)

    df = pd.concat([normal_df, abnormal_df], axis=0)
    df = df.reset_index(drop=True)

    yield ("MIT-BIH", df)
