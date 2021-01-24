
import pandas as pd



def normalize(df):
    # normalize data
    df_all = pd.concat(df)

    train_mean = df_all.mean()
    train_std = df_all.std()
    #for i in range(len(file_names)):
    for i in range(len(df)):
        df[i] = (df[i] - train_mean) / train_std

    return df_all, train_mean, train_std


