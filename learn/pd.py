import pandas as pd
import numpy as np


def get_mmd_per_minute(series):
    # Ref: Sleep Stage Classification Using EEG Signal Analysis A Comprehensive Survey and New Investigation
    return np.sqrt(((series.max() - series.min())**2) + ((series.idxmax() - series.idxmin())**2))


def main5():
    df = pd.DataFrame([
        ('id1', 1, 1, 10), ('id2', 1, 2, 11),
        ('id1', 1, 3, 15), ('id2', 1, 4, 11),
        ('id1', 2, 5, 10), ('id2', 2, 28, 10),
        ('id1', 2, 7, 11), ('id2', 2, 16, 13)
    ], columns=['series_id', 'time', 'anglez', 'awake'])
    print(df)
    df2 = df.groupby(['series_id', 'time'])['awake'].agg(
        get_mmd_per_minute).rename('awake2').reset_index()
    print(df2)

    df3 = df.merge(df2, on=['series_id', 'time'], how='left')
    print(df3)


def main4():
    df = pd.DataFrame([
        ('id1', 1, 1, 10), ('id2', 1, 2, 11),
        ('id1', 1, 3, 10), ('id2', 1, 4, 11),
        ('id1', 2, 5, 10), ('id2', 2, 8, 10),
        ('id1', 2, 7, 11), ('id2', 2, 16, 11)
    ], columns=['series_id', 'time', 'anglez', 'awake'])
    print(df)
    print(df.groupby(['series_id', 'time'])['awake'].agg(lambda x: x.value_counts().index.max()).rename('awake2'))


def main():
    df = pd.DataFrame([
        ('id1', 1), ('id2', 2),
        ('id1', 3), ('id2', 4),
        ('id1', 5), ('id2', 8),
        ('id1', 7), ('id2', 16)
    ], columns=['series_id', 'anglez'])
    print(df)
    
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_mean_right'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=3, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean_left'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=3, min_periods=1).mean().shift(-2).fillna(method='ffill')
        print(df)


def main3():
    df = pd.DataFrame([
        ('id1', 1), ('id2', 2),
        ('id1', 3), ('id2', 4),
        ('id1', 5), ('id2', 8),
        ('id1', 7), ('id2', 16)
    ], columns=['series_id', 'step'])
    print(df)
    print(
        df.groupby('series_id', as_index=False)['step'].rolling(
            window=3, min_periods=1).mean()
    )
    raise
    print(
        df.groupby('series_id', as_index=True)['step'].rolling(
            window=3, min_periods=1).mean().shift(-2)
    )
    df['step2'] = df.groupby('series_id', as_index=False)['step'].rolling(
        window=3, min_periods=1, center=True).mean()['step']
    print(df)


def main2():
    df = pd.DataFrame({'A': [2, 3, 6, 8, 20, 27]})
    print(df.rolling(window=3, min_periods=1).mean())
    print(df.rolling(window=3, min_periods=1, center=True).mean())
    print(df.rolling(window=3, min_periods=1).mean().shift(-2).fillna(method='ffill'))
    print(pd.__version__)


if __name__ == '__main__':
    main5()
