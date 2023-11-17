import pandas as pd


steps_per_min = 12

features = ['hour',
        'anglez',
        'enmo',
        'anglez_rolling_mean_right_aligned_1',
        'anglez_rolling_mean_left_aligned_1',
        'enmo_rolling_mean_right_aligned_1',
        'enmo_rolling_mean_left_aligned_1',
        'anglez_rolling_mean_diff_1',
        'enmo_rolling_mean_diff_1',
        'anglez_rolling_mean_right_aligned_2',
        'anglez_rolling_mean_left_aligned_2',
        'enmo_rolling_mean_right_aligned_2',
        'enmo_rolling_mean_left_aligned_2',
        'anglez_rolling_mean_diff_2',
        'enmo_rolling_mean_diff_2',
        'anglez_rolling_std_centered_1',
        'enmo_rolling_std_centered_1',
        'anglez_rolling_std_centered_2',
        'enmo_rolling_std_centered_2',
        ]


def make_features(df):
    return __make_features2(df)


def __make_features2(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    window = 1*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_1'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_1'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_1'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_1'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff_1'] = df['anglez_rolling_mean_left_aligned_1'] - df['anglez_rolling_mean_right_aligned_1']
    df['enmo_rolling_mean_diff_1'] = df['enmo_rolling_mean_left_aligned_1'] - df['enmo_rolling_mean_right_aligned_1']

    window = 5*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_2'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_2'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_2'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_2'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff_2'] = df['anglez_rolling_mean_left_aligned_2'] - df['anglez_rolling_mean_right_aligned_2']
    df['enmo_rolling_mean_diff_2'] = df['enmo_rolling_mean_left_aligned_2'] - df['enmo_rolling_mean_right_aligned_2']

    window = 1*steps_per_min
    df['anglez_rolling_std_centered_1'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered_1'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']

    window = 10*steps_per_min
    df['anglez_rolling_std_centered_2'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered_2'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']

    return df
