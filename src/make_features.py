import pandas as pd

steps_per_min = 12


def make_features(df):
    return __make_features2(df)


def __make_features1(df):
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp'].dt.hour
    features = ['hour', 'anglez', 'enmo']
    return df, features


def __make_features2(df):
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp'].dt.hour

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

    window = 30*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_3'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_3'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_3'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_3'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff_3'] = df['anglez_rolling_mean_left_aligned_3'] - df['anglez_rolling_mean_right_aligned_3']
    df['enmo_rolling_mean_diff_3'] = df['enmo_rolling_mean_left_aligned_3'] - df['enmo_rolling_mean_right_aligned_3']

    df['anglez_rolling_std_centered'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=1*steps_per_min, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=1*steps_per_min, min_periods=1, center=True).std()['enmo']

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
            'anglez_rolling_mean_right_aligned_3',
            'anglez_rolling_mean_left_aligned_3',
            'enmo_rolling_mean_right_aligned_3',
            'enmo_rolling_mean_left_aligned_3',
            'anglez_rolling_mean_diff_3',
            'enmo_rolling_mean_diff_3',
            'anglez_rolling_std_centered',
            'enmo_rolling_std_centered',
            ]
    return df, features
