import pandas as pd
from common import steps_per_min


features1 = ['hour',
        'anglez',
        'enmo',
        ]

features3 = ['hour',
        'anglez',
        'enmo',
        'anglez__std_centered',
        'enmo__std_centered',
        ]

features5 = ['hour',
        'anglez',
        'enmo',
        'anglez__std_centered',
        'enmo__std_centered',

        'anglez_rolling_mean_right_aligned_60',
        'anglez_rolling_mean_left_aligned_60',
        'enmo_rolling_mean_right_aligned_60',
        'enmo_rolling_mean_left_aligned_60',
        'anglez__std_centered_rolling_mean_right_aligned_60',
        'anglez__std_centered_rolling_mean_left_aligned_60',
        'enmo__std_centered_rolling_mean_right_aligned_60',
        'enmo__std_centered_rolling_mean_left_aligned_60',

        'anglez_rolling_mean_diff_60',
        'enmo_rolling_mean_diff_60',
        'anglez__std_centered_rolling_mean_diff_60',
        'enmo__std_centered_rolling_mean_diff_60',

        'anglez_rolling_mean_right_aligned_30',
        'anglez_rolling_mean_left_aligned_30',
        'enmo_rolling_mean_right_aligned_30',
        'enmo_rolling_mean_left_aligned_30',
        'anglez__std_centered_rolling_mean_right_aligned_30',
        'anglez__std_centered_rolling_mean_left_aligned_30',
        'enmo__std_centered_rolling_mean_right_aligned_30',
        'enmo__std_centered_rolling_mean_left_aligned_30',

        'anglez_rolling_mean_diff_30',
        'enmo_rolling_mean_diff_30',
        'anglez__std_centered_rolling_mean_diff_30',
        'enmo__std_centered_rolling_mean_diff_30',

        'anglez_rolling_mean_right_aligned_15',
        'anglez_rolling_mean_left_aligned_15',
        'enmo_rolling_mean_right_aligned_15',
        'enmo_rolling_mean_left_aligned_15',
        'anglez__std_centered_rolling_mean_right_aligned_15',
        'anglez__std_centered_rolling_mean_left_aligned_15',
        'enmo__std_centered_rolling_mean_right_aligned_15',
        'enmo__std_centered_rolling_mean_left_aligned_15',

        'anglez_rolling_mean_diff_15',
        'enmo_rolling_mean_diff_15',
        'anglez__std_centered_rolling_mean_diff_15',
        'enmo__std_centered_rolling_mean_diff_15',

        'anglez_rolling_mean_right_aligned_5',
        'anglez_rolling_mean_left_aligned_5',
        'enmo_rolling_mean_right_aligned_5',
        'enmo_rolling_mean_left_aligned_5',
        'anglez__std_centered_rolling_mean_right_aligned_5',
        'anglez__std_centered_rolling_mean_left_aligned_5',
        'enmo__std_centered_rolling_mean_right_aligned_5',
        'enmo__std_centered_rolling_mean_left_aligned_5',

        'anglez_rolling_mean_diff_5',
        'enmo_rolling_mean_diff_5',
        'anglez__std_centered_rolling_mean_diff_5',
        'enmo__std_centered_rolling_mean_diff_5',
        ]

features = features5


def make_features(df):
    df = __make_features1(df)
    df = __make_features3(df)
    df = __make_features5(df)
    return df

def __make_features1(df):
    return df


def __make_features3(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    # In short, don't use 'steps_per_min' without thinking.
    window = 5
    df['anglez__std_centered'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo__std_centered'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']
    return df


def __make_features5(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    # In short, don't use 'steps_per_min' without thinking.
    window = 60*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')

    df['anglez_rolling_mean_diff_60'] = df['anglez_rolling_mean_right_aligned_60'] - df['anglez_rolling_mean_left_aligned_60']
    df['enmo_rolling_mean_diff_60'] = df['enmo_rolling_mean_right_aligned_60'] - df['enmo_rolling_mean_left_aligned_60']
    df['anglez__std_centered_rolling_mean_diff_60'] = df['anglez__std_centered_rolling_mean_right_aligned_60'] - df['anglez__std_centered_rolling_mean_left_aligned_60']
    df['enmo__std_centered_rolling_mean_diff_60'] = df['enmo__std_centered_rolling_mean_right_aligned_60'] - df['enmo__std_centered_rolling_mean_left_aligned_60']

    window = 30*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')

    df['anglez_rolling_mean_diff_30'] = df['anglez_rolling_mean_right_aligned_30'] - df['anglez_rolling_mean_left_aligned_30']
    df['enmo_rolling_mean_diff_30'] = df['enmo_rolling_mean_right_aligned_30'] - df['enmo_rolling_mean_left_aligned_30']
    df['anglez__std_centered_rolling_mean_diff_30'] = df['anglez__std_centered_rolling_mean_right_aligned_30'] - df['anglez__std_centered_rolling_mean_left_aligned_30']
    df['enmo__std_centered_rolling_mean_diff_30'] = df['enmo__std_centered_rolling_mean_right_aligned_30'] - df['enmo__std_centered_rolling_mean_left_aligned_30']

    window = 15*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')

    df['anglez_rolling_mean_diff_15'] = df['anglez_rolling_mean_right_aligned_15'] - df['anglez_rolling_mean_left_aligned_15']
    df['enmo_rolling_mean_diff_15'] = df['enmo_rolling_mean_right_aligned_15'] - df['enmo_rolling_mean_left_aligned_15']
    df['anglez__std_centered_rolling_mean_diff_15'] = df['anglez__std_centered_rolling_mean_right_aligned_15'] - df['anglez__std_centered_rolling_mean_left_aligned_15']
    df['enmo__std_centered_rolling_mean_diff_15'] = df['enmo__std_centered_rolling_mean_right_aligned_15'] - df['enmo__std_centered_rolling_mean_left_aligned_15']

    window = 5*steps_per_min
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez__std_centered_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['anglez__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo__std_centered_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['enmo__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')

    df['anglez_rolling_mean_diff_5'] = df['anglez_rolling_mean_right_aligned_5'] - df['anglez_rolling_mean_left_aligned_5']
    df['enmo_rolling_mean_diff_5'] = df['enmo_rolling_mean_right_aligned_5'] - df['enmo_rolling_mean_left_aligned_5']
    df['anglez__std_centered_rolling_mean_diff_5'] = df['anglez__std_centered_rolling_mean_right_aligned_5'] - df['anglez__std_centered_rolling_mean_left_aligned_5']
    df['enmo__std_centered_rolling_mean_diff_5'] = df['enmo__std_centered_rolling_mean_right_aligned_5'] - df['enmo__std_centered_rolling_mean_left_aligned_5']

    return df


def filter_dataset(df):
    print('Begin to filter dataset')
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp'].dt.hour
    df['second'] = df['timestamp'].dt.second
    print(df)
    return df


def normalize(df):
    # Pandas automatically applies colomn-wise function.
    normalized_df = (df-df.mean())/df.std()
    return normalized_df


def extend_features(df):
    print('-'*50)
    print(f'df.shape: {df.shape}')
    df = make_features(df)
    X = df[features]
    y = df['awake']
    print(f'X.shape: {X.shape}')
    print(f'X.isnull().values.any(): {X.isnull().values.any()}')
    print(X)
    print('-'*50)
    return normalize(X), y
