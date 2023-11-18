import pandas as pd


features1 = ['hour',
        'anglez_mean',
        'anglez_std',
        'anglez_min',
        'anglez_max',
        'enmo_mean',
        'enmo_std',
        'enmo_min',
        'enmo_max',
        ]

features2 = ['hour',
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

features3 = ['hour',
        'anglez_rolling_std_centered',
        'enmo_rolling_std_centered',
        'anglez_rolling_mean_right_aligned_rolling_std_centered',
        'anglez_rolling_mean_left_aligned_rolling_std_centered',
        'enmo_rolling_mean_right_aligned_rolling_std_centered',
        'enmo_rolling_mean_left_aligned_rolling_std_centered',
        'anglez_rolling_mean_diff_rolling_std_centered',
        'enmo_rolling_mean_diff_rolling_std_centered',
        ]

features = features1


def make_features(df):
    return __make_features1(df)


def __make_features1(df):
    return df


def __make_features2(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    # In short, don't use 'steps_per_min' here.
    window = 1
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_1'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_1'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_1'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_1'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff_1'] = df['anglez_rolling_mean_left_aligned_1'] - df['anglez_rolling_mean_right_aligned_1']
    df['enmo_rolling_mean_diff_1'] = df['enmo_rolling_mean_left_aligned_1'] - df['enmo_rolling_mean_right_aligned_1']

    window = 5
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned_2'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned_2'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned_2'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned_2'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff_2'] = df['anglez_rolling_mean_left_aligned_2'] - df['anglez_rolling_mean_right_aligned_2']
    df['enmo_rolling_mean_diff_2'] = df['enmo_rolling_mean_left_aligned_2'] - df['enmo_rolling_mean_right_aligned_2']

    window = 1
    df['anglez_rolling_std_centered_1'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered_1'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']

    window = 10
    df['anglez_rolling_std_centered_2'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered_2'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']

    return df


def __make_features3(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    # In short, don't use 'steps_per_min' here.
    window = 5

    df['anglez_rolling_std_centered'] = df.groupby('series_id', as_index=False)['anglez'].rolling(window=window, min_periods=1, center=True).std()['anglez']
    df['enmo_rolling_std_centered'] = df.groupby('series_id', as_index=False)['enmo'].rolling(window=window, min_periods=1, center=True).std()['enmo']

    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_right_aligned'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_rolling_mean_left_aligned'] = df.loc[df['series_id']==series_id]['anglez'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_right_aligned'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_rolling_mean_left_aligned'] = df.loc[df['series_id']==series_id]['enmo'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    df['anglez_rolling_mean_diff'] = df['anglez_rolling_mean_left_aligned'] - df['anglez_rolling_mean_right_aligned']
    df['enmo_rolling_mean_diff'] = df['enmo_rolling_mean_left_aligned'] - df['enmo_rolling_mean_right_aligned']

    df['anglez_rolling_mean_right_aligned_rolling_std_centered'] = df.groupby('series_id', as_index=False)['anglez_rolling_mean_right_aligned'].rolling(window=window, min_periods=1, center=True).std()['anglez_rolling_mean_right_aligned']
    df['anglez_rolling_mean_left_aligned_rolling_std_centered'] = df.groupby('series_id', as_index=False)['anglez_rolling_mean_left_aligned'].rolling(window=window, min_periods=1, center=True).std()['anglez_rolling_mean_left_aligned']
    df['enmo_rolling_mean_right_aligned_rolling_std_centered'] = df.groupby('series_id', as_index=False)['enmo_rolling_mean_right_aligned'].rolling(window=window, min_periods=1, center=True).std()['enmo_rolling_mean_right_aligned']
    df['enmo_rolling_mean_left_aligned_rolling_std_centered'] = df.groupby('series_id', as_index=False)['enmo_rolling_mean_left_aligned'].rolling(window=window, min_periods=1, center=True).std()['enmo_rolling_mean_left_aligned']
    df['anglez_rolling_mean_diff_rolling_std_centered'] = df.groupby('series_id', as_index=False)['anglez_rolling_mean_diff'].rolling(window=window, min_periods=1, center=True).std()['anglez_rolling_mean_diff']
    df['enmo_rolling_mean_diff_rolling_std_centered'] = df.groupby('series_id', as_index=False)['enmo_rolling_mean_diff'].rolling(window=window, min_periods=1, center=True).std()['enmo_rolling_mean_diff']

    return df


def filter_dataset(df):
    print('Begin to filter dataset')
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['timestamp'] = df['timestamp'].dt.floor('Min')
    df = pd.concat([
        df.groupby(['series_id', 'timestamp'])['step'].min(),
        df.groupby(['series_id', 'timestamp'])[
            'anglez'].mean().rename('anglez_mean'),
        df.groupby(['series_id', 'timestamp'])[
            'anglez'].std().rename('anglez_std'),
        df.groupby(['series_id', 'timestamp'])[
            'anglez'].min().rename('anglez_min'),
        df.groupby(['series_id', 'timestamp'])[
            'anglez'].max().rename('anglez_max'),
        df.groupby(['series_id', 'timestamp'])[
            'enmo'].mean().rename('enmo_mean'),
        df.groupby(['series_id', 'timestamp'])[
            'enmo'].std().rename('enmo_std'),
        df.groupby(['series_id', 'timestamp'])[
            'enmo'].min().rename('enmo_min'),
        df.groupby(['series_id', 'timestamp'])[
            'enmo'].max().rename('enmo_max'),
        # If there are equal amount of 0 and 1, we choose 1 (awake).
        df.groupby(['series_id', 'timestamp'])['awake'].agg(
            lambda x: x.value_counts().index.max()).rename('awake'),
    ], axis=1).reset_index()
    df['hour'] = df['timestamp'].dt.hour
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
    print('-'*50)
    return normalize(X), y
