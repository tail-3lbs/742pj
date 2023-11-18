import pandas as pd


features1 = ['hour',
        'anglez_mean',
        'anglez_std',
        'enmo_mean',
        'enmo_std',
        ]

features3 = ['hour',
        'anglez_mean',
        'anglez_std',
        'enmo_mean',
        'enmo_std',

        'anglez_mean__std_centered',
        'enmo_mean__std_centered',
        ]

features5 = ['hour',
        'anglez_mean',
        'anglez_std',
        'enmo_mean',
        'enmo_std',

        'anglez_mean__std_centered',
        'enmo_mean__std_centered',

        'anglez_mean_rolling_mean_right_aligned_60',
        'anglez_mean_rolling_mean_left_aligned_60',
        'anglez_std_rolling_mean_right_aligned_60',
        'anglez_std_rolling_mean_left_aligned_60',
        'enmo_mean_rolling_mean_right_aligned_60',
        'enmo_mean_rolling_mean_left_aligned_60',
        'enmo_std_rolling_mean_right_aligned_60',
        'enmo_std_rolling_mean_left_aligned_60',
        'anglez_mean__std_centered_rolling_mean_right_aligned_60',
        'anglez_mean__std_centered_rolling_mean_left_aligned_60',
        'enmo_mean__std_centered_rolling_mean_right_aligned_60',
        'enmo_mean__std_centered_rolling_mean_left_aligned_60',

        'anglez_mean_rolling_mean_diff_60',
        'anglez_std_rolling_mean_diff_60',
        'enmo_mean_rolling_mean_diff_60',
        'enmo_std_rolling_mean_diff_60',
        'anglez_mean__std_centered_rolling_mean_diff_60',
        'enmo_mean__std_centered_rolling_mean_diff_60',

        'anglez_mean_rolling_mean_right_aligned_30',
        'anglez_mean_rolling_mean_left_aligned_30',
        'anglez_std_rolling_mean_right_aligned_30',
        'anglez_std_rolling_mean_left_aligned_30',
        'enmo_mean_rolling_mean_right_aligned_30',
        'enmo_mean_rolling_mean_left_aligned_30',
        'enmo_std_rolling_mean_right_aligned_30',
        'enmo_std_rolling_mean_left_aligned_30',
        'anglez_mean__std_centered_rolling_mean_right_aligned_30',
        'anglez_mean__std_centered_rolling_mean_left_aligned_30',
        'enmo_mean__std_centered_rolling_mean_right_aligned_30',
        'enmo_mean__std_centered_rolling_mean_left_aligned_30',

        'anglez_mean_rolling_mean_right_aligned_15',
        'anglez_mean_rolling_mean_left_aligned_15',
        'anglez_std_rolling_mean_right_aligned_15',
        'anglez_std_rolling_mean_left_aligned_15',
        'enmo_mean_rolling_mean_right_aligned_15',
        'enmo_mean_rolling_mean_left_aligned_15',
        'enmo_std_rolling_mean_right_aligned_15',
        'enmo_std_rolling_mean_left_aligned_15',
        'anglez_mean__std_centered_rolling_mean_right_aligned_15',
        'anglez_mean__std_centered_rolling_mean_left_aligned_15',
        'enmo_mean__std_centered_rolling_mean_right_aligned_15',
        'enmo_mean__std_centered_rolling_mean_left_aligned_15',

        'anglez_mean_rolling_mean_right_aligned_5',
        'anglez_mean_rolling_mean_left_aligned_5',
        'anglez_std_rolling_mean_right_aligned_5',
        'anglez_std_rolling_mean_left_aligned_5',
        'enmo_mean_rolling_mean_right_aligned_5',
        'enmo_mean_rolling_mean_left_aligned_5',
        'enmo_std_rolling_mean_right_aligned_5',
        'enmo_std_rolling_mean_left_aligned_5',
        'anglez_mean__std_centered_rolling_mean_right_aligned_5',
        'anglez_mean__std_centered_rolling_mean_left_aligned_5',
        'enmo_mean__std_centered_rolling_mean_right_aligned_5',
        'enmo_mean__std_centered_rolling_mean_left_aligned_5',
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
    # In short, don't use 'steps_per_min' here.
    window = 5
    df['anglez_mean__std_centered'] = df.groupby('series_id', as_index=False)['anglez_mean'].rolling(window=window, min_periods=1, center=True).std()['anglez_mean']
    df['enmo_mean__std_centered'] = df.groupby('series_id', as_index=False)['enmo_mean'].rolling(window=window, min_periods=1, center=True).std()['enmo_mean']
    return df


def __make_features5(df):
    # Note that this window is between rows, not steps.
    # So the interval here depends on the granularity of input dataframe.
    # In short, don't use 'steps_per_min' here.
    window = 60
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_right_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_left_aligned_60'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')

    df['anglez_mean_rolling_mean_diff_60'] = df['anglez_mean_rolling_mean_right_aligned_60'] - df['anglez_mean_rolling_mean_left_aligned_60']
    df['anglez_std_rolling_mean_diff_60'] = df['anglez_std_rolling_mean_right_aligned_60'] - df['anglez_std_rolling_mean_left_aligned_60']
    df['enmo_mean_rolling_mean_diff_60'] = df['enmo_mean_rolling_mean_right_aligned_60'] - df['enmo_mean_rolling_mean_left_aligned_60']
    df['enmo_std_rolling_mean_diff_60'] = df['enmo_std_rolling_mean_right_aligned_60'] - df['enmo_std_rolling_mean_left_aligned_60']
    df['anglez_mean__std_centered_rolling_mean_diff_60'] = df['anglez_mean__std_centered_rolling_mean_right_aligned_60'] - df['anglez_mean__std_centered_rolling_mean_left_aligned_60']
    df['enmo_mean__std_centered_rolling_mean_diff_60'] = df['enmo_mean__std_centered_rolling_mean_right_aligned_60'] - df['enmo_mean__std_centered_rolling_mean_left_aligned_60']

    window = 30
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_right_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_left_aligned_30'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    window = 15
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_right_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_left_aligned_15'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
    window = 5
    for series_id in df['series_id'].unique():
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_std_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_mean'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_std_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_std'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'anglez_mean__std_centered_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['anglez_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_right_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean()
        df.loc[df['series_id']==series_id, 'enmo_mean__std_centered_rolling_mean_left_aligned_5'] = df.loc[df['series_id']==series_id]['enmo_mean__std_centered'].rolling(window=window, min_periods=1).mean().shift(-window+1).fillna(method='ffill')
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
        df.groupby(['series_id', 'timestamp'])['anglez'].mean().rename('anglez_mean'),
        df.groupby(['series_id', 'timestamp'])['anglez'].std().rename('anglez_std'),
        df.groupby(['series_id', 'timestamp'])['enmo'].mean().rename('enmo_mean'),
        df.groupby(['series_id', 'timestamp'])['enmo'].std().rename('enmo_std'),
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
    print(X)
    print('-'*50)
    return normalize(X), y
