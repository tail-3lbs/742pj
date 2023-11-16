import pandas as pd


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
    # Ref: https://www.kaggle.com/code/carlmcbrideellis/zzzs-random-forest-model-starter.
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp'].dt.hour

    periods = 20
    df['anglez_abs'] = abs(df['anglez'])
    df['anglez_diff'] = df.groupby('series_id')['anglez'].diff(periods=periods).fillna(method='bfill').astype('float16')
    df['enmo_diff'] = df.groupby('series_id')['enmo'].diff(periods=periods).fillna(method='bfill').astype('float16')
    df['anglez_rolling_mean'] = df['anglez'].rolling(periods,center=True).mean().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['enmo_rolling_mean'] = df['enmo'].rolling(periods,center=True).mean().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['anglez_rolling_max'] = df['anglez'].rolling(periods,center=True).max().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['enmo_rolling_max'] = df['enmo'].rolling(periods,center=True).max().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['anglez_rolling_std'] = df['anglez'].rolling(periods,center=True).std().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['enmo_rolling_std'] = df['enmo'].rolling(periods,center=True).std().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['anglez_diff_rolling_mean'] = df['anglez_diff'].rolling(periods,center=True).mean().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['enmo_diff_rolling_mean'] = df['enmo_diff'].rolling(periods,center=True).mean().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['anglez_diff_rolling_max'] = df['anglez_diff'].rolling(periods,center=True).max().fillna(method='bfill').fillna(method='ffill').astype('float16')
    df['enmo_diff_rolling_max'] = df['enmo_diff'].rolling(periods,center=True).max().fillna(method='bfill').fillna(method='ffill').astype('float16')

    features = ['hour',
            'anglez',
            'anglez_abs',
            'anglez_rolling_mean',
            'anglez_rolling_max',
            'anglez_rolling_std',
            'anglez_diff',
            'anglez_diff_rolling_mean',
            'anglez_diff_rolling_max',
            'enmo',
            'enmo_rolling_mean',
            'enmo_rolling_max',
            'enmo_rolling_std',
            'enmo_diff',
            'enmo_diff_rolling_mean',
            'enmo_diff_rolling_max',
            ]
    return df, features
