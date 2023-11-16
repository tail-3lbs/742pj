import pandas as pd

def make_features(df):
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp'].dt.hour
    features = ['hour', 'anglez', 'enmo']
    return df, features
