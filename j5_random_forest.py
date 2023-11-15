from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from itertools import groupby
import gc


now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Lightweight training dataset from
# https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target?select=Zzzs_train_multi.parquet
train = pd.read_parquet('./Zzzs_train_multi.parquet')
train['awake'] = np.random.default_rng().integers(
    0, 2, size=len(train['awake']))
print(train['series_id'].nunique())
print(train)


def make_features(df):
    # It's important to drop timezone here, as there are different timezones in
    # the train dataset.
    df['timestamp2'] = pd.to_datetime(
        df['timestamp']).apply(lambda t: t.tz_localize(None))
    df['hour'] = df['timestamp2'].dt.hour
    return df


train = make_features(train)
features = ['hour', 'anglez', 'enmo']
X_train = train[features]
y_train = train['awake']


rf_classifier = RandomForestClassifier(
    n_estimators=30, min_samples_leaf=100, random_state=42, n_jobs=-1)
rf_classifier.fit(X_train, y_train)


test = pd.read_parquet(
    './child-mind-institute-detect-sleep-states/test_series.parquet')
test = make_features(test)
X_test = test[features]
test['not_awake'] = rf_classifier.predict_proba(X_test)[:, 0]
test['awake'] = rf_classifier.predict_proba(X_test)[:, 1]
test['insleep'] = (test['not_awake'] > test['awake']).astype('bool')
test.to_csv('./test.csv', index=False)


fig, ax = plt.subplots()
ax.bar(features, rf_classifier.feature_importances_)
ax.set_title('Random forest feature importances')
plt.savefig(f'./rf_feature_importances_{now_str}.jpg')
