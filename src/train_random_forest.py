'''
$ python train_random_forest.py
'''

from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import features


class Glob:
    mode = 0
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def load_dataset():
    # The lightweight training dataset 'Zzzs_train_multi.parquet' is from
    # https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target?select=Zzzs_train_multi.parquet
    train = pd.read_parquet(
        '../child-mind-institute-detect-sleep-states/Zzzs_train_multi.parquet')
    if Glob.mode == 2:
        train = train.head(1000).copy()
        train['awake'] = np.random.default_rng().integers(
            0, 2, size=len(train['awake']))
    print(train['series_id'].nunique())
    print(train)
    return train


def split_into_train_and_validation(train):
    series_ids = train['series_id'].unique()
    # Don't do random split. The last 8 series have no events.
    # train_ids, val_ids = train_test_split(series_ids, train_size=0.9)
    train_ids, val_ids = series_ids[5:], series_ids[:5]
    val = train[train['series_id'].isin(val_ids)].copy()
    train = train[train['series_id'].isin(train_ids)].copy()
    return train, val


def fit_classifier(X_train, y_train):
    if Glob.mode == 0:
        rf_classifier = RandomForestClassifier(
            n_estimators=10, min_samples_leaf=300, n_jobs=-1)
    elif Glob.mode == 1:
        rf_classifier = RandomForestClassifier(
            n_estimators=5, min_samples_leaf=5, n_jobs=-1)
    elif Glob.mode == 2:
        rf_classifier = RandomForestClassifier(
            n_estimators=5, min_samples_leaf=5, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def save_importance_plot(rf_classifier):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(features.features, rf_classifier.feature_importances_)
    ax.tick_params(axis='x', rotation=-35)
    ax.set_title('Random forest feature importance')
    plt.xticks(rotation=-30, ha='left')
    plt.tight_layout()
    plt.savefig(f'../outputs/rf_feature_importances_{Glob.now_str}.jpg')


def save_validation(rf_classifier, X_val, val):
    val['not_awake'] = rf_classifier.predict_proba(X_val)[:, 0]
    val['awake'] = rf_classifier.predict_proba(X_val)[:, 1]
    val['insleep'] = (val['not_awake'] > val['awake']).astype('bool')
    val.to_csv(f'../outputs/val.csv', index=False)


def save_prediction(rf_classifier):
    # Test file has time range less than 30 minutes. So the X matrix
    # will have NaN if we are using rolling with window = 30 minutes.
    # It's OK. Actually we should never run for test file. It's useless.
    test = pd.read_parquet(
        '../child-mind-institute-detect-sleep-states/test_series.parquet')
    test = features.make_features(test)
    X_test = test[features.features]
    test['not_awake'] = rf_classifier.predict_proba(X_test)[:, 0]
    test['awake'] = rf_classifier.predict_proba(X_test)[:, 1]
    test['insleep'] = (test['not_awake'] > test['awake']).astype('bool')
    test.to_csv(f'../outputs/test.csv', index=False)


def main():
    train = load_dataset()
    train = features.filter_dataset(train)
    train, val = split_into_train_and_validation(train)
    print('Begin to make features')
    X_train, y_train = features.extend_features(train)
    print('Begin to fit')
    rf_classifier = fit_classifier(X_train, y_train)
    save_importance_plot(rf_classifier)
    print('Begin to validate and predict')
    X_val, _ = features.extend_features(val)
    save_validation(rf_classifier, X_val, val)
    # save_prediction(rf_classifier)


if __name__ == '__main__':
    main()
