from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import make_features


class Glob:
    mode = 0
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def load_dataset():
    # The lightweight training dataset 'Zzzs_train_multi.parquet' is from
    # https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target?select=Zzzs_train_multi.parquet
    train = pd.read_parquet('./Zzzs_train_multi.parquet')
    if Glob.mode == 1:
        train = train.head(1000).copy()
        train['awake'] = np.random.default_rng().integers(
            0, 2, size=len(train['awake']))
    print(train['series_id'].nunique())
    print(train)
    return train


def extend_features(train):
    train, features = make_features.make_features(train)
    X_train = train[features]
    y_train = train['awake']
    return X_train, y_train, features


def fit_classifier(X_train, y_train):
    if Glob.mode == 0:
        rf_classifier = RandomForestClassifier(
            n_estimators=50, min_samples_leaf=300, n_jobs=-1)
    elif Glob.mode == 1:
        rf_classifier = RandomForestClassifier(
            n_estimators=5, min_samples_leaf=10, n_jobs=-1)
    rf_classifier.fit(X_train, y_train)
    return rf_classifier


def save_importance_plot(rf_classifier, features):
    fig, ax = plt.subplots()
    ax.bar(features, rf_classifier.feature_importances_)
    ax.set_title('Random forest feature importance')
    plt.savefig(f'./rf_feature_importances_{Glob.now_str}.jpg')


def predict(rf_classifier):
    test = pd.read_parquet(
        './child-mind-institute-detect-sleep-states/test_series.parquet')
    test, features = make_features.make_features(test)
    X_test = test[features]
    test['not_awake'] = rf_classifier.predict_proba(X_test)[:, 0]
    test['awake'] = rf_classifier.predict_proba(X_test)[:, 1]
    test['insleep'] = (test['not_awake'] > test['awake']).astype('bool')
    test.to_csv(f'./test_mode{Glob.mode}.csv', index=False)


def main():
    X_train, y_train, features = extend_features(load_dataset())
    rf_classifier = fit_classifier(X_train, y_train)
    save_importance_plot(rf_classifier, features)
    predict(rf_classifier)


if __name__ == '__main__':
    main()
