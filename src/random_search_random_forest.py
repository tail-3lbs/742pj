from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import features
import time
from sklearn.model_selection import RandomizedSearchCV


class Glob:
    mode = 0
    now_str = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')


def build_random_grid():
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=20, stop=1000, num=30)]
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
    max_depth.append(None)
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5, 50, 100, 200, 300]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 5, 50, 100, 200, 300]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    print(random_grid)
    return random_grid


def load_dataset():
    # The lightweight training dataset 'Zzzs_train.parquet' is from
    # https://www.kaggle.com/datasets/carlmcbrideellis/zzzs-lightweight-training-dataset-target?select=Zzzs_train.parquet
    train = pd.read_parquet(
        '../child-mind-institute-detect-sleep-states/Zzzs_train.parquet')
    if Glob.mode == 2:
        train = train.head(1000).copy()
        train['awake'] = np.random.default_rng().integers(
            0, 2, size=len(train['awake']))
    print(train['series_id'].nunique())
    print(train)
    return train


def main():
    train = load_dataset()
    print('Begin to make features')
    X_train, y_train = features.extend_features(train)
    print('Begin to fit')
    rf_random = RandomizedSearchCV(
        # error_score='raise',
        estimator=RandomForestClassifier(),
        param_distributions=build_random_grid(),
        n_iter=30, cv=5, verbose=2, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    print(f'best_params_: {rf_random.best_params_}')


if __name__ == '__main__':
    start = time.time()
    main()
    print(f'Total time: {time.time() - start:.2f} seconds')
