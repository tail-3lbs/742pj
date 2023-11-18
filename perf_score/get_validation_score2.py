'''
$ python get_validation_score2.py
'''
import os
import sys
sys.path.insert(1, os.getcwd()+'/../src')


import pandas as pd
import numpy as np
from metric import score
import matplotlib.pyplot as plt
import make_features
from common import steps_per_min


least_sleep_time = 90*steps_per_min
least_awake_time = 30*steps_per_min


def get_validation_score():
    column_names = {
        'series_id_column_name': 'series_id',
        'time_column_name': 'step',
        'event_column_name': 'event',
        'score_column_name': 'score',
    }
    tolerances = {
        'onset': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360],
        'wakeup': [12, 36, 60, 90, 120, 150, 180, 240, 300, 360]
    }
    submission = pd.read_csv('./submission_val.csv')
    val_truth = pd.read_csv('./submission_val_truth.csv')
    print('Val score: {:.3f}'.format(
        score(val_truth, submission, tolerances, **column_names)))
    for series_id in submission['series_id'].unique():
        perf_score = score(val_truth[val_truth['series_id'] == series_id],
                           submission[submission['series_id'] == series_id],
                           tolerances, **column_names)
        print('Val score of {}: {:.3f}'.format(
            series_id, perf_score))
        save_prediction(series_id,
                        perf_score,
                        submission[submission['series_id'] == series_id],
                        val_truth[val_truth['series_id'] == series_id])


def save_prediction(series_id, perf_score, submission, val_truth):
    fig, ax = plt.subplots(figsize=(16, 8))
    true_steps = list(val_truth['step'])
    for i in range(0, len(true_steps), 2):
        ax.fill_betweenx(
            np.linspace(.5, 1, 5),
            [true_steps[i]]*5, [true_steps[i+1]]*5,
            alpha=0.2, color='blue',
            hatch='x', label='_'*(1 if i > 0 else 0) + 'truth')
    for i in range(0, len(submission), 2):
        ax.fill_betweenx(
            np.linspace(0, .5, 5),
            # Here df.iloc does not use index. It's the n-th row.
            [submission.iloc[i]['step']] * \
            5, [submission.iloc[i+1]['step']]*5,
            alpha=0.2, color='green',
            label='_'*(1 if i > 0 else 0) + 'prediction')
        ax.text(
            (submission.iloc[i]['step']+submission.iloc[i+1]['step'])/2,
            1/4.,
            '{:.4f}'.format(
                (submission.iloc[i]['score'] +
                 submission.iloc[i+1]['score'])/2),
            rotation=90, ha='center')
    ax.legend()
    ax.set_title(f'Series ID: {series_id}, Perf score: {perf_score:.3f}')
    plt.savefig(f'./perf_score_{series_id}.jpg')
    plt.close()


def main():
    get_validation_score()


if __name__ == '__main__':
    main()
