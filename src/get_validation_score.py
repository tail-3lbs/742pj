'''
$ python get_validation_score.py
'''

import pandas as pd
import numpy as np
import get_submission
from metric import score
import matplotlib.pyplot as plt


threshold = 30*12  # In steps.


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
    val = pd.read_csv('../outputs/val.csv')
    submission = get_submission.get_submission(val, threshold)
    submission.to_csv(f'../outputs/submission_val.csv', index=False)
    val_truth = pd.read_csv(
        '../child-mind-institute-detect-sleep-states/train_events.csv')
    val_truth = val_truth[val_truth['series_id'].isin(
        val['series_id'].unique())]
    val_truth.to_csv(f'../outputs/submission_val_truth.csv', index=False)
    print('Val score: {}'.format(
        score(val_truth, submission, tolerances, **column_names)))
    for series_id in val['series_id'].unique():
        save_prediction(series_id,
                        val[val['series_id'] == series_id],
                        submission[submission['series_id'] == series_id],
                        val_truth[val_truth['series_id'] == series_id])


def save_prediction(series_id, val, submission, val_truth):
    fig, axs = plt.subplots(2, 1, figsize=(16, 8))
    axs[0].plot(val['step'], val['enmo'], 'r-', alpha=.6, label='enmo')
    axs[0].set_ylabel('enmo')
    axs[0].set_xlabel('step')
    true_steps = list(val_truth['step'])
    for i in range(0, len(true_steps), 2):
        axs[0].fill_betweenx(
            np.linspace((np.min(val['enmo'])+np.max(val['enmo']))/2,
                        np.max(val['enmo']), 5),
            [true_steps[i]]*5, [true_steps[i+1]]*5, alpha=0.2, color='blue',
            hatch='+', label="_"*(1 if i > 0 else 0) + "truth")
    pred_steps = list(submission['step'])
    for i in range(0, len(pred_steps), 2):
        axs[0].fill_betweenx(
            np.linspace(np.min(val['enmo']),
                        (np.min(val['enmo'])+np.max(val['enmo']))/2, 5),
            [pred_steps[i]]*5, [pred_steps[i+1]]*5, alpha=0.2, color='green',
            label="_"*(1 if i > 0 else 0) + "prediction")
    axs[0].set_ylim([np.min(val['enmo']), np.max(val['enmo'])])
    axs[0].legend()
    axs[0].set_title(f'Series ID: {series_id}')
    axs[1].plot(val['step'], val['anglez'], 'r-', alpha=.6, label='anglez')
    axs[1].set_ylabel('anglez')
    axs[1].set_xlabel('step')
    true_steps = list(val_truth['step'])
    for i in range(0, len(true_steps), 2):
        axs[1].fill_betweenx(
            np.linspace((np.min(val['anglez'])+np.max(val['anglez']))/2,
                        np.max(val['anglez']), 5),
            [true_steps[i]]*5, [true_steps[i+1]]*5, alpha=0.2, color='blue',
            hatch='+', label="_"*(1 if i > 0 else 0) + "truth")
    pred_steps = list(submission['step'])
    for i in range(0, len(pred_steps), 2):
        axs[1].fill_betweenx(
            np.linspace(np.min(val['anglez']),
                        (np.min(val['anglez'])+np.max(val['anglez']))/2, 5),
            [pred_steps[i]]*5, [pred_steps[i+1]]*5, alpha=0.2, color='green',
            label="_"*(1 if i > 0 else 0) + "prediction")
    axs[1].set_ylim([np.min(val['anglez']), np.max(val['anglez'])])
    axs[1].legend()
    plt.savefig(f'../outputs/predictions_{series_id}.jpg')


def main():
    get_validation_score()


if __name__ == '__main__':
    main()
