import pandas as pd
import get_submission
from metric import score


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
    print('Val score: {}'.format(
        score(val_truth, submission, tolerances, **column_names)))


def main():
    get_validation_score()


if __name__ == '__main__':
    main()
