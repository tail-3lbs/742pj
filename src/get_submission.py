import pandas as pd


def get_submission(test, threshold=30*12):
    series_ids = test['series_id'].unique()
    events = pd.DataFrame(
        columns=['series_id', 'step', 'event', 'score'])
    for series_id in series_ids:
        one_test = test[test['series_id'] == series_id].copy()
        one_events = pd.DataFrame(
            columns=['series_id', 'step', 'event', 'score'])
        last_onset = None
        last_wakeup = None
        for _, row in one_test.iterrows():
            if last_onset is None:
                if row['insleep'] == True:
                    last_onset = pd.DataFrame({
                        'series_id': [series_id],
                        'step': [row['step']],
                        'event': ['onset'],
                        'score': [row['not_awake']]
                    })
            else:
                if row['insleep'] == False:
                    last_wakeup = pd.DataFrame({
                        'series_id': [series_id],
                        'step': [row['step']],
                        'event': ['wakeup'],
                        'score': [row['awake']]
                    })
                    if last_wakeup.iloc[0][
                            'step'] - last_onset.iloc[0]['step'] > threshold:
                        one_events = pd.concat(
                            [one_events, last_onset, last_wakeup])
                    last_onset = None
                    last_wakeup = None
        events = pd.concat([events, one_events])

    events = events.astype({'step': 'int32'})
    return events.reset_index(
        drop=True).reset_index().rename(columns={'index': 'row_id'})
