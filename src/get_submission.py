import pandas as pd
from dataclasses import dataclass
import itertools


@dataclass
class Event:
    index: int
    series_id: str
    step: int
    event: str
    score: float

    def to_tuple(self):
        return (self.series_id, self.step, self.event, self.score)


def get_submission(test, least_sleep_time, least_awake_time):
    series_ids = test['series_id'].unique()
    all_events = pd.DataFrame(
        columns=['series_id', 'step', 'event', 'score'])
    for series_id in series_ids:
        one_test = test[test['series_id'] == series_id]
        one_windows = []
        last_onset = None
        last_wakeup = None
        for idx, row in one_test.iterrows():
            if last_onset is None:
                if row['insleep'] == True:
                    last_onset = Event(
                        idx, series_id, row['step'], 'onset', row['not_awake'])
            else:
                if row['insleep'] == False:
                    last_wakeup = Event(
                        idx, series_id, row['step'], 'wakeup', row['awake'])
                    if last_wakeup.step - last_onset.step > least_sleep_time:
                        # In pandas.loc[a:b], b is inclusive.
                        score = one_test.loc[last_onset.index:last_wakeup.index-1][
                            'not_awake'].median()
                        last_onset.score = score
                        last_wakeup.score = score
                        one_windows.append((last_onset, last_wakeup))
                    last_onset = None
                    last_wakeup = None

        smoothed_windows = []
        for current_window in one_windows:
            if not smoothed_windows:
                smoothed_windows.append(current_window)
                continue
            last_window = smoothed_windows[-1]
            if last_window[1].step + least_awake_time > current_window[0].step:
                score = (last_window[0].score + current_window[1].score)/2
                last_window[0].score, current_window[1].score = score, score
                smoothed_windows[-1] = (last_window[0], current_window[1])
            else:
                smoothed_windows.append(current_window)
        one_events = list(itertools.chain.from_iterable(smoothed_windows))
        one_events = [e.to_tuple() for e in one_events]

        all_events = pd.concat([
            all_events,
            pd.DataFrame(one_events,
                         columns=['series_id', 'step', 'event', 'score'])])

    all_events = all_events.astype({'step': 'int32'})
    return all_events.reset_index(
        drop=True).reset_index().rename(columns={'index': 'row_id'})
