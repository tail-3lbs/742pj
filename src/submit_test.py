import pandas as pd
import get_submission


threshold = 0  # In steps.


def submit_test():
    test = pd.read_csv('../outputs/test.csv')
    events = get_submission.get_submission(test, threshold)
    events.to_csv('../outputs/submission.csv', index=False)


def main():
    submit_test()


if __name__ == '__main__':
    main()
