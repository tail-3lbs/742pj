import pandas as pd
import get_submission


def submit_test():
    test = pd.read_csv('./test.csv')
    events = get_submission.get_submission(test, 0)
    events.to_csv('./submission.csv', index=False)


def main():
    submit_test()


if __name__ == '__main__':
    main()
