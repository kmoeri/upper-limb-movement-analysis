# src/data_split.py

# libraries
import pandas as pd

# modules
from src.config import config, project_path


def load_and_split_data(features_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads extracted features and splits into train and test sets.
    For test participants only the baseline visit (T1) is used. Subsequent visits (T2 & T3) are discarded.

    Args:
        features_path (str): Path to extracted features file.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames separating the train and test sets.
    """

    test_pids: list[str] = config['classification']['test_participants']
    test_visit: str = config['classification']['baseline_visit_id']

    df: pd.DataFrame = pd.read_csv(features_path)

    # isolate test participants
    df_test: pd.DataFrame = df[df['p_ID'].isin(test_pids)].copy()

    # isolate test visit (baseline)
    orig_test_rows = len(df_test)
    df_test = df_test[df_test['visit_ID'] == test_visit]

    dropped_visits = orig_test_rows - len(df_test)
    if dropped_visits > 0:
        print(f'Dataset preparation: isolating test participants complete. Dropped {dropped_visits} subsequent visits.')

    # isolate train participants (select all but the test participants)
    df_train = df[~df['p_ID'].isin(test_pids)].copy()

    print(f'Dataset split complete.\nTrain: {len(df_train)} samples,\nTest: {len(df_test)} samples')

    return df_train, df_test


