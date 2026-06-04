# src/reference_scores.py

# libraries
import os
import numpy as np
import pandas as pd

# modules
from src.config import project_path, config


def get_target_score_for_regression() -> tuple[pd.DataFrame, list[str]]:

    # reference score file
    ref_score_path: str = os.path.join(project_path,'reference_scores.csv')
    df_jt: pd.DataFrame = pd.read_csv(ref_score_path)

    # get the type of reference score to predict
    ref_type: str = config['regression'].get('reference_score', 'JT').upper()

    if ref_type == 'AHA':
        # check whether column with the corresponding reference scores exist
        if 'AHA_Score' not in df_jt.columns:
            raise ValueError(f'\nError: "AHA_Scores" column not found in file: {ref_score_path}.')

        target_col: list[str] = ['AHA_Score']
        df_out: pd.DataFrame = df_jt[['p_ID', 'visit_ID', 'AHA_Score']].copy()

        # drop rows with missing AHA Score
        df_out = df_out.dropna(subset=['AHA_Score'])
        return df_out, target_col

    elif ref_type == 'JT':

        # define the six task names
        subtests: list[str] = ['CardTurning', 'ObjectPicking', 'ChipStacking','BeanHandling', 'LCanGrasp', 'HCanGrasp']
        ratio_columns: list = []

        # loop through the subtests and calculate the ratios
        for idx, subtest in enumerate(subtests):

            # calculate the current affected and healthy Jebson&Taylor test numbers (goes from 01 to 12)
            aff_col: str = f'ex_JT-{idx*2 + 1:02d}'         # odd numbers
            health_col: str = f'ex_JT-{idx*2 + 2:02d}'      # even numbers

            # check whether column with the corresponding reference scores exist
            if aff_col not in df_jt.columns or health_col not in df_jt.columns:
                raise ValueError(f'\nError: Missing JT columns ({aff_col} or {health_col}).')

            # catch impossible exercise times (min. time > 1.0 s)
            if (df_jt[aff_col] <= 1.0).any() or (df_jt[health_col] <= 1.0).any():
                # catch negative log() values
                raise ValueError(f'\nError: Exercise times below 1.0 s detected in {aff_col} or {health_col}.')

            # apply log transform for the time
            log_aff = np.log(df_jt[aff_col] + 1e-8)
            log_health = np.log(df_jt[health_col] + 1e-8)

            # calculate the asymmetry ratios (affected / healthy)
            ratio_col_name = f'Asymmetry_JT_Ratio_{subtest}'
            df_jt[ratio_col_name] = log_aff / log_health
            ratio_columns.append(ratio_col_name)

        # calculate the total score (average of all six ratios)
        asym_ratio_tot: str = 'Asymmetry_JT_Ratio_Total'
        df_jt[asym_ratio_tot] = df_jt[ratio_columns].mean(axis=1)

        # reorder to put the total asymmetry ratio first
        all_target_col_names: list[str] = [asym_ratio_tot] + ratio_columns

        # return the IDs and the ratios; raw values are dropped
        col_to_keep = ['p_ID', 'visit_ID'] + all_target_col_names
        df_out: pd.DataFrame = df_jt[col_to_keep].copy()

        # drop rows where JT was not performed
        df_out = df_out.dropna(subset=[asym_ratio_tot])

        return df_out, all_target_col_names

    else:
        raise ValueError(f'\nError: {ref_type} is not supported.')


def merge_feature_with_targets(df_features: pd.DataFrame) -> tuple[pd.DataFrame, str, list[str]]:
    """
    Retrieves the reference scores and merges them with the feature DataFrame.

    Args:
        df_features: DataFrame containing the kinematic features.

    Returns:
        tuple[pd.DataFrame, list[str], list[str]]:
            - df_merged: The merged DataFrame.
            - primary_target: The column name of the main score to predict (e.g., Jebsen Taylor)
            - all_targets: A list of all score columns added
    """
    df_targets, target_cols = get_target_score_for_regression()

    # merge participant ID and visit ID
    df_merged = pd.merge(df_features, df_targets, on=['p_ID', 'visit_ID'], how='inner')

    # primary target is located first in the list
    primary_target: str = target_cols[0]

    return df_merged, primary_target, target_cols
