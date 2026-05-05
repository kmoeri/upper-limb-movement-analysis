#!/usr/bin/python3
# scripts/06_select_features.py


# libraries
import os
import pandas as pd
import numpy as np
from numpy.conftest import dtype

# modules
from src.config import config, project_path
from src.visualization import Visualizer

def remove_label_prefix(label: str) -> str:
    prefixes_to_remove: list[str] = ['idx_tap_', 'alt_tap_', 'open_close_', 'pro_sup_']
    for prefix in prefixes_to_remove:
        label = label.replace(prefix, '')
    return label


def run_feature_reduction():

    print(f'Starting Feature Reduction ...')

    # create visualizer object
    viz: Visualizer = Visualizer()

    # 1) load extracted movement features (raw CSV)
    features_path: str = os.path.join(project_path, 'data', '04_features', 'all_extracted_features.csv')
    df: pd.DataFrame = pd.read_csv(features_path)
    meta_cols = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', 'AHA_Score']

    # 2) calculate velocity ratio
    print('\nGlobal Features Processing ...')
    col_names: list[str] = df.columns.tolist()

    # find positive velocity columns
    pos_vel_cols: list[str] = [col for col in col_names if 'vel_pos' in col.lower() or '_pos' in col.lower()]

    for pos_col in pos_vel_cols:
        # find matching negative velocity columns
        neg_col = pos_col.replace('pos', 'neg')

        if neg_col in col_names:
            ratio_col_name = pos_col.replace('pos', 'ratio')

            # calculate absolute magnitude ratio (negative velocities are negative values)
            df[ratio_col_name] = df[neg_col].abs() / (df[pos_col].abs() + 1e-8)

            # drop positive velocity columns
            df.drop(columns=[pos_col], inplace=True)

    # drop other manually identified features
    drop_col_names: list[str] = ['_rep_num', '_rep_freq', '_mean']
    cols_to_drop: list[str] = [col for col in df.columns for sub in drop_col_names if sub in col]

    if cols_to_drop:
        df.drop(columns=cols_to_drop, inplace=True)

    # prepare directory for saving plots
    out_dir: str = os.path.join(project_path, 'data', '05_results', '04_classification')
    os.makedirs(out_dir, exist_ok=True)

    # 3) per-exercise processing
    print('\nExercise-Specific Features Processing ...')
    exercise = df['ex_name'].unique()
    for ex_name in exercise:

        # get row indices for the current exercise
        ex_mask = df['ex_name'] == ex_name
        df_ex = df[ex_mask].copy()

        # isolate numeric feature columns (excluding metadata)
        numeric_cols: list[str] = [col for col in df_ex.columns if col not in meta_cols]

        # A) drop zero-variance features
        low_var_cols: list = []
        for col in numeric_cols:
            col_var = df_ex[col].var()
            if col_var < 1e-8:
                low_var_cols.append(col)

        if low_var_cols:
            print(f'\nWarning: Zero-variance features dropped: {low_var_cols}')
            numeric_cols = [col for col in numeric_cols if col not in low_var_cols]
            # set low var to NaN
            df.loc[ex_mask, low_var_cols] = np.nan

        # B) calculate Spearman correlation matrix
        corr_matrix = df[numeric_cols].corr(method='spearman').abs()

        # C) visualize and safe heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        labels = [remove_label_prefix(col) for col in corr_matrix.columns]
        viz.corr_matrix_heatmap(matrix_data=corr_matrix, mask=mask, ex_name=ex_name, labels=labels)
        corr_matrix.to_csv(os.path.join(out_dir, f'corr_matrix_{ex_name}.csv'), index=False)

        # D) drop multicollinearity (r > 0.85) for the current exercise
        # select upper triangle to avoid dropping both features
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

        # find features with correlation r > 0.85
        drop_collinear = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.85)]

        if drop_collinear:
            df.loc[ex_mask, drop_collinear] = np.nan

    # 5) save clean version of features to CSV
    df.to_csv(os.path.join(out_dir, 'clean_features.csv'), index=False)
    print('\nFeature Reduction Done. Results saved.')


if __name__ == '__main__':
    run_feature_reduction()
