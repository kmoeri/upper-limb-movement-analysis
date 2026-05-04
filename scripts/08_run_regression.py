#!/usr/bin/python3
# scripts/08_run_regression.py

# libraries
import os
import pandas as pd

# modules
from src.config import config, project_path
from src.data_split import load_and_split_data
from src.ml_pipeline_manager import EnsembleManager


def calc_asymmetry_ratios(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:

    # separate rows by condition (healthy vs. affected)
    df_affected = df[df['side_condition'] == 'Affected'].set_index(['p_ID', 'visit_ID', 'ex_name'])
    df_healthy = df[df['side_condition'] == 'Healthy'].set_index(['p_ID', 'visit_ID', 'ex_name'])

    # find common indices (exercise exists for both sides)
    common_idx: int = df_affected.index.intersection(df_healthy.index)

    # calculate ratio (prevent division by zero)
    df_ratio = df_affected.loc[common_idx, feature_cols] / (df_healthy.loc[common_idx, feature_cols] + 1e-8)

    # merge meta back to final dataframe
    df_meta = df_affected.loc[common_idx, ['AHA_Score']]
    df_final = df_ratio.join(df_meta).reset_index()

    return df_final


def run_regression_pipeline():

    print(f'Starting Machine Learning Pipeline: Phase 2 - AHA-Score Regression ...')

    # 1) load and split dataset
    features_path: str = os.path.join(project_path, 'data', '04_features', 'clean_features.csv')
    test_participants = config['classification']['test_participants']
    baseline_visit = config['classification']['baseline_visit_id']
    df_train_raw, df_test_raw = load_and_split_data(features_path, test_participants, baseline_visit)

    # define features
    meta_cols = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', 'AHA_Score']
    feature_cols = [c for c in df_train_raw.columns if c not in meta_cols]

    # 2) transforming raw kinematics into asymmetry ratios
    print('\nTransforming kinematics into Asymmetry Ratios ...')
    df_train = calc_asymmetry_ratios(df_train_raw, feature_cols)
    df_test = calc_asymmetry_ratios(df_test_raw, feature_cols)
    target_col = 'AHA_Score'    # target column to predict

    # 3) Train base models via the pipeline manager
    print('\nTraining base regression models ...')
    # select method to use: 'catboost', 'xgb' or 'rf'
    ensemble = EnsembleManager(model_type='catboost', task_type='regression', n_trials=30)

    # loop exercises, run optuna, perform SHAP reducing
    ensemble.train_all_exercises(df_train, feature_cols, target_col)

    # 4) Evaluate test set
    print('\nEvaluating test set ...')
    results: list[dict] = []

    # group by participant ID to evaluate visit by visit
    for pid, df_visit in df_test.groupby('p_ID'):

        # ensure only baseline visits are used
        df_baseline = df_visit[df_visit['visit_ID'] == baseline_visit]
        if df_baseline.empty:
            continue

        # get ground truth
        real_aha = df_baseline[target_col].iloc[0]

        # get ensemble prediction
        predicted_aha = ensemble.predict_visit(df_baseline, feature_cols)

        if pd.isna(predicted_aha):
            continue

        # append current results
        results.append({'p_ID': pid,
                        'Real_AHA': real_aha,
                        'Predicted_AHA': round(predicted_aha, 2),
                        'Error': round(abs(real_aha - predicted_aha), 2),
                        'Exercises_Used': list(df_baseline['ex_name'].unique())})

        print(f'Participant ID: {pid} | Real AHA: {real_aha} | '
              f'Predicted AHA: {predicted_aha:.2f} | Used: {len(df_baseline)} exercises.')

    # save final results
    results_df = pd.DataFrame(results)
    out_dir: str = os.path.join(project_path, 'data', '05_results', '04_classification')
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'regression_predictions.csv'), index=False)
    print('\nRegression complete. Results saved.')


if __name__ == '__main__':
    run_regression_pipeline()
