#!/usr/bin/python3
# scripts/08_run_regression.py

# libraries
import os
import pandas as pd

# modules
from src.config import config, project_path
from src.data_split import load_and_split_data
from src.ml_pipeline_manager import EnsembleManager
from src.reference_scores import get_target_score_for_regression


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

    # target score
    target_type: str = config['regression'].get('reference_score', 'JT')
    print(f'Starting Machine Learning Pipeline: Phase 2 - {target_type}-Score Regression ...')

    # 1) load and split dataset
    features_path: str = os.path.join(project_path, 'data', '04_features', 'clean_features.csv')
    test_participants = config['classification']['test_participants']
    baseline_visit = config['classification']['baseline_visit_id']
    df_train_raw, df_test_raw = load_and_split_data(features_path, test_participants, baseline_visit)

    # define features
    meta_cols: list[str] = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID']
    feature_cols: list[str] = [c for c in df_train_raw.columns if c not in meta_cols]

    # 2) transforming raw kinematics into asymmetry ratios and merge with target scores
    print('\nTransforming kinematics into Asymmetry Ratios ...')
    df_train_kinematics: pd.DataFrame = calc_asymmetry_ratios(df_train_raw, feature_cols)
    df_test_kinematics: pd.DataFrame = calc_asymmetry_ratios(df_test_raw, feature_cols)

    # get regression target
    df_targets, target_columns = get_target_score_for_regression()

    # merge kinematics with targets
    df_train: pd.DataFrame = pd.merge(df_train_kinematics, df_targets, on=['p_ID', 'visit_ID'], how='inner')
    df_test: pd.DataFrame = pd.merge(df_test_kinematics, df_targets, on=['p_ID', 'visit_ID'], how='inner')

    # 3) loop through targets
    for target_col in target_columns:

        # 3.1) train base models via the pipeline manager
        print(f'\nTraining Ensemble for: {target_col} ... ')

        # select method to use: 'catboost', 'xgb' or 'rf'
        ensemble = EnsembleManager(model_type='catboost', task_type='regression', n_trials=30)

        # loop exercises, run optuna, perform SHAP reducing
        ensemble.train_all_exercises(df_train, feature_cols, target_col)

        # 3.2) Evaluate test set
        print(f'\nEvaluating test set for {target_col} ...')
        results: list[dict] = []

        # group by participant ID to evaluate visit by visit
        for pid, df_visit in df_test.groupby('p_ID'):

            # ensure only baseline visits are used
            df_baseline = df_visit[df_visit['visit_ID'] == baseline_visit]
            if df_baseline.empty:
                continue

            # get ground truth
            real_score = df_baseline[target_col].iloc[0]

            # get ensemble prediction
            predicted_score = ensemble.predict_visit(df_baseline, feature_cols)

            if pd.isna(predicted_score):
                continue

            # append current results
            results.append({'Target': target_col,
                            'p_ID': pid,
                            'Real_Score': real_score,
                            'Predicted_Score': round(predicted_score, 3),
                            'Error': round(abs(real_score - predicted_score), 3),
                            'Exercises_Used': list(df_baseline['ex_name'].unique())})

            print(f'Participant ID: {pid} | Real Score: {real_score} | '
                  f'Predicted Score: {predicted_score:.2f} | Used: {len(df_baseline)} exercises.')

        # save final results
        results_df = pd.DataFrame(results)
        out_dir: str = os.path.join(project_path, 'data', '05_results', '06_regression')
        os.makedirs(out_dir, exist_ok=True)
        results_df.to_csv(os.path.join(out_dir, f'predictions_{target_col}.csv'), index=False)

    print(f'\nRegression completed for {target_type}. Results saved.')


if __name__ == '__main__':
    run_regression_pipeline()
