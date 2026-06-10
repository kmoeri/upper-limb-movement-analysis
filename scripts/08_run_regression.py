#!/usr/bin/python3
# scripts/08_run_regression.py

# libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
from scipy.stats import bootstrap

# modules
from src.config import config, project_path
from src.utils import ToolBox
from src.visualization import Visualizer
from src.ml_pipeline_manager import EnsembleManager
from src.reference_scores import merge_features_with_targets


def calc_asymmetry_ratios(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:

    # separate rows by condition (healthy vs. affected)
    df_affected = df[df['side_condition'] == 'Affected'].set_index(['p_ID', 'visit_ID', 'ex_name'])
    df_healthy = df[df['side_condition'] == 'Healthy'].set_index(['p_ID', 'visit_ID', 'ex_name'])

    # find common indices (exercise exists for both sides)
    common_idx: int = df_affected.index.intersection(df_healthy.index)

    # calculate ratio (prevent division by zero)
    df_ratio = df_affected.loc[common_idx, feature_cols] / (df_healthy.loc[common_idx, feature_cols] + 1e-8)

    # reset the index to convert back to standard columns
    df_final = df_ratio.reset_index()

    return df_final


def calc_regression_metrics(df: pd.DataFrame) -> dict:
    y_true = df['Real_Score']
    y_pred = df['Predicted_Score']

    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    # percentage accuracy (1 - (error / real))
    accuracies = (1 - (abs(y_true - y_pred) / y_true)) * 100
    mean_accuracy = accuracies.mean()

    # BCa bootstrap 95% CI for accuracy
    boot_res = bootstrap((accuracies,), np.mean, confidence_level=0.95,
                         method='BCa', n_resamples=1000, random_state=42)
    ci_lower = boot_res.confidence_interval.low
    ci_upper = boot_res.confidence_interval.high

    return {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Mean_Accuracy_%': mean_accuracy,
            'Acc_CI_lower': ci_lower, 'Acc_CI_upper': ci_upper}


def evaluate_regression_models(res_dir: str) -> None:

    # initialize Visualizer
    viz: Visualizer = Visualizer()

    # get the model algo name (catboost, xgboost, rf)
    model_algo: str = os.path.basename(res_dir).split('_')[0]

    # master metrics table
    oof_file_lst = [os.path.join(res_dir, f) for f in os.listdir(res_dir)
                    if f.startswith(f'{model_algo}_predictions_') and f.endswith('.csv')]

    metrics_lst = []
    df_all_subtests: pd.DataFrame = pd.DataFrame()

    for file in oof_file_lst:
        df = pd.read_csv(file)
        target = df['Target'].iloc[0]
        metrics = calc_regression_metrics(df)
        metrics['Target'] = target
        metrics_lst.append(metrics)

        if 'Total' not in target:
            df_all_subtests = pd.concat([df_all_subtests, df])
        else:
            df_total = df.copy()

    metrics_df = pd.DataFrame(metrics_lst)
    print('\nMaster Metrics Table:')
    print(metrics_df.round(3).to_string(index=False))
    metrics_df.to_csv(os.path.join(res_dir, f'{model_algo}_metrics.csv'))

    # create identity plot (all subtests)
    if not df_all_subtests.empty:
        viz.viz_regression_identity_plot(df_all_subtests, model_algo)

    # create bland-altman plot (total score)
    if 'df_total' in locals():
        viz.viz_regression_bland_altman(df_total, model_algo)


def run_regression_pipeline():

    # model choice and target score
    model_algo: str = config['regression'].get('model_choice', 'catboost')
    target_type: str = config['regression'].get('reference_score', 'JT')
    baseline_visit: str = config['classification']['baseline_visit_id']
    print(f'Starting Machine Learning Pipeline: Phase 2 - {target_type}-Score Regression ...')

    # define output folder
    out_dir: str = os.path.join(project_path, 'data', '05_results', '06_regression', f'{model_algo}_regression')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

        # 1) load feature dataset
        features_path: str = os.path.join(project_path, 'data', '05_results', '04_classification', 'clean_features.csv')
        df_raw: pd.DataFrame = pd.read_csv(features_path)

        # extract raw feature names
        meta_cols: list[str] = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID']
        feature_cols: list[str] = [c for c in df_raw.columns if c not in meta_cols]

        # 2) transforming raw kinematics into asymmetry ratios and merge with target scores
        print('\nTransforming kinematics into Asymmetry Ratios ...')
        df_kinematics: pd.DataFrame = calc_asymmetry_ratios(df_raw, feature_cols)

        # get the reference scores merged with the features
        df_master, primary_target, target_columns = merge_features_with_targets(df_kinematics)

        # 3) loop through targets
        for target_col in target_columns:

            # 3.1) train base models via the pipeline manager
            print(f'\nTraining Ensemble for: {target_col} ... ')

            # select method to use: 'catboost', 'xgb' or 'rf'
            ensemble = EnsembleManager(model_type=model_algo,
                                       task_type='regression',
                                       n_trials=config['regression']['optuna_n_trials'])

            # call inner- and outer-loop cross validation function; returns validation results and SHAP values
            df_oof, df_shap, df_feat = ensemble.train_with_nested_cv(df_master, feature_cols, target_col,
                                                                     baseline_visit, n_splits=5, out_dir=out_dir)

            # save pooled out-of-fold results
            df_oof.to_csv(os.path.join(out_dir, f'{model_algo}_predictions_{target_col}.csv'), index=False)
            df_shap.to_csv(os.path.join(out_dir, f'{model_algo}_shap_vals_{target_col}.csv'), index=False)
            df_feat.to_csv(os.path.join(out_dir, f'{model_algo}_shap_feats_{target_col}.csv'), index=False)

        print(f'\nRegression completed for {target_type}. Results saved.')

    else:
        print(f'\nRegression results directory already exist. Regression was skipped.')

    print(f'\nEvaluating models ...')
    evaluate_regression_models(out_dir)

    print(f'\nEvaluating SHAP ...')
    tb: ToolBox = ToolBox()
    tb.evaluate_shap(out_dir)

    print(f'\nEvaluation was successful. Results saved.')


if __name__ == '__main__':
    run_regression_pipeline()
