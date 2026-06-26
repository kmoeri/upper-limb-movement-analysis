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
from src.data_logging import DataLogger


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
    y_true = df['Real_Score'].values
    y_pred = df['Predicted_Score'].values

    # standard regression metrics
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    results = {}

    # helper function for SciPy bootstrap
    def _metric_statistic(y_t, y_p, metric_func):
        # prevent zero variance
        if np.var(y_t) == 0:
            return np.nan
        return metric_func(y_t, y_p)

    # calculate BCa confidence intervals
    try:
        # R2
        res_r2 = bootstrap((y_true, y_pred), lambda t, p: _metric_statistic(t, p, r2_score),
                           paired=True, method='BCa', n_resamples=1000, random_state=42)
        results[
            'Pooled_Test_R2 [95% CI]'] = f"{r2:.3f} [{res_r2.confidence_interval.low:.3f} - {res_r2.confidence_interval.high:.3f}]"

        # RMSE
        res_rmse = bootstrap((y_true, y_pred), lambda t, p: _metric_statistic(t, p, root_mean_squared_error),
                             paired=True, method='BCa', n_resamples=1000, random_state=42)
        results[
            'Pooled_Test_RMSE [95% CI]'] = f"{rmse:.3f} [{res_rmse.confidence_interval.low:.3f} - {res_rmse.confidence_interval.high:.3f}]"

        # MAE
        res_mae = bootstrap((y_true, y_pred), lambda t, p: _metric_statistic(t, p, mean_absolute_error),
                            paired=True, method='BCa', n_resamples=1000, random_state=42)
        results[
            'Pooled_Test_MAE [95% CI]'] = f"{mae:.3f} [{res_mae.confidence_interval.low:.3f} - {res_mae.confidence_interval.high:.3f}]"

    except Exception as e:
        # fallback if BCa fails due to extreme variance edge cases
        print(f"BCa Bootstrap failed: {e}. Falling back to N/A.")
        results['Pooled_Test_R2 [95% CI]'] = f"{r2:.3f} [N/A]"
        results['Pooled_Test_RMSE [95% CI]'] = f"{rmse:.3f} [N/A]"
        results['Pooled_Test_MAE [95% CI]'] = f"{mae:.3f} [N/A]"

    return results


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

        # get pooled test metrics
        metrics = calc_regression_metrics(df)
        metrics['Target'] = target

        # get fold-level test metrics (Mean +/- STD)
        fold_r2s, fold_rmses, fold_maes = [], [], []
        for fold in sorted(df['Fold'].unique()):
            df_fold = df[df['Fold'] == fold]
            fold_r2s.append(r2_score(df_fold['Real_Score'], df_fold['Predicted_Score']))
            fold_rmses.append(root_mean_squared_error(df_fold['Real_Score'], df_fold['Predicted_Score']))
            fold_maes.append(mean_absolute_error(df_fold['Real_Score'], df_fold['Predicted_Score']))

        metrics['Test_R2 (Mean±STD)'] = f"{np.mean(fold_r2s):.3f} ± {np.std(fold_r2s):.3f}"
        metrics['Test_RMSE (Mean±STD)'] = f"{np.mean(fold_rmses):.3f} ± {np.std(fold_rmses):.3f}"
        metrics['Test_MAE (Mean±STD)'] = f"{np.mean(fold_maes):.3f} ± {np.std(fold_maes):.3f}"

        # read matching fold file, compute Mean +/- STD for train and update metrics table
        train_metrics_file: str = os.path.join(res_dir, f'{model_algo}_fold_train_metrics_{target}.csv')
        if os.path.exists(train_metrics_file):
            df_train_folds = pd.read_csv(train_metrics_file)
            mean_r2, std_r2 = df_train_folds['Train_R2'].mean(), df_train_folds['Train_R2'].std()
            mean_rmse, std_rmse = df_train_folds['Train_RMSE'].mean(), df_train_folds['Train_RMSE'].std()
            mean_mae, std_mae = df_train_folds['Train_MAE'].mean(), df_train_folds['Train_MAE'].std()

            # formatting
            metrics['Train_R2 (Mean±STD)'] = f"{mean_r2:.3f} ± {std_r2:.3f}"
            metrics['Train_RMSE (Mean±STD)'] = f"{mean_rmse:.3f} ± {std_rmse:.3f}"
            metrics['Train_MAE (Mean±STD)'] = f"{mean_mae:.3f} ± {std_mae:.3f}"
        else:
            metrics['Train_R2 (Mean±STD)'] = "N/A"
            metrics['Train_RMSE (Mean±STD)'] = "N/A"
            metrics['Train_MAE (Mean±STD)'] = "N/A"

        if 'Total' not in target:
            df_all_subtests = pd.concat([df_all_subtests, df])
        else:
            df_total = df.copy()

        metrics_lst.append(metrics)

    metrics_df = pd.DataFrame(metrics_lst)

    # re-order columns to display train properties before test properties
    ordered_cols = ['Target',
                    'Train_R2 (Mean±STD)', 'Test_R2 (Mean±STD)', 'Pooled_Test_R2',
                    'Train_RMSE (Mean±STD)', 'Test_RMSE (Mean±STD)', 'Pooled_Test_RMSE',
                    'Train_MAE (Mean±STD)', 'Test_MAE (Mean±STD)', 'Pooled_Test_MAE']

    # ensure all remaining active columns are mapped correctly
    final_cols = [c for c in ordered_cols if c in metrics_df.columns]
    metrics_df = metrics_df[final_cols]

    print('\nMaster Metrics Table:')
    print(metrics_df.round(3).to_string(index=False))
    metrics_df.to_csv(os.path.join(res_dir, f'{model_algo}_metrics.csv'))

    # create identity plot (all subtests)
    if not df_all_subtests.empty:
        viz.viz_regression_identity_plot(df_all_subtests, model_algo)

    # create bland-altman plot (total score)
    if 'df_total' in locals():
        # Bland-Altmann
        viz.viz_regression_bland_altman(df_total, model_algo)

        # generalization gap
        viz.viz_regression_generalization_gap(res_dir=res_dir, model_algo=model_algo, task_type='regression')

        # feature stability boxplot
        total_shap_file = os.path.join(res_dir, f'{model_algo}_shap_vals_Asymmetry_JT_Ratio_Total.csv')
        if os.path.exists(total_shap_file):
            viz.viz_regression_feature_stability_boxplot(model_algo=model_algo,
                                                         f_path=total_shap_file,
                                                         task_type='regression')

        # fold-based distribution plot of train and test data
        viz.viz_regression_fold_distributions(df_total, df_total['Target'].iloc[0], model_algo, res_dir)

        # learning curves
        log_file:str = os.path.join(res_dir, f'{model_algo}_Asymmetry_JT_Ratio_Total_log.json')
        if os.path.exists(log_file):
            viz.viz_regression_learning_curves(log_filepath=log_file, out_dir=res_dir)


def run_regression_pipeline(model_algo: str):

    # model choice and target score
    #model_algo: str = config['regression'].get('model_choice', 'catboost')
    target_type: str = config['regression'].get('reference_score', 'JT')
    baseline_visit: str = config['classification']['baseline_visit_id']
    print(f'Starting Machine Learning Pipeline: Phase 2 - {target_type}-Score Regression ...')

    # define output folder
    out_dir: str = os.path.join(project_path, 'data', '05_results', '06_regression', f'{model_algo}_regression')
    os.makedirs(out_dir, exist_ok=True)

    # SHAP and Optuna parameters are stored and can be retrieved to jump over the evaluation of those parameters
    pred_file = os.path.join(out_dir, f'{model_algo}_predictions_Asymmetry_JT_Ratio_Total.csv')

    if not os.path.exists(pred_file):

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

        # to run only one JTHFT exercise at a time
        #target_columns = ['Asymmetry_JT_Ratio_BeanHandling']

        # 3) loop through targets
        for target_col in target_columns:

            # 3.1) train base models via the pipeline manager
            print(f'\nTraining Ensemble for: {target_col} ... ')

            # select method to use: 'catboost', 'xgb' or 'rf'
            ensemble = EnsembleManager(model_type=model_algo,
                                       task_type='regression',
                                       n_trials=config['regression']['optuna_n_trials'])

            logger: DataLogger = DataLogger(out_dir=out_dir, model_algo=model_algo, task_type='regression',
                                            target=target_col, n_trials=config['regression']['optuna_n_trials'])

            # call inner- and outer-loop cross validation function; returns validation results and SHAP values
            df_oof, df_shap, df_feat = ensemble.train_with_nested_cv(df_master, feature_cols, target_col,
                                                                     baseline_visit, n_splits=5, out_dir=out_dir,
                                                                     logger=logger)

            # save the data logger to JSON
            logger.save()

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
    model_algo_lst: list[str] = ['xgboost']    # ['catboost', 'xgboost', 'rf']
    for model_algo in model_algo_lst:
        print(f'Starting Regression with {model_algo} ...')
        run_regression_pipeline(model_algo)
