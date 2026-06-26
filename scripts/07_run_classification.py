#!/usr/bin/python3
# scripts/07_run_classification.py

# libraries
import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.utils import resample

# modules
from src.config import config, project_path
from src.utils import ToolBox
from src.visualization import Visualizer
from src.ml_pipeline_manager import EnsembleManager
from src.data_logging import DataLogger


def calc_classification_metrics(df: pd.DataFrame, n_bootstraps: int = 1000) -> dict:
    y_true = df['Real_Score'].values
    y_prob = df['Predicted_Score'].values
    y_pred = (y_prob > 0.5).astype(int)

    # metrics
    metrics = {'Accuracy': accuracy_score(y_true, y_pred),
               'Precision': precision_score(y_true, y_pred, zero_division=0),
               'Recall': recall_score(y_true, y_pred, zero_division=0),
               'F1_Score': f1_score(y_true, y_pred, zero_division=0),
               'MCC': matthews_corrcoef(y_true, y_pred),
               'ROC_AUC': roc_auc_score(y_true, y_prob)}

    # bootstrapping for 95% CI
    bootstrapped_scores = {k: [] for k in metrics.keys()}

    for _ in range(n_bootstraps):
        # resample with replacement
        indices = resample(np.arange(len(y_true)), replace=True)
        y_true_b = y_true[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = (y_prob_b > 0.5).astype(int)

        # skip iteration if only one class is present (prevents AUC crash)
        if len(np.unique(y_true_b)) < 2:
            continue

        bootstrapped_scores['Accuracy'].append(accuracy_score(y_true_b, y_pred_b))
        bootstrapped_scores['ROC_AUC'].append(roc_auc_score(y_true_b, y_prob_b))
        bootstrapped_scores['F1_Score'].append(f1_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['Precision'].append(precision_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['Recall'].append(recall_score(y_true_b, y_pred_b, zero_division=0))
        bootstrapped_scores['MCC'].append(matthews_corrcoef(y_true_b, y_pred_b))

    # compile the final formatted strings: "Mean [95% CI]"
    results = {}
    for metric_name, point_est in metrics.items():
        if len(bootstrapped_scores[metric_name]) > 0:
            ci_lower = np.percentile(bootstrapped_scores[metric_name], 2.5)
            ci_upper = np.percentile(bootstrapped_scores[metric_name], 97.5)
            results[f'{metric_name} [95% CI]'] = f"{point_est:.3f} [{ci_lower:.3f} - {ci_upper:.3f}]"
        else:
            # fallback if bootstrapping fails completely
            results[f'{metric_name} [95% CI]'] = f"{point_est:.3f} [N/A]"

    return results


def evaluate_classification_models(res_dir: str) -> None:

    # initialize Visualizer
    viz: Visualizer = Visualizer()

    # get the model algo name (catboost, xgboost, rf)
    model_algo: str = os.path.basename(res_dir).split('_')[0]
    task_type: str = os.path.basename(res_dir).split('_')[1]

    # identify the parent directory to check for sister model folders
    parent_dir: str = os.path.dirname(res_dir)

    # master metrics table
    oof_file_lst = [os.path.join(res_dir, f) for f in os.listdir(res_dir)
                    if f.startswith(f'{model_algo}_predictions_') and f.endswith('.csv')]

    metrics_lst = []

    for file in oof_file_lst:
        df = pd.read_csv(file)

        if df.empty:
            continue

        # extract the target from the filename
        filename = os.path.basename(file)
        target = filename.replace(f'{model_algo}_predictions_', '').replace('.csv', '')

        # calculate metrics
        metrics = calc_classification_metrics(df)

        # insert 'Target' key-value pair to dictionary
        metrics = {'Target': target, **metrics}
        metrics_lst.append(metrics)

        # plot confusion matrix
        viz.viz_classification_confusion_matrix(df, model_algo)

        # roc curve
        viz.viz_roc_curve(df, model_algo, target, res_dir)

        # learning curves
        log_file: str = os.path.join(res_dir, f'{model_algo}_{target}_log.json')
        if os.path.exists(log_file):
            viz.viz_regression_learning_curves(log_filepath=log_file, out_dir=res_dir)

        # combined roc curve
        models_to_check = ['xgboost', 'catboost', 'rf']
        model_display_names = {'xgboost': 'XGBoost', 'catboost': 'CatBoost', 'rf': 'Random Forest'}

        all_dirs_exist = True
        oof_csv_paths = {}

        # check if all 3 model folders and their corresponding predictions CSVs exist
        for m in models_to_check:
            m_dir = os.path.join(parent_dir, f"{m}_{task_type}")
            m_csv = os.path.join(m_dir, f"{m}_predictions_{target}.csv")

            if not os.path.exists(m_dir) or not os.path.exists(m_csv):
                all_dirs_exist = False
                break

            oof_csv_paths[m] = m_csv

        # if all exist, generate the combined ROC curve
        if all_dirs_exist:
            predictions_dict = {}

            # load the prediction data for all 3 models
            for m, csv_path in oof_csv_paths.items():
                df_m = pd.read_csv(csv_path)
                predictions_dict[model_display_names[m]] = df_m

            # generate the plot and save it in the parent directory
            viz.viz_combined_roc_curve(
                predictions_dict=predictions_dict,
                target=target,
                out_dir=parent_dir
            )

    if metrics_lst:
        metrics_df = pd.DataFrame(metrics_lst)
        print('\nMaster Metrics Table:')
        print(metrics_df.round(3).to_string(index=False))
        metrics_df.to_csv(os.path.join(res_dir, f'{model_algo}_metrics.csv'), index=False)
    else:
        print('\nNo metrics were found. No table is created.')


def run_classification_pipeline():

    print(f'Starting Machine Learning Pipeline: Phase 1 - Condition Classification ...')

    # model choice and setup
    model_algo: str = config['classification'].get('model_choice', 'catboost')
    baseline_visit = config['classification']['baseline_visit_id']
    res_dir: str = os.path.join(project_path, 'data', '05_results', '04_classification')
    model_dir: str = os.path.join(res_dir, f'{model_algo}_classification')

    # create model directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)

    # 1) load and split dataset
    features_path: str = os.path.join(res_dir, 'clean_features.csv')
    df_raw: pd.DataFrame = pd.read_csv(features_path)

    # binary mapping of condition: 1 = Affected, 0 = Healthy
    df_raw['is_affected'] = df_raw['side_condition'].apply(lambda x: 1 if x == 'Affected' else 0)
    target_col = 'is_affected'    # target column to predict

    # check whether prediction file already exists
    prediction_file: str = os.path.join(model_dir, f'{model_algo}_predictions_{target_col}.csv')

    if not os.path.exists(prediction_file):

        # define features
        meta_cols = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', target_col]
        feature_cols = [c for c in df_raw.columns if c not in meta_cols]

        # 2) train base models via the pipeline manager
        print('\nTraining base classification models ...')
        ensemble = EnsembleManager(model_type=model_algo,
                                   task_type='classification',
                                   n_trials=config['classification']['optuna_n_trials'])

        # instantiate the DataLogger
        logger: DataLogger = DataLogger(out_dir=model_dir,
                                        model_algo=model_algo,
                                        task_type='classification',
                                        target=target_col,
                                        n_trials=config['classification']['optuna_n_trials'])

        # call inner- and outer-loop cross validation
        df_oof, df_shap, df_feat = ensemble.train_with_nested_cv(df_raw, feature_cols, target_col, baseline_visit,
                                                                 n_splits=5, out_dir=model_dir, logger=logger)

        # save logger data to JSON
        logger.save()

        # 3) save Out-Of-Fold (OOF) results and SHAP
        df_oof.to_csv(os.path.join(model_dir, f'{model_algo}_predictions_{target_col}.csv'), index=False)
        df_shap.to_csv(os.path.join(model_dir, f'{model_algo}_shap_vals_{target_col}.csv'), index=False)
        df_feat.to_csv(os.path.join(model_dir, f'{model_algo}_shap_feats_{target_col}.csv'), index=False)

        print('\nClassification complete. Results saved.')

    else:
        print('\nClassification results already exists. Classification skipped.')

    print(f'\nEvaluating models ...')
    evaluate_classification_models(model_dir)

    print('\nEvaluating SHAP ...')
    tb: ToolBox = ToolBox()
    tb.evaluate_shap(model_dir)

    print('\nEvaluation was successful. Results saved.')


if __name__ == '__main__':
    run_classification_pipeline()
