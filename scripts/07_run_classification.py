#!/usr/bin/python3
# scripts/07_run_classification.py

# libraries
import os
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, matthews_corrcoef

# modules
from src.config import config, project_path
from src.utils import ToolBox
from src.visualization import Visualizer
from src.ml_pipeline_manager import EnsembleManager


def calc_classification_metrics(df: pd.DataFrame) -> dict:
    y_true = df['Real_Score']
    y_prob = df['Predicted_Score']
    y_pred = (y_prob > 0.5).astype(int)

    # metrics
    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    matthews_corr = matthews_corrcoef(y_true, y_pred)

    return {'Accuracy': accuracy, 'ROC_AUC': roc_auc, 'F1_Score': f1,
            'Precision': precision, 'Recall': recall, 'MCC': matthews_corr}


def evaluate_classification_models(res_dir: str) -> None:

    # initialize Visualizer
    viz: Visualizer = Visualizer()

    # get the model algo name (catboost, xgboost, rf)
    model_algo: str = os.path.basename(res_dir).split('_')[0]

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

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

        # 1) load and split dataset
        features_path: str = os.path.join(res_dir, 'clean_features.csv')
        df_raw: pd.DataFrame = pd.read_csv(features_path)

        # binary mapping of condition: 1 = Affected, 0 = Healthy
        df_raw['is_affected'] = df_raw['side_condition'].apply(lambda x: 1 if x == 'Affected' else 0)
        target_col = 'is_affected'    # target column to predict

        # define features
        meta_cols = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', target_col]
        feature_cols = [c for c in df_raw.columns if c not in meta_cols]

        # 2) train base models via the pipeline manager
        print('\nTraining base classification models ...')
        ensemble = EnsembleManager(model_type=model_algo,
                                   task_type='classification',
                                   n_trials=config['classification']['optuna_n_trials'])

        # call inner- and outer-loop cross validation
        df_oof, df_shap, df_feat = ensemble.train_with_nested_cv(df_raw, feature_cols, target_col, baseline_visit,
                                                                 n_splits=5)

        # 3) save Out-Of-Fold (OOF) results and SHAP
        df_oof.to_csv(os.path.join(model_dir, f'{model_algo}_predictions_{target_col}.csv'), index=False)
        df_shap.to_csv(os.path.join(model_dir, f'{model_algo}_shap_vals_{target_col}.csv'), index=False)
        df_feat.to_csv(os.path.join(model_dir, f'{model_algo}_shap_feats_{target_col}.csv'), index=False)

        print('\nClassification complete. Results saved.')

    else:
        print('\nClassification results directory already exists. Classification skipped.')

    print(f'\nEvaluating models ...')
    evaluate_classification_models(model_dir)

    print('\nEvaluating SHAP ...')
    tb: ToolBox = ToolBox()
    tb.evaluate_shap(model_dir)

    print('\nEvaluation was successful. Results saved.')


if __name__ == '__main__':
    run_classification_pipeline()
