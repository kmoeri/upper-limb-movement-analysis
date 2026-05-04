#!/usr/bin/python3
# scripts/07_run_classification.py

# libraries
import os
import pandas as pd

# modules
from src.config import config, project_path
from src.data_split import load_and_split_data
from src.ml_pipeline_manager import EnsembleManager


def run_classification_pipeline():

    print(f'Starting Machine Learning Pipeline: Phase 1 - Condition Classification ...')

    # 1) load and split dataset
    features_path: str = os.path.join(project_path, 'data', '04_features', 'clean_features.csv')
    test_participants = config['classification']['test_participants']
    baseline_visit = config['classification']['baseline_visit_id']
    df_train, df_test = load_and_split_data(features_path, test_participants, baseline_visit)

    # binary mapping of condition: 1 = Affected, 0 = Healthy
    df_train['is_affected'] = df_train['side_condition'].apply(lambda x: 1 if x == 'Affected' else 0)
    df_test['is_affected'] = df_test['side_condition'].apply(lambda x: 1 if x == 'Affected' else 0)

    # define features
    meta_cols = ['p_ID', 'visit_ID', 'ex_name', 'affected_side', 'side_focus', 'side_condition', 'cam_ID', 'AHA_Score']
    target_col = 'is_affected'    # target column to predict
    feature_cols = [c for c in df_train.columns if c not in meta_cols and c != target_col]

    # 2) Train base models via the pipeline manager
    print('\nTraining base classification models ...')
    # select method to use: 'catboost', 'xgb' or 'rf'
    ensemble = EnsembleManager(model_type='catboost', task_type='classification', n_trials=30)

    # loop exercises, run optuna, perform SHAP reducing
    ensemble.train_all_exercises(df_train, feature_cols, target_col)

    # 3) Evaluate test set
    print('\nEvaluating test set ...')
    results: list[dict] = []

    # group by participant ID to evaluate visit by visit
    for (pid, side), df_limb in df_test.groupby(['p_ID', 'side_focus']):

        # ensure only baseline visits are used
        df_baseline = df_limb[df_limb['visit_ID'] == baseline_visit]
        if df_baseline.empty:
            continue

        # get ground truth
        real_condition = df_baseline[target_col].iloc[0]

        # get ensemble prediction (output: probability of class 1)
        predicted_probability = ensemble.predict_visit(df_baseline, feature_cols)

        if pd.isna(predicted_probability):
            continue

        # get class from predicted probability
        predicted_class = 1 if predicted_probability > 0.5 else 0

        # append current results
        results.append({'p_ID': pid,
                        'Real_Affected': real_condition,
                        'Predicted_Probability': round(predicted_probability, 3),
                        'Predicted_Class': predicted_class,
                        'Correct': real_condition == predicted_class,
                        'Exercises_Used': list(df_baseline['ex_name'].unique())})

        status = 'Correct' if real_condition == predicted_class else 'Incorrect'
        print(f'Participant ID: {pid} ({side}) | Real: {real_condition} | '
              f'Pred: {predicted_probability:.2f} -> {predicted_class} | {status}.')

    # save final results
    results_df = pd.DataFrame(results)
    out_dir: str = os.path.join(project_path, 'data', '05_results', '04_classification')
    os.makedirs(out_dir, exist_ok=True)
    results_df.to_csv(os.path.join(out_dir, 'classification_predictions.csv'), index=False)
    print('\nClassification complete. Results saved.')


if __name__ == '__main__':
    run_classification_pipeline()
