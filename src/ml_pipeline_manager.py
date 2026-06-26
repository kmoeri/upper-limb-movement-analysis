# src/ml_pipeline_manager.py

# libraries
import os
import json
import shap
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, StratifiedKFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

# modules
from src.config import config
from src.model_wrapper import ModelWrapper
from src.visualization import Visualizer


class EnsembleManager:
    def __init__(self, model_type: str = 'catboost', task_type: str = 'classification', n_trials: int = 30):
        """
        model_algo: 'xgboost', 'catboost', or 'rf'
        task_type: 'classification' or 'regression'
        n_trials: Number of optuna optimization rounds
        """
        # member variables
        self.model_type = model_type.lower()
        self.task_type = task_type.lower()
        self.n_trials = n_trials

        # stores models and their CV errors
        self.trained_models: dict[str, ModelWrapper] = {}
        self.exercise_errors: dict[str, float] = {}

        # define exercises
        self.expected_exercises: list[str] = ['FingerTapping', 'FingerAlternation', 'HandOpening', 'ProSup']

    def train_all_exercises(self, df_train: pd.DataFrame, feature_cols: list[str],
                            target_col: str, pre_tuned_params: dict = None,
                            logger=None, fold_idx: int = None) -> dict:
        """
        Trains models for all exercises. If pre_tuned_params is provided (data from a previous run),
        the tuning is skipped.

        Args:
            df_train (pd.DataFrame): Training data.
            feature_cols (list[str]): List of features to train on.
            target_col (str): Target column.
            pre_tuned_params (dict): Dictionary of tuning parameters from a previous run.
            logger (DataLogger): DataLogger object tracking training information. Default is None.
            fold_idx (int): index of the current fold. Default is None.

        Returns:
            newly_tuned_params (dict): Dictionary of tuned parameters and selected features.
        """

        self.trained_models = {}
        self.exercise_errors = {}
        newly_tuned_params = {}

        for ex_name in self.expected_exercises:
            print(f'Processing exercise: {ex_name} ...')

            # filter data for the current exercise
            df_ex: pd.DataFrame = df_train[df_train['ex_name'] == ex_name].dropna(subset=[target_col])
            df_ex = df_ex.dropna(axis=1, how='all')

            if df_ex.empty:
                print(f'Warning: No training data found for exercise: {ex_name}. Skipping.')
                continue

            # get feature matrix, target vector, and participant groups
            X: pd.DataFrame = df_ex[[col for col in feature_cols if col in df_ex.columns]]
            y: pd.Series = df_ex[target_col]
            groups: pd.Series = df_ex['p_ID']   # required for GroupKFold

            # instantiate and run ModelWrapper (optuna and SHAP)
            wrapper: ModelWrapper = ModelWrapper(model_type=self.model_type,
                                                 task_type=self.task_type,
                                                 n_trials=self.n_trials)

            if pre_tuned_params and ex_name in pre_tuned_params:
                # load from cache
                cached_data = pre_tuned_params[ex_name]
                wrapper.fit_with_predefined(X, y, cached_data['params'], cached_data['features'])
            else:
                # fit and reduce from scratch
                max_features: int = config['classification']['max_features']
                wrapper.fit_and_reduce(X, y, groups, max_features=max_features,
                                       logger=logger, fold_idx=fold_idx, ex_name=ex_name)

                # bundle parameters to save to JSON
                params_to_save = wrapper.best_params.copy()
                params_to_save['_cv_error'] = wrapper.study_best_value if hasattr(wrapper, 'study_best_value') else 1.0

                newly_tuned_params[ex_name] = {'params': params_to_save,
                                               'features': wrapper.selected_features}

            # store trained wrapper
            self.trained_models[ex_name] = wrapper

            # extract the lowest CV error from optuna for the weights
            best_error = wrapper.study_best_value if hasattr(wrapper, 'study_best_value') else 1.0
            self.exercise_errors[ex_name] = best_error

        print('\nAll exercises trained successfully.')
        return newly_tuned_params

    def predict_and_explain_visit(self, df_visit: pd.DataFrame, feature_cols: list[str]) -> tuple:
        """
        Calculates the weighted ensemble prediction and the weighted SHAP values for the entire participant visit.

        Args:
            df_visit (pd.DataFrame): subset of feature and target data of a baseline visit (T1).
            feature_cols (list[str]): List of features to train on.

        Returns:
            tuple: Weighted ensemble prediction, participant SHAP values, and participant features.
        """

        # A) weighted ensemble prediction

        predictions = []
        base_weights = []
        models_used = []
        exercise_X_news = []

        complete_exercises: pd.DataFrame = df_visit['ex_name'].unique()
        for ex_name in complete_exercises:
            if ex_name not in self.trained_models:
                continue

            df_ex_row = df_visit[df_visit['ex_name'] == ex_name]
            X_new = df_ex_row[feature_cols]
            model = self.trained_models[ex_name]

            if self.task_type == 'regression':
                pred = model.predict(X_new)[0]
            else:
                pred = model.predict_proba(X_new)[0, 1]

            error = self.exercise_errors[ex_name]
            weight = 1.0 / (error + 1e-8)

            predictions.append(pred)
            base_weights.append(weight)
            models_used.append((ex_name, model))
            exercise_X_news.append(X_new)

        if not predictions:
            return np.nan, {}, {}

        # normalize weights
        total_weight = sum(base_weights)
        normalized_weights = [weight / total_weight for weight in base_weights]

        # final ensemble prediction
        final_prediction = sum(p * w for p, w in zip(predictions, normalized_weights))

        # B) SHAP values extraction
        participant_shap = {}
        participant_features = {}

        for weight, (ex_name, model), X_new in zip(normalized_weights, models_used, exercise_X_news):
            X_reduced = X_new[model.selected_features]
            explainer = shap.TreeExplainer(model.final_model)
            shap_vals = explainer.shap_values(X_reduced)

            # handle varying SHAP output formats
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]
            elif len(shap_vals.shape) == 3:
                shap_vals = shap_vals[:, :, 1]

            # apply ensemble weight to the SHAP values
            weighted_shap = shap_vals[0] * weight

            for feat_name, s_val, f_val in zip(model.selected_features, weighted_shap, X_reduced.iloc[0].values):
                # if multiple exercises select the same feature, sum their weighted SHAP contribution
                participant_shap[feat_name] = participant_shap.get(feat_name, 0.0) + s_val
                participant_features[feat_name] = f_val

        return final_prediction, participant_shap, participant_features

    def train_with_nested_cv(self, df: pd.DataFrame, feature_cols: list[str], target_col: str, baseline_visit: str,
                             n_splits: int = 5, out_dir: str = None, logger=None) -> tuple:
        """
        Executes a nested Cross-Validation (CV) with StratifiedGroupKFold. Dynamically bins the target variable
        (e.g., Jebson Taylor assessment scores) to ensure balanced severity (mild, moderate, severe) across folds.
        Trains on all available visits (T1, T2, T3) but evaluates Out-Of-Fold (OOF) predictions strictly on T1 visits.

        Args:
            df (pd.DataFrame): Dataset including all features and target variables.
            feature_cols (list[str]): List of feature names of the dataset.
            target_col (str): The target column with the scores to be predicted (also used for training).
            baseline_visit (str): Visit ID information to only use the baseline data for evaluation (test fold).
            n_splits (int, optional): Number of folds. Defaults to 5.
            out_dir (str, optional): Directory to store the CV results. Defaults to None.
            logger (DataLogger): DataLogger object tracking training information. Default is None.

        Returns:
            tuple: (OOF predictions for all patients, SHAP value DataFrame, SHAP feature DataFrame).
        """

        # splitting logic to handle classification and regression differently

        if self.task_type == 'regression':

            """
            ####################### Testing Flattened Data #######################
            # create a unique ID for every single visit (e.g., "P001_T1")
            df['unique_visit_id'] = df['p_ID'] + '_' + df['visit_ID']

            # isolate the 62 unique visits for severity binning
            df_unique_visits = df.drop_duplicates(subset=['unique_visit_id']).copy()
            df_unique_visits = df_unique_visits.dropna(subset=[target_col])

            # bin the 62 visits
            df_unique_visits['severity_bin'] = pd.qcut(df_unique_visits[target_col], q=3,
                                                       labels=['Mild', 'Moderate', 'Severe'])

            # map the bins back to the main long-format dataframe
            bin_mapping = dict(zip(df_unique_visits['unique_visit_id'], df_unique_visits['severity_bin']))
            df['severity_bin'] = df['unique_visit_id'].map(bin_mapping)
            df = df.dropna(subset=['severity_bin']).copy()

            X = df[feature_cols]
            y = df['severity_bin']

            # group by the unique visit instead of the participant
            groups = df['unique_visit_id']

            cv_splitter = StratifiedGroupKFold(n_splits=n_splits)
            split_iterator = cv_splitter.split(X, y=y, groups=groups)
            ####################### Testing Flattened Data #######################
            """

            # dynamically determine the chronologically first visit per participant
            first_visit_series = df.groupby('p_ID')['visit_ID'].transform('min')
            df_first_visits: pd.DataFrame = df[df['visit_ID'] == first_visit_series].copy()

            # baseline visit data
            df_baseline: pd.DataFrame = df_first_visits.drop_duplicates(subset=['p_ID']).copy()

            # create 3 equally sized percentiles with "qcut"
            df_baseline['severity_bin'] = pd.qcut(df_baseline[target_col], q=3, labels=['Mild', 'Moderate', 'Severe'])

            # visualize regression target distribution
            viz: Visualizer = Visualizer()
            viz.viz_regression_target_distribution(df_baseline, target_col, self.model_type)

            # map the bins back to the main dataframe
            bin_mapping = dict(zip(df_baseline['p_ID'], df_baseline['severity_bin']))
            df['severity_bin'] = df['p_ID'].map(bin_mapping)

            # drop any participants that lack a baseline visit
            df = df.dropna(subset=['severity_bin']).copy()

            X = df[feature_cols]
            y = df['severity_bin']
            groups = df['p_ID']

            cv_splitter = StratifiedGroupKFold(n_splits=n_splits)
            split_iterator = cv_splitter.split(X, y=df['severity_bin'], groups=groups)

        else:
            # classification - naturally balanced group as each participant has a healthy and an affected side
            X = df[feature_cols]
            y = df[target_col]
            groups = df['p_ID']

            cv_splitter = GroupKFold(n_splits=n_splits)
            split_iterator = cv_splitter.split(X, y=y, groups=groups)

        # json cache loader
        all_folds_params = {}
        json_path = os.path.join(out_dir, f'{self.model_type}_tuned_params_{target_col}.json') if out_dir else None

        if json_path and os.path.exists(json_path):
            with open(json_path, 'r') as f:
                all_folds_params = json.load(f)
            print('Found cached hyperparameter file. Optuna tuning will be bypassed.')

        # list definition to store the oof results and SHAP results
        oof_results_lst: list = []
        shap_values_lst: list = []
        shap_features_lst: list = []

        # track fold-specific training performance metrics
        fold_train_metrics_lst: list = []

        # nested cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(split_iterator):
            print(f'\nFold {fold + 1}/{n_splits}')

            # df_train_fold contains all visits (T1, T2, and T3) for the training folds (e.g., 20 participants)
            df_train_fold = df.iloc[train_idx].copy()
            df_val_fold = df.iloc[val_idx].copy()

            # log the train and test participant IDs for the current fold
            if logger:
                train_pids = df_train_fold['p_ID'].unique().tolist()
                test_pids = df_val_fold['p_ID'].unique().tolist()
                logger.log_fold_splits(fold + 1, train_pids, test_pids)

            # pass cache and save updates
            fold_key: str = f'fold_{fold}'
            cached_fold_data = all_folds_params.get(fold_key, None)

            # handles both cached training and training from scratch
            new_params = self.train_all_exercises(df_train_fold, feature_cols, target_col,
                                                  pre_tuned_params=cached_fold_data, logger=logger, fold_idx=fold+1)

            if new_params and json_path:
                all_folds_params[fold_key] = new_params
                with open(json_path, 'w') as f:
                    json.dump(all_folds_params, f, indent=4)

            # 1) train the ensemble on the augmented training data
            #self.train_all_exercises(df_train_fold, feature_cols, target_col, logger=logger, fold_idx=fold+1)

            # compute and track training ensemble performance metrics
            if self.task_type == 'regression':
                fold_train_preds = []
                fold_train_reals = []
                for pid, df_visit in df_train_fold.groupby('p_ID'):
                    real_score_t = df_visit[target_col].iloc[0]
                    pred_score_t, _, _ = self.predict_and_explain_visit(df_visit, feature_cols)
                    if not pd.isna(pred_score_t):
                        fold_train_preds.append(pred_score_t)
                        fold_train_reals.append(real_score_t)

                if fold_train_preds:
                    f_r2 = r2_score(fold_train_reals, fold_train_preds)
                    f_rmse = root_mean_squared_error(fold_train_reals, fold_train_preds)
                    f_mae = mean_absolute_error(fold_train_reals, fold_train_preds)
                    fold_train_metrics_lst.append({'Target': target_col,
                                                   'Fold': fold + 1,
                                                   'Train_R2': f_r2,
                                                   'Train_RMSE': f_rmse,
                                                   'Train_MAE': f_mae})

            # 2) filter the validation fold to only use baseline data

            val_first_visit_series = df_val_fold.groupby('p_ID')['visit_ID'].transform('min')
            df_val_baseline = df_val_fold[df_val_fold['visit_ID'] == val_first_visit_series].copy()
            """
            ####################### Testing Flattened Data #######################
            df_val_baseline = df_val_fold.copy()
            df_val_baseline['severity_bin'] = 'Flattened_Test'
            ####################### Testing Flattened Data #######################
            """
            # 3) grouping logic (separate limbs for classification)
            group_keys = ['p_ID', 'side_focus'] if self.task_type == 'classification' else ['p_ID']

            # 4) predict the baseline OOF scores and extract SHAP
            for keys, df_visit in df_val_baseline.groupby(group_keys):

                if self.task_type == 'classification':
                    pid = keys[0]
                    side = keys[1]
                else:
                    pid = keys[0] if isinstance(keys, tuple) else keys
                    side = 'Ratio'

                # get the groundtruth score
                real_score = df_visit[target_col].iloc[0]

                # get the ensemble prediction using the models trained in this fold
                predicted_score, p_shap, p_feat = self.predict_and_explain_visit(df_visit, feature_cols)

                if pd.isna(predicted_score):
                    continue

                # append current results
                if self.task_type == 'regression':
                    oof_results_lst.append({'Target': target_col,
                                            'p_ID': pid,
                                            'Fold': fold + 1,
                                            'Real_Score': real_score,
                                            'Predicted_Score': round(predicted_score, 3),
                                            'Error': round(abs(float(real_score - predicted_score)), 3),
                                            'Severity_Class': df_visit['severity_bin'].iloc[0],
                                            'Exercises_Used': list(df_visit['ex_name'].unique())})
                else:
                    # classification
                    oof_results_lst.append({'p_ID': pid,
                                            'Side': side,
                                            'Real_Score': real_score,
                                            'Predicted_Score': round(predicted_score, 3),
                                            'Exercises_Used': list(df_visit['ex_name'].unique())})

                # attach tracking metadata to dictionaries
                p_shap['p_ID'] = pid
                p_feat['p_ID'] = pid

                # track validation fold number for stability analysis
                if self.task_type == 'regression':
                    p_shap['Fold'] = fold + 1
                    p_feat['Fold'] = fold + 1

                shap_values_lst.append(p_shap)
                shap_features_lst.append(p_feat)

                print(f'Participant ID: {pid} | Real Score: {real_score:.2f} | '
                      f'Predicted Score: {predicted_score:.2f} | Used: {len(df_visit)} exercises.')

        # compile the OOF result DataFrame
        oof_results_df = pd.DataFrame(oof_results_lst)

        # compile SHAP DataFrames
        df_shap: pd.DataFrame = pd.DataFrame(shap_values_lst).fillna(0.0)
        df_feat: pd.DataFrame = pd.DataFrame(shap_features_lst).fillna(np.nan)

        if self.task_type == 'regression' and out_dir and fold_train_metrics_lst:
            df_fold_train = pd.DataFrame(fold_train_metrics_lst)
            df_fold_train.to_csv(os.path.join(out_dir, f'{self.model_type}_fold_train_metrics_{target_col}.csv'), index=False)

        if self.task_type == 'regression' and not oof_results_df.empty:
            viz: Visualizer = Visualizer()
            viz.viz_regression_split_distribution(df_train=df,
                                                  df_test=oof_results_df,
                                                  target_col=target_col,
                                                  model_algo=self.model_type,
                                                  task_type=self.task_type)

        print(f'\nSuccessfully evaluated {len(oof_results_lst)} baseline OOF predictions.')

        return oof_results_df, df_shap, df_feat
