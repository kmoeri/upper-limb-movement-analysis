# src/ml_pipeline_manager.py

# libraries
import pandas as pd
import numpy as np
import shap
from sklearn.model_selection import StratifiedGroupKFold

# modules
from src.config import config
from src.model_wrapper import ModelWrapper


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

    # def predict_visit(self, df_visit: pd.DataFrame, feature_cols: list[str]) -> float:
    #     """
    #     Takes all rows for a single participant visit (1-4 rows depending on completed exercises) and
    #     outputs the weighted ensemble prediction.
    #
    #     Args:
    #         df_visit (pd.DataFrame): Visit data.
    #         feature_cols (list[str]): List of features to train on.
    #
    #     Returns:
    #         float: Weighted ensemble prediction.
    #     """
    #
    #     predictions = []
    #     base_weights = []
    #
    #     # identify which exercises were completed
    #     completed_exercises = df_visit['ex_name'].unique()
    #
    #     for ex_name in completed_exercises:
    #         if ex_name not in self.trained_models:
    #             continue    # skip if there is no model for this exercise
    #
    #         # get the row for this exercise
    #         df_ex_row: pd.DataFrame = df_visit[df_visit['ex_name'] == ex_name]
    #         X_new: pd.DataFrame = df_ex_row[feature_cols]
    #
    #         # get prediction from the corresponding model
    #         model = self.trained_models[ex_name]
    #
    #         if self.task_type == 'regression':
    #             pred = model.predict(X_new)[0]
    #         else:
    #             pred = model.predict_proba(X_new)[0, 1]     # get probability of class 1
    #
    #         # calculate inverse-error weight (1/RMSE)
    #         error = self.exercise_errors[ex_name]
    #         weight = 1.0 / (error + 1e-8)   # prevent division by zero
    #
    #         predictions.append(pred)
    #         base_weights.append(weight)
    #
    #     if not predictions:
    #         return np.nan
    #
    #     # dynamic normalization
    #     total_weight = sum(base_weights)
    #     normalized_weights = [weight / total_weight for weight in base_weights]
    #
    #     # calculate the final weighted prediction
    #     final_prediction = sum(prediction * weight for prediction, weight in zip(predictions, normalized_weights))
    #
    #     return final_prediction
    #
    def train_all_exercises(self, df_train: pd.DataFrame, feature_cols: list[str], target_col: str):
        """
        Loops across each exercise, filtering data and training a model using a hold-out validation approach.

        Args:
            df_train (pd.DataFrame): Training data.
            feature_cols (list[str]): List of features to train on.
            target_col (str): Target column.

        Returns:

        """

        self.trained_models = {}
        self.exercise_errors = {}

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

            # fit and reduce
            max_features: int = config['classification']['max_features']
            wrapper.fit_and_reduce(X, y, groups, max_features=max_features)

            # store trained wrapper
            self.trained_models[ex_name] = wrapper

            # extract the lowest CV error from optuna for the weights
            best_error = wrapper.study_best_value if hasattr(wrapper, 'study_best_value') else 1.0
            self.exercise_errors[ex_name] = best_error

        print('\nAll exercises trained successfully.')

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

            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]

            # apply ensemble weight to the SHAP values
            weighted_shap = shap_vals[0] * weight

            for feat_name, s_val, f_val in zip(model.selected_features, weighted_shap, X_reduced.iloc[0].values):
                # if multiple exercises select the same feature, sum their weighted SHAP contribution
                participant_shap[feat_name] = participant_shap.get(feat_name, 0.0) + s_val
                participant_features[feat_name] = f_val

        return final_prediction, participant_shap, participant_features

    def train_with_nested_cv(self, df: pd.DataFrame, feature_cols: list[str], target_col: str, baseline_visit: str,
                             n_splits: int = 5) -> tuple:
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

        Returns:
            tuple: (OOF predictions for all patients, SHAP value DataFrame, SHAP feature DataFrame).
        """

        # create dynamic severity bins based strictly on the baseline visit (T1) -> mild, moderate, and severe

        # baseline visit data
        df_baseline: pd.DataFrame = df[df['visit_ID'] == baseline_visit].drop_duplicates(subset=['p_ID']).copy()

        # create 3 equally sized percentiles with "qcut"
        df_baseline['severity_bin'] = pd.qcut(df_baseline[target_col], q=3, labels=['Mild', 'Moderate', 'Severe'])

        # map the bins back to the main dataframe
        bin_mapping = dict(zip(df_baseline['p_ID'], df_baseline['severity_bin']))
        df['severity_bin'] = df['p_ID'].map(bin_mapping)

        # drop any participants that lack a baseline visit
        df = df.dropna(subset=['severity_bin']).copy()

        # setup stratified group k-fold
        sgkf = StratifiedGroupKFold(n_splits=n_splits)

        groups = df['p_ID']
        X = df[feature_cols]
        y = df[target_col]
        bins = df['severity_bin']

        # list definition to store the oof results and SHAP results
        oof_results_lst: list = []
        shap_values_lst: list = []
        shap_features_lst: list = []

        # nested cross-validation loop
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y=bins, groups=groups)):
            print(f'\nFold {fold + 1}/{n_splits}')

            # df_train_fold contains all visits (T1, T2, and T3) for the training folds (e.g., 20 participants)
            df_train_fold = df.iloc[train_idx].copy()
            df_val_fold = df.iloc[val_idx].copy()

            # 1) train the ensemble on the augmented training data
            self.train_all_exercises(df_train_fold, feature_cols, target_col)

            # 2) filter the validation fold to only use baseline data
            df_val_baseline = df_val_fold[df_val_fold['visit_ID'] == baseline_visit].copy()

            # 3) predict the baseline OOF scores and extract SHAP
            for pid, df_visit in df_val_baseline.groupby('p_ID'):

                # get the groundtruth score
                real_score = df_visit[target_col].iloc[0]

                # get the ensemble prediction using the models trained in this fold
                predicted_score, p_shap, p_feat = self.predict_and_explain_visit(df_visit, feature_cols)

                if pd.isna(predicted_score):
                    continue

                # append current results
                oof_results_lst.append({'Target': target_col,
                                        'p_ID': pid,
                                        'Real_Score': real_score,
                                        'Predicted_Score': round(predicted_score, 3),
                                        'Error': round(abs(float(real_score - predicted_score)), 3),
                                        'Severity_Class': df_visit['severity_bin'].iloc[0],
                                        'Exercises_Used': list(df_visit['ex_name'].unique())})

                # attach tracking metadata to dictionaries
                p_shap['p_ID'] = pid
                p_feat['p_ID'] = pid
                shap_values_lst.append(p_shap)
                shap_features_lst.append(p_feat)

                print(f'Participant ID: {pid} | Real Score: {real_score} | '
                      f'Predicted Score: {predicted_score:.2f} | Used: {len(df_visit)} exercises.')

        # compile the OOF result DataFrame
        oof_results_df = pd.DataFrame(oof_results_lst)

        # compile SHAP DataFrames
        df_shap: pd.DataFrame = pd.DataFrame(shap_values_lst).fillna(0.0)
        df_feat: pd.DataFrame = pd.DataFrame(shap_features_lst).fillna(np.nan)

        print(f'\nSuccessfully evaluated {len(oof_results_lst)} baseline OOF predictions.')

        return oof_results_df, df_shap, df_feat
