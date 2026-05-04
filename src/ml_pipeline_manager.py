# src/ml_pipeline_manager.py

# libraries
import pandas as pd
import numpy as np

# modules
from src.config import config
from src.model_wrapper import ModelWrapper


class EnsembleManager:
    def __init__(self, model_type: str = 'xgboost', task_type: str = 'classification', n_trials: int = 30):
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

    # exercise pipeline manager
    def train_all_exercises(self, df_train: pd.DataFrame, feature_cols: list[str], target_col: str):
        """
        Loops across each exercise, filtering data and training a model.

        Args:
            df_train (pd.DataFrame): Training data.
            feature_cols (list[str]): List of features to train on.
            target_col (str): Target column.

        Returns:

        """

        for ex_name in self.expected_exercises:
            print(f'Processing exercise: {ex_name} ...')

            # filter data for the current exercise
            df_ex: pd.DataFrame = df_train[df_train['ex_name'] == ex_name].dropna(subset=[target_col])

            if df_ex.empty:
                print(f'Warning: No training data found for exercise: {ex_name}. Skipping.')
                continue

            # get feature matrix, target vector, and participant groups
            X: pd.DataFrame = df_ex[feature_cols]
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

    def predict_visit(self, df_visit: pd.DataFrame, feature_cols: list[str]) -> float:
        """
        Takes all rows for a single participant visit (1-4 rows depending on completed exercises) and
        outputs the weighted ensemble prediction.

        Args:
            df_visit (pd.DataFrame): Visit data.
            feature_cols (list[str]): List of features to train on.

        Returns:
            float: Weighted ensemble prediction.
        """

        predictions = []
        base_weights = []

        # identify which exercises were completed
        completed_exercises = df_visit['ex_name'].unique()

        for ex_name in completed_exercises:
            if ex_name not in self.trained_models:
                continue    # skip if there is no model for this exercise

            # get the row for this exercise
            df_ex_row: pd.DataFrame = df_visit[df_visit['ex_name'] == ex_name]
            X_new: pd.DataFrame = df_ex_row[feature_cols]

            # get prediction from the corresponding model
            model = self.trained_models[ex_name]

            if self.task_type == 'regression':
                pred = model.predict(X_new)[0]
            else:
                pred = model.predict_proba(X_new)[0, 1]     # get probability of class 1

            # calculate inverse-error weight (1/RMSE)
            error = self.exercise_errors[ex_name]
            weight = 1.0 / (error + 1e-8)   # prevent division by zero

            predictions.append(pred)
            base_weights.append(weight)

        if not predictions:
            return np.nan

        # dynamic normalization
        total_weight = sum(base_weights)
        normalized_weights = [weight / total_weight for weight in base_weights]

        # calculate the final weighted prediction
        final_prediction = sum(prediction * weight for prediction, weight in zip(predictions, normalized_weights))

        return final_prediction
