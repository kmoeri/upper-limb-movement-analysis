# src/model_wrapper.py

# libraries
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, root_mean_squared_error

# models
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ModelWrapper:
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

        self.best_params = None
        self.study_best_value = None
        self.final_model = None
        self.selected_features = None

    def _get_estimator(self, params: dict = None):

        params = params or {}

        # classifiers
        if self.task_type == 'classification':
            # type selection
            if self.model_type == 'xgboost':
                return XGBClassifier(eval_metric='logloss', random_state=42, **params)
            elif self.model_type == 'catboost':
                return CatBoostClassifier(verbose=False, random_state=42, **params)
            elif self.model_type == 'rf':
                return RandomForestClassifier(random_state=42, **params)

        # regressors
        elif self.task_type == 'regression':
            # type selection
            if self.model_type == 'xgboost':
                return XGBRegressor(random_state=42, **params)
            elif self.model_type == 'catboost':
                return CatBoostRegressor(verbose=False, random_state=42, **params)
            elif self.model_type == 'rf':
                return RandomForestRegressor(random_state=42, **params)

        raise ValueError(f'Unknown combination: {self.model_type} / {self.task_type}')

    def _optuna_objective(self, trial, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        """
        Optuna objective function using GroupKFold to prevent data leakage.

        Args:
            trial: current trial object
            X (pd.DataFrame): feature matrix
            y (pd.Series): target vector
            groups (pd.Series): groups of trials

        Returns:
            float: average score of either log_loss (classification) or mean_squared_error (regression)
        """

        params: dict = {}
        if self.model_type == 'xgboost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 6)              # keep low to prevent overfitting
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)
            params['subsample'] = trial.suggest_float('subsample', 0.6, 1.0)

        elif self.model_type == 'catboost':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['depth'] = trial.suggest_int('depth', 2, 6)
            params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.2, log=True)

        elif self.model_type == 'rf':
            params['n_estimators'] = trial.suggest_int('n_estimators', 50, 300)
            params['max_depth'] = trial.suggest_int('max_depth', 2, 6)
            params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 10)

        # evaluation using GroupKFold
        gkf = GroupKFold(n_splits=4)
        scores = []

        for train_idx, val_idx in gkf.split(X, y, groups=groups):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self._get_estimator(params)
            model.fit(X_train, y_train)

            if self.task_type == 'classification':
                preds = model.predict_proba(X_val)[:, 1]
                scores.append(log_loss(y_val, preds))
            else:
                preds = model.predict(X_val)
                scores.append(root_mean_squared_error(y_val, preds))

        return np.mean(scores)  # minimize log_loss or RMSE

    def fit_and_reduce(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, max_features: int = 6):
        """
        Runs Optuna, extracts SHAP values, drops weak features, and refits the model.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): target vector
            groups (pd.Series): groups of trials
            max_features (int): maximum number of features to use

        Returns:
            self
        """

        print(f'Training {self.model_type.upper()} ({self.task_type}) on {len(X.columns)} features ...')

        # 1) Hyperparameter Tuning (optuna)
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self._optuna_objective(trial, X, y, groups), n_trials=self.n_trials)
        self.best_params = study.best_params
        self.study_best_value = study.best_value

        # 2) SHAP Feature reduction
        # train model on all features
        interim_model = self._get_estimator(self.best_params)
        interim_model.fit(X, y)

        # extract SHAP values
        explainer = shap.TreeExplainer(interim_model)

        # classification may return list of arrays; regression usually returns one array
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]    # get class 1

        # calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({'feature': X.columns,
                                        'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)

        # reduce to 'max_features'
        self.selected_features = shap_importance['feature'].head(max_features).tolist()
        print(f'Top {max_features} features selected via SHAP: {self.selected_features}')

        # 3) Final Model Fit
        # refit the final model on best features
        X_reduced = X[self.selected_features]
        self.final_model = self._get_estimator(self.best_params)
        self.final_model.fit(X_reduced, y)

        return self

    # inference methods
    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        return self.final_model.predict(X_new[self.selected_features])

    def predict_proba(self, X_new: pd.DataFrame) -> pd.DataFrame:
        return self.final_model.predict_proba(X_new[self.selected_features])

    def get_final_explainer(self):
        return shap.TreeExplainer(self.final_model, feature_perturbation='tree_path_dependent')

