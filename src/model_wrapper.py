# src/model_wrapper.py

# libraries
import numpy as np
import pandas as pd
import optuna
import shap
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import log_loss, root_mean_squared_error

# modules
from src.config import config

# models
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ModelWrapper:
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

        try:
            space_dict = config[self.task_type]['optuna_spaces'][self.model_type]
        except KeyError:
            raise ValueError(f'Optuna space not found in config for {self.task_type} -> {self.model_type}')

        # dynamically parse TOML definitions into Optuna trial suggestions
        for p_name, p_rules in space_dict.items():
            p_type = p_rules.get('type')

            if p_type == 'int':
                params[p_name] = trial.suggest_int(name=p_name, low=p_rules['low'], high=p_rules['high'],
                                                   log=p_rules.get('log', False))
            elif p_type == 'float':
                params[p_name] = trial.suggest_float(name=p_name, low=p_rules['low'], high=p_rules['high'],
                                                     log=p_rules.get('log', False))
            elif p_type == 'categorical':
                params[p_name] = trial.suggest_categorical(name=p_name, choices=p_rules['choices'])
            elif p_type == 'fixed':
                params[p_name] = p_rules['value']
            else:
                print(f'Warning: Unknown parameter type {p_type} for {p_name}')

        # if self.model_type == 'xgboost':
        #     params['n_estimators'] = trial.suggest_int('n_estimators', 20, 60) # [Learning Mechanic] number of trees in the model: high = risk of overfitting
        #     params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)  # [Learning Mechanic] step size: low = more robust learning
        #     params['max_depth'] = trial.suggest_int('max_depth', 2, 3)  # [Complexity] max depth per tree: low = forces generalization
        #     params['min_child_weight'] = trial.suggest_int('min_child_weight', 4, 10)    # [Complexity] minimum weight required to create a new split
        #     params['reg_lambda'] = trial.suggest_float('reg_lambda', 5.0, 40.0, log=True)  # [Complexity] mathematical L2 penalty (regularization) on weights
        #     params['reg_alpha'] = trial.suggest_float('reg_alpha', 1.0, 20.0, log=True) # [Complexity] mathematical L1 penalty (Lasso)
        #     params['subsample'] = trial.suggest_float('subsample', 0.5, 0.8)    # [Randomness] fraction of rows (participants) used per tree
        #     params['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.4, 0.7)  # [Randomness] fraction of columns (features) used per tree
        #
        # elif self.model_type == 'catboost':
        #     params['n_estimators'] = trial.suggest_int('n_estimators', 20, 60) # [Learning Mechanic] number of trees in the model: high = risk of overfitting
        #     params['learning_rate'] = trial.suggest_float('learning_rate', 0.01, 0.1, log=True)  # [Learning Mechanic] step size: low = more robust learning
        #     params['depth'] = trial.suggest_int('depth', 2, 3)  # [Complexity] max depth per tree: low = forces generalization
        #     params['l2_leaf_reg'] = trial.suggest_float('l2_leaf_reg', 5.0, 40.0, log=True) # [Complexity] mathematical L2 penalty (regularization) on weights
        #     params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.4, 0.7)  # [Randomness] fraction of columns (features) used per tree level
        #     params['random_strength'] = trial.suggest_float('random_strength', 1.0, 8.0)   # calculate the best split, then add a random noise multiplier to the score
        #     params['bootstrap_type'] = 'Bernoulli'  # required for subsampling
        #     params['subsample'] = trial.suggest_float('subsample', 0.5, 0.7)  # Only look at 50-80% of participants per tree
        #
        # elif self.model_type == 'rf':
        #     params['n_estimators'] = trial.suggest_categorical('n_estimators', [100, 200]) # [Learning Mechanic] number of trees in the model: high = better stability
        #     params['max_depth'] = trial.suggest_int('max_depth', 2, 3)  # [Complexity] max depth per tree: low = forces generalization
        #     params['min_samples_split'] = trial.suggest_int('min_samples_split', 5, 8) # [Complexity] Min samples required to split a node.
        #     params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 3, 5)    # [Complexity] Min samples required to form a final leaf.
        #     params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2'])    # [Randomness] Max features considered at each split.
        #     params['max_samples'] = trial.suggest_float('max_samples', 0.4, 0.7)

        # evaluation using GroupKFold
        gkf = GroupKFold(n_splits=4)
        scores = []

        for step, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = self._get_estimator(params)
            model.fit(X_train, y_train)

            if self.task_type == 'classification':
                preds = model.predict_proba(X_val)[:, 1]
                score = log_loss(y_val, preds)
            else:
                preds = model.predict(X_val)
                score = root_mean_squared_error(y_val, preds)

            scores.append(score)

            # reports the intermediate fold score to optuna at current step for pruning decision
            trial.report(score, step)

            # check for pruning
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(scores)  # minimize log_loss or RMSE

    def _extract_learning_curve(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series,
                                logger, fold_idx: int, ex_name: str):
        """
        Internally splits data 80:20 to record the iteration-by-iteration learning curve for the logger.
        """
        metric_name: str = 'logloss' if self.task_type == 'classification' else 'rmse'

        # create an internal split just for evaluation
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        try:
            train_idx, val_idx = next(gss.split(X, y, groups=groups))
        except ValueError:
            print('Not enough groups to split for learning curve. Skipping curve generation ...')
            return

        X_train_lc, X_val_lc = X.iloc[train_idx], X.iloc[val_idx]
        y_train_lc, y_val_lc = y.iloc[train_idx], y.iloc[val_idx]

        train_loss = []
        val_loss = []

        if self.model_type == 'xgboost':
            model = self._get_estimator(self.best_params)
            model.fit(X_train_lc, y_train_lc, eval_set=[(X_train_lc, y_train_lc), (X_val_lc, y_val_lc)], verbose=False)
            results = model.evals_result()
            train_loss = results['validation_0'][metric_name]
            val_loss = results['validation_1'][metric_name]

        elif self.model_type == 'catboost':
            model = self._get_estimator(self.best_params)
            model.fit(X_train_lc, y_train_lc, eval_set=(X_val_lc, y_val_lc), verbose=False)
            results = model.get_evals_result()

            # catboost extraction
            train_dict = results.get('learn', {})
            val_dict = results.get('validation', {})

            if train_dict and val_dict:
                # dynamically get the first array
                t_key = list(train_dict.keys())[0]
                v_key = list(val_dict.keys())[0]
                train_loss = train_dict[t_key]
                val_loss = val_dict[v_key]
            else:
                print(f"CatBoost keys missing! Found: {results.keys()}")

        elif self.model_type == 'rf':
            # random forest workaround using warm_start
            params = self.best_params.copy()
            total_trees = params.get('n_estimators', 100)
            params['n_estimators'] = 1
            params['warm_start'] = True
            model = self._get_estimator(params)

            for i in range(1, total_trees + 1):
                model.n_estimators = i
                model.fit(X_train_lc, y_train_lc)

                if self.task_type == 'classification':
                    t_pred = model.predict_proba(X_train_lc)[:, 1]
                    v_pred = model.predict_proba(X_val_lc)[:, 1]
                    train_loss.append(log_loss(y_train_lc, t_pred))
                    val_loss.append(log_loss(y_val_lc, v_pred))
                else:
                    t_pred = model.predict(X_train_lc)
                    v_pred = model.predict(X_val_lc)
                    train_loss.append(root_mean_squared_error(y_train_lc, t_pred))
                    val_loss.append(root_mean_squared_error(y_val_lc, v_pred))

        # log data to JSON file
        logger.log_learning_curve(fold_idx, ex_name, metric_name.upper(), train_loss, val_loss)

    def fit_and_reduce(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, max_features: int = 6,
                       logger=None, fold_idx: int = None, ex_name: str = None):
        """
        Extracts SHAP values to drops weak features, runs Optuna on the reduced feature set and refits the final model.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): target vector
            groups (pd.Series): groups of trials
            max_features (int): maximum number of features to use. Default is 6.
            logger (DataLogger): instance of DataLogger class to log data during model fitting. Default is None.
            fold_idx (int): index of the current fold. Default is None.
            ex_name (str): name of the exercise. Default is None.

        Returns:
            self
        """

        print(f'Training {self.model_type.upper()} ({self.task_type}) on {len(X.columns)} features ...')

        # 1) SHAP feature reduction
        # train model on all features using the default parameters
        interim_model = self._get_estimator({})
        interim_model.fit(X, y)

        # extract SHAP values
        explainer = shap.TreeExplainer(interim_model)

        # classification may return list of arrays; regression usually returns one array
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]        # get class 1 from 2D arrays [class_0, class_1]
        elif len(shap_values.shape) == 3:
            shap_values = shap_values[:, :, 1]  # random forest may return 3D arrays (n_samples, n_features, n_classes)

        # calculate mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        shap_importance = pd.DataFrame({'feature': X.columns,
                                        'importance': mean_abs_shap}).sort_values(by='importance', ascending=False)

        # reduce to 'max_features'
        self.selected_features = shap_importance['feature'].head(max_features).tolist()
        print(f'Top {max_features} features selected via SHAP: {self.selected_features}')

        # apply reduction to the dataset
        X_reduced = X[self.selected_features]

        # 2) hyperparameter tuning with optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        # initialize Median Pruner (requires 5 full trials to set a baseline, waits until fold 1 is done
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
        study = optuna.create_study(direction='minimize', pruner=pruner)

        study.optimize(lambda trial: self._optuna_objective(trial, X_reduced, y, groups), n_trials=self.n_trials)
        self.best_params = study.best_params
        self.study_best_value = study.best_value

        # 3) log training information
        if logger and fold_idx is not None and ex_name is not None:
            # features and parameters
            logger.log_exercise_data(fold_idx, ex_name, self.selected_features, self.best_params)
            # learning curves
            self._extract_learning_curve(X_reduced, y, groups, logger, fold_idx, ex_name)

        # 4) final model fit
        # refit the final model on best features (uses the entire training fold)
        self.final_model = self._get_estimator(self.best_params)
        self.final_model.fit(X_reduced, y)

        return self

    def fit_with_predefined(self, X: pd.DataFrame, y: pd.Series, best_params: dict, selected_features: list[str]):
        """
        Bypasses SHAP reduction and Optuna tuning when optimal hyperparameters data already exists from a previous run.

        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target vector
            best_params (dict): The best hyperparameters evaluated in the previous run.
            selected_features (list[str]): The 6 selected features from previous run.

        Returns:
            self
        """

        print(f'Loading cached parameters from previous run. Optuna tuning for {self.model_type.upper()} is skipped ...')

        self.selected_features = selected_features

        # extract the cached cross-validation error (will be used for ensemble weighting)
        self.study_best_value = best_params.get('_cv_error', 1.0)

        # clean the parameter dictionary before passing it to the model
        clean_params = {k: v for k, v in best_params.items() if k != '_cv_error'}
        self.best_params = clean_params

        # refit the final model instantly
        X_reduced = X[self.selected_features]
        self.final_model = self._get_estimator(clean_params)
        self.final_model.fit(X_reduced, y)

        return self

    # inference methods
    def predict(self, X_new: pd.DataFrame) -> pd.DataFrame:
        return self.final_model.predict(X_new[self.selected_features])

    def predict_proba(self, X_new: pd.DataFrame) -> pd.DataFrame:
        return self.final_model.predict_proba(X_new[self.selected_features])

    def get_final_explainer(self):
        return shap.TreeExplainer(self.final_model, feature_perturbation='tree_path_dependent')

