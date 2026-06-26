# src/data_logging.py

# libraries
import os
import json
import datetime


class DataLogger:
    def __init__(self, out_dir: str, model_algo: str, task_type: str, target: str, n_trials: int) -> None:
        """
        Initialize the logger.
        """
        self.out_dir = out_dir
        self.filename = f'{model_algo}_{target}_log.json'
        self.filepath = os.path.join(self.out_dir, self.filename)

        # initialize the dictionary
        self.data: dict = {'Metadata':{'Timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                                       'Algorithm': model_algo.upper(),
                                       'Target': target,
                                       'Task_Type': task_type.capitalize(),
                                       'Optuna_Trials': n_trials},
                           'Outer_Folds': {}}

    def _ensure_fold_exists(self, fold_idx: int):
        fold_key = f'Fold_{fold_idx}'
        if fold_key not in self.data['Outer_Folds']:
            self.data['Outer_Folds'][fold_key] = {'Data_Splits': {},
                                                  'Exercises': {}}
        return fold_key

    def log_fold_splits(self, fold_idx: int, train_p_ids: list[str], test_p_ids: list[str]):
        """
        Logs the participant IDs used in the Train and Test splits for a given fold.
        """
        fold_key = self._ensure_fold_exists(fold_idx)
        self.data['Outer_Folds'][fold_key]['Data_Splits'] = {'Train_P_IDs': list(train_p_ids),
                                                             'Test_P_IDs': list(test_p_ids)}

    def log_exercise_data(self, fold_idx: int, ex_name: str, selected_features: list[str], best_params: dict):
        """
        Logs the selected features and optimal hyperparameters for a specific exercise.
        """
        fold_key = self._ensure_fold_exists(fold_idx)
        if ex_name not in self.data['Outer_Folds'][fold_key]['Exercises']:
            self.data['Outer_Folds'][fold_key]['Exercises'][ex_name] = {}

        self.data['Outer_Folds'][fold_key]['Exercises'][ex_name]['Selected_Features'] = selected_features

        # filter internal keys like '_cv_error' before saving
        clean_params = {k: v for k, v in best_params.items() if not k.startswith('_')}
        self.data['Outer_Folds'][fold_key]['Exercises'][ex_name]['Hyperparameters'] = clean_params

    def log_learning_curve(self, fold_idx: int, ex_name: str, metric: str, train_loss: list[float], val_loss: list[float]):
        """
        Logs the iteration-by-iteration learning curve for investigating overfitting
        """
        fold_key = self._ensure_fold_exists(fold_idx)
        if ex_name not in self.data['Outer_Folds'][fold_key]['Exercises']:
            self.data['Outer_Folds'][fold_key]['Exercises'][ex_name] = {}

        self.data['Outer_Folds'][fold_key]['Exercises'][ex_name]['Learning_Curve'] = {
            'Metric': metric,
            'Train_Loss': [round(float(v), 4) for v in train_loss],
            'Val_Loss': [round(float(v), 4) for v in val_loss]
        }

    def save(self):
        """
        Dumps the current state to the JSON file.
        """
        os.makedirs(self.out_dir, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self.data, f, indent=4)
        print(f'Run log updated: {self.filepath}')
