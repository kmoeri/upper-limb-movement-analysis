#!/usr/bin/python3
# scripts/04_extract_kinematics.py

# libraries
import os
from tqdm import tqdm

# modules
from src.config import config, project_path
from src.core import Participant
from src.exercise_evaluation import ExerciseEvaluator
from src.visualization import Visualizer
from src.utils import save_extracted_data_to_csv


def run_kinematics_extractor(save_plots: bool = True):

    # create class instance from ExerciseEvaluator and Visualizer
    ex_eval: ExerciseEvaluator = ExerciseEvaluator(config['camera_param']['fps'])
    viz: Visualizer = Visualizer()

    # load participant objects
    participant_objs_path: str = os.path.join(project_path, 'data', '03_processed')
    participant_pickle_lst: list = [x for x in sorted(os.listdir(participant_objs_path))
                                    if x.startswith('P') and x.endswith('.pickle')]

    all_extracted_features_lst: list = []

    # run for each participant and each visit
    for pickle_file in tqdm(participant_pickle_lst, desc='Extracting Kinematic Features'):
        p: Participant = Participant.load(os.path.join(participant_objs_path, pickle_file))

        # loop through every exercise this participant performed
        for ex_key, exercise in p.exercises.items():

            # metadata
            row_meta_data = {'p_ID': p.pid,
                             'visit_ID': p.visit_id,
                             'affected_side': p.affected_side,
                             'ex_name': exercise.exercise_id,
                             'side_focus': exercise.side_focus,
                             'side_condition': exercise.side_condition,
                             'cam_ID': exercise.cam_id}

            # get the hand size
            active_hand_size: float = exercise.left_hand_size if exercise.side_focus == 'L' else exercise.right_hand_size

            # safety fallback
            if active_hand_size == 0.0:
                print(f'Warning: Hand size 0.0 for {ex_key}. Check loader output.')
                active_hand_size = 1e-8

            try:
                metrics: dict = {}
                if 'FingerTapping' in ex_key:
                    metrics = ex_eval.analyze_finger_tapping(exercise, p.pid, active_hand_size, save_plots)
                elif 'FingerAlternation' in ex_key:
                    metrics = ex_eval.analyze_finger_alternation(exercise, p.pid, active_hand_size, save_plots)
                elif 'HandOpening' in ex_key:
                    metrics = ex_eval.analyze_hand_opening(exercise, p.pid, save_plots)
                elif 'ProSup' in ex_key:
                    metrics = ex_eval.analyze_pronation_supination(exercise, p.pid, save_plots)

                # add the extracted metrics to the metadata
                if metrics:
                    # save all metrics to current exercise
                    exercise.metrics = metrics
                    # filter out the time series data for the extracted features
                    csv_features = {key: value for key, value in metrics.items() if 'time_series' not in key}
                    row_meta_data.update(csv_features)
                    all_extracted_features_lst.append(row_meta_data)

            except Exception as e:
                print(f'Error extracting {ex_key} for {p.pid}: {e}')

        # 'update' (overwrites) the participant object file with the new metrics data.
        p.save(participant_objs_path)

    # save the extracted features to a csv file
    feature_dir: str = os.path.join(project_path, 'data', '04_features')
    os.makedirs(feature_dir, exist_ok=True)
    feature_file_path = os.path.join(feature_dir, 'all_extracted_features.csv')
    save_extracted_data_to_csv(all_extracted_features_lst, out_file_path=feature_file_path)

    if os.path.exists(feature_file_path):
        viz.feature_zscore_heatmap(feature_file_path)


if __name__ == '__main__':
    run_kinematics_extractor()
