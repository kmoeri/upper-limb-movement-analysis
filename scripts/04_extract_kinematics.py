#!/usr/bin/python3
# scripts/04_extract_kinematics.py

# libraries
import os

# modules
from src.config import config, project_path
from src.core import Participant
from src.exercise_evaluation import ExerciseEvaluator
from src.utils import save_extracted_data_to_csv


def run_kinematics_extractor(save_plots: bool = True):

    # create class instance from ExerciseEvaluator
    ex_eval: ExerciseEvaluator = ExerciseEvaluator(config['camera_param']['fps'])

    # load participant objects
    participant_objs_path: str = os.path.join(project_path, 'data', '03_processed')
    participant_pickle_lst: list = [x for x in sorted(os.listdir(participant_objs_path))
                                    if x.startswith('P') and x.endswith('.pickle')]

    all_extracted_features_lst: list = []

    # run for each participant and each visit
    for pickle_file in participant_pickle_lst:
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
            try:
                metrics: dict = {}
                if 'FingerTapping' in ex_key:
                    metrics = ex_eval.analyze_finger_tapping(exercise, p.pid, save_plots)
                    exercise.metrics = metrics
                elif 'FingerAlternation' in ex_key:
                    metrics = ex_eval.analyze_finger_alternation(exercise)
                    exercise.metrics = metrics
                elif 'HandOpening' in ex_key:
                    metrics = ex_eval.analyze_hand_opening(exercise, p.pid, save_plots)
                    exercise.metrics = metrics
                elif 'ProSup' in ex_key:
                    metrics = ex_eval.analyze_pronation_supination(exercise, p.pid, save_plots)
                    exercise.metrics = metrics

                # add the extracted metrics to the metadata
                row_meta_data.update(metrics)

                # append the row data to the main list
                all_extracted_features_lst.append(row_meta_data)

            except Exception as e:
                print(f'Error extracting {ex_key} for {p.pid}: {e}')

    # save the extracted features to a csv file
    save_extracted_data_to_csv(all_extracted_features_lst, out_dir=os.path.join(project_path, 'data', '04_features'))


if __name__ == '__main__':
    run_kinematics_extractor()
