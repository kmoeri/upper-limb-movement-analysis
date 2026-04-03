#!/usr/bin/python3
# scripts/05_check_kinematics.py

# libraries
import os

# modules
from src.config import project_path
from src.core import Participant, Exercise
from src.visualization import Visualizer


def _prepare_dashboard_data(exercise: Exercise) -> dict:
    """
    Extracts the 3D landmarks of hands and pose including the 1D metrics for the dashboard.
    Returns a dictionary comprising the items necessary for rendering the dashboard:
    - Pose Landmarks
    - Active Hand Landmarks
    - Passive Hand Landmarks
    - Time series of the metrics extracted from the respective exercise
    """

    # filter the Pose landmarks (keep only upper limbs)
    desired_pose_lms = [
        'shoulder_left', 'shoulder_right',
        'elbow_left', 'elbow_right',
        'wrist_left', 'wrist_right'
    ]
    filtered_pose = {k: v for k, v in exercise.clean_pose_landmarks.items() if k in desired_pose_lms}

    # helper function to find the first digit in the landmark name (e.g., 'ftip15' -> returns '1')
    def get_hand_id(key_str):
        for char in key_str:
            if char.isdigit():
                return char
        return None

    # split the hands landmark dictionary using the left-right coding convention (1: Left, 2: Right)
    left_hand_lm: dict = {k: v for k, v in
                          getattr(exercise, 'aligned_hand_landmarks', exercise.clean_hand_landmarks).items() if
                          get_hand_id(k) == '1'}
    right_hand_lm: dict = {k: v for k, v in
                           getattr(exercise, 'aligned_hand_landmarks', exercise.clean_hand_landmarks).items() if
                           get_hand_id(k) == '2'}

    # create dashboard data dict template
    dashboard_data: dict = {'pose': filtered_pose,
                            'left_hand': left_hand_lm,
                            'right_hand': right_hand_lm,
                            'metrics': {}                   # {'ExerciseName': (time_array, signal_array)}
                            }

    # associate hand side to active/passive based on the exercise focus
    if exercise.side_focus == 'L':
        dashboard_data['active_hand'] = left_hand_lm
        dashboard_data['passive_hand'] = right_hand_lm
    else:
        dashboard_data['active_hand'] = right_hand_lm
        dashboard_data['passive_hand'] = left_hand_lm

    # extract the metrics
    metrics = getattr(exercise, 'metrics', {})
    if not metrics:
        return dashboard_data

    ex_id = exercise.exercise_id

    # add the exercise metrics based on the exercise name
    if 'FingerTapping' in ex_id:
        t = metrics.get('idx_tap_time_series_x')
        y = metrics.get('idx_tap_time_series_y')
        if t is not None and y is not None:
            dashboard_data['metrics']['Index-Thenar_Distance'] = (t, y)

    elif 'HandOpening' in ex_id:
        t = metrics.get('open_close_time_series_x')
        y = metrics.get('open_close_time_series_y')
        if t is not None and y is not None:
            dashboard_data['metrics']['Mean-Finger-Wrist_Distance'] = (t, y)

    elif 'ProSup' in ex_id:
        t = metrics.get('pro_sup_time_series_x')
        y = metrics.get('pro_sup_time_series_y')
        if t is not None and y is not None:
            dashboard_data['metrics']['Pronation-Supination_Rotation-Angle'] = (t, y)

    # FingerAlternation

    return dashboard_data


def _render_single_exercise(exercise: Exercise, pid: str, visit_id: str, ex_key: str, viz=None) -> None:

    if viz is None:
        viz = Visualizer()

    try:
        # get data for dashboard
        dashboard_data = _prepare_dashboard_data(exercise)

        # if no metrics were found (e.g., extraction failed) -> skip plotting
        if not dashboard_data.get('metrics'):
            print(f"Skipping {ex_key} for {pid}...: No metrics found")
            return

        # render the dashboard
        viz.viz_render_dashboard(dashboard_data, pid, visit_id, ex_key, exercise.side_focus, skip_frames=2)
        print(f"Successfully queued data for {ex_key}")

    except Exception as e:
        print(f"Error processing {ex_key} for {pid}: {e}")


def check_kinematics():

    # initialize Visualizer object
    viz = Visualizer()

    # load participant objects
    participant_objs_path: str = os.path.join(project_path, 'data', '03_processed')
    participant_pickle_lst: list = [x for x in sorted(os.listdir(participant_objs_path))
                                    if x.startswith('P') and x.endswith('.pickle')]

    # run for each participant and each visit
    for pickle_file in participant_pickle_lst:
        p = Participant.load(os.path.join(participant_objs_path, pickle_file))

        # loop through every exercise this participant performed
        for ex_key, ex in p.exercises.items():

            # render single exercise
            _render_single_exercise(ex, p.pid, p.visit_id, ex_key, viz)


if __name__ == '__main__':
    check_kinematics()
