# src/loader.py

# libraries
import os
import numpy as np
import pandas as pd

# modules
from src.config import config
from src.core import Participant, Exercise

# look-up-table to map file names with exercise names
EXERCISE_LUT = {
    # WT-01 & WT-02 pair -> Index Finger Tapping on Thenar
    'WT-01': 'FingerTapping',
    'WT-02': 'FingerTapping',

    # WT-03 & WT-04 pair -> Finger Alternation Tapping
    'WT-03': 'FingerAlternation',
    'WT-04': 'FingerAlternation',

    # WT-05 & WT-06 pair -> Hand Opening and Closing
    'WT-05': 'HandOpening',
    'WT-06': 'HandOpening',

    # WT-07 & WT-08 pair -> Hand Pronation/Supination
    'WT-07': 'ProSup',
    'WT-08': 'ProSup',

    # WT-09 & WT-10 pair -> Finger Tapping on Table
    'WT-09': 'TableTapping',
    'WT-10': 'TableTapping',
}


def load_video_files(video_path: str) -> list[str]:
    """
    Imports video files from a given project folder.

    Args:
        video_path (str): Path to the project folder.

    Returns:
        list: List of video file paths.
    """

    if not os.path.isdir(video_path):
        raise ValueError(f'Not a valid project path: {video_path}')

    video_files: list[str] = [os.path.join(video_path, x) for x in os.listdir(video_path)
                              if x.endswith('.mp4') or x.endswith('.avi')]

    if not video_files:
        raise ValueError(f'No video files found in {video_path}')

    return sorted(video_files)


def parse_filename(video_fpath: str) -> tuple:
    """
    Parses the base video file name into exercise information elements:
    - participant ID
    - visit ID
    - affected side of participant (R or L)
    - exercise ID
    - exercise condition (Healthy or Affected)
    - exercise side (R or L)

    Args:
        video_fpath (str): video file path.

    Returns:
        tuple: tuple containing exercise information.
    """
    # Filename: Project_PID_CamType_VisitID_ExerciseID_CamID
    filename: str = os.path.basename(video_fpath)
    f_splits: list = filename.split('_')
    p_id: str = f_splits[1]
    visit_id: str = f_splits[3]
    ex_id: str = f_splits[4]

    # get the exercise name from the mapping
    ex_name: str = EXERCISE_LUT.get(ex_id, 'Unknown')

    # check whether the current side is 'Healthy' or 'Affected'
    ex_num: int = int(ex_id.split('-')[1])
    ex_condition: str = 'Healthy' if ex_num % 2 == 0 else 'Affected'

    # check which side ('R' or 'L') corresponds to the current 'side_condition'
    affected_sides_lst: list[list] = config['participant_info']['affected_side']
    affected_side: list = [x for x in affected_sides_lst if x[0] == p_id][0]

    if len(affected_side) == 0:
        raise ValueError(f'Participant {p_id} was not found.')

    if ex_condition == 'Affected':
        ex_side: str = affected_side[1]
    else:
        ex_side: str = 'L' if affected_side == 'R' else 'R'

    return p_id, visit_id, affected_side, ex_name, ex_condition, ex_side


def load_landmarks_to_dict(csv_file: str) -> dict:
    """"
    Loads landmark data stored in a csv file into a dictionary.
    Each dict element has a label (e.g., wrist) as key and a list of ndarray for each axis (x,y,z) as value.

    Args:
        csv_file (str): Absolute path of the csv file.

    Returns:
        dict: Dictionary of landmarks and their corresponding 3D coordinates.
    """

    # read csv data in a pandas DataFrame
    landmarks_df: pd.DataFrame = pd.read_csv(csv_file)
    landmarks_dict: dict = dict()

    # get a list of the base label names (without axis appendix)
    base_names: list[str] = [label[:-2] for label in landmarks_df.columns if label.endswith('_x')]

    for label in base_names:

        try:
            # get single axis arrays
            x_landmark_data: np.ndarray = landmarks_df[f'{label}_x'].values
            y_landmark_data: np.ndarray = landmarks_df[f'{label}_y'].values
            z_landmark_data: np.ndarray = landmarks_df[f'{label}_z'].values

            # store all axes in dict using the corresponding label
            landmarks_dict[label] = [x_landmark_data, y_landmark_data, z_landmark_data]

        except KeyError as e:
            print(f'Warning: Missing coordinate column for {label}: {e}')
            continue

    return landmarks_dict


def load_participants(csv_file_paths: list) -> None:
    """"
    Loads csv files with landmark coordinate of a movement exercise and passes the data to a Participant object.
    The created Participant object is stored as a pickle file for efficient handling of different exercises
    and participants.

    Args:
        csv_file_paths (list): List with absolute paths of csv files with movement data (raw normalized from MediaPipe).

    Returns:
        None
    """

    all_participants: dict = {}

    # loop through all csv files
    for csv_file_path in csv_file_paths:

        # 1) parse file name
        p_id, visit_id, affected_side, ex_name, side_condition, ex_side = parse_filename(csv_file_path)

        # 2) load or create participant object
        if p_id not in all_participants:
            p: Participant = Participant(p_id, visit_id, affected_side)
            all_participants[p_id] = p

        # 3) load csv data
        raw_landmarks: dict = load_landmarks_to_dict(csv_file_path)

        # 4) create exercise object
        ex: Exercise = Exercise(visit_id=visit_id, exercise_id=ex_name,
                                side_condition=side_condition, side_focus=ex_side,
                                raw_landmarks=raw_landmarks)

        # add exercise object to participant
        all_participants[p_id].add_exercise(ex)

    # 5) save participant objects
    for p in all_participants.values():
        p.save(f'data/03_processed/{p.pid}.pickle')

