# src/loader.py

# libraries
import os
import pandas as pd

# modules
from src.config import config, project_path
from src.core import Participant, Exercise
from src.utils import ToolBox

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
    'WT-08': 'ProSup'
}


def parse_filename(fpath: str) -> tuple:
    """
    Parses the base file name into exercise information elements:
    - participant ID
    - visit ID
    - affected side of participant (R or L)
    - exercise ID
    - exercise condition (Healthy or Affected)
    - exercise side (R or L)
    - camera ID (e.g., camZ)

    Args:
        fpath (str): video file path.

    Returns:
        tuple: tuple containing exercise information.
    """
    # Filename: Project_PID_CamType_VisitID_ExerciseID_CamID
    filename: str = os.path.basename(fpath)
    f_splits: list = filename.split('_')
    p_id: str = f_splits[1]
    visit_id: str = f_splits[3]
    ex_id: str = f_splits[4]
    cam_id: str = f_splits[5]

    # get the exercise name from the mapping
    ex_name: str = EXERCISE_LUT.get(ex_id, 'Unknown')

    # check whether the current side is 'Healthy' or 'Affected'
    ex_num: int = int(ex_id.split('-')[1])
    ex_condition: str = 'Healthy' if ex_num % 2 == 0 else 'Affected'

    # check which side ('R' or 'L') corresponds to the current 'side_condition'
    affected_sides_lst: list[list] = config['participant_info']['affected_side']
    affected_side: str = [x for x in affected_sides_lst if x[0] == p_id][0][1]

    if len(affected_side) == 0:
        raise ValueError(f'Participant {p_id} was not found.')

    if ex_condition == 'Affected':
        ex_side: str = affected_side
    else:
        ex_side: str = 'L' if affected_side == 'R' else 'R'

    return p_id, visit_id, affected_side, ex_name, ex_condition, ex_side, cam_id


def load_participants(parquet_file_paths: list) -> None:
    """
    Loads csv files with landmark coordinate of a movement exercise and passes the data to a Participant object.
    The created Participant object is stored as a pickle file for efficient handling of different exercises
    and participants.

    Args:
        parquet_file_paths (list): List with absolute paths of csv files with movement data (raw from MediaPipe).

    Returns:
        None
    """

    all_participants: dict = {}

    # create ToolBox object for utility function calling
    tb: ToolBox = ToolBox()

    # loop through all csv files and add all exercises to the corresponding Participant
    for parquet_file_path in parquet_file_paths:

        # 1) parse file name
        p_id, visit_id, affected_side, ex_name, side_condition, ex_side, cam_id = parse_filename(parquet_file_path)

        # 2) create unique keys
        session_key = f'{p_id}_{visit_id}'

        # 3) load or create participant object
        if session_key not in all_participants:
            all_participants[session_key] = Participant(p_id, visit_id, affected_side)

        # 4) create Exercise object
        ex: Exercise = Exercise(visit_id=visit_id, exercise_id=ex_name, side_condition=side_condition,
                                side_focus=ex_side, cam_id=cam_id)

        # 5) add path to file pointer
        ex.raw_landmark_data_path = parquet_file_path

        # 6) data processing phase
        # load
        raw_df: pd.DataFrame = ex.load_dataframe('raw')
        # filter
        clean_df: pd.DataFrame = tb.filter_landmark_dataframe(raw_df)
        # save
        ex.save_dataframe(clean_df, stage='clean')

        # add exercise object to participant
        all_participants[session_key].add_exercise(ex)

    # calculate the reference hand size for each participant, add it to each exercise, and save the participant objects
    for p in all_participants.values():

        # 1) calculate the global hand sizes (left and right) for the current Participant
        best_hand_ref_dict = tb.determine_best_hand_reference(p)    # Returns: {p.pid: {'Affected': X, 'Healthy': Y}}
        participant_refs = best_hand_ref_dict.get(p.pid, {})

        # 2) map the 'Affected' and 'Healthy' keys back to Left/Right
        if p.affected_side == 'L':
            left_size = participant_refs.get('Affected', 0.0)
            right_size = participant_refs.get('Healthy', 0.0)
        else:
            left_size = participant_refs.get('Healthy', 0.0)
            right_size = participant_refs.get('Affected', 0.0)

        # 3) add the calculated sizes to every exercise of the current Participant
        for ex_key, ex in p.exercises.items():
            ex.left_hand_size = left_size
            ex.right_hand_size = right_size

        # 4) Save the current participant object
        p.save(os.path.join(project_path,'data', '03_processed'))
