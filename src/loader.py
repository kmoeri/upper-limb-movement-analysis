# src/loader.py

# libraries
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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

    print('Smoothing and registering participant landmarks ...')
    # loop through all csv files and add all exercises to the corresponding Participant
    for raw_path in tqdm(parquet_file_paths, desc='Filtering Landmark Coordinates'):

        # 1) parse file name
        p_id, visit_id, affected_side, ex_name, side_condition, ex_side, cam_id = parse_filename(raw_path)

        # 2) create unique keys
        session_key = f'{p_id}_{visit_id}'

        # 3) load or create participant object
        if session_key not in all_participants:
            all_participants[session_key] = Participant(p_id, visit_id, affected_side)

        # 4) create Exercise object
        ex: Exercise = Exercise(visit_id=visit_id, exercise_id=ex_name, side_condition=side_condition,
                                side_focus=ex_side, cam_id=cam_id)

        # 5) add path to file pointer
        ex.raw_landmark_data_path = raw_path
        clean_path = raw_path.replace('_raw.parquet', '_clean.parquet')
        ex.clean_landmark_data_path = clean_path

        if not os.path.exists(clean_path):
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
    print('Calculating anatomical reference hand sizes ...')
    for p in tqdm(all_participants.values(), desc='Calculating Participant Hand Size.'):

        left_sizes: dict = {}
        right_sizes: dict = {}

        for ex_key, ex in p.exercises.items():
            left_size, right_size = tb.calc_anatomical_hand_sizes(ex)

            # store hand sizes in corresponding Exercise member variable
            ex.left_hand_size = left_size
            ex.right_hand_size = right_size

            if left_size > 0:
                left_sizes[ex_key] = left_size
            if right_size > 0:
                right_sizes[ex_key] = right_size

        # plausibility check
        STD_THRESH: float = 0.02

        if len(left_sizes) > 1:
            l_std = np.std(list(left_sizes.values()))
            if l_std > STD_THRESH:
                print(f'\n[!] Warning: High left hand size variance (STD: {l_std:.4f} for {p.pid}.')
                for key, size in left_sizes.items():
                    print(f'\t - {key}: {size:.4f}')

        if len(right_sizes) > 1:
            r_std = np.std(list(right_sizes.values()))
            if r_std > STD_THRESH:
                print(f'\n[!] Warning: High right hand size variance (STD: {r_std:.4f} for {p.pid}.')
                for key, size in right_sizes.items():
                    print(f'\t - {key}: {size:.4f}')

        # 4) Save the current participant object
        p.save(os.path.join(project_path,'data', '03_processed'))

    # generate participant master file
    print('Generating Participant overview CSV ...')
    csv_rows: list = []
    for p in all_participants.values():
        session_key: str = f'{p.pid}_{p.visit_id}'

        for ex_key, ex in p.exercises.items():
            try:
                filename: str = os.path.basename(ex.raw_landmark_data_path)
                trial_code = filename.split('_')[4]
            except IndexError:
                trial_code = 'Unknown'

            csv_rows.append({'session_key': session_key,
                             'trial_code': trial_code,
                             'visit_id': ex.visit_id,
                             'exercise_id': ex.exercise_id,
                             'side_condition': ex.side_condition,
                             'side_focus': ex.side_focus,
                             'left_hand_size': round(ex.left_hand_size, 4),
                             'right_hand_size': round(ex.right_hand_size, 4)})

    if csv_rows:
        df_overview: pd.DataFrame = pd.DataFrame(csv_rows)

        # sort by participant -> trial code
        df_overview.sort_values(by=['session_key', 'trial_code'], inplace=True)

        out_file: str = os.path.join(project_path, 'data', '05_results', '01_ul_tracking', 'participant_overview.csv')
        df_overview.to_csv(out_file, index=False)
        print(f'Successfully saved overview for {len(df_overview)} trials.')
