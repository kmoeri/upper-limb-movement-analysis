# Libraries
import os
import numpy as np
import pandas as pd

# look-up-table to map file names with exercise names
EXERCISE_LUT = {
    # WT-01 & WT-02 pair -> Index Finger Tapping on Thenar
    "WT-01": "Ex_FingerTapping",
    "WT-02": "Ex_FingerTapping",

    # WT-03 & WT-04 pair -> Finger Alternation Tapping
    "WT-03": "Ex_FingerAlternation",
    "WT-04": "Ex_FingerAlternation",

    # WT-05 & WT-06 pair -> Hand Opening and Closing
    "WT-05": "Ex_HandOpening",
    "WT-06": "Ex_HandOpening",

    # WT-07 & WT-08 pair -> Hand Pronation/Supination
    "WT-07": "Ex_ProSup",
    "WT-08": "Ex_ProSup",

    # WT-09 & WT-10 pair -> Finger Tapping on Table
    "WT-09": "Ex_TableTapping",
    "WT-10": "Ex_TableTapping",
}


def import_video_files(video_path: str) -> list[str]:
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


def parse_filename(video_fpath: str, affected_sides_lst: list) -> tuple[str, str, str, str, str]:
    """
    Parses the base video file name into exercise information elements:
    - participant ID
    - visit ID
    - exercise ID
    - side condition (Healthy or Affected)
    - exercise side (R or L)

    Args:
        video_fpath (str): video file path.
        affected_sides_lst (list): List of lists. Each list contains the participant ID and affected side ('R' or 'L').

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
    side_condition: str = 'Healthy' if ex_num % 2 == 0 else 'Affected'

    # check which side ('R' or 'L') corresponds to the current 'side_condition'
    affected_side: list = [x for x in affected_sides_lst if x[0] == p_id][0]

    if len(affected_side) == 0:
        raise ValueError(f'Participant {p_id} was not found.')

    if side_condition == 'Affected':
        ex_side: str = affected_side[1]
    else:
        ex_side: str = 'L' if affected_side == 'R' else 'R'

    return p_id, visit_id, ex_name, side_condition, ex_side


def save_hands_to_csv(video_name: str, res_dir: str, marker_dict: dict, body_part_name_lst: list,
                      multi_header: bool = False) -> None:
    """
    Saves motion data of the tracked hand landmarks to a csv file.

    Args:
        video_name (str): Absolute path of the video file.
        res_dir (str): Directory to save the csv file.
        marker_dict (dict): Dictionary of landmarks and their corresponding 3D coordinates.
        body_part_name_lst (list): List of body part names corresponding to the landmarks.
        multi_header (bool, optional): Whether to include multiple headers. Defaults to False.

    Returns:
        None
    """

    # flatten marker data returned from mediapipe's landmark detection
    flattened_data: dict = {}
    for frame, landmarks in marker_dict.items():
        flattened_data[frame] = [coord for landmark in landmarks for coord in landmark]

    if multi_header:
        # Create the multi level header for the columns
        first_level_names: list = [name for name in body_part_name_lst for _ in range(3)]
        second_level_names: list[str] = ['x', 'y', 'z'] * len(body_part_name_lst)
        multi_index: pd.MultiIndex = pd.MultiIndex.from_arrays([first_level_names, second_level_names],
                                                               names=['bodypart', 'axis'])

        # Create the DataFrame with the flattened data and the multi level header
        marker_df: pd.DataFrame = pd.DataFrame.from_dict(flattened_data, orient='index', columns=multi_index)

    else:

        col_names: list[str] = []
        for bodypart in body_part_name_lst:
            col_names.append(f'{bodypart}_x')
            col_names.append(f'{bodypart}_y')
            col_names.append(f'{bodypart}_z')

        marker_df = pd.DataFrame.from_dict(flattened_data, orient='index', columns=col_names)

    marker_df.index.name = 'frame'

    # Save the DataFrame to a CSV file
    marker_df.to_csv(os.path.join(res_dir, video_name))


def save_pose_to_csv(video_name: str, res_dir: str, marker_dict: dict, body_part_name_lst: list) -> None:
    """
    Saves motion data of the tracked pose landmarks to a csv file.

    Args:
        video_name (str): Absolute path of the video file.
        res_dir (str): Directory to save the csv file.
        marker_dict (dict): Dictionary of landmarks and their corresponding 3D coordinates.
        body_part_name_lst (list): List of body part names corresponding to the landmarks.

    Returns:
        None
    """

    # Extract only shoulders, elbows, and wrists (idc: 11-16) and add the hips (idc: 23,24)
    pose_marker_name_lst = body_part_name_lst[11:17] + body_part_name_lst[23:25]

    # flatten pose marker data
    flattened_ul_data: dict = {}
    for frame, landmarks in marker_dict.items():
        upper_body_landmarks = landmarks[11:17] + landmarks[23:25]
        flattened_ul_data[frame] = [coord for landmark in upper_body_landmarks for coord in landmark]

    col_names: list[str] = []
    for bodypart in pose_marker_name_lst:
        col_names.append(f'{bodypart}_x')
        col_names.append(f'{bodypart}_y')
        col_names.append(f'{bodypart}_z')

    marker_df = pd.DataFrame.from_dict(flattened_ul_data, orient='index', columns=col_names)
    marker_df.index.name = 'frame'

    # Save the DataFrame to a CSV file
    marker_df.to_csv(os.path.join(res_dir, video_name))


def infer_focus_side(df: pd.DataFrame, model_type: str = 'Hand') -> str | None:
    """
    Infers the focus side ('Left' or 'Right') by checking column names
    against known starting landmarks.

    Args:
        df (pd.DataFrame): DataFrame (either hand_df or pose_df).
        model_type (str): 'Hand' or 'Pose' to check the correct set of labels.

    Returns:
        str | None: 'Left', 'Right', or None if side cannot be determined.
    """

    # check hand model labels ('wrist1_x' or 'wrist2_x')
    if model_type.capitalize() == 'Hand':
        if 'wrist1_x' in df.columns:
            return 'Left'
        elif 'wrist2_x' in df.columns:
            return 'Right'

    # check pose model labels ('wrist_left_x' or 'wrist_right_x')
    elif model_type.capitalize() == 'Pose':
        if 'wrist_left_x' in df.columns:
            return 'Left'
        elif 'wrist_right_x' in df.columns:
            return 'Right'

    return None


def group_motion_files_by_exercise(file_path_lst: list[str]) -> dict:
    """
    Groups file paths based on a two-digit ExerciseNumber embedded in the basename of the motion data file.

    File name structure:
    ProjectName_ParticipantID_CameraType_VisitNumber_Assessment-ExerciseNumber_CameraAngle_ModelType_Status.csv

    Args:
        file_path_lst (list[str]): A list of absolute file paths.

    Returns:
        Dict[str, list[str]]: A dictionary where keys are 'ExerciseNumber' (e.g., '01', '02') and values are lists
        of corresponding file paths.
    """

    exercise_dict: dict[str, list[str]] = {}

    for file_path in file_path_lst:

        file_basename = os.path.basename(file_path)

        try:
            # get exercise number from file name
            exercise_name: str = file_basename.split('_')[4]
            exercise_num: str = exercise_name.split('-')[1]

            # check for valid exercise number
            if exercise_num.isdigit() and len(exercise_num) == 2:
                exercise_dict[exercise_num].append(file_path)
            else:
                print(f'Warning: File skipped due to invalid exercise number: {file_basename}.')

        except IndexError:
            print(f'Warning: File skipped due to invalid exercise number: {file_basename}.')
            continue

    # returns a dictionary sorted by its keys
    return dict(sorted(exercise_dict.items(), key=lambda item: item[0]))


def group_motion_files_by_participants(file_path_lst: list[str]) -> dict:
    """
    Groups file paths based on the participant ID embedded in the basename of the motion data file.

    File name structure:
    ProjectName_ParticipantID_CameraType_VisitNumber_Assessment-ExerciseNumber_CameraAngle_ModelType_Status.csv

    Args:
        file_path_lst (list[str]): A list of absolute file paths.

    Returns:
        Dict[str, list[str]]: A dictionary where keys are 'ParticipantID' (e.g., 'P001', 'P002') and values are lists
        of corresponding file paths.
    """

    participant_dict: dict[str, list[str]] = {}

    for file_path in file_path_lst:

        file_basename = os.path.basename(file_path)

        try:
            # get participant id from file name
            participant_id: str = file_basename.split('_')[1]

            # check for valid participant id
            if participant_id.startswith('P') and participant_id[1:].isdigit():
                participant_dict[participant_id].append(file_path)
            else:
                print(f'Warning: File skipped due to invalid participant ID: {file_basename}.')

        except IndexError:
            print(f'Warning: File skipped due to invalid participant ID: {file_basename}.')
            continue

    # returns a dictionary sorted by its keys
    return dict(sorted(participant_dict.items(), key=lambda item: item[0]))


def get_descriptive_stats(data: np.ndarray, prefix: str = ''):
    """
    Calculates descriptive statistics - normal version (mean, standard deviation, min, max).

    Args:
        data (np.ndarray): A numpy array holding values of an extracted parameter (e.g., amplitudes)
        prefix (str, optional): The prefix used to name the metric. Defaults to ''.

    Returns:
        dict: Dictionary with descriptive statistics.
    """

    if len(data) == 0:
        return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0, f'{prefix}_min': 0.0, f'{prefix}_max': 0.0}
    return {
        f'{prefix}_mean': round(float(np.mean(data)), 3),
        f'{prefix}_std': round(float(np.std(data)), 3),
        f'{prefix}_min': round(float(np.min(data)), 3),
        f'{prefix}_max': round(float(np.max(data)), 3)
    }


def get_descriptive_stats_short(data: np.ndarray, prefix: str = ''):
    """
    Calculates descriptive statistics - short version (mean and standard deviation).

    Args:
        data (np.ndarray): A numpy array holding values of an extracted parameter (e.g., amplitudes)
        prefix (str, optional): The prefix used to name the metric. Defaults to ''.

    Returns:
        dict: Dictionary with descriptive statistics.
    """

    if len(data) == 0:
        return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0}
    return {
        f'{prefix}_mean': round(float(np.mean(data)), 3),
        f'{prefix}_std': round(float(np.std(data)), 3),
    }


def save_extracted_data_to_csv(feature_list_of_dicts: list[dict], conf_dict: dict,
                               out_dir: str, overwrite: bool = False) -> None:
    """
    Saves extracted movement parameters by updating a csv main file with features from a specific trial run.
    Each time this function is called, it adds new information to the csv main file. Content that already exists
    can be overwritten by setting the 'overwrite' parameter to 'True'.

    Args:
        feature_list_of_dicts: List of dicts containing features from a specific trial.
        conf_dict: Dictionary containing configuration information.
        out_dir: Directory where 'extracted_movement_features.csv' is stored.
        overwrite: If 'True', overwrites existing data for the matching IDs. If 'False', only updates empty rows

    Returns:
        None
    """

    # csv main file
    out_path = os.path.join(out_dir, 'extracted_movement_features.csv')

    n_participants: int = conf_dict['participant_info']['n_participants']   # 20
    n_trials: int = conf_dict['participant_info']['n_trials']               # 10
    age: list[list] = conf_dict['participant_info']['age']

    info_col_names: list[str] = ['p_ID', 't_ID', 'f_exists', 'cam_ID', 'f_path', 'side']
    params_col_names: list[str] = ['repetition_freq', 'num_repetitions',
                                   'period_mean', 'period_std', 'period_min', 'period_max',
                                   'amplitude_mean', 'amplitude_std', 'amplitude_min', 'amplitude_max',
                                   'velocity_pos_mean', 'velocity_pos_std', 'velocity_neg_mean', 'velocity_neg_std']

    all_col_names: list[str] = info_col_names + params_col_names

    # load csv main file - load with str dtype to maintain leading zeros
    if os.path.exists(out_path):
        main_df = pd.read_csv(out_path, dtype={'t_ID': str, 'p_ID': str})

    # create a new main dataframe
    else:
        # create rows for participant ID (participant * number_of_trials)
        unique_p_ID_lst: list[str] = ['P{:03d}'.format(i) for i in range(1, n_participants + 1)]
        final_p_ID_arr: np.ndarray = np.array([x for x in unique_p_ID_lst for _ in range(n_trials)])

        # create rows for trial ID (trials 01 - 10)
        final_t_ID_arr: np.ndarray = np.array(['{:02d}'.format(i) for i in range(1, n_trials + 1)] * n_participants)

        # create a template dataframe with all possible participant trial combinations
        main_df: pd.DataFrame = pd.DataFrame({'p_ID': final_p_ID_arr, 't_ID': final_t_ID_arr})

        # initialize all columns with np.nan
        for col in all_col_names:
            if col not in main_df.columns:
                main_df[col] = np.nan

        # flag handling overwrites
        main_df['f_exists'] = 'no'

    # This allows them to hold both np.nan AND strings without generating a warning.
    for col in info_col_names:
        if col in main_df.columns:
            main_df[col] = main_df[col].astype(object)

    # main_df is indexed by IDs for updating functionality
    main_df.set_index(['p_ID', 't_ID'], inplace=True)

    # handle empty list
    if not feature_list_of_dicts:
        print("No features provided in input list. Skipping update.")
        return

    # load passed list of dicts as dataframe
    new_param_df = pd.DataFrame(feature_list_of_dicts)

    # extract participant IDs and trial IDs from filenames
    new_param_df['fname'] = new_param_df['f_path'].apply(os.path.basename)

    # use regex to identify the respective pattern (p_ID: P and 3 digits)
    new_param_df['p_ID'] = new_param_df['fname'].str.extract(r'(P\d{3})')

    # use regex to identify the respective pattern (t_ID: WT and 2 digits following a hyphen)
    new_param_df['t_ID'] = new_param_df['fname'].str.extract(r'WT-(\d{2})')

    # use regex to identify the respective pattern (cam_ID: 'cam' followed by one capital letter)
    new_param_df['cam_ID'] = new_param_df['fname'].str.extract(r'(cam[A-Z])')

    # even trial ids indicate exercises performed with the healthy side; odd: affected side
    new_param_df['side'] = new_param_df['t_ID'].astype(int).apply(lambda x: 'healthy' if x % 2 == 0 else 'affected')

    # update file status
    new_param_df['f_exists'] = 'yes'

    # filter matching columns between the new parameter dataframe and the main dataframe
    cols_to_use = [col for col in all_col_names if col in new_param_df.columns]
    new_param_df = new_param_df[cols_to_use]

    # index the new dataframe to match the main dataframe
    new_param_df.set_index(['p_ID', 't_ID'], inplace=True)

    # incoming values replace existing data in csv main file
    if overwrite:
        main_df.update(new_param_df)

    # only update rows with missing data
    else:
        # filter main dataframe for empty rows
        rows_to_update = main_df.index[main_df['f_exists'] == 'no']

        # filter new params dataframe for empty rows
        safe_new_params = new_param_df.loc[new_param_df.index.intersection(rows_to_update)]

        # notify in case new parameters were skipped due to overwrite prevention
        if len(safe_new_params) < len(new_param_df):
            skipped = len(new_param_df) - len(safe_new_params)
            print(f"Skipped {skipped} rows because data already existed there.")

        # update the main dataframe with new, non-existing data
        main_df.update(safe_new_params)

    # reset index to make p_ID and t_ID normal columns again
    main_df.reset_index(inplace=True)

    # reorder columns
    main_df = main_df[all_col_names]

    # create the path (if not existing) and save the csv file (overwrites if existing)
    os.makedirs(out_dir, exist_ok=True)
    main_df.to_csv(out_path, index=False)
    print("csv main file successfully updated.")
