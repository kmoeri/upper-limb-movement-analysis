# Libraries
import os
import pandas as pd


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

