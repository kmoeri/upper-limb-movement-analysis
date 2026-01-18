#!/usr/bin/python3
# scripts/01_run_tracking.py

# libraries
import os
import pandas as pd
import concurrent.futures

# modules
from src.config import config, project_path
from src.tracking import hand_landmark_extractor, pose_landmark_extractor
from src.loader import load_video_files


def save_tracked_landmarks_to_csv(video_name: str, out_dir: str, marker_dict: dict, body_part_name_lst: list,
                                  tracking_selection: int) -> None:
    """
    Saves tracked landmark coordinates to a csv file.

    Args:
        video_name (str): Absolute path of the video file.
        out_dir (str): Directory to save the csv file.
        marker_dict (dict): Dictionary of landmarks and their corresponding 3D coordinates.
        body_part_name_lst (list): List of body part names corresponding to the landmarks.
        tracking_selection (int): Whether hands or pose were tracked. 1: hand tracking, 2: pose tracking.

    Returns:
        None
    """

    # flatten pose marker data
    flattened_data: dict = {}
    for frame, landmarks in marker_dict.items():
        landmark_lst = landmarks

        # if pose was tracked: keep upper limb landmarks only
        if tracking_selection == 2:
            landmark_lst = landmarks[11:17] + landmarks[23:25]

        flattened_data[frame] = [coord for landmark in landmark_lst for coord in landmark]

    # get landmark names
    landmark_name_lst: list[str] = body_part_name_lst

    # if pose was tracked: keep upper limb landmark names only
    if tracking_selection == 2:
        # Extract only shoulders, elbows, and wrists (idc: 11-16) and add the hips (idc: 23,24)
        landmark_name_lst = body_part_name_lst[11:17] + body_part_name_lst[23:25]

    # creat column names
    col_names: list[str] = []
    for bodypart in landmark_name_lst:
        col_names.append(f'{bodypart}_x')
        col_names.append(f'{bodypart}_y')
        col_names.append(f'{bodypart}_z')

    marker_df = pd.DataFrame.from_dict(flattened_data, orient='index', columns=col_names)
    marker_df.index.name = 'frame'

    # Save the DataFrame to a CSV file
    marker_df.to_csv(os.path.join(out_dir, video_name))


def process_single_video(video_task) -> str:
    """
    Worker function to process a single video for hand and/or pose tracking.

    Args:
        video_task (list): list of video path and video name.

    Returns:
        str: Message for finished tracking including the video base name.
    """

    # get info from video task
    vid_path, out_path, out_path_lst, model_path, name_lst, tracking_selection = video_task

    # A) hand tracking
    if tracking_selection == 1:
        vid_name_hands: str = (os.path.basename(vid_path)).split('.')[0] + '_hands.csv'

        if vid_name_hands not in out_path_lst:
            # apply hands tracking model
            hands_data_dict: dict = hand_landmark_extractor(vid_path, out_path, model_path, n_hands=2,
                                                            min_hand_detect_conf=0.5, min_hand_track_conf=0.8,
                                                            normalize=True, visualize=False, save_video=True)
            # save marker data to csv
            save_tracked_landmarks_to_csv(vid_name_hands, out_path, hands_data_dict, name_lst, tracking_selection)
            return f'Finished hand tracking for {os.path.basename(vid_path)}.'

    # B) pose tracking
    elif tracking_selection == 2:
        vid_name_pose: str = (os.path.basename(vid_path)).split('.')[0] + '_pose.csv'

        if vid_name_pose not in out_path_lst:
            # apply pose tracking model
            pose_data_dict: dict = pose_landmark_extractor(vid_path, out_path, model_path,
                                                           min_pose_detect_conf=0.5, min_pose_track_conf=0.8,
                                                           normalize=True, visualize=False, save_video=True)
            # save marker data to csv
            save_tracked_landmarks_to_csv(vid_name_pose, out_path, pose_data_dict, name_lst, tracking_selection)
            return f'Finished pose tracking for {os.path.basename(vid_path)}.'

    return f'Warning: Nothing was processed for {os.path.basename(vid_path)}.'


def run_batch_tracking():

    # load hand and pose landmark names
    hand_name_lst: list = config['body_parts']['hands_landmark_lst']
    pose_name_lst: list = config['body_parts']['pose_landmark_lst']

    # load hand and pose tracking model paths
    hand_model_path: str = 'mp_models/hand_landmarker.task'
    pose_model_path: str = 'mp_models/pose_landmarker_heavy.task'

    # source and destination path
    tracking_src_path: str = os.path.join(project_path, 'data', '01_raw_videos')
    tracking_out_path: str = os.path.join(project_path, 'data', '02_mediapipe_raw')

    # load all file names in source directory
    participant_fname_lst: list = [x for x in sorted(os.listdir(tracking_src_path)) if x.startswith('P')]

    # list with all videos of exercises from all participants
    all_tasks = []

    # process for each participant
    for participant_fname in participant_fname_lst:

        # update source and result paths
        src_fpath = os.path.join(tracking_src_path, participant_fname)
        out_fpath = os.path.join(tracking_out_path, participant_fname)

        # create a list of video file paths
        video_path_lst: list[str] = load_video_files(src_fpath)

        # create a directory for the resulting files (skip if already existing)
        os.makedirs(out_fpath, exist_ok=True)

        # create a list of already existing result files
        res_path_lst: list[str] = [x.split('.')[0] for x in os.listdir(out_fpath)
                                   if x.endswith('.csv')] if os.listdir(out_fpath) else []

        print(f'Queueing videos of {participant_fname}...')

        # create tasks for both (1) hand and (2) pose tracking
        for video_path in video_path_lst:
            # (1) task for hand tracking
            all_tasks.append((video_path, out_fpath, res_path_lst, 1, hand_model_path, hand_name_lst))
            # (2) task for pose Tracking
            all_tasks.append((video_path, out_fpath, res_path_lst, 2, pose_model_path, pose_name_lst))

    # define multiple workers for parallel execution
    MAX_WORKERS = config['batch_tracking']['max_workers']

    print(f'\nStarting parallel processing of {len(all_tasks)} tasks with {MAX_WORKERS} workers...')

    with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Distribute tasks to the workers
        results = executor.map(process_single_video, all_tasks)

        # Print results as they complete
        for result in results:
            if result:
                print(f'Done: {result}')

    print('\n --------- All tracking complete! ---------')


if __name__ == "__main__":
    run_batch_tracking()
