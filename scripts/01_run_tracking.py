#!/usr/bin/python3
# scripts/01_run_tracking.py

# libraries
import os
import pandas as pd
import concurrent.futures

# modules
from src.config import config, project_path
from src.tracking import combined_landmark_extractor
from src.loader import load_video_files


def save_tracked_landmarks_to_csv(video_name: str, out_dir: str,
                                  pose_dict: dict, hands_dict: dict,
                                  pose_name_lst: list, hand_name_lst: list) -> None:
    """
    Saves tracked landmark coordinates of hands and pose to a single csv file.
    Lower limb pose landmarks (idc 25-32) are explicitly removed before saving.

    Args:
        video_name (str): Base name of the video file to be saved.
        out_dir (str): Directory to save the csv file.
        pose_dict (dict): Dictionary of pose landmarks and their corresponding 3D coordinates.
        hands_dict (dict): Dictionary of hand landmarks and their corresponding 3D coordinates.
        pose_name_lst (list): List of body part names corresponding to the pose landmarks.
        hand_name_lst (list): List of body part names corresponding to the hand landmarks.

    Returns:
        None
    """

    # flatten marker data
    flattened_data: dict = {}

    # hands and pose landmarks should have the same number of frames as they were extracted simultaneously
    for frame in pose_dict.keys():

        # get the upper limb pose landmarks
        pose_lst = pose_dict[frame][:25]

        # get the hand landmarks
        hand_lst = hands_dict[frame]

        # flatten the lists of tuples into a single list of coordinates for the current frame
        flat_pose = [coord for landmark in pose_lst for coord in landmark]
        flat_hand = [coord for landmark in hand_lst for coord in landmark]

        # append the flat lists
        flattened_data[frame] = flat_pose + flat_hand

    # get landmark names of hands and pose
    combined_name_lst: list[str] = pose_name_lst[:25] + hand_name_lst

    # creat column names
    col_names: list[str] = []
    for bodypart in combined_name_lst:
        col_names.extend([f'{bodypart}_x', f'{bodypart}_y', f'{bodypart}_z'])

    marker_df = pd.DataFrame.from_dict(flattened_data, orient='index', columns=col_names)
    marker_df.index.name = 'frame'

    # Save the DataFrame to a CSV file
    marker_df.to_csv(os.path.join(out_dir, video_name))


def process_single_video(video_task) -> str:
    """
    Worker function to process a single video for combined (hand and pose) landmark tracking.

    Args:
        video_task (tuple): tuple containing all paths and configuration lists for a single video.

    Returns:
        str: Message for finished tracking including the video base name.
    """

    # get info from video task
    vid_path, out_path, out_path_lst, model_paths, pose_name_lst, hand_name_lst = video_task

    vid_name_base: str = (os.path.basename(vid_path)).split('.')[0]
    vid_name_full: str = f'{vid_name_base}_tracked-raw.csv'

    # process this video if csv is missing
    if vid_name_full not in out_path_lst:

        # apply the combined tracking model
        pose_data_dict, hands_data_dict = combined_landmark_extractor(
            video_path=vid_path,
            res_path=out_path,
            model_paths=model_paths,
            n_hands=2,
            min_hand_detect_conf=0.5,
            min_pose_detect_conf=0.5,
            min_track_conf=0.8,
            normalize=False,
            visualize=False,
            save_video=True
        )

        # save marker data to csv for both outputs
        save_tracked_landmarks_to_csv(vid_name_full, out_path,
                                      pose_data_dict, hands_data_dict,
                                      pose_name_lst, hand_name_lst)

        return f'Finished combined tracking for {os.path.basename(vid_path)}.'

    return f'Skipped {os.path.basename(vid_path)} (already processed).'


def run_batch_tracking():

    # load hand and pose landmark names
    hand_name_lst: list = config['body_parts']['hands_landmark_lst']
    pose_name_lst: list = config['body_parts']['pose_landmark_lst']

    # load hand and pose tracking model paths
    hand_model_path: str = os.path.join(project_path,'mp_models', 'hand_landmarker.task')
    pose_model_path: str = os.path.join(project_path,'mp_models', 'pose_landmarker_heavy.task')

    # source and destination path
    tracking_src_path: str = os.path.join(project_path, 'data', '01_videos-raw')
    tracking_out_path: str = os.path.join(project_path, 'data', '02_mediapipe-raw')

    # load all participant folders (e.g., P001, P002, etc.) in the tracking src path
    participant_fname_lst: list = [x for x in sorted(os.listdir(tracking_src_path)) if x.startswith('P')]

    # list with all participant folders
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

        # create a list of already existing result files (including both pose and hands csv)
        res_path_lst: list[str] = os.listdir(out_fpath) if os.path.exists(out_fpath) else []

        print(f'Queueing videos of {participant_fname}...')

        # create a single task for each video containing both models and name lists
        for video_path in video_path_lst:
            all_tasks.append((
                video_path,
                out_fpath,
                res_path_lst,
                (pose_model_path, hand_model_path),
                pose_name_lst,
                hand_name_lst
            ))

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
