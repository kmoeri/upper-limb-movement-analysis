#!/usr/bin/python3
# scripts/02_load_participants.py

# libraries
import os

# modules
from src.config import config, project_path
from src.loader import load_participants


def run_participant_loader():

    # load source data
    project_name = config['batch_tracking']['project_name']
    tracked_data_src_path: str = os.path.join(project_path, 'data', '02_mediapipe_raw')
    tracked_data_dirs: list[str] = [os.path.join(tracked_data_src_path, x)
                                    for x in sorted(os.listdir(tracked_data_src_path)) if x.startswith('P')]

    # list to hold all paths of all participants for each exercise
    tracked_data_path_lst: list[str] = []

    for tracked_data_dir in tracked_data_dirs:
        [tracked_data_path_lst.append(os.path.join(tracked_data_dir, x)) for x in sorted(os.listdir(tracked_data_dir))
                                      if x.startswith(project_name) and x.endswith('.csv')]

    # load Participant objects
    load_participants(tracked_data_path_lst)

    print('Loading finished')


if __name__ == "__main__":
    run_participant_loader()
