#!/usr/bin/python3
# scripts/01_prepare_data.py

# libraries
import os
import pandas as pd
import tqdm as tqdm

# modules
from src.config import config, project_path
from src.loader import parse_filename


def run_data_preparation():
    print('\nInitializing workspace and building trim registry ...')

    project_name: str = config['project_cfg']['project_name']
    tracked_data_src_path: str = os.path.join(project_path, 'data', '02_tracking_data')
    registry_path: str = os.path.join(project_path, 'data', '03_processed', 'trim_registry.csv')

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)

    tracked_data_dirs: list[str] = [os.path.join(tracked_data_src_path, x)
                                    for x in sorted(os.listdir(tracked_data_src_path)) if x.startswith('P')]

    new_registry_rows: list = []
    for tracked_data_dir in tqdm.tqdm(tracked_data_dirs, desc='Scanning raw data'):
        for x in sorted(os.listdir(tracked_data_dir)):
            if not x.startswith(project_name) or not x.endswith('.parquet'):
                continue
            if x.endswith('_clean.parquet') or x.endswith('_trimmed.parquet'):
                continue

            file_path: str = os.path.join(tracked_data_dir, x)

            # ensure '_raw' suffix exists
            if not x.endswith('_raw.parquet'):
                new_name = x[:-8] + '_raw.parquet'
                new_filepath = os.path.join(tracked_data_dir, new_name)
                os.rename(file_path, new_filepath)
                file_path = new_filepath

            # parse metadata
            p_id, visit_id, affected_side, ex_name, side_condition, ex_side, cam_id = parse_filename(file_path)

            # load raw dataframe to get the max length
            try:
                df_raw = pd.read_parquet(file_path)
                max_frame = len(df_raw) - 1
            except Exception as e:
                print(f'Failed to read {file_path}: {e}')
                max_frame = 0

            new_registry_rows.append({
                'p_ID': p_id,
                'visit_ID': visit_id,
                'ex_name': ex_name,
                'side_focus': ex_side,
                'start_frame': 0,
                'end_frame': max_frame
            })

    new_df: pd.DataFrame = pd.DataFrame(new_registry_rows)

    # merge with existing registry
    if os.path.exists(registry_path):
        existing_df: pd.DataFrame = pd.read_csv(registry_path)
        combined_df = pd.concat([existing_df, new_df]).drop_duplicates(subset=['p_ID', 'visit_ID', 'ex_name', 'side_focus'], keep='first')

    else:
        combined_df: pd.DataFrame = new_df

    combined_df.sort_values(by=['p_ID', 'visit_ID', 'ex_name'], inplace=True)
    combined_df.to_csv(registry_path, index=False)

    print(f'\nDone. Trim Registry updated with {len(combined_df)} records.')


if __name__ == "__main__":
    run_data_preparation()
