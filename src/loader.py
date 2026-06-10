# src/loader.py

# libraries
import os
import re
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

    # load the trim registry
    registry_path: str = os.path.join(project_path, 'data', '03_processed', 'trim_registry.csv')
    if os.path.exists(registry_path):
        reg_df = pd.read_csv(registry_path)
    else:
        reg_df = pd.DataFrame()

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
        trimmed_path = raw_path.replace('_raw.parquet', '_trimmed.parquet')
        ex.trimmed_landmark_data_path = trimmed_path
        clean_path = raw_path.replace('_raw.parquet', '_clean.parquet')
        ex.clean_landmark_data_path = clean_path

        # 6) data processing phase
        if not os.path.exists(clean_path):
            # load
            raw_df: pd.DataFrame = ex.load_dataframe('raw')

            # --- TEMPORARY FIX: Left/Right Label Swapper ---
            # Create a dictionary to map the old reversed names to the correct new names
            swap_map = {}
            # all lateralized joint prefixes (hands and pose)
            prefixes = r'(wrist|mcp|pip|dip|ip|cmc|ftip|shoulder|elbow|hip|knee|ankle|heel|foot|eye|clavicle)'

            for col in raw_df.columns:
                if 'left' in col:
                    swap_map[col] = col.replace('left', 'right')
                elif 'right' in col:
                    swap_map[col] = col.replace('right', 'left')
                elif re.search(f'^{prefixes}1', col):
                    # swaps '1' to '2' only if it immediately follows a known prefix at the start of the string
                    swap_map[col] = re.sub(f'^{prefixes}1', r'\g<1>2', col)
                elif re.search(f'^{prefixes}2', col):
                    # swaps '2' to '1' only if it immediately follows a known prefix at the start of the string
                    swap_map[col] = re.sub(f'^{prefixes}2', r'\g<1>1', col)

            # apply the renaming
            raw_df.rename(columns=swap_map, inplace=True)
            # -----------------------------------------------

            # apply trimming
            start_frame = 0
            end_frame = len(raw_df) - 1
            if not reg_df.empty:
                # find matching row in registry
                mask = ((reg_df['p_ID'] == p_id) & (reg_df['visit_ID'] == visit_id) &
                        (reg_df['ex_name'] == ex_name) & (reg_df['side_focus'] == ex_side))
                if mask.any():
                    start_frame = int(reg_df.loc[mask, 'start_frame'].values[0])
                    end_frame = int(reg_df.loc[mask, 'end_frame'].values[0])

            # slice and save trimmed
            trimmed_df = raw_df.iloc[start_frame:end_frame+1].reset_index(drop=True)
            ex.save_dataframe(trimmed_df, stage='trimmed')

            # filter and save clean
            clean_df, tracking_stats = tb.filter_landmark_dataframe(trimmed_df)
            ex.save_dataframe(clean_df, stage='clean')
            ex.tracking_stats = tracking_stats          # store dict for later aggregation

        # add exercise object to participant
        all_participants[session_key].add_exercise(ex)

    # dict for aggregated hand sizes
    global_hand_sizes: dict = {}

    # calculate the reference hand size for each participant, add it to each exercise, and save the participant objects
    for p in tqdm(all_participants.values(), desc='Calculating Participant Hand Sizes ...'):

        left_sizes: dict = {}
        right_sizes: dict = {}

        # add participant ID to global aggregation dict
        if p.pid not in global_hand_sizes:
            global_hand_sizes[p.pid] = {'Affected': [], 'Healthy': []}

        for ex_key, ex in p.exercises.items():
            left_size, right_size = tb.calc_anatomical_hand_sizes(ex)

            # store hand sizes in corresponding Exercise member variable
            ex.left_hand_size = left_size
            ex.right_hand_size = right_size

            if left_size > 0:
                left_sizes[ex_key] = left_size
                if p.affected_side == 'L':
                    global_hand_sizes[p.pid]['Affected'].append(left_size)
                else:
                    global_hand_sizes[p.pid]['Healthy'].append(left_size)
            if right_size > 0:
                right_sizes[ex_key] = right_size
                if p.affected_side == 'R':
                    global_hand_sizes[p.pid]['Affected'].append(right_size)
                else:
                    global_hand_sizes[p.pid]['Healthy'].append(right_size)

        # plausibility check
        STD_THRESH: float = 0.02

        if len(left_sizes) > 1:
            l_std = np.std(list(left_sizes.values()))
            if l_std > STD_THRESH:
                print(f'\nWarning: High left hand size variance (STD: {l_std:.4f} for {p.pid}.')
                for key, size in left_sizes.items():
                    print(f'\t - {key}: {size:.4f}')

        if len(right_sizes) > 1:
            r_std = np.std(list(right_sizes.values()))
            if r_std > STD_THRESH:
                print(f'\nWarning: High right hand size variance (STD: {r_std:.4f} for {p.pid}.')
                for key, size in right_sizes.items():
                    print(f'\t - {key}: {size:.4f}')

        # 4) Save the current participant object
        p.save(os.path.join(project_path,'data', '03_processed'))

    # calculate descriptive statistics for the hand size variance
    hand_size_stats = []
    for pid, sizes in global_hand_sizes.items():
        for side in ['Affected', 'Healthy']:
            arr = np.array(sizes[side])
            if len(arr) > 1:
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                cov_val = float((std_val / mean_val) * 100) if mean_val > 0 else 0.0
                hand_size_stats.append({'p_ID': pid,
                                        'Condition': side,
                                        'N_Trials': len(arr),
                                        'Mean_Size': mean_val,
                                        'STD': std_val,
                                        'CoV_Pct': cov_val,
                                        'Min_Size': float(np.min(arr)),
                                        'Max_Size': float(np.max(arr))})

    # generate participant master file
    print('Generating Participant overview CSV and tracking quality stats...')

    csv_rows: list = []
    tracking_log_rows = []

    # global aggregators
    glob_total_cells = 0
    glob_nan_cells = 0
    glob_nan_crit_cells = 0
    glob_nan_non_crit_cells = 0

    total_exercises = 0
    exercises_with_dropped = 0
    exercises_with_crit_dropped = 0

    for p in all_participants.values():
        session_key: str = f'{p.pid}_{p.visit_id}'

        for ex_key, ex in p.exercises.items():
            total_exercises += 1
            try:
                filename: str = os.path.basename(ex.raw_landmark_data_path)
                trial_code = filename.split('_')[4]
            except IndexError:
                trial_code = 'Unknown'

            # overview row
            csv_rows.append({'session_key': session_key,
                             'trial_code': trial_code,
                             'visit_id': ex.visit_id,
                             'exercise_id': ex.exercise_id,
                             'side_condition': ex.side_condition,
                             'side_focus': ex.side_focus,
                             'left_hand_size': round(ex.left_hand_size, 4),
                             'right_hand_size': round(ex.right_hand_size, 4)})

            # tracking log row
            stats = getattr(ex, 'tracking_stats', None)
            if stats:
                glob_total_cells += stats['total_cells']
                glob_nan_cells += stats['total_nans']
                glob_nan_crit_cells += stats['nan_crit_cells']
                glob_nan_non_crit_cells += stats['nan_non_crit_cells']

                crit_drops = stats['dropped_crit_lms']
                non_crit_drops = stats['dropped_non_crit_lms']

                if crit_drops or non_crit_drops:
                    exercises_with_dropped += 1
                if crit_drops:
                    exercises_with_crit_dropped += 1

                tracking_log_rows.append({
                    'session_key': session_key, 'trial_code': trial_code, 'ex_name': ex.exercise_id,
                    'frames': stats['frames'],
                    'missing_overall_pct': round((stats['total_nans'] / max(stats['total_cells'], 1)) * 100, 3),
                    'dropped_critical': ", ".join(crit_drops) if crit_drops else "None",
                    'dropped_non_critical': ", ".join(non_crit_drops) if non_crit_drops else "None"
                })

    # output
    res_dir: str = os.path.join(project_path, 'data', '05_results', '01_ul_tracking')
    os.makedirs(res_dir, exist_ok=True)

    if csv_rows:
        df_overview: pd.DataFrame = pd.DataFrame(csv_rows)
        df_overview.sort_values(by=['session_key', 'trial_code'], inplace=True) # sort by participant -> trial code
        # save to csv
        df_overview.to_csv(os.path.join(res_dir, 'participant_overview.csv'), index=False)
        print(f'Successfully saved overview for {len(df_overview)} trials.')

    if tracking_log_rows:
        df_tracking_log: pd.DataFrame = pd.DataFrame(tracking_log_rows)
        df_tracking_log.sort_values(by=['session_key', 'trial_code'], inplace=True)
        df_tracking_log.to_csv(os.path.join(res_dir, 'tracking_quality_log.csv'), index=False)

    if hand_size_stats:
        df_hand_stats = pd.DataFrame(hand_size_stats)
        df_hand_stats.sort_values(by=['p_ID', 'Condition'], inplace=True)
        df_hand_stats.to_csv(os.path.join(res_dir, 'hand_size_variance_log.csv'), index=False)

    # log results to text file
    if glob_total_cells > 0:
        overall_missing_pct = (glob_nan_cells / glob_total_cells) * 100
        pct_missing_is_non_crit = (glob_nan_non_crit_cells / glob_nan_cells) * 100 if glob_nan_cells > 0 else 0.0
        pct_missing_is_crit = (glob_nan_crit_cells / glob_nan_cells) * 100 if glob_nan_cells > 0 else 0.0

        max_gap = config['preprocessing']['max_gap_threshold']

        # compile the summary into a single string
        summary_text = (
            f"{'=' * 80}\n"
            f" METHODOLOGICAL RESULTS SUMMARY (Tracking Performance)\n"
            f"{'=' * 80}\n\n"
            f"Overall, {overall_missing_pct:.3f}% of tracking data was missing.\n"
            f"Of that missing data, {pct_missing_is_non_crit:.1f}% concerned landmarks NOT used in downstream calculations, "
            f"while only {pct_missing_is_crit:.1f}% affected critical analysis landmarks.\n\n"
            f"There were {exercises_with_dropped} out of {total_exercises} exercise trials where one or multiple "
            f"landmarks exceeded the maximum continuous occlusion threshold of {max_gap} frames and were successfully dropped.\n"
        )

        if exercises_with_crit_dropped == 0:
            summary_text += "NONE of the dropped landmarks were used for anatomical normalization or feature calculations.\n"
        else:
            summary_text += (
                f"Only {exercises_with_crit_dropped} of those trials resulted in dropping a critical landmark "
                f"required for downstream calculation.\n")

        if hand_size_stats:
            df_aff = df_hand_stats[df_hand_stats['Condition'] == 'Affected']
            df_hel = df_hand_stats[df_hand_stats['Condition'] == 'Healthy']

            summary_text += (
                f"\n{'=' * 80}\n"
                f" Hand Size Stability (Intra-Participant Hand Size Variation)\n"
                f"{'=' * 80}\n\n"
            )

            if not df_aff.empty:
                aff_max = df_aff.loc[df_aff['CoV_Pct'].idxmax()]
                aff_min = df_aff.loc[df_aff['CoV_Pct'].idxmin()]
                aff_avg_cov = df_aff['CoV_Pct'].mean()

                summary_text += (
                    f"--- PARETIC HANDS ---\n"
                    f"Overall Average CoV: {aff_avg_cov:.2f}%\n"
                    f"Largest Variation: CoV = {aff_max['CoV_Pct']:.2f}% (Participant {aff_max['p_ID']})\n"
                    f"Smallest Variation: CoV = {aff_min['CoV_Pct']:.2f}% (Participant {aff_min['p_ID']})\n\n"
                )

            if not df_hel.empty:
                hel_max = df_hel.loc[df_hel['CoV_Pct'].idxmax()]
                hel_min = df_hel.loc[df_hel['CoV_Pct'].idxmin()]
                hel_avg_cov = df_hel['CoV_Pct'].mean()

                summary_text += (
                    f"--- NON-PARETIC HANDS ---\n"
                    f"Overall Average CoV: {hel_avg_cov:.2f}%\n"
                    f"Largest Variation: CoV = {hel_max['CoV_Pct']:.2f}% (Participant {hel_max['p_ID']})\n"
                    f"Smallest Variation: CoV = {hel_min['CoV_Pct']:.2f}% (Participant {hel_min['p_ID']})\n\n"
                )

        summary_text += (f"{'=' * 80}\n"
                         f"Detailed tracking statistics exported to: tracking_quality_log.csv\n")

        # 2. Print to the terminal
        print(f"\n{summary_text}")

        # 3. Save to a .txt file alongside the CSVs
        txt_out_path = os.path.join(res_dir, 'methodological_summary.txt')
        with open(txt_out_path, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        print(f"Methodological summary successfully saved to: {txt_out_path}")
