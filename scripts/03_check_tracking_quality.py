#!/usr/bin/python3
# scripts/03_check_tracking_quality.py

# libraries
import os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# modules
from src.config import config, project_path
from src.core import Participant, Exercise
from src.utils import ToolBox
from src.visualization import Visualizer


def determine_best_hand_reference(p: Participant) -> dict:
    """
    Calculates the median hand sizes of the left and the right hand sides across all exercises of the given participant
    for a specific visit (e.g., T1 or T2).

    Args:
        p (Participant): participant object of the class Participant.

    Returns:
        best_hand_ref_dict (dict): dictionary holding the largest median hand size for the affected and healthy side
    """

    # load lists from config file
    link_lst: list[list[str]] = config['body_parts']['hands_link_lst']
    l_hand_size_link_lst: list[str] = config['body_parts']['hand_size_left']
    r_hand_size_link_lst: list[str] = config['body_parts']['hand_size_right']

    def get_med_hand_size(exercise: Exercise, target_side: str, hand_specific_link_lst: list[list[str]],
                          hand_size_link_lst: list[str], occlusion_threshold: float = 0.85,
                          framerate: float = 90.0) -> float:
        """
        Calculates the median hand size using the anatomical hand size (sum of wrist to middle finger knuckle
        plus each middle finger segment) and the direct distance (wrist to middle fingertip) as a threshold measure.
        This function is intended for single hand processing.

        Args:
            exercise (Exercise): Exercise object with tracked landmarks and exercise information.
            target_side (str): Whether the active or passive hand is targeted ('L' or 'R').
            hand_specific_link_lst (list): List of connected landmark pairs, e.g., [['wrist1', 'cmc11'], ...].
            hand_size_link_lst (list): List of segments from wrist to middle fingertip, e.g., ['wrist1-mcp13', ...].
            occlusion_threshold: Minimum acceptable ratio of straight-line hand size to segmented hand size.
                                 (e.g., 0.7 means straight line must be at least 70% of the segmented length).
            framerate (float): Number of frames per second of the underlying data.

        Returns:
            median_hand_size (float): The median size of the given hand.
        """

        # length of each hand segment
        tb: ToolBox = ToolBox()
        exercise_dict: dict = exercise.clean_hand_landmarks

        df_cols = {}
        for label, axes in exercise_dict.items():
            if label == 'frame':                    # do not extract the column named 'frame' (usually first)
                continue
            df_cols[f'{label}_x'] = axes[0]
            df_cols[f'{label}_y'] = axes[1]
            df_cols[f'{label}_z'] = axes[2]

        exercise_df: pd.DataFrame = pd.DataFrame(df_cols)
        segment_len_df: pd.DataFrame = tb.calculate_3d_segment_lengths(exercise_df, hand_specific_link_lst)

        # calculate the hand size
        try:
            # get handedness: index 1 -> left hand, index 2 -> right hand. First column is 'wrist1' or 'wrist2'
            side: int = 1 if target_side == 'L' else 2

            # drop rows where any of the segments to calculate the hand size are missing (NaN)
            clean_segment_len: pd.DataFrame = segment_len_df.dropna(subset=hand_size_link_lst)

            # distance from wrist to fingertip by the sum of each segment inbetween
            hand_size_segment_len: pd.Series = clean_segment_len[hand_size_link_lst].sum(axis=1)

            # distance from wrist to fingertip by direct connection
            wrist_middle_finger_dist = tb.calculate_3d_segment_lengths(exercise_df.loc[clean_segment_len.index],
                                                                       [[f'wrist{side}', f'ftip{side}3']])
        except KeyError as e:
            print(
                f"Error: Could not calculate hand size. Missing one or more required middle finger segment columns: {e}")
            del tb
            return 0.0

        # direct wrist to middle fingertip distance
        hand_size_direct_len: pd.Series = wrist_middle_finger_dist[f'wrist{side}-ftip{side}3']

        # occlusion filter: hand size segment lengths of flexed hands, i.e., fists should not be included
        diff_dist_series: pd.Series = hand_size_segment_len - hand_size_direct_len
        if (diff_dist_series.values < 0).any():
            print('Warning: tracking detected abnormal values for hand size calculation. '
                  'Direct wrist-ftip distance was larger than sum of segments.')

        # occlusion metric
        occlusion_ratio = hand_size_direct_len / hand_size_segment_len
        filt_mask_lst = occlusion_ratio > occlusion_threshold

        # find at least a sum of frames equal or larger than 1s of 'flat' hand positions (e.g., 90 fps -> 90 frames)
        if sum(filt_mask_lst) < int(framerate):
            print(f'Warning: current exercise was skipped for median hand size calculation.'
                  f'Not enough flat hand positions found (found: {sum(filt_mask_lst)}).')
            del tb
            return 0.0

        # get the median hand size of the 'flat' hand positions
        median_hand_size: float = np.median(hand_size_segment_len[filt_mask_lst].tolist())

        del tb
        return median_hand_size

    # variables for results
    best_hand_ref_dict: dict = dict()
    med_results = {'L': [], 'R': []}

    # run for each exercise
    for ex_key in p.exercises.keys():

        # extract side of focus
        hand_of_focus: str = p.exercises[ex_key].side_focus

        # determine the links based on the side of focus ('L' or 'R')
        if hand_of_focus == 'L':
            passive_hand = 'R'
            active_hand_link_lst: list = link_lst[:len(link_lst)//2]
            passive_hand_link_lst: list = link_lst[len(link_lst)//2:]
            active_hand_size_link_lst: list = l_hand_size_link_lst
            passive_hand_size_link_lst: list = r_hand_size_link_lst

        else:
            passive_hand = 'L'
            active_hand_link_lst: list = link_lst[len(link_lst)//2:]
            passive_hand_link_lst: list = link_lst[:len(link_lst)//2]
            active_hand_size_link_lst: list = r_hand_size_link_lst
            passive_hand_size_link_lst: list = l_hand_size_link_lst

        # relax the threshold for more severe spasticity affecting the hand
        thresh_lst = config['tracking_consistency']['thresh_lst']
        active_med_hand_size: float = 0.0
        passive_med_hand_size: float = 0.0

        # calculate median hand sizes for different hand aperture threshold until return variable is > 0.0
        # active side
        for thresh in thresh_lst:
            active_med_hand_size = get_med_hand_size(p.exercises[ex_key],
                                                     target_side=hand_of_focus,
                                                     hand_specific_link_lst=active_hand_link_lst,
                                                     hand_size_link_lst=active_hand_size_link_lst,
                                                     occlusion_threshold=thresh,
                                                     framerate=config['camera_param']['fps'])

            if active_med_hand_size > 0.0:
                break

        # passive side
        for thresh in thresh_lst:
            passive_med_hand_size = get_med_hand_size(p.exercises[ex_key],
                                                      target_side=passive_hand,
                                                      hand_specific_link_lst=passive_hand_link_lst,
                                                      hand_size_link_lst=passive_hand_size_link_lst,
                                                      occlusion_threshold=thresh,
                                                      framerate=config['camera_param']['fps'])

            if passive_med_hand_size > 0.0:
                break

            if active_med_hand_size == 0.0 or passive_med_hand_size == 0.0:
                print(f'Warning: Exercise "{ex_key}" of participant {p.pid} for visit {p.visit_id} yielded no median for either hand size.')

        med_results[hand_of_focus].append(active_med_hand_size)
        med_results[passive_hand].append(passive_med_hand_size)

    # select the largest median hand size
    max_left_hand_size: float = max(med_results['L']) if med_results['L'] else 0.0
    max_right_hand_size: float = max(med_results['R']) if med_results['R'] else 0.0

    # add the selected hand size to the dictionary
    if p.affected_side == 'L':
        best_hand_ref_dict[p.pid] = {'Affected': max_left_hand_size, 'Healthy': max_right_hand_size}

    elif p.affected_side == 'R':
        best_hand_ref_dict[p.pid] = {'Affected': max_right_hand_size, 'Healthy': max_left_hand_size}

    return best_hand_ref_dict


def get_hand_segment_stability(p: Participant, ref_hand_size_dict: dict) -> pd.DataFrame:
    """
    Calculates segment lengths and normalizes them by anatomical hand size (sum of wrist to middle finger knuckle plus
    each middle finger segment). Calculates the Coefficient of Variance (CoV) for each body segment for both the
    active and the passive hands.

    Args:
        p (Participant): list of participant objects of the class Participant.
        ref_hand_size_dict (dict): dictionary of reference hand sizes for each participant.

    Returns:
        pd.dataframe: A dataframe containing the CoV of each body segment for both hands.
    """

    # configuration definitions
    finger_names_dict = {'1': 'Thumb', '2': 'Index', '3': 'Middle', '4': 'Ring', '5': 'Pinky'}
    link_lst: list[list[str]] = config['body_parts']['hands_link_lst']

    # tracking consistency test using all participants and exercises
    results_lst: list = []
    tb: ToolBox = ToolBox()

    # get the reference dict of the current participant
    participant_ref_dict: dict = ref_hand_size_dict.get(p.pid, {})

    # run for each exercise
    for ex_key in p.exercises.keys():

        # determine the active and passive side ('L' or 'R')
        active_side: str = p.exercises[ex_key].side_focus
        passive_side: str = 'R' if active_side == 'L' else 'L'

        # map the active and passive sides to the affected and healthy sides by participant info
        active_condition: str = 'Affected' if active_side == p.affected_side else 'Healthy'
        passive_condition: str = 'Healthy' if active_side == p.affected_side else 'Affected'

        # configuration for the roles of both hands for a given exercise
        hand_roles = {
            'Active': {
                'side': active_side,
                'condition': active_condition,
                'link_idx': link_lst[:len(link_lst)//2] if active_side == 'L' else link_lst[len(link_lst)//2:]
            },
            'Passive': {
                'side_lr': passive_side,
                'condition': passive_condition,
                'link_idx': link_lst[:len(link_lst)//2] if passive_side == 'L' else link_lst[len(link_lst)//2:]
            }
        }

        # load the exercise values in a dataframe
        exercise_dict = p.exercises[ex_key].clean_hand_landmarks
        df_cols = {}

        for label, axes in exercise_dict.items():
            if label == 'frame':
                continue
            df_cols[f'{label}_x'] = axes[0]
            df_cols[f'{label}_y'] = axes[1]
            df_cols[f'{label}_z'] = axes[2]

        exercise_df: pd.DataFrame = pd.DataFrame(df_cols)

        for role, role_info in hand_roles.items():
            curr_link_lst = role_info['link_idx']
            condition = role_info['condition']

            # get the median hand size using the correct keys ('Affected' or 'Healthy')
            med_hand_size: float = participant_ref_dict.get(condition, 0.0)

            # handle missing median hand size

            if med_hand_size == 0.0:
                opposite_condition = 'Healthy' if condition == 'Affected' else 'Affected'
                med_hand_size = participant_ref_dict.get(opposite_condition, 0.0)

                if med_hand_size > 0.0:
                    print(f'Warning: Participant {p.pid} used the "{opposite_condition}" hand size to normalize '
                          f'the "{condition}" ({role}) segments in "{ex_key}".')
                else:
                    print(f'Error: No median hand size for participant {p.pid}. Skipping {role} hand in {ex_key}.')
                    continue

            # calculate the length of each hand segment
            segment_len_df: pd.DataFrame = tb.calculate_3d_segment_lengths(exercise_df, curr_link_lst)

            # anatomical normalization - only needed when comparing hand sizes between participants, otherwise no effect.
            norm_segment_df: pd.DataFrame = segment_len_df.div(med_hand_size, axis=0)

            # calculation of segment CoV (%)
            body_segments_cov_series: pd.Series = ((norm_segment_df.std() / norm_segment_df.mean()) * 100)

            for segment_name, cov_value in body_segments_cov_series.items():
                finger_name = finger_names_dict.get(str(segment_name)[-1], 'Unknown')

                results_lst.append({
                    'Participant': p.pid,                   # e.g, P001
                    'Trial': ex_key,                        # e.g., 'FingerTapping_Affected'
                    'Active_Side': active_side,             # 'L' or 'R'
                    'Passive_Side': passive_side,           # 'L' or 'R'
                    'Hand_Role': role,                      # 'Active' or 'Passive'
                    'Hand_Condition': condition,            # 'Affected' or 'Healthy'
                    'Finger': finger_name,                  # 'Thumb', 'Index', 'Middle', 'Ring', 'Pinky'
                    'Segment': segment_name,                # e.g., 'wrist1-mcp13'
                    'CoV': round(cov_value, 3)
                })

    return pd.DataFrame(results_lst)


def get_lmm_consistency_statistics(hand_df: pd.DataFrame, results_dir: str):

    # clean the dataframe: drop rows where test variables are NaN
    lmm_df: pd.DataFrame = hand_df.dropna(subset=['CoV', 'Hand_Condition', 'Hand_Role', 'Participant']).copy()

    # categorical ordering to make the 'Passive' Role and 'Healthy' Condition the baselines
    lmm_df['Hand_Role'] = pd.Categorical(lmm_df['Hand_Role'], categories=['Passive', 'Active'], ordered=True)
    lmm_df['Hand_Condition'] = pd.Categorical(lmm_df['Hand_Condition'], categories=['Healthy', 'Affected'], ordered=True)

    # fit the model
    try:
        model = smf.mixedlm('CoV ~C(Hand_Condition) * C(Hand_Role)', data=lmm_df, groups=lmm_df['Participant'])
        result = model.fit()

        # extract the statistics table
        stats_df: pd.DataFrame = result.summary().tables[1]

        # save results to csv
        csv_path: str = os.path.join(results_dir, 'lmm_temporal_consistency_stats.csv')
        if not os.path.exists(csv_path):
            stats_df.to_csv(csv_path)

        # save full summary as txt
        txt_path = os.path.join(results_dir, 'lmm_temporal_consistency_stats_full.txt')
        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as f:
                f.write(result.summary().as_text())

        print(f"Linear Mixed Model results saved successfully to:\n {csv_path}\n {txt_path}")
        return result

    except Exception as e:
        print(f'Linear Mixed Model failed to converge or encountered an error: {e}')
        return None


def run_temporal_consistency_check():

    # load participant objects
    participant_objs_path: str = os.path.join(project_path, 'data', '03_processed')
    participant_pickle_lst: list = [x for x in sorted(os.listdir(participant_objs_path))
                                    if x.startswith('P') and x.endswith('.pickle')]

    all_cov_df_lst: list = []

    # run for each participant and each visit
    for pickle_file in participant_pickle_lst:
        p: Participant = Participant.load(os.path.join(participant_objs_path, pickle_file))

        # determine coefficients of variation for each hand segment
        hand_size_dict: dict = determine_best_hand_reference(p)
        cov_res_df: pd.DataFrame = get_hand_segment_stability(p, hand_size_dict)
        all_cov_df_lst.append(cov_res_df)

    # combine all participants into one dataframe
    full_cov_df: pd.DataFrame = pd.concat(all_cov_df_lst, ignore_index=True)

    # order of fingers for plotting
    finger_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    # get the average coefficients of variance of each finger grouping by condition and role
    finger_df: pd.DataFrame = full_cov_df.groupby(['Participant', 'Trial', 'Hand_Condition', 'Hand_Role', 'Finger']
                                                  )['CoV'].mean().reset_index()

    # create combined 'State' column for the 4-way comparison
    finger_df['State'] = finger_df['Hand_Condition'] + " (" + finger_df['Hand_Role'] + ")"

    # pivot DataFrame table to adjust parameters for raincloud plot
    reformat_finger_df: pd.DataFrame = finger_df.pivot_table(
        index=['Participant', 'Trial', 'Hand_Condition', 'Hand_Role'],
        columns='Finger',
        values='CoV'
    ).reset_index()

    # filter states for Raincloud plots
    active_healthy_df = reformat_finger_df[(reformat_finger_df['Hand_Condition'] == 'Healthy') &
                                           (reformat_finger_df['Hand_Role'] == 'Active')]
    active_affected_df = reformat_finger_df[(reformat_finger_df['Hand_Condition'] == 'Affected') &
                                            (reformat_finger_df['Hand_Role'] == 'Active')]

    # create visualizer object
    vis_util: Visualizer = Visualizer()

    # plot raincloud (Active Healthy)
    vis_util.vis_consistency_raincloud(link_df=active_healthy_df,
                                       columns_to_plot=finger_order,
                                       title='Task Consistency of Healthy Hand (Active)',
                                       x_label='Finger',
                                       y_label='Mean CoV (%)')

    # plot raincloud (affected)
    vis_util.vis_consistency_raincloud(link_df=active_affected_df,
                                       columns_to_plot=finger_order,
                                       title='Task Consistency of Affected Hand (Active)',
                                       x_label='Finger',
                                       y_label='Mean CoV (%)')

    # box plot comparison of all 4 states
    vis_util.viz_comparison_boxplot(finger_df, finger_order)

    # run the LMM statistical test
    lmm_results = get_lmm_consistency_statistics(finger_df, vis_util.temp_consistency_res_path)

    # generate the LLM's interaction trajectories
    vis_util.viz_lmm_interaction_trajectories(finger_df)


if __name__ == '__main__':
    run_temporal_consistency_check()
