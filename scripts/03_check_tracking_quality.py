#!/usr/bin/python3
# scripts/03_check_tracking_quality.py

# libraries
import os
import pandas as pd
import statsmodels.formula.api as smf
from tqdm import tqdm
from scipy.stats import chi2

# modules
from src.config import config, project_path
from src.core import Participant
from src.utils import ToolBox
from src.visualization import Visualizer


def get_hand_segment_stability(p: Participant) -> pd.DataFrame:
    """
    Calculates segment lengths and normalizes them by anatomical hand size (sum of wrist to middle finger knuckle plus
    each middle finger segment). Calculates the Coefficient of Variance (CoV) for each body segment for both the
    active and the passive hands.

    Args:
        p (Participant): list of participant objects of the class Participant.

    Returns:
        pd.dataframe: A dataframe containing the CoV of each body segment for both hands.
    """

    # configuration definitions
    finger_names_dict = {'1': 'Thumb', '2': 'Index', '3': 'Middle', '4': 'Ring', '5': 'Pinky'}
    link_lst: list[list[str]] = config['body_parts']['hands_link_lst']

    # tracking consistency test using all participants and exercises
    results_lst: list = []
    tb: ToolBox = ToolBox()

    # run for each exercise
    for ex_key, ex in p.exercises.items():

        # determine the active and passive side ('L' or 'R')
        active_side: str = ex.side_focus
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
                'side': passive_side,
                'condition': passive_condition,
                'link_idx': link_lst[:len(link_lst)//2] if passive_side == 'L' else link_lst[len(link_lst)//2:]
            }
        }

        # load the raw parquet dataframe
        try:
            exercise_df: pd.DataFrame = ex.load_dataframe('raw')
        except FileNotFoundError:
            print(f'Warning: No raw data found for {ex_key} of {p.pid}. Skipping exercise.')
            continue

        for role, role_info in hand_roles.items():
            curr_link_lst = role_info['link_idx']
            condition = role_info['condition']
            current_side = role_info['side']   # 'L' or 'R'

            # get the pre-calculated, exercise-specific median hand size
            med_hand_size: float = ex.left_hand_size if current_side == 'L' else ex.right_hand_size

            # handle missing median hand size
            if med_hand_size == 0.0:
                print(f'Error: Hand size is 0.0 for {current_side} hand in {ex_key}. Skipping {role} role.')
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
                    'Segment': segment_name,                # e.g., 'wrist1-mcp15'
                    'CoV': round(cov_value, 3)
                })

    results_df: pd.DataFrame = pd.DataFrame(results_lst)
    #print(results_df['Segment'].unique())

    return results_df


def get_arm_segment_stability(p: Participant) -> pd.DataFrame:
    """
    Calculates segment lengths of the arms and calculates the Coefficient of Variance (CoV) for each body segment for
    both the active and the passive arms.

    Args:
        p (Participant): list of participant objects of the class Participant.

    Returns:
        pd.dataframe: a dataframe containing the CoV of each body segment for both arms.
    """

    # configuration definitions
    ul_seg_names_dict = {'1': 'Torso', '2': 'Upper_Arm', '3': 'Lower_Arm'}
    link_lst: list[list[str]] = config['body_parts']['pose_link_lst']

    # tracking consistency test using all participants and exercises
    results_lst: list = []
    tb: ToolBox = ToolBox()

    # run for each exercise
    for ex_key, ex in p.exercises.items():

        # determine the active and passive side ('L' or 'R')
        active_side: str = ex.side_focus
        passive_side: str = 'R' if active_side == 'L' else 'L'

        # map the active and passive sides to the affected and healthy sides by participant info
        active_condition: str = 'Affected' if active_side == p.affected_side else 'Healthy'
        passive_condition: str = 'Healthy' if active_side == p.affected_side else 'Affected'

        # configuration for the roles of both arms for a given exercise
        arm_roles = {
            'Active': {
                'side': active_side,
                'condition': active_condition,
                'link_idx': link_lst[:3] if active_side == 'L' else [link_lst[0]] + link_lst[3:]
            },
            'Passive': {
                'side_lr': passive_side,
                'condition': passive_condition,
                'link_idx': link_lst[:3] if passive_side == 'L' else [link_lst[0]] + link_lst[3:]
            }
        }

        # load the raw parquet dataframe
        try:
            exercise_df: pd.DataFrame = ex.load_dataframe('raw')
        except FileNotFoundError:
            print(f'Warning: No raw data found for {ex_key} of {p.pid}. Skipping exercise.')
            continue

        for role, role_info in arm_roles.items():
            curr_link_lst = role_info['link_idx']
            condition = role_info['condition']

            # calculate the length of each arm segment
            segment_len_df: pd.DataFrame = tb.calculate_3d_segment_lengths(exercise_df, curr_link_lst)

            # calculation of segment CoV (%)
            body_segments_cov_series: pd.Series = ((segment_len_df.std() / segment_len_df.mean()) * 100)

            # identify the segment by checking the connected string names
            for segment_name, cov_value in body_segments_cov_series.items():
                seg_str = str(segment_name).lower()

                # check if 'shoulder' appears twice (left and right)
                if seg_str.count('shoulder') == 2:
                    ul_seg_name = 'Torso'
                elif 'shoulder' in seg_str and 'elbow' in seg_str:
                    ul_seg_name = 'Upper_Arm'
                elif 'elbow' in seg_str and 'wrist' in seg_str:
                    ul_seg_name = 'Lower_Arm'
                else:
                    ul_seg_name = 'Unknown'

                results_lst.append({
                    'Participant': p.pid,                   # e.g, P001
                    'Trial': ex_key,                        # e.g., 'FingerTapping_Affected'
                    'Active_Side': active_side,             # 'L' or 'R'
                    'Passive_Side': passive_side,           # 'L' or 'R'
                    'Hand_Role': role,                      # 'Active' or 'Passive'
                    'Hand_Condition': condition,            # 'Affected' or 'Healthy'
                    'Upper_Limb_Segment_Name': ul_seg_name, # 'Torso', 'Upper_Arm', 'Lower_Arm'
                    'Segment': segment_name,                # e.g., 'shoulder1-elbow1'
                    'CoV': round(cov_value, 3)
                })

    return pd.DataFrame(results_lst)


def get_lmm_consistency_statistics(hand_df: pd.DataFrame, results_dir: str, body_part: str = "Hand"):
    """
    This function fits a Linear Mixed Model (LMM), performs model selection using a likelihood ratio test (LRT),
    calculates the ICC, and saves the results to a text and a csv file.

    Args:
        hand_df (pd.DataFrame): Table containing the Participant, CoV, Hand_Condition, and Hand_Role
        results_dir (str): Directory where the results are saved.
        body_part (str, optional): Body part name to add to filename. Defaults to 'Hand'.

    Returns:
        final_model: Linear Mixed Model fitted on the data (generated with statsmodels)
    """
    def report_lmm_diagnostic_stats(main_effect_llf: float, interaction_effect_llf: float, likelihood_ratio_test: float,
                                    df_diff: int, p_value, model_select_txt: str, select_formula: str,
                                    random_effect_var: float, res_var: float, icc: float):

        # prepare the text to save in text file
        report_text = (
            "====================================================================\n"
            "     LINEAR MIXED MODEL (LMM) - STATISTICAL DIAGNOSTICS REPORT\n"
            "====================================================================\n\n"
            "1. MODEL SELECTION (Likelihood Ratio Test)\n"
            "------------------------------------------\n"
            f"   - Null Model (Main Effects Only) LLF: {main_effect_llf:.4f}\n"
            f"   - Full Model (With Interaction) LLF:  {interaction_effect_llf:.4f}\n"
            f"   - LR Statistic (Chi-Square):          {likelihood_ratio_test:.4f}\n"
            f"   - Degrees of Freedom Diff:            {df_diff}\n"
            f"   - P-value:                            {p_value:.4e}\n"
            f"   - Conclusion:\n"
            f"   -> {model_select_txt}\n"
            f"   - Selected Formula:\n"
            f"   -> {select_formula}\n\n"
            "2. VARIANCE PARTITIONING (ICC)\n"
            "------------------------------\n"
            f"   - Random Effect Variance (Participants): {random_effect_var:.4f}\n"
            f"   - Residual Variance (Unexplained Noise): {res_var:.4f}\n"
            f"   - ICC: {icc:.3f} ({(icc * 100):.1f}% of unexplained variance is due to participant baselines)\n\n\n"
            "====================================================================\n"
            "SUMMARY TABLE\n"
            "====================================================================\n"
        )

        report_text += final_model.summary().as_text()

        # save full report in text file
        txt_path = os.path.join(results_dir, f'lmm_temporal_consistency_stats_{body_part}_full.txt')
        if not os.path.exists(txt_path):
            with open(txt_path, 'w') as f:
                f.write(report_text)

        print(f"Linear Mixed Model results saved successfully to:\n {txt_path}")

    # clean the dataframe: drop rows where test variables are NaN
    lmm_df: pd.DataFrame = hand_df.dropna(subset=['CoV', 'Hand_Condition', 'Hand_Role', 'Participant']).copy()

    # categorical ordering to make the 'Passive' Role and 'Healthy' Condition the baselines
    # pd.Categorical assigns values '0' and '1' to 'Passive' (0) and Active (1), and to 'Healthy' (0) and 'Affected' (1)
    # crucial for the model to correctly interpret the baseline ('Passive' and 'Healthy')
    lmm_df['Hand_Role'] = pd.Categorical(lmm_df['Hand_Role'], categories=['Passive', 'Active'], ordered=True)
    lmm_df['Hand_Condition'] = pd.Categorical(lmm_df['Hand_Condition'], categories=['Healthy', 'Affected'], ordered=True)

    # Likelihood Ratio Test (LRT)
    main_formula: str = 'CoV ~C(Hand_Condition) + C(Hand_Role)'
    interaction_formula: str = 'CoV ~C(Hand_Condition) * C(Hand_Role)'

    try:
        main_model = smf.mixedlm(formula=main_formula, data=lmm_df, groups=lmm_df['Participant']).fit(reml=False)
        inter_model = smf.mixedlm(formula=interaction_formula, data=lmm_df, groups=lmm_df['Participant']).fit(reml=False)

        # get the log likelihood function (llf) attribute of the fitted models - how well the models explain the data
        llf_main = main_model.llf
        llf_int = inter_model.llf
        # get the difference in degrees of freedom between the models - dof: number of variables the model estimates
        dof_diff = inter_model.df_modelwc - main_model.df_modelwc

        # calculate Chi-Square statistic and p-value
        likelihood_ratio_stat = -2 * (llf_main - llf_int)
        p_val = chi2.sf(likelihood_ratio_stat, dof_diff)

        # determine whether the interaction term significantly improves the model or not
        if p_val < 0.05:
            better_formula = 'CoV ~ Hand_Condition * Hand_Role'
            model_selection_txt = 'The interaction term significantly improves the model.'
        else:
            better_formula = 'CoV ~ Hand_Condition + Hand_Role'
            model_selection_txt = 'The interaction term does not significantly improve the model.'

        # re-fit the model with the better formula (reml=True, default)
        final_model = smf.mixedlm(formula=better_formula, data=lmm_df, groups=lmm_df['Participant']).fit()

        # extract variances: cov_re is the random effect variance and scale is the residual variance
        rand_effect_var = final_model.cov_re.iloc[0, 0]
        residual_var = final_model.scale
        icc = rand_effect_var / (rand_effect_var + residual_var)

        report_lmm_diagnostic_stats(llf_main, llf_int, likelihood_ratio_stat, dof_diff, p_val,
                                    model_selection_txt, better_formula, rand_effect_var, residual_var, icc)

        # extract the statistics table
        stats_df: pd.DataFrame = final_model.summary().tables[1]

        # save results to csv
        csv_path: str = os.path.join(results_dir, f'lmm_temporal_consistency_stats_{body_part}.csv')
        if not os.path.exists(csv_path):
            stats_df.to_csv(csv_path)

        print(f"Linear Mixed Model results saved successfully to:\n {csv_path}")
        return final_model

    except Exception as e:
        print(f'Linear Mixed Model failed to converge or encountered an error: {e}')
        return None


def run_temporal_consistency_check():
    """
    Main function that loads participant data, calculates segment stability (CoV)
    for both arms and hands, performs statistical evaluations using Linear Mixed Models (LMM),
    and generates all diagnostic and comparative visualizations.

    Args:
        None

    Returns:
        None
    """
    # load participant objects
    participant_objs_path: str = os.path.join(project_path, 'data', '03_processed')
    participant_pickle_lst: list = [x for x in sorted(os.listdir(participant_objs_path))
                                    if x.startswith('P') and x.endswith('.pickle')]

    all_hand_cov_df_lst: list = []
    all_arm_cov_df_lst: list = []

    # init ToolBox objects
    tb = ToolBox()

    # run for each participant and each visit
    for pickle_file in tqdm(participant_pickle_lst, desc='Running temporal consistency check...'):
        p: Participant = Participant.load(os.path.join(participant_objs_path, pickle_file))

        # determine coefficients of variation (CoV) for each hand segment
        hand_cov_res_df: pd.DataFrame = get_hand_segment_stability(p)
        all_hand_cov_df_lst.append(hand_cov_res_df)

        # determine coefficients of variation (CoV) for each arm segment
        arm_cov_res_df: pd.DataFrame = get_arm_segment_stability(p)
        all_arm_cov_df_lst.append(arm_cov_res_df)

    # HAND CONSISTENCY CHECK

    hand_states: list[str] = ['Healthy (Active)', 'Healthy (Passive)', 'Affected (Active)', 'Affected (Passive)']

    # combine all participants into one dataframe
    full_hand_cov_df: pd.DataFrame = pd.concat(all_hand_cov_df_lst, ignore_index=True)
    hand_df = full_hand_cov_df.copy()

    # create combined 'State' column for the 4-way comparison
    hand_df['State'] = hand_df['Hand_Condition'] + " (" + hand_df['Hand_Role'] + ")"

    # prepare aggregated data for the raincloud plots: -> 1 mean value per trial
    reformat_hand_df: pd.DataFrame = hand_df.groupby(['Participant', 'Trial', 'Hand_Condition',
                                                      'Hand_Role', 'State'])['CoV'].mean().reset_index()

    # filter states for Raincloud plots
    #active_healthy_hand_df = reformat_hand_df[(reformat_hand_df['State'] == 'Healthy (Active)')]
    #active_affected_hand_df = reformat_hand_df[(reformat_hand_df['State'] == 'Affected (Active)')]


    # # order of fingers for plotting
    # finger_order = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    #
    # # get the average coefficients of variance of each finger grouping by condition and role
    # finger_df: pd.DataFrame = full_hand_cov_df.groupby(['Participant', 'Trial', 'Hand_Condition',
    #                                                     'Hand_Role', 'Finger'])['CoV'].mean().reset_index()
    #
    # # create combined 'State' column for the 4-way comparison
    # finger_df['State'] = finger_df['Hand_Condition'] + " (" + finger_df['Hand_Role'] + ")"
    #
    # # pivot DataFrame table to adjust parameters for raincloud plot
    # reformat_finger_df: pd.DataFrame = finger_df.pivot_table(
    #     index=['Participant', 'Trial', 'Hand_Condition', 'Hand_Role'],
    #     columns='Finger',
    #     values='CoV'
    # ).reset_index()
    #
    # # filter states for Raincloud plots
    # active_healthy_hand_df = reformat_finger_df[(reformat_finger_df['Hand_Condition'] == 'Healthy') &
    #                                             (reformat_finger_df['Hand_Role'] == 'Active')]
    # active_affected_hand_df = reformat_finger_df[(reformat_finger_df['Hand_Condition'] == 'Affected') &
    #                                              (reformat_finger_df['Hand_Role'] == 'Active')]

    # create visualizer object
    vis_util: Visualizer = Visualizer()

    # plot raincloud of hand detection
    vis_util.vis_hand_consistency_raincloud(df=hand_df,
                                            states_to_plot=hand_states,
                                            title='Temporal Consistency of Hands by Task',
                                            x_label='State (Role)',
                                            y_label='Mean CoV (%)')

    # box plot comparison of all 4 states
    vis_util.viz_comparison_boxplot(segment_df=hand_df, body_part='Hand')

    # run the LMM statistical test
    lmm_results_hands = get_lmm_consistency_statistics(hand_df, vis_util.temp_consistency_res_path, body_part='Hand')

    # generate the LLM's interaction trajectories
    vis_util.viz_lmm_interaction_trajectories(segment_df=hand_df, body_part='Hand')

    # generate Q-Q plot and Tukey-Anscombe (homoscedasticity) to test model assumptions
    vis_util.viz_lmm_normality_and_homoscedasticity(model=lmm_results_hands, body_part='Hand')

    # ARM CONSISTENCY CHECK

    # combine all participants into one dataframe
    full_arm_cov_df: pd.DataFrame = pd.concat(all_arm_cov_df_lst, ignore_index=True)

    # get the average coefficients of variance of each upper limb segment by condition and role
    arm_df = full_arm_cov_df[full_arm_cov_df['Upper_Limb_Segment_Name'] != 'Torso'].copy()

    # create combined 'State' column for the 4-way comparison
    arm_df['State'] = arm_df['Hand_Condition'] + " (" + arm_df['Hand_Role'] + ")"

    # aggregate data for raincloud plot
    reformat_arm_df: pd.DataFrame = arm_df.groupby(['Participant', 'Trial', 'Hand_Condition', 'Hand_Role',
                                                    'State'])['CoV'].mean().reset_index()

    # plot raincloud (Active Healthy)
    vis_util.vis_hand_consistency_raincloud(df=reformat_arm_df,
                                            states_to_plot=hand_states,
                                            title='Temporal Consistency of Arms by Task',
                                            x_label='State (Role)',
                                            y_label='Mean CoV (%)')

    # box plot comparison of all 4 states
    vis_util.viz_comparison_boxplot(segment_df=arm_df, body_part='Arm')

    # run the LMM statistical test
    lmm_results_arms = get_lmm_consistency_statistics(arm_df, vis_util.temp_consistency_res_path, body_part='Arm')

    if lmm_results_arms:
        # generate the LLM's interaction trajectories
        vis_util.viz_lmm_interaction_trajectories(segment_df=arm_df, body_part='Arm')

        # generate Q-Q plot and Tukey-Anscombe (homoscedasticity) to test model assumptions
        vis_util.viz_lmm_normality_and_homoscedasticity(model=lmm_results_arms, body_part='Arm')


if __name__ == '__main__':
    run_temporal_consistency_check()
