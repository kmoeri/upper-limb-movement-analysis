# src/exercise_evaluation.py
import pydoc

# libraries
import numpy as np
import editdistance
import pandas as pd
from scipy.ndimage import median_filter

# modules
from src.config import config
from src.core import Exercise
from src.utils import ToolBox
from src.kinematic_features import KinematicFeatures
from src.visualization import Visualizer


class ExerciseEvaluator:
    def __init__(self, fps=config['camera_param']['fps']):
        self.fps = fps
        # instantiate globally
        self.tb = ToolBox(fps=self.fps)
        self.kf = KinematicFeatures(fps=self.fps)
        self.viz = Visualizer()
        # default config for adaptive peak detection
        self.peak_cfg: dict = {'min_segment_length': 0.05,
                               'min_peak_amp_diff': 0.15,
                               'min_peak_dur_diff': 0.00,
                               'min_valley_amp_diff': 0.15,
                               'min_valley_dur_diff': 0.00,
                               'max_peak_amp_diff': 0.00,
                               'max_peak_dur_diff': 0.0,
                               'max_valley_amp_diff': 0.0,
                               'prominence_factor': 0.35,
                               'distance_factor': 0.35}

    def _extract_distance_based_kinematics(self, df: pd.DataFrame, static_hand_size: float, ref_landmark_name: str,
                                           target_landmark_names: list, custom_peak_cfg: dict = None,
                                           min_offset: float = 0.0, max_offset: float = 0.0) -> dict:
        """
        Helper function to calculate normalized Euclidean distance and extract kinematic
        peaks for arbitrary combination of reference landmark and target landmarks combinations.
        Uses dynamic palm lengths for smooth peak detection and static anatomical hand sizes for feature scaling.

        Args:
            df (pd.DataFrame): Pandas DataFrame with 3D landmark coordinates.
            static_hand_size (float): Static hand size.
            ref_landmark_name (str): The reference point (e.g., 'wrist1', 'cmc1', 'ftip1').
            target_landmark_names (list): The moving points (e.g., ['ftip2', 'ftip3']).
            custom_peak_cfg (dict): A dictionary with customized peak detection parameters.
            min_offset (float, optional): The minimal distance offset of an anatomically unreachable distance.
            max_offset (float, optional): The maximal distance offset of an anatomically unreachable distance.

        Returns:
            dict: Nested dictionary containing normalized distances and feature parameters.
        """
        performance_dict = {}

        # determine the config to use
        current_cfg = custom_peak_cfg if custom_peak_cfg else self.peak_cfg

        # extract the (frames, 3) tensor for the reference landmarks
        ref_cols: list = [f'{ref_landmark_name}_x', f'{ref_landmark_name}_y', f'{ref_landmark_name}_z']

        try:
            ref_pos: np.ndarray = df[ref_cols].to_numpy()
        except KeyError:
            print(f'Error: reference landmark {ref_landmark_name} not found in dataframe.')
            return {}

        for target_name in target_landmark_names:
            target_cols = [f'{target_name}_x', f'{target_name}_y', f'{target_name}_z']
            try:
                target_pos: np.ndarray = df[target_cols].to_numpy()
            except KeyError:
                print(f'Error: target landmark {target_cols} not found. Skipping.')
                continue

            # calculate the absolute raw distance
            raw_dist = np.linalg.norm(ref_pos - target_pos, axis=1)

            # distance used for final distance measurement
            norm_dist_arr = raw_dist / max(static_hand_size, 1e-8)

            # apply anatomical offset correction
            norm_dist_arr = np.clip((norm_dist_arr - min_offset) / (max_offset - min_offset), 0.0, 1.0)

            # peak extraction
            feature_dict = self.kf.calc_kinematic_parameters(norm_dist_arr, current_cfg)

            finger_key = f'{ref_landmark_name}-{target_name}'
            performance_dict[finger_key] = {'normalized_distance': norm_dist_arr,
                                            'features': feature_dict}

        return performance_dict

    def analyze_finger_tapping(self, exercise: Exercise, p_id: str, p_hand_size: float, save_plots: bool = False) -> dict:

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx: int = 1 if exercise.side_focus == 'L' else 2

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names: list = config['index_ftap']['landmark_names']
        lm_corr_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names]
        anchor = f'cmc{active_side_idx}1'       # thenar
        target = lm_corr_names[1]               # index fingertip
        pair_key: str = f'{anchor}-{target}'

        # select exercise-specific config (if not defined, fall back to default self.peak_cfg)
        ex_peak_cfg: dict = config['index_ftap'].get('peak_cfg', self.peak_cfg)

        # load the cleaned DataFrame
        try:
            df: pd.DataFrame = exercise.load_dataframe('clean')
        except FileNotFoundError:
            print(f'Error: Clean data not found for {exercise.exercise_id}. Skipping.')
            return {}

        # offset adjustment: reference point is anatomically unreachable
        min_offset: float = config['index_ftap'].get('dist_offset_min', 0.18)
        max_offset: float = config['index_ftap'].get('dist_offset_max', 0.82)

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict: dict = self._extract_distance_based_kinematics(df,
                                                                         p_hand_size,       # for feature scaling
                                                                         anchor,
                                                                         [target],
                                                                         ex_peak_cfg,
                                                                         min_offset, max_offset)

        # handle missing key pair
        if pair_key not in active_dist_dict:
            return {}

        # 1.2) correctness of the tapping (spatial accuracy)
        valid_valley_idc: list = active_dist_dict[pair_key]['features']['valid_valleys_idx']
        norm_dist_arr: np.ndarray = active_dist_dict[pair_key]['normalized_distance']

        if len(valid_valley_idc) > 0:
            valid_dist: float = config['index_ftap']['min_dist_thresh']

            # Euclidean distance at each event (valley)
            valley_dist_arr: np.ndarray = norm_dist_arr[valid_valley_idc]

            # results
            mean_target_error: float = float(np.mean(valley_dist_arr))
            successful_taps: int = np.sum(valley_dist_arr < valid_dist)
            accuracy_percentage: float = float((successful_taps / len(valid_valley_idc)) * 100)
        else:
            mean_target_error: float = 0.0
            accuracy_percentage: float = 0.0

        # 1.3) quality: assess rhythm with CoV (period time between taps), isolation (variance of other fingers)
        # 1.3.1) calculate the tapping rhythm with CoV
        if len(valid_valley_idc) > 1:
            tapping_diff: np.ndarray = np.diff(valid_valley_idc)
            tapping_mean: float = float(np.mean(tapping_diff))
            tapping_cov: float = float((float(np.std(tapping_diff)) / tapping_mean) * 100) if tapping_mean > 0 else 0.0

            # dynamic window for isolation (33% of a period)
            half_window: int = max(int(tapping_mean / 6.0), 2)
        else:
            tapping_cov: float = 0.0
            half_window: int = int(self.fps * 0.05)

        # 1.3.2) calculate the tapping isolation
        passive_fingers: list = lm_corr_names[2:]
        wrist_key = f'wrist{active_side_idx}'

        # calculate Euclidean distance for passive fingers
        passive_kinematics: dict = self._extract_distance_based_kinematics(df,
                                                                           p_hand_size,
                                                                           wrist_key,
                                                                           passive_fingers,
                                                                           ex_peak_cfg,
                                                                           min_offset, max_offset)

        # calculate the isolation variances
        isolation_variances: list = []
        for tap_idx in valid_valley_idc:
            for passive_kin_key, p_data in passive_kinematics.items():
                dist_arr: np.ndarray = p_data['normalized_distance']
                start: int = max(0, tap_idx - half_window)
                end: int = min(len(dist_arr), tap_idx + half_window)
                if start < end:
                    isolation_variances.append(np.var(dist_arr[start:end]))

        isolation_score: float = float(np.mean(isolation_variances)) if isolation_variances else 0.0

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        # 5) create result dictionary

        # flatten the performance metrics of the active finger
        performance_features = active_dist_dict[pair_key]['features']

        # add exercise specific prefix-key 'idx_tap'
        results = {
            # time series
            'idx_tap_time_series_y': performance_features.get('signal_original', []),
            'idx_tap_time_series_x': performance_features.get('time_axis', []),
            # general kinematic features
            'idx_tap_rep_num': performance_features.get('repetition_num', 0.0),
            'idx_tap_rep_freq': performance_features.get('repetition_freq', 0.0),
            'idx_tap_amp_mean': performance_features.get('amplitude_mean', 0.0),
            'idx_tap_amp_pct_90': performance_features.get('amplitude_pct_90', 0.0),
            'idx_tap_amp_cov': performance_features.get('amplitude_cov', 0.0),
            'idx_tap_period_mean': performance_features.get('period_mean', 0.0),
            'idx_tap_period_pct_90': performance_features.get('period_pct_90', 0.0),
            'idx_tap_period_cov': performance_features.get('period_cov', 0.0),
            'idx_tap_vel_pos_mean': performance_features.get('velocity_pos_mean', 0.0),
            'idx_tap_vel_pos_pct_90': performance_features.get('velocity_pos_pct_90', 0.0),
            'idx_tap_vel_pos_cov': performance_features.get('velocity_pos_cov', 0.0),
            'idx_tap_vel_neg_mean': performance_features.get('velocity_neg_mean', 0.0),
            'idx_tap_vel_neg_pct_90': performance_features.get('velocity_neg_pct_90', 0.0),
            'idx_tap_vel_neg_cov': performance_features.get('velocity_neg_cov', 0.0),
            # correctness of the tapping (spatial accuracy)
            'idx_tap_accuracy': accuracy_percentage,
            'idx_tap_target_error': mean_target_error,
            # rhythm with CoV (period time between taps)
            'idx_tap_cov': tapping_cov,
            # isolation (variance of other fingers)
            'idx_tap_isolation': isolation_score,
        }

        # plot visualization
        if save_plots and performance_features['extraction_status'] == 'success':

            self.viz.viz_repetitive_binary_exercises(time_axis=performance_features['time_axis'],
                                                     signal=performance_features['signal_original'],
                                                     features=performance_features,
                                                     p_id=p_id,
                                                     visit_id=exercise.visit_id,
                                                     ex_id=f'{exercise.exercise_id}_{exercise.side_condition}')

        return results

    def analyze_finger_alternation(self, exercise: Exercise, p_id: str, p_hand_size: float, save_plots: bool = False) -> dict:

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names = config['ftap_alter']['landmark_names']
        lm_corr_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names]
        thumb = lm_corr_names[0]
        finger_names = lm_corr_names[1:]

        # select exercise-specific config (if not defined, fall back to default self.peak_cfg)
        ex_peak_cfg: dict = config['ftap_alter'].get('peak_cfg', self.peak_cfg)

        # load the cleaned DataFrame directly
        try:
            df: pd.DataFrame = exercise.load_dataframe('clean')
        except FileNotFoundError:
            print(f'Error: Clean data not found for {exercise.exercise_id}. Skipping.')
            return {}

        # dynamic palm scaling
        dynamic_palm_arr: np.ndarray = np.ones(len(df))

        # offset adjustment: reference point is anatomically unreachable -> offset of ~0.0
        min_offset: float = config['ftap_alter'].get('dist_offset_min', 0.00)
        max_offset: float = config['ftap_alter'].get('dist_offset_max', 1.00)

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict: dict = self._extract_distance_based_kinematics(df,
                                                                         p_hand_size,
                                                                         thumb,
                                                                         finger_names,
                                                                         ex_peak_cfg,
                                                                         min_offset, max_offset)

        # 1.2) correctness of the tapping order (extract tapping sequence and score with Levenshtein Distance)

        # extract tapping idc list for each finger digit
        finger_tapping_idc_lst: list = []
        for finger_key, val in active_dist_dict.items():
            finger_tapping_idc_lst += [(int(finger_key[-1]), x) for x in val['features']['valid_valleys_idx']]

        # sort the indices of each finger by the second tuple element (index number): [(1, '153'), (2, '153'), ...]
        finger_tapping_idc_lst = sorted(finger_tapping_idc_lst, key=lambda x: x[1])

        # extract the finger digit sequence
        tapping_sequence_digit_lst: list = [x[0] for x in finger_tapping_idc_lst]

        pat_len = len(tapping_sequence_digit_lst)

        # safety: for zero movement
        if pat_len == 0:
            return {'errors': 0, 'accuracy': 0.0, 'intended_target_len': 0}

        # template of possible target tapping sequences - one full repetition
        target_templates = [
            [2,3,4,5],              # resetting
            [2, 3, 4, 5, 4, 3]      # looping
        ]

        # initialize rating variables
        best_accuracy = -1.0
        best_errors = 0
        best_target_len = 0

        # define a search window for the estimated taps intended
        min_search_len = max(1, pat_len // 2)   # x0.5 the taps if every finger was double-tapped
        max_search_len = int(pat_len * 1.5)     # x1.5 the taps if there are many skipped fingers

        for template in target_templates:
            # create an oversized sequence (30 repetitions of the sequence)
            oversized_target = template * 30

            # check every possible target length from the min to max window
            for test_len in range(min_search_len, max_search_len + 1):

                # slice the perfect sequence to this test length
                test_target = oversized_target[:test_len]

                # calculate the Levenshtein distance errors
                errors = editdistance.eval(test_target, tapping_sequence_digit_lst)

                # normalize to an accuracy percentage
                max_possible_errors = max(test_len, pat_len)
                accuracy = ((max_possible_errors - errors) / max_possible_errors) * 100

                # keep the window that aligns the best and the corresponding accuracy and errors
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_errors = errors
                    best_target_len = test_len

        tapping_results: dict = {
            'errors': best_errors,
            'accuracy': round(best_accuracy, 2),
            'intended_target_len': best_target_len
        }

        # 1.3) quality: assess rhythm with CoV (period time between taps), isolation (variance of other fingers)

        # extract the idc of the tapping sequence
        tapping_sequence_idc_lst: list = [x[1] for x in finger_tapping_idc_lst]

        # calculate the tapping rhythm with CoV
        if len(tapping_sequence_idc_lst) > 1:
            tapping_diff: np.ndarray = np.diff(tapping_sequence_idc_lst)
            tapping_mean: float = float(np.mean(tapping_diff))
            tapping_cov: float = float((float(np.std(tapping_diff)) / tapping_mean) * 100) if tapping_mean > 0 else 0.0
        else:
            tapping_mean: float = 0.0
            tapping_cov: float = 0.0

        # calculate the isolation score
        isolation_variances = []

        # calculate a dynamic window ca. 33% of the mean tapping period (tapping_mean / 3).
        if 'tapping_mean' in locals() and tapping_mean > 0:
            half_window_frames = int(tapping_mean / 6.0)        # backward and forward window
            half_window_frames = max(half_window_frames, 2)     # limit minimum window size
        else:
            # use 50ms half-window if there is only one tap (no mean period)
            half_window_frames = int(self.fps * 0.05)

        for active_finger_id, tap_idx in finger_tapping_idc_lst:

            # find the dictionary keys for the 3 fingers that are not supposed to be moving right now
            passive_finger_keys = [key for key in active_dist_dict.keys() if str(active_finger_id) not in key]

            for finger_key in passive_finger_keys:
                dist_array = active_dist_dict[finger_key]['normalized_distance']

                # array sliced dynamically based on participant-specific tapping speed
                start_idx = max(0, tap_idx - half_window_frames)
                end_idx = min(len(dist_array), tap_idx + half_window_frames)

                if start_idx < end_idx:
                    snippet = dist_array[start_idx:end_idx]
                    # calculate the movement variance of each 'resting' finger
                    isolation_variances.append(np.var(snippet))

        # isolation score: The average variance of all non-active fingers across all taps (lower: better isolation)
        isolation_score: float = float(np.mean(isolation_variances)) if isolation_variances else 0.0

        # extract all four time series (from each finger pair)
        alt_tap_y: list = []
        alt_tap_x: list = []
        alt_tap_features: list = []

        for fn in finger_names:
            p_feat = active_dist_dict[f'{thumb}-{fn}']['features']
            alt_tap_y.append(p_feat.get('signal_original', []))
            alt_tap_x.append(p_feat.get('time_axis', []))
            alt_tap_features.append(p_feat)

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        # 5) Create result dictionary

        # flatten the performance metrics of the active finger
        performance_features = active_dist_dict[f'{thumb}-{finger_names[0]}']['features']

        # add exercise specific prefix-key 'alt_tap'
        results = {
            # time series
            'alt_tap_time_series_y': alt_tap_y,
            'alt_tap_time_series_x': alt_tap_x,
            # general kinematic features
            'alt_tap_rep_num': performance_features.get('repetition_num', 0.0),
            'alt_tap_rep_freq': performance_features.get('repetition_freq', 0.0),
            'alt_tap_amp_mean': performance_features.get('amplitude_mean', 0.0),
            'alt_tap_amp_pct_90': performance_features.get('amplitude_pct_90', 0.0),
            'alt_tap_amp_cov': performance_features.get('amplitude_cov', 0.0),
            'alt_tap_period_mean': performance_features.get('period_mean', 0.0),
            'alt_tap_period_pct_90': performance_features.get('period_pct_90', 0.0),
            'alt_tap_period_cov': performance_features.get('period_cov', 0.0),
            'alt_tap_vel_pos_mean': performance_features.get('velocity_pos_mean', 0.0),
            'alt_tap_vel_pos_pct_90': performance_features.get('velocity_pos_pct_90', 0.0),
            'alt_tap_vel_pos_cov': performance_features.get('velocity_pos_cov', 0.0),
            'alt_tap_vel_neg_mean': performance_features.get('velocity_neg_mean', 0.0),
            'alt_tap_vel_neg_pct_90': performance_features.get('velocity_neg_pct_90', 0.0),
            'alt_tap_vel_neg_cov': performance_features.get('velocity_neg_cov', 0.0),
            # correctness of the tapping sequence (tapping order)
            'alt_tap_accuracy': tapping_results['accuracy'],
            'alt_tap_target_error': tapping_results['errors'],
            # rhythm with CoV (period time between taps)
            'alt_tap_cov': tapping_cov,
            # isolation (variance of other fingers)
            'alt_tap_isolation': isolation_score,
        }

        # plot visualization
        if save_plots and performance_features['extraction_status'] == 'success':

            self.viz.viz_repetitive_binary_exercises(time_axis=alt_tap_x,
                                                     signal=alt_tap_y,
                                                     features=alt_tap_features,
                                                     p_id=p_id,
                                                     visit_id=exercise.visit_id,
                                                     ex_id=f'{exercise.exercise_id}_{exercise.side_condition}')

        return results

    def analyze_hand_opening(self, exercise: Exercise, p_id: str, p_hand_size: float, save_plots: bool = False) -> dict:

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names = config['open_close']['landmark_names']
        wrist_name: str = f'{lm_base_names[0]}{active_side_idx}'
        finger_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names[1:]]

        # select exercise-specific config (if not defined, fall back to default self.peak_cfg)
        ex_peak_cfg: dict = config['open_close'].get('peak_cfg', self.peak_cfg)

        # load the cleaned DataFrame directly
        try:
            df: pd.DataFrame = exercise.load_dataframe('clean')
        except FileNotFoundError:
            print(f'Error: Clean data not found for {exercise.exercise_id}. Skipping.')
            return {}

        active_dist_dict: dict = {}
        finger_digits = ['2', '3', '4', '5']        # index, middle, ring, pinky
        all_kcr_arrays = []
        all_ang_arrays = []

        for digit in finger_digits:
            # vectorized matrix extraction from dataframe
            wrist_arr = df[[f'{wrist_name}_x', f'{wrist_name}_y', f'{wrist_name}_z']].to_numpy()
            mcp_arr = df[[f'mcp{active_side_idx}{digit}_x', f'mcp{active_side_idx}{digit}_y', f'mcp{active_side_idx}{digit}_z']].to_numpy()
            pip_arr = df[[f'pip{active_side_idx}{digit}_x', f'pip{active_side_idx}{digit}_y', f'pip{active_side_idx}{digit}_z']].to_numpy()
            dip_arr = df[[f'dip{active_side_idx}{digit}_x', f'dip{active_side_idx}{digit}_y', f'dip{active_side_idx}{digit}_z']].to_numpy()
            ftip_arr = df[[f'ftip{active_side_idx}{digit}_x', f'ftip{active_side_idx}{digit}_y', f'ftip{active_side_idx}{digit}_z']].to_numpy()

            # calculate the total kinematic chain ratio (KCR)

            # direct distance from fingertip landmarks to the wrist landmark
            dist_ftip_wrist = np.linalg.norm(ftip_arr - wrist_arr, axis=1)

            # distance measured by the sum of each segment - reference length for each frame
            sum_segments = (np.linalg.norm(mcp_arr - wrist_arr, axis=1) +
                            np.linalg.norm(pip_arr - mcp_arr, axis=1) +
                            np.linalg.norm(dip_arr - pip_arr, axis=1) +
                            np.linalg.norm(ftip_arr - dip_arr, axis=1))

            # clipping the KCR to the range 0.0-1.0 prevents values > 1.0 due to noise or hyperextension
            kcr_arr = np.clip(dist_ftip_wrist / np.clip(sum_segments, 1e-8, None), 0.0, 1.0)

            # offset adjustment: reference point is anatomically unreachable -> offset of ~0.35
            min_offset: float = config['open_close'].get('dist_offset_min', 0.35)
            max_offset: float = config['open_close'].get('dist_offset_max', 1.00)

            kcr_arr = np.clip((kcr_arr - min_offset) / (max_offset - min_offset), 0.0, 1.0)     # rescale

            # calculate the angle composite score
            vec_mw: np.ndarray = mcp_arr - wrist_arr
            vec_pm: np.ndarray = pip_arr - mcp_arr
            vec_dp: np.ndarray = dip_arr - pip_arr
            vec_fd: np.ndarray = ftip_arr - dip_arr

            # get flexion angle of each finger joint
            mcp_flex: np.ndarray = self.kf.calc_flexion_angle(vec_mw, vec_pm)
            pip_flex: np.ndarray = self.kf.calc_flexion_angle(vec_pm, vec_dp)
            dip_flex: np.ndarray = self.kf.calc_flexion_angle(vec_dp, vec_fd)

            # normalize flexion angles to 0-1 (1.0: perfectly extended, 0.0: max flexion)
            score_mcp = 1.0 - np.clip(mcp_flex / config['open_close'].get('mcp_flex', 90.0), 0, 1)
            score_pip = 1.0 - np.clip(pip_flex / config['open_close'].get('pip_flex', 100.0), 0, 1)
            score_dip = 1.0 - np.clip(dip_flex / config['open_close'].get('dip_flex', 80.0), 0, 1)
            angle_score_arr = (score_mcp + score_pip + score_dip) / 3.0

            # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
            feature_dict: dict = self.kf.calc_kinematic_parameters(kcr_arr, ex_peak_cfg)

            finger_key = f'{wrist_name}-ftip{active_side_idx}{digit}'
            active_dist_dict[finger_key] = {
                'normalized_distance': kcr_arr,
                'angle_score': angle_score_arr,
                'features': feature_dict
            }
            all_kcr_arrays.append(kcr_arr)
            all_ang_arrays.append(angle_score_arr)

        # 1.2) average signal extraction
        # calculate the average signals across all four fingers for whole-hand scoring
        avg_kcr_arr = np.mean(all_kcr_arrays, axis=0)
        avg_ang_arr = np.mean(all_ang_arrays, axis=0)

        # extract features on the averaged signal
        avg_performance_features = self.kf.calc_kinematic_parameters(avg_kcr_arr, ex_peak_cfg)

        # re-calculate amplitudes for the average signal
        avg_valid_peaks = avg_performance_features.get('valid_peaks_idx', [])
        avg_valid_valleys = avg_performance_features.get('valid_valleys_idx', [])

        # 1.3) Correctness of Opening & Closing (Completeness)
        # extension score (hand open) - graded average signal peaks
        if len(avg_valid_peaks) > 0:
            extension_score: float = float(
                (np.mean(avg_kcr_arr[avg_valid_peaks]) * 100 + np.mean(avg_ang_arr[avg_valid_peaks]) * 100) / 2.0)
        else:
            extension_score: float = 0.0

        # flexion score (hand closed) - graded average signal valleys
        if len(avg_valid_valleys) > 0:
            min_dist_kcr: float = config['open_close'].get('fist_min_dist', 0.25)
            kcr_mapped: np.ndarray = np.clip(
                1.0 - (avg_kcr_arr[avg_valid_valleys] - min_dist_kcr) / (1.0 - min_dist_kcr), 0.0, 1.0)
            ang_mapped: np.ndarray = np.clip(1.0 - avg_ang_arr[avg_valid_valleys], 0.0, 1.0)
            flexion_score: float = float((np.mean(kcr_mapped) * 100 + np.mean(ang_mapped) * 100) / 2.0)
        else:
            flexion_score: float = 0.0

        # 1.4) Quality: Temporal Dispersion (Synchronization)
        dispersion_variances = []

        # index finger remains the synchronization reference
        index_key = f'{wrist_name}-{finger_names[0]}'
        index_peaks = active_dist_dict[index_key]['features']['valid_peaks_idx']
        passive_finger_keys = [f'{wrist_name}-{pf}' for pf in finger_names[1:]]
        max_lag_frames: int = int(self.fps * 0.5)  # allow max. 0.5 seconds of lag

        for index_peak in index_peaks:
            rep_timing = [index_peak]

            for passive_fkey in passive_finger_keys:
                passive_peaks = active_dist_dict[passive_fkey]['features']['valid_peaks_idx']
                if len(passive_peaks) == 0:
                    continue

                closest_peak = min(passive_peaks, key=lambda x: abs(x - index_peak))
                if abs(closest_peak - index_peak) < max_lag_frames:
                    rep_timing.append(closest_peak)

            if len(rep_timing) == 4:
                dispersion_sec = np.std(rep_timing) / self.fps
                dispersion_variances.append(dispersion_sec)

        synchronization_score: float = float(np.mean(dispersion_variances)) if dispersion_variances else 0.0

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        # 5) Create result dictionary

        # flatten the performance metrics using the average hand movement
        performance_features = avg_performance_features

        # add exercise specific prefix-key 'open_close'
        results = {
            # time series
            'open_close_time_series_y': performance_features.get('signal_original', []),
            'open_close_time_series_x': performance_features.get('time_axis', []),
            # general kinematic features
            'open_close_rep_num': performance_features.get('repetition_num', 0.0),
            'open_close_rep_freq': performance_features.get('repetition_freq', 0.0),
            'open_close_amp_mean': performance_features.get('amplitude_mean', 0.0),
            'open_close_amp_pct_90': performance_features.get('amplitude_pct_90', 0.0),
            'open_close_amp_cov': performance_features.get('amplitude_cov', 0.0),
            'open_close_period_mean': performance_features.get('period_mean', 0.0),
            'open_close_period_pct_90': performance_features.get('period_pct_90', 0.0),
            'open_close_period_cov': performance_features.get('period_cov', 0.0),
            'open_close_vel_pos_mean': performance_features.get('velocity_pos_mean', 0.0),
            'open_close_vel_pos_pct_90': performance_features.get('velocity_pos_pct_90', 0.0),
            'open_close_vel_pos_cov': performance_features.get('velocity_pos_cov', 0.0),
            'open_close_vel_neg_mean': performance_features.get('velocity_neg_mean', 0.0),
            'open_close_vel_neg_pct_90': performance_features.get('velocity_neg_pct_90', 0.0),
            'open_close_vel_neg_cov': performance_features.get('velocity_neg_cov', 0.0),
            # correctness of opening & closing (completeness of movement)
            'open_close_extension_score': extension_score,
            'open_close_flexion_score': flexion_score,
            # synchronization of finger movement using temporal dispersion
            'open_close_sync': synchronization_score
        }

        # plot visualization
        if save_plots and performance_features['extraction_status'] == 'success':

            self.viz.viz_repetitive_binary_exercises(time_axis=performance_features['time_axis'],
                                                     signal=performance_features['signal_original'],
                                                     features=performance_features,
                                                     p_id=p_id,
                                                     visit_id=exercise.visit_id,
                                                     ex_id=f'{exercise.exercise_id}_{exercise.side_condition}')

        return results

    def analyze_pronation_supination(self, exercise: Exercise, p_id: str, save_plots: bool = False) -> dict:

        # 1) extract the exercise-specific metric

        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # load the cleaned DataFrame directly
        try:
            df: pd.DataFrame = exercise.load_dataframe('clean')
        except FileNotFoundError:
            print(f'Error: Clean data not found for {exercise.exercise_id}. Skipping.')
            return {}

        # assuming that the y-axis is the longitudinal axis of the forearm
        euler_x, euler_y, euler_z = self.tb.calculate_3d_hand_rotation(df, active_side_idx)

        # select exercise-specific config (if not defined, fall back to default self.peak_cfg)
        ex_peak_cfg: dict = config['pro_sup'].get('peak_cfg', self.peak_cfg)

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict = self.kf.calc_kinematic_parameters(euler_y, ex_peak_cfg)

        # 1.2) correctness of the rotation (completeness of movement)

        # get peak and valley values of the pronation-supination movement
        tot_active_rom = active_dist_dict.get('raw_amplitudes', np.array([]))

        # get config thresholds for valid opening and closing
        tot_active_rom_thresh: float = config['pro_sup'].get('total_active_rom', 140.0)

        # calculate pronation and supination scores
        active_rom_score = (np.sum(tot_active_rom > tot_active_rom_thresh) / len(tot_active_rom) * 100) if len(tot_active_rom) > 0 else 0.0

        # 1.3) quality: assess rhythm with CoV (period time between rotations), stability (out-of-plane compensation)
        # 1.3.1) calculate the rotation rhythm with CoV
        rotation_cov: float = active_dist_dict.get('period_cov', 0.0)

        # 1.3.2) calculate the rotation stability - the larger the values of the other euler angles, the lower the score
        x_stability_score: float = float(np.std(euler_x))
        z_stability_score: float = float(np.std(euler_z))

        # the total out-of-plane compensation movement during the pronation-supination exercise
        total_comp_movement: float = x_stability_score + z_stability_score

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        # 5) Create result dictionary

        # flatten the performance metrics of the active finger
        performance_features = active_dist_dict

        # add exercise specific prefix-key 'pro_sup'
        results = {
            # time series
            'pro_sup_time_series_y': performance_features.get('signal_original', []),
            'pro_sup_time_series_x': performance_features.get('time_axis', []),
            # general kinematic features
            'pro_sup_rep_num': performance_features.get('repetition_num', 0.0),
            'pro_sup_rep_freq': performance_features.get('repetition_freq', 0.0),
            'pro_sup_amp_mean': performance_features.get('amplitude_mean', 0.0),
            'pro_sup_amp_pct_90': performance_features.get('amplitude_pct_90', 0.0),
            'pro_sup_amp_cov': performance_features.get('amplitude_cov', 0.0),
            'pro_sup_period_mean': performance_features.get('period_mean', 0.0),
            'pro_sup_period_pct_90': performance_features.get('period_pct_90', 0.0),
            'pro_sup_period_cov': performance_features.get('period_cov', 0.0),
            'pro_sup_vel_pos_mean': performance_features.get('velocity_pos_mean', 0.0),
            'pro_sup_vel_pos_pct_90': performance_features.get('velocity_pos_pct_90', 0.0),
            'pro_sup_vel_pos_cov': performance_features.get('velocity_pos_cov', 0.0),
            'pro_sup_vel_neg_mean': performance_features.get('velocity_neg_mean', 0.0),
            'pro_sup_vel_neg_pct_90': performance_features.get('velocity_neg_pct_90', 0.0),
            'pro_sup_vel_neg_cov': performance_features.get('velocity_neg_cov', 0.0),
            # correctness of pronation & supination(completeness of movement)
            'pro_sup_active_rom_score': active_rom_score,
            # rhythm with CoV (period time between rotations)
            'pro_sup_rot_cov': rotation_cov,
            # rotation stability
            'pro_sup_isolation': total_comp_movement,
        }

        # plot visualization
        if save_plots and performance_features['extraction_status'] == 'success':

            self.viz.viz_repetitive_binary_exercises(time_axis=performance_features['time_axis'],
                                                     signal=performance_features['signal_original'],
                                                     features=performance_features,
                                                     p_id=p_id,
                                                     visit_id=exercise.visit_id,
                                                     ex_id=f'{exercise.exercise_id}_{exercise.side_condition}')

        return results
