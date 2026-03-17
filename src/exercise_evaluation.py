# src/analyze.py

# libraries
import os
import numpy as np
import editdistance

# modules
from src.config import config
from src.core import Exercise
from src.utils import ToolBox
from src.kinematic_features import KinematicFeatures


class ExerciseEvaluator:
    def __init__(self, fps=config['camera_param']['fps']):
        self.fps = fps
        # instantiate globally
        self.tb = ToolBox(fps=self.fps)
        self.kf = KinematicFeatures(fps=self.fps)
        # default config for adaptive peak detection
        self.peak_cfg: dict = {'min_segment_length': 0.05, 'min_peak_amp_diff': 0.15,
                               'min_valley_amp_diff': 0.15, 'max_peak_amp_diff': None,
                               'max_valley_amp_diff': None, 'prominence_factor': 0.35,
                               'distance_factor': 0.35}

    def _extract_base_kinematics(self, landmark_data: dict, ref_hand_size: float,
                                 ref_landmark_name: str, target_landmark_names: list) -> dict:
        """
        Helper function to calculate normalized Euclidean distance and extract kinematic
        peaks for arbitrary combination of reference landmark and target landmarks combinations.

        Args:
            landmark_data (dict): 3D coordinate arrays.
            ref_hand_size (float): Scalar for normalization.
            ref_landmark_name (str): The reference point (e.g., 'wrist1', 'cmc1', 'ftip1').
            target_landmark_names (list): The moving points (e.g., ['ftip2', 'ftip3']).

        Returns:
            dict: Nested dictionary containing normalized distances and feature parameters.
        """
        performance_dict = {}
        ref_point_lst = landmark_data[ref_landmark_name]

        for target_name in target_landmark_names:
            target_point_lst = landmark_data[target_name]

            # distance and normalization
            euclidean_dist = self.tb.calc_euclidean_dist(ref_point_lst, target_point_lst)
            norm_dist_arr = euclidean_dist / ref_hand_size

            # peak extraction
            feature_dict = self.kf.calc_kinematic_parameters(norm_dist_arr, self.peak_cfg)

            finger_key = f'{ref_landmark_name}-{target_name}'
            performance_dict[finger_key] = {'normalized_distance': norm_dist_arr,
                                            'features': feature_dict}

        return performance_dict

    def analyze_finger_tapping(self, exercise: Exercise):

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # participant-specific anatomical normalization
        ref_hand_size: float = exercise.left_hand_size if active_side_idx == 1 else exercise.right_hand_size

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names = config['index_ftap']['landmark_names']
        lm_corr_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names]
        anchor, target = lm_corr_names[0], lm_corr_names[1]
        pair_key: str = f'{anchor}-{target}'

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict: dict = self._extract_base_kinematics(exercise.clean_hand_landmarks, ref_hand_size,
                                                               anchor, [target])

        # 1.2) correctness of the tapping (spatial accuracy)
        valid_valley_idc: list = active_dist_dict[pair_key]['features']['valid_valley_idx']
        norm_dist_arr: np.ndarray = active_dist_dict[pair_key]['features']['normalized_distance']

        if len(valid_valley_idc) > 0:
            valid_dist: float = config['index_ftap']['min_dist_thresh']

            # Euclidean distance at each event (valley)
            valley_dist_arr: np.ndarray = norm_dist_arr[valid_valley_idc]

            # results
            mean_target_error: float = float(np.mean(valley_dist_arr))
            successful_taps: int = np.sum(np.where(valley_dist_arr < valid_dist, 1, 0))
            accuracy_percentage: float = float((successful_taps / len(valid_valley_idc)) * 100)
        else:
            mean_target_error: float = 0.0
            successful_taps: int = 0
            accuracy_percentage: float = 0.0

        # 1.3) quality: assess rhythm with CoV (period time between taps), isolation (variance of other fingers)

        # 1.3.1) calculate the tapping rhythm with CoV
        if len(valid_valley_idc) > 1:
            tapping_diff: np.ndarray = np.diff(valid_valley_idc)
            tapping_mean: float = float(np.mean(tapping_diff))
            tapping_std: float = float(np.std(tapping_diff))
            tapping_cov: float = float((tapping_std / tapping_mean) * 100) if tapping_mean > 0 else 0.0

            # dynamic window for isolation (33% of a period)
            half_window: int = max(int(tapping_mean / 6.0), 2)
        else:
            tapping_mean: float = 0.0
            half_window: int = int(self.fps * 0.05)

        # 1.3.2) calculate the tapping isolation
        wrist_name: str = f'wrist{active_side_idx}'
        passive_fingers: list = lm_corr_names[2:]

        # calculate Euclidean distance for passive fingers
        passive_kinematics: dict = self._extract_base_kinematics(exercise.clean_hand_landmarks, ref_hand_size,
                                                                 wrist_name, passive_fingers)

        # calculate the isolation variances
        isolation_variances: list = []
        for tap_idx in valid_valley_idc:
            for passive_kin_key in passive_kinematics.keys():
                dist_arr: np.ndarray = passive_kinematics[passive_kin_key]['normalized_distance']
                start: int = max(0, tap_idx - half_window)
                end: int = min(len(dist_arr), tap_idx + half_window)
                if start < end:
                    isolation_variances.append(np.var(dist_arr[start:end]))

        isolation_score: float = float(np.mean(isolation_variances)) if isolation_variances else 0.0

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass

    def analyze_finger_alternation(self, exercise: Exercise):

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # participant-specific anatomical normalization
        ref_hand_size: float = exercise.left_hand_size if active_side_idx == 1 else exercise.right_hand_size

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names = config['ftap_alter']['landmark_names']
        lm_corr_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names]
        thumb = lm_corr_names[0]
        finger_names = lm_corr_names[1:]
        pair_key: str = f'{thumb}-{finger_names}'

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict: dict = self._extract_base_kinematics(exercise.clean_hand_landmarks, ref_hand_size,
                                                               thumb, finger_names)

        # 1.2) correctness of the tapping order (extract tapping sequence and score with Levenshtein Distance)

        # extract tapping idc list for each finger digit
        finger_tapping_idc_lst: list = []
        for finger_key, val in active_dist_dict.items():
            tapping_idc_lst: list = [(int(finger_key[-1]), x) for x in val['features']['valid_valleys_idx']]
            finger_tapping_idc_lst += tapping_idc_lst

        # sort the indices of each finger by the second tuple element (index number): [(1, '153'), (2, '153'), ...]
        finger_tapping_idc_lst = sorted(finger_tapping_idc_lst, key=lambda x: x[1])

        # extract the finger digit sequence
        tapping_sequence_digit_lst: list = [x[0] for x in finger_tapping_idc_lst]

        pat_len = len(tapping_sequence_digit_lst)

        # safety: for zero movement
        if pat_len == 0:
            return {'errors': 0, 'accuracy': 0.0, 'intended_target_len': 0}

        # target tapping sequence - single full repetition
        target_sequence = [2, 3, 4, 5, 4, 3]

        # create an oversized sequence (30 repetitions of the sequence)
        oversized_target = target_sequence * 30

        # initialize rating variables
        best_accuracy = -1.0
        best_errors = 0
        best_target_len = 0

        # define a search window for the estimated taps intended
        min_search_len = max(1, pat_len // 2)  # x0.5 the taps if every finger was double-tapped
        max_search_len = int(pat_len * 1.5)  # x1.5 the taps if there are many skipped fingers

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
            tapping_std: float = float(np.std(tapping_diff))
            tapping_cov: float = float((tapping_std / tapping_mean) * 100) if tapping_mean > 0 else 0.0
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

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass

    def analyze_hand_opening(self, exercise: Exercise):

        # 1) extract the exercise-specific metric
        # get current active side
        active_side_idx = 1 if exercise.side_focus == 'L' else 2

        # participant-specific anatomical normalization
        ref_hand_size: float = exercise.left_hand_size if active_side_idx == 1 else exercise.right_hand_size

        # modify landmark names to hold the side information (left: 1, right: 2)
        lm_base_names = config['open_close']['landmark_names']
        wrist_name: str = f'{lm_base_names[0]}{active_side_idx}'
        finger_names: list = [f'{x[:-1]}{active_side_idx}{x[-1]}' for x in lm_base_names[1:]]

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        active_dist_dict: dict = self._extract_base_kinematics(exercise.clean_hand_landmarks, ref_hand_size,
                                                               wrist_name, finger_names)

        # 1.2) correctness of the closing (completeness of movement)

        # get all peak and valley amplitudes
        all_peak_amps: list = []
        all_valley_amps: list = []
        for finger_id in active_dist_dict.keys():
            dist_arr = active_dist_dict[finger_id]['normalized_distance']
            peaks = active_dist_dict[finger_id]['features']['valid_peaks_idx']
            valleys = active_dist_dict[finger_id]['features']['valid_valleys_idx']

            all_peak_amps.extend(dist_arr[peaks])
            all_valley_amps.extend(dist_arr[valleys])

        # get config thresholds for valid opening and closing
        open_thresh = config['open_close'].get('hand_opening_thresh', 0.8)
        close_thresh = config['open_close'].get('hand_closing_thresh', 0.2)

        # calculate opening (extension) and closing (flexion) scores
        extension_score = (np.sum(np.array(all_peak_amps) > open_thresh) / len(all_peak_amps) * 100) if all_peak_amps else 0.0
        flexion_score = (np.sum(np.array(all_valley_amps) < close_thresh) / len(all_valley_amps) * 100) if all_valley_amps else 0.0

        # 1.3) quality: temporal dispersion
        dispersion_variances = []

        # index finger is the reference to solve the problem of missing events (peaks/valleys) of the other fingers
        index_key = f'{wrist_name}-{finger_names[0]}'
        index_peaks = active_dist_dict[index_key]['features']['valid_peaks_idx']

        passive_finger_keys = [f'{wrist_name}-{pf}' for pf in finger_names[1:]]
        max_lag_frames: int = int(self.fps * 0.5)   # allow max. 0.5 seconds of lag

        for index_peak in index_peaks:
            rep_timing = [index_peak]

            for passive_fkey in passive_finger_keys:
                passive_peaks = active_dist_dict[passive_fkey]['features']['valid_peaks_idx']
                if len(passive_peaks) == 0:
                    continue

                # find the closest peak between the index finger and the passive fingers
                closest_peak = min(passive_peaks, key=lambda x: abs(x - index_peak))

                # closest peak belongs to this repetition if it appears within the same half of a second
                if abs(closest_peak - index_peak) < max_lag_frames:
                    rep_timing.append(closest_peak)

            # calculate the standard deviation if all 4 fingers participated in the current repetition
            if len(rep_timing) == 4:
                dispersion_sec = np.std(rep_timing) / self.fps
                dispersion_variances.append(dispersion_sec)

        # calculate the final synchronization score (lower is better)
        synchronization_score: float = float(np.mean(dispersion_variances)) if dispersion_variances else 0.0

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass

    def analyze_pronation_supination(self, exercise: Exercise):

        # 1) extract the exercise-specific metric

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass

    def analyze_table_tapping(self, exercise):

        # 1) extract the exercise-specific metric

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass
