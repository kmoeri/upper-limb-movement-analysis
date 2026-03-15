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

    def analyze_finger_tapping(self, exercise: Exercise):

        # 1) extract the exercise-specific metric

        # 2) extract general metrics (e.g., task impairment by spectrogram)

        # 3) associated reactions (passive hand movement -> mirror movement)

        # 4) spasticity/cramping (dynamic behavior of affected side while active or passive)

        pass

    def analyze_finger_alternation(self, exercise: Exercise):

        # 1) extract the exercise-specific metric
        tb = ToolBox(fps=self.fps)
        kf = KinematicFeatures(fps=self.fps)

        active_side_idx = 1 if exercise.side_focus == 'L' else 2
        passive_side_idx = 2 if active_side_idx == 1 else 1

        # 1.1) performance: extract amplitude, period time, velocity, etc. (using peak detection)
        def _calc_performance_metrics(landmark_data: dict, side_idx: int) -> dict:
            """
            Calculates Euclidean distances between the thumb and each fingertip,
            normalizes by hand size, and extracts peak-based kinematic features.

            Args:
                landmark_data (dict): Dictionary of 3D landmark coordinate arrays.
                side_idx (int): 1 for Left side, 2 for Right side.

            Returns:
                alt_performance_dict (dict): Nested dictionary containing the normalized distance arrays
                                             and extracted kinematic features for each finger pair.
            """
            landmark_base_names = config['ftap_alter']['landmark_names']
            lm_corr_names: list = [f'{x[:-1]}{side_idx}{x[-1]}' for x in landmark_base_names]

            alt_performance_dict = {}
            thumb_point_lst: list = landmark_data[lm_corr_names[0]]
            for lm_name in lm_corr_names[1:]:
                # extract the point lists from the dictionary
                curr_point_lst: list = landmark_data[lm_name]
                # calculate the Euclidean distance for each finger
                curr_euclidean_dist: np.ndarray = tb.calc_euclidean_dist(thumb_point_lst, curr_point_lst)

                # participant-specific anatomical normalization
                ref_hand_size: float = exercise.left_hand_size if side_idx == 1 else exercise.right_hand_size
                norm_dist_arr: np.ndarray = curr_euclidean_dist / ref_hand_size

                # extract peaks
                adaptive_peak_cfg: dict = {'min_segment_length': 0.05, 'min_peak_amp_diff': 0.15,
                                           'min_valley_amp_diff': 0.15, 'max_peak_amp_diff': None,
                                           'max_valley_amp_diff': None, 'prominence_factor': 0.35,
                                           'distance_factor': 0.35}

                feature_dict: dict = kf.calc_kinematic_parameters(norm_dist_arr, adaptive_peak_cfg)

                # create new dict key and add Euclidean distance and peaks
                finger_key = f'{lm_corr_names[0]}-{lm_name}'
                alt_performance_dict[finger_key] = {'normalized_distance': norm_dist_arr,
                                                    'features': feature_dict}

            return alt_performance_dict

        # calculate basic metrics for the active and passive side
        active_dist_dict: dict = _calc_performance_metrics(exercise.clean_hand_landmarks, active_side_idx)
        passive_dist_dict: dict = _calc_performance_metrics(exercise.clean_hand_landmarks, passive_side_idx)

        # 1.2) correctness of the tapping order (extract tapping sequence and score with Levenshtein Distance)
        def _extract_tapping_sequence(active_alt_tapping_dict: dict) -> list:
            """
            Aggregates the identified taps (valleys) across all fingers into a single chronological sequence
            to evaluate tapping coordination.

            Args:
                active_alt_tapping_dict (dict): Dictionary containing the extracted features for the active hand.

            Returns:
                finger_tapping_idc_lst (list): A list of tuples ordered by time, e.g., [(2, 45), (3, 89)], where the
                                               first element is the finger ID and the second is the frame index.
            """
            # extract tapping idc list for each finger digit
            finger_tapping_idc_lst: list = []
            for finger_key, val in active_alt_tapping_dict.items():
                tapping_idc_lst: list = [(int(finger_key[-1]), x) for x in val['features']['valid_valleys_idx']]
                finger_tapping_idc_lst += tapping_idc_lst

            # sort the indices of each finger by the second tuple element (index number): [(1, '153'), (2, '153'), ...]
            finger_tapping_idc_lst = sorted(finger_tapping_idc_lst, key=lambda x: x[1])

            return finger_tapping_idc_lst

        def _calculate_tapping_accuracy(source_sequence: list) -> dict:
            """
            Finds the optimal Levenshtein sequence match for varying repetition lengths.

            Args:
                source_sequence (list): The sequence of taps the participant actually performed.

            Returns:
                dict: The minimal errors, maximum accuracy, and how many intended taps were matched.
            """
            pat_len = len(source_sequence)

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
            min_search_len = max(1, pat_len // 2)   # x0.5 the taps if every finger was double-tapped
            max_search_len = int(pat_len * 1.5)     # x1.5 the taps if there are many skipped fingers

            # check every possible target length from the min to max window
            for test_len in range(min_search_len, max_search_len + 1):

                # slice the perfect sequence to this test length
                test_target = oversized_target[:test_len]

                # calculate the Levenshtein distance errors
                errors = editdistance.eval(test_target, source_sequence)

                # normalize to an accuracy percentage
                max_possible_errors = max(test_len, pat_len)
                accuracy = ((max_possible_errors - errors) / max_possible_errors) * 100

                # keep the window that aligns the best and the corresponding accuracy and errors
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_errors = errors
                    best_target_len = test_len

            return {'errors': best_errors,
                    'accuracy': round(best_accuracy, 2),
                    'intended_target_len': best_target_len}

        # get the tapping sequence of the active hand
        tapping_sequence_tuple_lst: list = _extract_tapping_sequence(active_dist_dict)

        # extract the finger digit sequence
        tapping_sequence_digit_lst: list = [x[0] for x in tapping_sequence_tuple_lst]

        tapping_results: dict = _calculate_tapping_accuracy(tapping_sequence_digit_lst)

        # 1.3) quality: assess rhythm with CoV (period time between taps), isolation (variance of other fingers)

        # extract the idc of the tapping sequence
        tapping_sequence_idc_lst: list = [x[1] for x in tapping_sequence_tuple_lst]

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

        for active_finger_id, tap_idx in tapping_sequence_tuple_lst:

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
