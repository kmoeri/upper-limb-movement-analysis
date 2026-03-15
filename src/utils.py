# src/utils.py

# libraries
# standard
import os
import numpy as np
import pandas as pd

# preprocessing
from hampel import hampel
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.signal import savgol_filter
from scipy.signal import ShortTimeFFT
from scipy.spatial.transform import Rotation

# modules
from src.config import config
from src.core import Participant
from src.core import Exercise


class ToolBox:
    def __init__(self, video_width: int = config['camera_param']['width'],
                 video_height: int = config['camera_param']['height'],
                 fps: float = config['camera_param']['fps']):
        """
        Constructor to initialize class variables.

        Args:
            video_width (int, optional): Video image width. Defaults to 2064.
            video_height (int, optional): Video image height. Defaults to 1544.
            fps (float, optional): Video framerate. Defaults to 90.0.
        """
        self.video_width: int = video_width
        self.video_height: int = video_height
        self.fps: float = fps

    @staticmethod
    def shift_origin_to_shoulders(landmarks_dict: dict, shoulder_name_l: str, shoulder_name_r: str) -> dict:
        # TODO: docstring
        # calculate the center between both shoulders
        center_x: np.ndarray = (landmarks_dict[shoulder_name_l][0] + landmarks_dict[shoulder_name_r][0]) / 2.0
        center_y: np.ndarray = (landmarks_dict[shoulder_name_l][1] + landmarks_dict[shoulder_name_r][1]) / 2.0
        center_z: np.ndarray = (landmarks_dict[shoulder_name_l][2] + landmarks_dict[shoulder_name_r][2]) / 2.0

        # subtract the calculated center from all landmarks
        for landmark_name in landmarks_dict.keys():
            landmarks_dict[landmark_name][0] -= center_x
            landmarks_dict[landmark_name][1] -= center_y
            landmarks_dict[landmark_name][2] -= center_z

        return landmarks_dict

    @staticmethod
    def snap_hands_to_pose(landmarks_dict: dict, pose_wrist_name: str, hand_wrist_name: str,
                           target_hand_landmarks: list) -> dict:
        # TODO: docstring
        # calculate the spatial offset between the pose wrist and the hand wrist landmark
        offset_x: np.ndarray = landmarks_dict[pose_wrist_name][0] - landmarks_dict[hand_wrist_name][0]
        offset_y: np.ndarray = landmarks_dict[pose_wrist_name][1] - landmarks_dict[hand_wrist_name][1]
        offset_z: np.ndarray = landmarks_dict[pose_wrist_name][2] - landmarks_dict[hand_wrist_name][2]

        # this offset is added to every landmark of the hand model to snapp the wrist landmarks on top of each other
        for landmark_name in target_hand_landmarks:
            if landmark_name in landmarks_dict:
                landmarks_dict[landmark_name][0] += offset_x
                landmarks_dict[landmark_name][1] += offset_y
                landmarks_dict[landmark_name][2] += offset_z

        return landmarks_dict

    def normalize_to_aspect_ratio(self, raw_landmarks: dict) -> dict:
        """
        Converts normalized MediaPipe landmark coordinates to pixel space coordinates (3D coordinates).
        Corrects via the aspect ratio of the camera image.
        Do not apply this conversion to MediaPipe's world coordinate landmarks, as it is not needed.

        Args:
            raw_landmarks (dict): Dictionary of MediaPipe (normalized) landmarks (3D coordinates).

        Returns:
            pixel_landmarks (dict): Dictionary of aspect ratio corrected pixel space landmarks (3D coordinates).
        """

        pixel_landmarks: dict = dict()

        for label in raw_landmarks.keys():

            # get list with all three axes
            landmark_lst: list = raw_landmarks[label]

            # handle landmarks that do not have exactly 3 coordinates (x, y, z)
            if len(landmark_lst) != 3:
                print(f'Warning: Expected 3 coordinate arrays. '
                      f'Landmark {label} has {len(landmark_lst)} landmark coordinates.')
                continue

            # do not process labels without data
            if len(landmark_lst[0]) == 0:
                print(f'Warning: Landmark {label} has no landmark coordinates.')
                continue

            # correct the data in each axes
            x_pixel: np.ndarray = landmark_lst[0] * self.video_width
            y_pixel: np.ndarray = landmark_lst[1] * self.video_height
            z_pixel: np.ndarray = landmark_lst[2] * self.video_width

            # store pixel landmarks
            pixel_landmarks[label] = [x_pixel, y_pixel, z_pixel]

        return pixel_landmarks

    def filter_landmarks(self, landmarks_dict: dict) -> dict:
        """
            Preprocess ndarray time series of landmark data in three distinct stages:
            - Hampel: detects outliers (large jumps) and replaces them with a local median.
            - Kalman: smooths signal, predicts values for missing data, and provides better estimates.
            - Savitzky-Golay: clean-up residual noise/jitter

            Args:
                landmarks_dict (dict): Dictionary of normalized landmarks (3D coordinates).

            Returns:
                data_processed_df (pd.DataFrame): Processed motion data.
                max_nan_gap_overall (int): the maximum number of consecutive nans in the entire DataFrame.
            """

        # configurations
        MAX_GAP_THRESHOLD: int = config['preprocessing']['max_gap_threshold']
        DT: float = 1/self.fps

        def _max_repeated_nan(arr: np.ndarray) -> int:
            """
            Calculates the maximum gap (number of consecutive) NaN values in an array.

            Args:
                arr (np.ndarray): 1D array.

            Returns:
                gap (int): maximum count of consecutive nans.
            """
            mask = np.concatenate(([False], np.isnan(arr), [False]))
            if ~mask.any():
                return 0
            else:
                idx = np.nonzero(mask[1:] != mask[:-1])[0]
                gap: int = (idx[1::2] - idx[::2]).max()
                return gap

        def _const_acc_kalman_filter(data_arr: np.ndarray, dt: float,
                                     Q_scale: float = 100.0, R_scale: float = 0.0001) -> np.ndarray:
            """
            Applies a Constant Acceleration Kalman Filter using FilterPy with NaN handling.

            Args:
                data_arr: 1D array of position measurements (position_x or position_y).
                dt: Time step (1 / FRAMERATE).
                Q_scale: Scaling factor for Process Noise (Q). High Q = more responsiveness.
                R_scale: Scaling factor for Measurement Noise (R). Low R = trust the measurement.

            Returns:
                np.ndarray: The filtered position estimates.
            """

            # find the index and value of the first valid measurement for initialization
            valid_indices = np.where(~np.isnan(data_arr))[0]
            if not valid_indices.size:
                return np.full_like(data_arr, np.nan)

            # state initialization
            dim_x = 3   # pos, vel, acc
            dim_z = 1   # only position is measured

            # set the starting position at first valid value
            start_idx = valid_indices[0]
            initial_pos = data_arr[start_idx]

            # initialize the Kalman Filter object
            kf: KalmanFilter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

            # initial state (x): start with position, zero velocity, zero acceleration
            kf.x = np.array([[initial_pos], [0.], [0.]])

            # initial covariance (P): small initial confidence
            kf.P = np.eye(dim_x) * 1000.0

            # measurement function (H): only measure position
            kf.H = np.array([[1., 0., 0.]]) # classic H matrix

            # measurement noise (R): low R_scale tuning for trusting the measurement
            kf.R = np.array([[R_scale]])

            # transition matrix (F): constant acceleration model
            kf.F = np.array([[1., dt, 0.5 * dt ** 2],
                             [0., 1., dt],
                             [0., 0., 1.]])

            # process noise (Q): high Q_scale tuning for responsiveness
            # Q_discrete_white_noise generates a numerically stable Q matrix for kinematic models
            kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_scale, block_size=1)

            # Filtering loop
            # array to store the filtered position estimates
            filtered_pos_estimates = np.zeros(len(data_arr))

            for i in range(len(data_arr)):
                z_k = data_arr[i]

                # 1) predict step (always runs)
                kf.predict()

                # 2) update step (only runs when measurement is valid)
                if not np.isnan(z_k):
                    # measurement is a 1x1 array for the measured position
                    kf.update(np.array([[z_k]]))

                # store the current filtered position estimate
                filtered_pos_estimates[i] = kf.x[0, 0]

            return filtered_pos_estimates

        # initialize max gap count variable
        filtered_landmarks: dict = {}

        # iterate over items (key, value)
        for landmark_name, landmark_axes in landmarks_dict.items():

            landmark_data_lst: list = []      # reset list
            is_gap_exceeded: bool = False     # gap safety flag

            # check for gaps
            for landmark_axis in landmark_axes:
                if _max_repeated_nan(landmark_axis) > MAX_GAP_THRESHOLD:
                    is_gap_exceeded = True

            # handle axes with gaps
            if is_gap_exceeded:
                print(f'Landmark {landmark_name} was flagged as invalid due to missing data > {MAX_GAP_THRESHOLD}.')
                nan_arr: np.ndarray = np.full_like(landmark_axes[0], np.nan)
                filtered_landmarks[landmark_name] = [nan_arr, nan_arr, nan_arr]
                continue

            # preprocessing loop
            for data_arr in landmark_axes:

                # interpolate gaps for Hampel filter
                data_series: pd.Series = pd.Series(data_arr)
                interp_data: np.ndarray = data_series.interpolate(method='linear', limit_direction='both').to_numpy()

                # Hampel filtering
                try:
                    hampel_filt_arr: hampel.hampel = hampel(interp_data, config['preprocessing']['hampel_window_size'],
                                                            config['preprocessing']['hampel_n_sigma']).filtered_data
                except Exception:
                    hampel_filt_arr: np.ndarray = interp_data

                # Kalman filtering
                kalman_filt_arr: np.ndarray = _const_acc_kalman_filter(hampel_filt_arr, DT,
                                                                       config['preprocessing']['kalman_Q_scale'],
                                                                       config['preprocessing']['kalman_R_scale'])

                # Savitzky-Golay filtering
                sg_filt_arr: np.ndarray = savgol_filter(kalman_filt_arr,
                                                        config['preprocessing']['savgol_window_length'],
                                                        config['preprocessing']['savgol_polyorder'])

                landmark_data_lst.append(sg_filt_arr)

            filtered_landmarks[landmark_name] = landmark_data_lst

        return filtered_landmarks

    def determine_best_hand_reference(self, p: Participant) -> dict:
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
                              framerate: float = self.fps) -> float:
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
                if label == 'frame':  # do not extract the column named 'frame' (usually first)
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
                active_hand_link_lst: list = link_lst[:len(link_lst) // 2]
                passive_hand_link_lst: list = link_lst[len(link_lst) // 2:]
                active_hand_size_link_lst: list = l_hand_size_link_lst
                passive_hand_size_link_lst: list = r_hand_size_link_lst

            else:
                passive_hand = 'L'
                active_hand_link_lst: list = link_lst[len(link_lst) // 2:]
                passive_hand_link_lst: list = link_lst[:len(link_lst) // 2]
                active_hand_size_link_lst: list = r_hand_size_link_lst
                passive_hand_size_link_lst: list = l_hand_size_link_lst

            # relax the threshold for more severe spasticity affecting the hand
            thresh_lst = config['preprocessing']['thresh_lst']
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
                                                         framerate=self.fps)

                if active_med_hand_size > 0.0:
                    break

            # passive side
            for thresh in thresh_lst:
                passive_med_hand_size = get_med_hand_size(p.exercises[ex_key],
                                                          target_side=passive_hand,
                                                          hand_specific_link_lst=passive_hand_link_lst,
                                                          hand_size_link_lst=passive_hand_size_link_lst,
                                                          occlusion_threshold=thresh,
                                                          framerate=self.fps)

                if passive_med_hand_size > 0.0:
                    break

                if active_med_hand_size == 0.0 or passive_med_hand_size == 0.0:
                    print(
                        f'Warning: Exercise "{ex_key}" of participant {p.pid} for visit {p.visit_id} yielded no median for either hand size.')

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

    @staticmethod
    def calculate_3d_segment_lengths(landmarks_df: pd.DataFrame, landmark_link_lst: list[list[str]]) -> pd.DataFrame:
        """
        Calculates the 3D segment lengths for all frames in the DataFrame using vectorized operations.

        Args:
            landmarks_df (pd.DataFrame): dataframe containing 'landmark_x', 'landmark_y', 'landmark_z' landmarks.
            landmark_link_lst (list): List of connected landmark pairs, e.g., [['wrist1', 'cmc11'], ...].

        Returns:
            pd.DataFrame: A new DataFrame with columns for each segment length.
        """

        segment_len_dict: dict = {}

        # run for each body part combination (list) in segment list
        for bp1_name, bp2_name in landmark_link_lst:

            segment_name = f'{bp1_name}-{bp2_name}'

            # extract landmark coordinate columns (x, y, z) from landmark dataframe (movement data)
            try:
                # extract x, y, and z coordinates for both landmarks across all rows (frames)
                p1 = landmarks_df[[f'{bp1_name}_x', f'{bp1_name}_y', f'{bp1_name}_z']].values
                p2 = landmarks_df[[f'{bp2_name}_x', f'{bp2_name}_y', f'{bp2_name}_z']].values
            except KeyError as e:
                print(f'Warning: missing column {e} for segment {segment_name}. Skipping.')
                continue

            # subtraction (diffs is an N x 3 array, where N is the number of frames)
            diffs = p1 - p2

            # squared difference
            diffs_squared = diffs ** 2

            # Euclidean distance (L2 norm)
            lengths = np.sqrt(np.sum(diffs_squared, axis=1))

            # store the calculated lengths (an array of length N)
            segment_len_dict[segment_name] = lengths

        return pd.DataFrame(segment_len_dict, index=landmarks_df.index)

    def init_short_time_fft(self) -> ShortTimeFFT:

        # configuration
        fs = self.fps                                               # sampling rate [Hz]
        nperseg = config['parameter_extraction']['nperseg']         # number of points per segment
        noverlap = config['parameter_extraction']['noverlap']       # number of overlapping points
        mfft = config['parameter_extraction']['mfft']               # number of FFT points
        window = config['parameter_extraction']['window']           # window to smooth signal edges

        # initialize object with configuration parameters
        SFT = ShortTimeFFT.from_window(window, fs=fs, nperseg=nperseg, noverlap=noverlap, mfft=mfft,
                                       fft_mode='onesided', scale_to='magnitude', phase_shift=None)

        return SFT

    def adaptive_stft_filter(self, signal: np.ndarray) -> np.ndarray:

        # detrending the signal (remove spatial offset)
        signal_det = signal - np.mean(signal)

        # initialize the stft class object
        SFT = self.init_short_time_fft()

        # calculate the complex stft
        Sx_complex = SFT.stft(signal_det)

        # calculate the power from the complex numbers by squaring the magnitude
        Sxx_power = np.abs(Sx_complex) ** 2

        # upper and lower bound
        upper_limit = np.percentile(Sxx_power, config['parameter_extraction']['vmax_percentile'])
        n_order_mag: int = config['parameter_extraction']['vmin_factor']
        lower_limit = upper_limit * pow(1e1, (-1)*n_order_mag)

        # mask values above the noise floor (lower limit) as 'True' and the rest below as 'False'
        spectral_mask = Sxx_power > lower_limit

        # adaptive filtering by multiplication with the mask: noise (below lower_limit) becomes 0 + 0j
        Sx_filtered_complex = Sx_complex * spectral_mask

        # reconstruct the timeseries 'landmark_arr' using inverse stft
        signal_filtered_detrended = SFT.istft(Sx_filtered_complex, k1=len(signal))  # k1: trims to signal length

        # add the previously remove spatial mean -> move signal back to its previous spatial coordinate
        signal_filtered_final = signal_filtered_detrended + np.mean(signal)

        return signal_filtered_final

    @staticmethod
    def calc_euclidean_dist(landmark_a: list, landmark_b: list) -> np.ndarray:

        # create a point vector and transpose (row vec -> column vec)
        thumb_point_vec = np.array([landmark_a]).T
        finger_point_vec = np.array([landmark_b]).T

        # calculate the Euclidean difference
        diff_vec = thumb_point_vec - finger_point_vec
        euclidean_dist = np.linalg.norm(diff_vec, axis=1)

        return euclidean_dist

    @staticmethod
    def get_descriptive_stats(data: np.ndarray, prefix: str = ''):
        """
        Calculates descriptive statistics - normal version (mean, standard deviation, min, max).

        Args:
            data (np.ndarray): A numpy array holding values of an extracted parameter (e.g., amplitudes)
            prefix (str, optional): The prefix used to name the metric. Defaults to ''.

        Returns:
            dict: Dictionary with descriptive statistics.
        """

        if len(data) == 0:
            return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0, f'{prefix}_min': 0.0, f'{prefix}_max': 0.0}
        return {
            f'{prefix}_mean': round(float(np.mean(data)), 3),
            f'{prefix}_std': round(float(np.std(data)), 3),
            f'{prefix}_min': round(float(np.min(data)), 3),
            f'{prefix}_max': round(float(np.max(data)), 3)
        }

    @staticmethod
    def get_descriptive_stats_short(data: np.ndarray, prefix: str = ''):
        """
        Calculates descriptive statistics - short version (mean and standard deviation).

        Args:
            data (np.ndarray): A numpy array holding values of an extracted parameter (e.g., amplitudes)
            prefix (str, optional): The prefix used to name the metric. Defaults to ''.

        Returns:
            dict: Dictionary with descriptive statistics.
        """

        if len(data) == 0:
            return {f'{prefix}_mean': 0.0, f'{prefix}_std': 0.0}
        return {
            f'{prefix}_mean': round(float(np.mean(data)), 3),
            f'{prefix}_std': round(float(np.std(data)), 3),
        }

    def calculate_3d_hand_rotation(self, landmarks_dict: dict) -> np.ndarray:

        def _get_wrist_coordinate_system(landmarks_dict: dict) -> tuple:
            """
                Calculates the coordinate system of the wrist by spanning a triangular surface:
                wrist -> index finger mcp -> little finger mcp <- wrist.
                The x-axis is aligned from the wrist distal to the center between index mcp and little finger mcp.
                The z-axis vector is represented by the normal of the back of the hand.
                The y-axis results from the cross product of the other two vectors and is directed medially.

                Args:
                    landmarks_dict (dict): Dictionary of normalized landmarks (3D coordinates).

                Returns:
                    x_vec_norm (np.ndarray): x-axis vector of wrist.
                    y_vec_norm (np.ndarray): y-axis vector of wrist.
                    z_vec_norm (np.ndarray): z-axis vector of wrist.
                    wrist_coord (np.ndarray): mediapipe coordinate of wrist.
                    norm_x (np.ndarray): L2 distance from wrist to midpoint of mcp.
                """

            hand_of_focus: str = landmark_lst[0][-1]

            coord_suffix = ['x', 'y', 'z']
            landmark_name_lst: list = [f'{landmark}_{axis}' for landmark in landmark_lst for axis in coord_suffix]

            # landmark coordinates
            wrist_coord: np.ndarray = motion_df[
                [landmark_name_lst[0], landmark_name_lst[1], landmark_name_lst[2]]].values
            index_base_coord: np.ndarray = motion_df[
                [landmark_name_lst[3], landmark_name_lst[4], landmark_name_lst[5]]].values
            pinky_base_coord: np.ndarray = motion_df[
                [landmark_name_lst[6], landmark_name_lst[7], landmark_name_lst[8]]].values

            # get the center between index and pinky to align the x vector with the axis of the hand
            index_pinky_center_coord_out: np.ndarray = index_base_coord + 0.5 * (pinky_base_coord - index_base_coord)
            index_pinky_center_coord_in: np.ndarray = pinky_base_coord - 0.5 * (pinky_base_coord - index_base_coord)

            # local hand vectors with origin at wrist
            y_vec: np.ndarray = 0.5 * (index_pinky_center_coord_out + index_pinky_center_coord_in) - wrist_coord
            z_vec: np.ndarray = np.cross((index_base_coord - wrist_coord), (pinky_base_coord - wrist_coord))

            if hand_of_focus == '1':
                # Left Hand
                z_vec = z_vec * (-1)
                x_vec: np.ndarray = np.cross(y_vec, z_vec)
            else:
                # Right hand
                x_vec: np.ndarray = np.cross(y_vec, z_vec)

            # L2 norm / vector magnitude
            norm_x: np.ndarray = np.linalg.norm(x_vec, axis=1)
            norm_y: np.ndarray = np.linalg.norm(y_vec, axis=1)
            norm_z: np.ndarray = np.linalg.norm(z_vec, axis=1)

            # normalize by broadcasting (catch divisions by zero --> returns a vector of zeros)
            x_vec_norm: np.ndarray = np.divide(x_vec, norm_x[:, np.newaxis], out=np.zeros_like(x_vec),
                                               where=norm_x[:, np.newaxis] != 0)
            y_vec_norm: np.ndarray = np.divide(y_vec, norm_y[:, np.newaxis], out=np.zeros_like(y_vec),
                                               where=norm_y[:, np.newaxis] != 0)
            z_vec_norm: np.ndarray = np.divide(z_vec, norm_z[:, np.newaxis], out=np.zeros_like(z_vec),
                                               where=norm_z[:, np.newaxis] != 0)

            return x_vec_norm, y_vec_norm, z_vec_norm, wrist_coord, norm_y

        # get hand landmarks for wrist coordinate system calc

        pass


def infer_focus_side(df: pd.DataFrame, model_type: str = 'Hand') -> str | None:
    """
    Infers the focus side ('Left' or 'Right') by checking column names
    against known starting landmarks.

    Args:
        df (pd.DataFrame): DataFrame (either hand_df or pose_df).
        model_type (str): 'Hand' or 'Pose' to check the correct set of labels.

    Returns:
        str | None: 'Left', 'Right', or None if side cannot be determined.
    """

    # check hand model labels ('wrist1_x' or 'wrist2_x')
    if model_type.capitalize() == 'Hand':
        if 'wrist1_x' in df.columns:
            return 'Left'
        elif 'wrist2_x' in df.columns:
            return 'Right'

    # check pose model labels ('wrist_left_x' or 'wrist_right_x')
    elif model_type.capitalize() == 'Pose':
        if 'wrist_left_x' in df.columns:
            return 'Left'
        elif 'wrist_right_x' in df.columns:
            return 'Right'

    return None


def save_extracted_data_to_csv(feature_list_of_dicts: list[dict], conf_dict: dict,
                               out_dir: str, overwrite: bool = False) -> None:
    """
    Saves extracted movement parameters by updating a csv main file with features from a specific trial run.
    Each time this function is called, it adds new information to the csv main file. Content that already exists
    can be overwritten by setting the 'overwrite' parameter to 'True'.

    Args:
        feature_list_of_dicts: List of dicts containing features from a specific trial.
        conf_dict: Dictionary containing configuration information.
        out_dir: Directory where 'extracted_movement_features.csv' is stored.
        overwrite: If 'True', overwrites existing data for the matching IDs. If 'False', only updates empty rows

    Returns:
        None
    """

    # csv main file
    out_path = os.path.join(out_dir, 'extracted_movement_features.csv')

    n_participants: int = conf_dict['participant_info']['n_participants']   # 20
    n_trials: int = conf_dict['participant_info']['n_trials']               # 10
    age: list[list] = conf_dict['participant_info']['age']

    info_col_names: list[str] = ['p_ID', 't_ID', 'f_exists', 'cam_ID', 'f_path', 'side']
    params_col_names: list[str] = ['repetition_freq', 'num_repetitions',
                                   'period_mean', 'period_std', 'period_min', 'period_max',
                                   'amplitude_mean', 'amplitude_std', 'amplitude_min', 'amplitude_max',
                                   'velocity_pos_mean', 'velocity_pos_std', 'velocity_neg_mean', 'velocity_neg_std']

    all_col_names: list[str] = info_col_names + params_col_names

    # load csv main file - load with str dtype to maintain leading zeros
    if os.path.exists(out_path):
        main_df = pd.read_csv(out_path, dtype={'t_ID': str, 'p_ID': str})

    # create a new main dataframe
    else:
        # create rows for participant ID (participant * number_of_trials)
        unique_p_ID_lst: list[str] = ['P{:03d}'.format(i) for i in range(1, n_participants + 1)]
        final_p_ID_arr: np.ndarray = np.array([x for x in unique_p_ID_lst for _ in range(n_trials)])

        # create rows for trial ID (trials 01 - 10)
        final_t_ID_arr: np.ndarray = np.array(['{:02d}'.format(i) for i in range(1, n_trials + 1)] * n_participants)

        # create a template dataframe with all possible participant trial combinations
        main_df: pd.DataFrame = pd.DataFrame({'p_ID': final_p_ID_arr, 't_ID': final_t_ID_arr})

        # initialize all columns with np.nan
        for col in all_col_names:
            if col not in main_df.columns:
                main_df[col] = np.nan

        # flag handling overwrites
        main_df['f_exists'] = 'no'

    # This allows them to hold both np.nan AND strings without generating a warning.
    for col in info_col_names:
        if col in main_df.columns:
            main_df[col] = main_df[col].astype(object)

    # main_df is indexed by IDs for updating functionality
    main_df.set_index(['p_ID', 't_ID'], inplace=True)

    # handle empty list
    if not feature_list_of_dicts:
        print("No features provided in input list. Skipping update.")
        return

    # load passed list of dicts as dataframe
    new_param_df = pd.DataFrame(feature_list_of_dicts)

    # extract participant IDs and trial IDs from filenames
    new_param_df['fname'] = new_param_df['f_path'].apply(os.path.basename)

    # use regex to identify the respective pattern (p_ID: P and 3 digits)
    new_param_df['p_ID'] = new_param_df['fname'].str.extract(r'(P\d{3})')

    # use regex to identify the respective pattern (t_ID: WT and 2 digits following a hyphen)
    new_param_df['t_ID'] = new_param_df['fname'].str.extract(r'WT-(\d{2})')

    # use regex to identify the respective pattern (cam_ID: 'cam' followed by one capital letter)
    new_param_df['cam_ID'] = new_param_df['fname'].str.extract(r'(cam[A-Z])')

    # even trial ids indicate exercises performed with the healthy side; odd: affected side
    new_param_df['side'] = new_param_df['t_ID'].astype(int).apply(lambda x: 'healthy' if x % 2 == 0 else 'affected')

    # update file status
    new_param_df['f_exists'] = 'yes'

    # filter matching columns between the new parameter dataframe and the main dataframe
    cols_to_use = [col for col in all_col_names if col in new_param_df.columns]
    new_param_df = new_param_df[cols_to_use]

    # index the new dataframe to match the main dataframe
    new_param_df.set_index(['p_ID', 't_ID'], inplace=True)

    # incoming values replace existing data in csv main file
    if overwrite:
        main_df.update(new_param_df)

    # only update rows with missing data
    else:
        # filter main dataframe for empty rows
        rows_to_update = main_df.index[main_df['f_exists'] == 'no']

        # filter new params dataframe for empty rows
        safe_new_params = new_param_df.loc[new_param_df.index.intersection(rows_to_update)]

        # notify in case new parameters were skipped due to overwrite prevention
        if len(safe_new_params) < len(new_param_df):
            skipped = len(new_param_df) - len(safe_new_params)
            print(f"Skipped {skipped} rows because data already existed there.")

        # update the main dataframe with new, non-existing data
        main_df.update(safe_new_params)

    # reset index to make p_ID and t_ID normal columns again
    main_df.reset_index(inplace=True)

    # reorder columns
    main_df = main_df[all_col_names]

    # create the path (if not existing) and save the csv file (overwrites if existing)
    os.makedirs(out_dir, exist_ok=True)
    main_df.to_csv(out_path, index=False)
    print("csv main file successfully updated.")
