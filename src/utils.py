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
from plotly.graph_objs.indicator.gauge import axis
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
        #thumb_point_vec = np.array([landmark_a]).T
        #finger_point_vec = np.array([landmark_b]).T

        # calculate the Euclidean difference
        #diff_vec = thumb_point_vec - finger_point_vec
        #euclidean_dist = np.linalg.norm(diff_vec, axis=1)

        # TEMP TEST
        a_2d = np.array(landmark_a)[:2]
        b_2d = np.array(landmark_b)[:2]

        # Calculate the 2D Euclidean difference
        diff_vec = a_2d - b_2d
        euclidean_dist = np.linalg.norm(diff_vec, axis=0)

        return euclidean_dist

    @staticmethod
    def get_descriptive_stats(data: np.ndarray, prefix: str = ''):
        """
        Calculates the descriptive statistics: mean, 90th percentile, and coefficient of variation.
        - The mean captures the overall performance and is sensitive to outlier performances
        - The 90th percentile captures the biomechanical capacity (the best possible performance)
        - The CoV represents the consistency and rhythmicity

        Args:
            data (np.ndarray): A numpy array holding values of an extracted parameter (e.g., amplitudes)
            prefix (str, optional): The prefix used to name the metric. Defaults to ''.

        Returns:
            dict: Dictionary with descriptive statistics.
        """

        if len(data) == 0:
            return {f'{prefix}_mean': 0.0, f'{prefix}_pct_90': 0.0, f'{prefix}_cov': 0.0}
        return {
            f'{prefix}_mean': round(float(np.mean(data)), 3),
            f'{prefix}_pct_90': round(np.percentile(data, 90), 3),
            f'{prefix}_cov': round((float(np.std(data)) / float(np.mean(data))) if np.mean(data) != 0.0 else 0.0, 3)
        }

    @staticmethod
    def calculate_3d_hand_rotation(landmarks_dict: dict, landmark_name_lst) -> tuple:
        """
        Calculates Euler angles by accumulating relative angles between frames. Ensures shortest-path
        rotation to prevent signal rectification.

        Args:
            landmarks_dict (dict): Dictionary of landmarks (3D coordinates).
            landmark_name_lst (list): List of landmark names (e.g., "wrist1", "mcp12", "mcp15").

        Returns:
            euler_angles (tuple): Angle around x-axis, y-axis, and z-axis.
        """

        def _get_wrist_coordinate_system(lm_dict: dict, lm_name_lst: list) -> tuple:
            """
                Calculates the coordinate system of the wrist by spanning a triangular surface:
                wrist -> index finger mcp -> little finger mcp <- wrist.
                The x-axis is aligned from the wrist distal to the center between index mcp and little finger mcp.
                The z-axis vector is represented by the normal of the back of the hand.
                The y-axis results from the cross product of the other two vectors and is directed medially.

                Args:
                    lm_dict (dict): Dictionary of landmarks (3D coordinates).
                    lm_name_lst (list): List of landmark names (e.g., "wrist1", "mcp12", "mcp15").

                Returns:
                    x_vec_norm (np.ndarray): x-axis vector of wrist.
                    y_vec_norm (np.ndarray): y-axis vector of wrist.
                    z_vec_norm (np.ndarray): z-axis vector of wrist.
                    wrist_coord (np.ndarray): mediapipe coordinate of wrist.
                    norm_x (np.ndarray): L2 distance from wrist to midpoint of mcp.
                """

            # landmark coordinates
            wrist_lm: np.ndarray = lm_dict[lm_name_lst[0]]
            index_lm: np.ndarray = lm_dict[lm_name_lst[1]]
            pinky_lm: np.ndarray = lm_dict[lm_name_lst[2]]

            # get the center between index and pinky knuckle to align the x vector with the axis of the hand
            midpoint: np.ndarray = 0.5 * (index_lm + pinky_lm)

            # local hand vectors with origin at wrist
            y_vec: np.ndarray = midpoint - wrist_lm
            z_vec: np.ndarray = np.cross((index_lm - wrist_lm), (pinky_lm - wrist_lm))

            # align coordinate system depending on the current hand side
            if lm_name_lst[0][-1] == '1': # left hand
                z_vec = z_vec * (-1)

            # calculate the x_vector using the cross product of the other two vectors
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

            # also returns norm_x to identify tracking loss
            return x_vec_norm, y_vec_norm, z_vec_norm, norm_x

        # get basis vectors (y-distal, x-medial, z-palm)
        x_n, y_n, z_n, norm_x = _get_wrist_coordinate_system(landmarks_dict, landmark_name_lst)

        # stack vectors to matrix
        mats: np.ndarray = np.stack((x_n, y_n, z_n), axis=-1)

        # replace zero-matrices with the identity matrix
        invalid_mask = (norm_x == 0)
        mats[invalid_mask] = np.eye(3)

        # convert to rotation object
        rot_objs = Rotation.from_matrix(mats)

        # calculate the absolute rotation relative to the first valid frame
        valid_idc: int = np.where(~invalid_mask)[0]
        if len(valid_idc) == 0:
            return np.zeros(len(mats)), np.zeros(len(mats)), np.zeros(len(mats))

        # get the first frame with a valid tracking to define the base reference
        base_rot = rot_objs[valid_idc[0]]

        # calculate delta rotation between current and previous frame (R_delta = R_prev^-1 * R_curr)
        rel_rot_objs = base_rot.inv() * rot_objs

        # extract Euler angles: pronation-supination is around y-axis -> start with y rotation to prevent gimbal lock
        euler_angles = rel_rot_objs.as_euler('YXZ', degrees=True)

        # fix the +/- 180° jumps (jitter handling)
        # 1) np.unwrap connects angles that flip from 179 to -179
        # 2) convert angles to rad for np.unwrap and then back to degrees
        euler_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(euler_angles), axis=0))

        # create dataframe with euler angles using the correct order (YXZ)
        euler_df: pd.DataFrame = pd.DataFrame(euler_unwrapped, columns=['y', 'x', 'z'])

        # mask tracking failures
        euler_df.loc[invalid_mask] = np.nan

        # apply a velocity limit
        velocity = euler_df.diff().abs()
        limit_mask = (velocity > 38).any(axis=1)
        euler_df.loc[limit_mask] = np.nan

        # interpolate (monotonic cubic)
        euler_df = euler_df.interpolate(method='pchip', limit_direction='both')

        # extract the angle for pronation-supination and return in standard XYZ order
        euler_angles: tuple = (euler_df['x'].to_numpy(), euler_df['y'].to_numpy(), euler_df['z'].to_numpy())

        return euler_angles


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


def save_extracted_data_to_csv(feature_list_of_dicts: list[dict], out_dir: str) -> None:
    """
    Saves extracted movement parameters to a csv file with features from all exercises.
    Relies on Pandas dynamic DataFrame creation to handle varying exercise columns.

    Args:
        feature_list_of_dicts: List of dicts containing features from a specific trial.
        out_dir: Directory where 'extracted_movement_features.csv' is stored.

    Returns:
        None
    """

    if not feature_list_of_dicts:
        print('No features provided. Skipping.')
        return

    # csv main file
    out_path: str = os.path.join(out_dir, 'extracted_movement_features.csv')

    # load metric data into a dataframe
    data_df: pd.DataFrame = pd.DataFrame(feature_list_of_dicts)

    # if the file already exists load and update the file
    if os.path.exists(out_path):
        main_df: pd.DataFrame = pd.read_csv(out_path)

        combined_df: pd.DataFrame = pd.concat([data_df, main_df]).drop_duplicates(
            subset=['p_ID', 'visit_ID', 'ex_name', 'side_focus'],
            keep='first',   # keeps the new calculated data
        )
    else:
        combined_df: pd.DataFrame = data_df

    # sort table
    combined_df.sort_values(by=['p_ID', 'visit_ID', 'ex_name'], inplace=True)

    # save table
    os.makedirs(out_dir, exist_ok=True)
    combined_df.to_csv(out_path, index=False)
    print(f'Successfully saved {len(combined_df)} trial records of extracted movement parameters to {out_path}')
