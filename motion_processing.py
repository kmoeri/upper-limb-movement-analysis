# libraries
import numpy as np
import pandas as pd
from hampel import hampel
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation


def extract_pose_of_focus(motion_df: pd.DataFrame, curr_affected_side: str, trial_id: int) -> tuple:

    shoulders_df: pd.DataFrame = motion_df[motion_df.columns[1:7]]          # save shoulders as both are needed
    pose_of_focus_df: pd.DataFrame = motion_df[motion_df.columns[7:-6]]     # keep elbows and wrists

    if trial_id % 2 == 0:

        if curr_affected_side == 'L':
            # extract right side
            mask = pose_of_focus_df.columns.str.contains('right', case=False)
            pose_of_focus_str: str = 'R'

        else:
            # extract left side
            mask = pose_of_focus_df.columns.str.contains('left', case=False)
            pose_of_focus_str: str = 'L'

    else:

        if curr_affected_side == 'R':
            # extract right side
            mask = pose_of_focus_df.columns.str.contains('right', case=False)
            pose_of_focus_str: str = 'R'

        else:
            # extract left side
            mask = pose_of_focus_df.columns.str.contains('left', case=False)
            pose_of_focus_str: str = 'L'

    # filter position of focus and combine with shoulders (both sides needed)
    pose_of_focus_df = pose_of_focus_df[pose_of_focus_df.columns[mask]]
    pose_of_focus_df = pd.concat([shoulders_df, pose_of_focus_df], axis=1)

    return pose_of_focus_df, pose_of_focus_str


def extract_hand_of_focus(motion_df: pd.DataFrame, curr_affected_side: str, trial_id: int) -> tuple:

    # number of landmarks (equivalent to columns minus the frame number column)
    n_cols = motion_df.shape[1] - 1
    left_hand_df: pd.DataFrame = motion_df[motion_df.columns[1:(n_cols//2)+1]]
    right_hand_df: pd.DataFrame = motion_df[motion_df.columns[(n_cols//2):]]

    if trial_id % 2 == 0:

        if curr_affected_side == 'L':
            # extract right side
            hand_of_focus_df: pd.DataFrame = right_hand_df
            hand_of_focus_str: str = 'R'

        else:
            # extract left side
            hand_of_focus_df: pd.DataFrame = left_hand_df
            hand_of_focus_str: str = 'L'

    else:

        if curr_affected_side == 'R':
            # extract right side
            hand_of_focus_df: pd.DataFrame = right_hand_df
            hand_of_focus_str: str = 'R'

        else:
            # extract left side
            hand_of_focus_df: pd.DataFrame = left_hand_df
            hand_of_focus_str: str = 'L'

    return hand_of_focus_df, hand_of_focus_str


def const_acc_kalman_filter(data_arr: np.ndarray, dt: float, Q_scale: float = 0.1, R_scale: float = 0.01) -> np.ndarray:
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
    N = len(data_arr)

    # 1) State Initialization

    # state is [position, velocity, acceleration]
    dim_x = 3
    dim_z = 1   # only position is measured

    # find the index and value of the first valid measurement for initialization
    valid_indices = np.where(~np.isnan(data_arr))[0]

    # if the whole array is NaN, return it as is
    if not valid_indices.size:
        return np.full_like(data_arr, np.nan)

    # set the starting position at first valid value
    start_idx = valid_indices[0]
    initial_pos = data_arr[start_idx]

    # initialize the Kalman Filter object
    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # initial state (x): start with position, zero velocity, zero acceleration
    kf.x = np.array([[initial_pos], [0.], [0.]])

    # initial covariance (P): small initial confidence
    kf.P = np.eye(dim_x) * 1000.0

    # measurement function (H): only measure position
    kf.H = np.array([[1., dt, 0.5 * dt ** 2]])  # this H maps the state [p, v, a] to the measurement [p_measured]
    #kf.H = np.array([[1., 0., 0.]]) # classic H matrix

    # measurement noise (R): low R_scale tuning for trusting the measurement
    kf.R = np.array([[R_scale]])

    # transition matrix (F): constant acceleration model
    kf.F = np.array([[1., dt, 0.5 * dt ** 2],
                     [0., 1., dt],
                     [0., 0., 1.]])

    # process noise (Q): high Q_scale tuning for responsiveness
    # Q_discrete_white_noise generates a numerically stable Q matrix for kinematic models
    kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=Q_scale, block_size=1)

    # 2) Filtering Loop

    # array to store the filtered position estimates
    filtered_pos_estimates = np.zeros(N)

    for i in range(N):
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


def preprocess_motion_data(df: pd.DataFrame, framerate: float) -> tuple:
    """
    Preprocess time series of motion data with (1) Hampel filter, (2) Kalman filter, and (3) Median filter.
    - Hampel: detects outliers (large jumps) and replaces them with a local median.
    - Kalman: smooths the signal, predicts values for gaps, and provides overall better estimates for the movement signal
    - Median: removes sparse noise

    Args:
        df (pd.DataFrame): Motion data.
        framerate (float): Framerate of the motion data.

    Returns:
        data_processed_df (pd.DataFrame): Processed motion data.
        max_nan_gap_overall (int): the maximum number of consecutive nans in the entire DataFrame.
    """

    def max_repeated_nan(arr: np.ndarray) -> int:
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

    # prepare column names for new dataframe
    col_name_lst: list[str] = df.columns.tolist()
    if col_name_lst[0] == 'frame':
        col_name_lst.remove('frame')

    # initialize empty DataFrame for each filter
    raw_df: pd.DataFrame = pd.DataFrame()
    hampel_filtered_df: pd.DataFrame = pd.DataFrame()
    kalman_filtered_df: pd.DataFrame = pd.DataFrame()

    # run signal processing
    max_nan_gap_overall: int = 0
    processed_df: pd.DataFrame = pd.DataFrame()
    for col_name in col_name_lst:
        try:

            # convert to numpy
            data_arr: np.ndarray = df[col_name].values

            # update the max nan gap identified
            current_max_nan_gap: int = max_repeated_nan(data_arr)
            max_nan_gap_overall: int = max(max_nan_gap_overall, current_max_nan_gap)

            # 2) Hampel Filtering: Handling outliers (Median Absolute Deviation)
            hampel_filt_arr: hampel.hampel = hampel(data_arr, window_size=7, n_sigma=2.0).filtered_data

            # 3) Kalman Filtering:
            kalman_filt_arr: np.ndarray = const_acc_kalman_filter(hampel_filt_arr, dt=(1/framerate),
                                                                  Q_scale=100.0, R_scale=.0001)

            # 4) Savgol Filtering: clean-up residual noise/jitter
            sg_filt_arr: np.ndarray = savgol_filter(kalman_filt_arr, window_length=9, polyorder=2)

        except Exception as error:
            print(f"Error processing column {col_name}: {error}")
            # fill output for this column with NaNs
            data_arr = np.full(len(df), np.nan)
            hampel_filt_arr = np.full(len(df), np.nan)
            kalman_filt_arr = np.full(len(df), np.nan)
            sg_filt_arr = np.full(len(df), np.nan)

        # store all filter stages
        raw_df[col_name] = data_arr
        hampel_filtered_df[col_name] = hampel_filt_arr
        kalman_filtered_df[col_name] = kalman_filt_arr

        # add last filter stage to processed DataFrame
        processed_df[col_name] = sg_filt_arr

    return processed_df, max_nan_gap_overall,


def get_wrist_coordinate_system(motion_df: pd.DataFrame, landmark_lst: list[str]) -> tuple:
    """
    Calculates the coordinate system of the wrist by spanning a triangular surface:
    wrist -> index finger mcp -> little finger mcp <- wrist.
    The x-axis is aligned from the wrist distal to the center between index mcp and little finger mcp.
    The z-axis vector is represented by the normal of the back of the hand.
    The y-axis results from the cross product of the other two vectors and is directed medially.

    Args:
        motion_df (pd.DataFrame): Motion data.
        landmark_lst (list[str]): List of landmark names.

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
    wrist_coord: np.ndarray = motion_df[[landmark_name_lst[0], landmark_name_lst[1], landmark_name_lst[2]]].values
    index_base_coord: np.ndarray = motion_df[[landmark_name_lst[3], landmark_name_lst[4], landmark_name_lst[5]]].values
    pinky_base_coord: np.ndarray = motion_df[[landmark_name_lst[6], landmark_name_lst[7], landmark_name_lst[8]]].values

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
    x_vec_norm: np.ndarray = np.divide(x_vec, norm_x[:, np.newaxis], out=np.zeros_like(x_vec), where=norm_x[:, np.newaxis] != 0)
    y_vec_norm: np.ndarray = np.divide(y_vec, norm_y[:, np.newaxis], out=np.zeros_like(y_vec), where=norm_y[:, np.newaxis] != 0)
    z_vec_norm: np.ndarray = np.divide(z_vec, norm_z[:, np.newaxis], out=np.zeros_like(z_vec), where=norm_z[:, np.newaxis] != 0)

    return x_vec_norm, y_vec_norm, z_vec_norm, wrist_coord, norm_y


def calculate_3d_hand_rotation(motion_df: pd.DataFrame, landmark_lst: list) -> np.ndarray:
    """
    Calculates Euler angles by accumulating relative angles between frames. Ensures shortest-path
    rotation to prevent signal rectification.

    Args:
        motion_df (pd.DataFrame): Motion data.
        landmark_lst (list[str]): List of landmark names.

    Returns:
        euler_df (pd.DataFrame): Euler angles.
    """
    # 1. Get basis vectors (Y-distal, X-medial, Z-palm)
    x_n, y_n, z_n, _, _ = get_wrist_coordinate_system(motion_df, landmark_lst)
    frame_num = motion_df.shape[0]

    # convert to rotation object
    mats = np.stack((x_n, y_n, z_n), axis=-1)
    rot_objs = Rotation.from_matrix(mats)

    # initialize arrays
    euler_accumulated = np.zeros((frame_num, 3))

    # initialize mask to track max velocity violations
    glitch_mask = np.zeros(frame_num, dtype=bool)

    # 2. Iterate and accumulate
    for i in range(1, frame_num):
        # calculate delta rotation between current and previous frame
        # R_delta = R_prev^-1 * R_curr
        delta_rot_obj = rot_objs[i - 1].inv() * rot_objs[i]
        delta_euler = delta_rot_obj.as_euler('xyz', degrees=True)

        # shortest path correction - wrap delta to [-180, 180]
        delta_euler = (delta_euler + 180) % 360 - 180

        # check for glitch - rotations exceeding 35 degrees per frame
        if np.any(np.abs(delta_euler) > 38):
            glitch_mask[i] = True
            delta_euler = np.zeros(3)

        # ignore tiny rotations (jitter)
        # if np.all(np.abs(delta_euler) < 0.2):
        #     delta_euler = np.zeros(3)

        # Add to total angle
        euler_accumulated[i] = euler_accumulated[i - 1] + delta_euler

    # 3. Apply the interpolation
    euler_df = pd.DataFrame(euler_accumulated, columns=['x', 'y', 'z'])

    # set identified glitches to NaN (for interpolate() function)
    euler_df.loc[glitch_mask] = np.nan

    # interpolate
    euler_df = euler_df.interpolate(method='pchip', limit_direction='both')

    # extract the angle for pronation-supination
    return euler_df['y'].values

