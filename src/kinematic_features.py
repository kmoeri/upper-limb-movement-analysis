# src/kinematic_features.py

# libraries
import numpy as np
import pandas as pd
import editdistance
from scipy.stats import entropy
from scipy.signal import find_peaks, butter, sosfiltfilt

# modules
from src.config import config
from src.utils import ToolBox


class KinematicFeatures:
    def __init__(self, fps: float = config['camera_param']['fps']):
        self.fps = fps

    def calc_spectral_entropy(self, sig_active: np.ndarray, sig_passive: np.ndarray) -> dict:
        """
        Calculates the global and time-varying Spectral Shannon Entropy for both the active
        and passive limbs. Uses a shared noise floor (calibrated on the active limb) to
        suppress tracking noise and isolate the true biomechanical movement.

        Args:
            sig_active (np.ndarray): 1D array of kinematic time-series data for the active/task limb.
            sig_passive (np.ndarray): 1D array of kinematic time-series data for the passive/resting limb.

        Returns:
            dict: A dictionary containing four key-value pairs:
                - 'Spectral_Entropy_Active_OT': 1D array of entropy values over time.
                - 'Spectral_Entropy_Active_Global': Float representing the overall trial entropy.
                - 'Spectral_Entropy_Passive_OT': 1D array of entropy values over time.
                - 'Spectral_Entropy_Passive_Global': Float representing the overall trial entropy.
        """

        # get the spectral frequencies (limited to 20 Hz) for each landmark and calculate the shannon-entropy
        tb = ToolBox(fps=self.fps)

        # detrending the signal (remove spatial offset)
        sig_active_det = sig_active - np.mean(sig_active)
        sig_passive_det = sig_passive - np.mean(sig_passive)

        # initialize a short time fft object
        SFT_active = tb.init_short_time_fft()
        SFT_passive = tb.init_short_time_fft()

        # calculate the complex stft
        Sx_complex_active = SFT_active.stft(sig_active_det)
        Sx_complex_passive = SFT_passive.stft(sig_passive_det)

        # calculate the power from the complex numbers by squaring the magnitude
        Sxx_power_active = np.abs(Sx_complex_active) ** 2
        Sxx_power_passive = np.abs(Sx_complex_passive) ** 2

        # upper and lower limits
        global_ceiling = np.percentile(Sxx_power_active, config['parameter_extraction'].get('vmax_percentile', 95.0))
        n_order_mag: int = config['parameter_extraction'].get('vmin_factor', 4)
        global_floor = global_ceiling * pow(1e1, (-1) * n_order_mag)

        # set values below the global floor to zero
        power_active_clean = Sxx_power_active.copy()
        power_active_clean[power_active_clean < global_floor] = 0.0

        power_passive_clean = Sxx_power_passive.copy()
        power_passive_clean[power_passive_clean < global_floor] = 0.0

        # get 1D frequency (f) array from the SFT object
        f = SFT_active.f

        # limit frequency to 20 Hz
        f_limit = 20
        f_idx = f <= f_limit

        # slice the arrays to limit the values to 20 Hz
        Sxx_filtered_active_plot = power_active_clean[f_idx, :]
        Sxx_filtered_passive_plot = power_passive_clean[f_idx, :]

        # calculate the maximum possible entropy (number of frequency bins (rows) in the filtered array
        n_bins = Sxx_filtered_active_plot.shape[0]
        max_entropy = np.log2(n_bins) if n_bins > 1 else 1.0

        # prevent the entropy function from returning NaN if a hand is perfectly still
        def _get_safe_norm_global_entropy(power_matrix):
            """
                Calculates the normalized global spectral Shannon Entropy and total Energy for a given
                time-frequency power matrix and safely handles 'resting' conditions to prevent NaN errors.

                Args:
                    power_matrix (np.ndarray): 2D array of noice filtered STFT power values.

                Returns:
                    tuple[float, float]: A tuple containing:
                        - norm_entropy (float): The normalized spectral entropy (0.0 to 1.0).
                        - tot_energy (float): The total spectral energy across the evaluated frequency band.
                """
            # average to get a global average 1D spectrum
            mean_spectrum = np.mean(power_matrix, axis=1)

            # calculate the total energy
            tot_energy = np.sum(mean_spectrum)

            # check the energy sum
            if tot_energy == 0:
                return 0.0, 0.0                  # return 0 entropy for 0 energy
            else:
                raw_entropy = entropy(mean_spectrum, base=2)
                norm_entropy = raw_entropy / max_entropy
                return norm_entropy, tot_energy

        def _get_safe_norm_ot_entropy(power_matrix):
            """
                Calculates the normalized spectral Shannon Entropy over time (for each time window).
                Suppresses SciPy division warnings caused by empty time windows.

                Args:
                    power_matrix (np.ndarray): 2D array of cropped, noise-gated STFT power values.

                Returns:
                    np.ndarray: 1D array of normalized entropy values (0.0 to 1.0) corresponding to each time step.
                """
            # prevent division by zero warnings from SciPy
            with np.errstate(divide='ignore', invalid='ignore'):
                raw_entropy = entropy(power_matrix, axis=0, base=2)

            # replace NaNs caused by empty time windows with 0.0
            clean_raw_entropy = np.nan_to_num(raw_entropy, nan=0.0)
            norm_entropy = clean_raw_entropy / max_entropy
            return norm_entropy

        # global entropy - low number indicates highly rhythmic movement
        spect_entropy_active_global, energy_active_global = _get_safe_norm_global_entropy(Sxx_filtered_active_plot)
        spect_entropy_passive_global, energy_passive_global = _get_safe_norm_global_entropy(Sxx_filtered_passive_plot)

        # entropy value for each window - changes in entropy allow to spot characteristics such as fatigue
        spect_entropy_active_ot: np.ndarray = _get_safe_norm_ot_entropy(Sxx_filtered_active_plot)
        spect_entropy_passive_ot: np.ndarray = _get_safe_norm_ot_entropy(Sxx_filtered_passive_plot)

        entropy_dict: dict = dict({
            'Spectral_Entropy_Active_OT': spect_entropy_active_ot,
            'Spectral_Entropy_Active_Global': spect_entropy_active_global,
            'Total_Energy_Active_Global': energy_active_global,
            'Spectral_Entropy_Passive_OT': spect_entropy_passive_ot,
            'Spectral_Entropy_Passive_Global': spect_entropy_passive_global,
            'Total_Energy_Passive_Global': energy_passive_global,
        })

        return entropy_dict

    def calc_associated_reactions(self, segment_df: pd.DataFrame) -> dict:

        # calculate the velocity magnitudes of all features for both hands (active hand vs. passive hand)

        # calculate the cross-correlation

        pass

    def calc_spastic_drift(self, segment_df: pd.DataFrame) -> dict:

        # calculate the average finger to palm distance

        # calculate the angular change between the hand and the lower arm (wrist flexion)

        # calculate the angular changes in the elbow joint and the shoulder joint

        pass

    def calc_kinematic_parameters(self, raw_signal: np.ndarray, custom_cfg: dict = None) -> dict:
        """
        Extracts kinematic parameters from a 1D time-series signal using scipy find_peaks.
        Identifies valid alternating peaks and valleys and calculates amplitude, period, velocity, and (CoV) metrics.

        Args:
            raw_signal (np.ndarray): 1D array of kinematic data (e.g., Euclidean distance).
            custom_cfg (dict, optional): Custom configuration dictionary to override default
                                         peak detection parameters. Defaults to None.

        Returns:
            dict: Dictionary containing descriptive statistics for period, amplitude,
                  velocity, CoV, the indices of valid peaks/valleys, and an extraction
                  status flag ('success' or failure reason).
        """
        # initialize ToolBox object
        tb = ToolBox(fps=self.fps)

        # default config used for peak detection
        cfg: dict = {'lowpass_cutoff': 15.0,        # Hz
                     'min_prominence': 0.1,         # minimum peak-to-valley delta
                     'min_distance_sec': 0.1,       # minimum time between the same events (peak-peak or valley-valley)
                     'max_distance_sec': 5.0}       # max time between events to count as a repetition

        # update if a custom config exists
        if custom_cfg:
            cfg.update(custom_cfg)

        features = {'repetition_freq': 0.0, 'repetition_num': 0.0,
                    'period_mean': 0.0, 'period_pct_90': 0.0, 'period_cov': 0.0,
                    'amplitude_mean': 0.0, 'amplitude_pct_90': 0.0, 'amplitude_cov': 0.0,
                    'velocity_pos_mean': 0.0, 'velocity_pos_pct_90': 0.0, 'velocity_pos_cov': 0.0,
                    'velocity_neg_mean': 0.0, 'velocity_neg_pct_90': 0.0, 'velocity_neg_cov': 0.0,
                    'extraction_status': 'failed',
                    'valid_peaks_idx': [],
                    'valid_valleys_idx': []}

        raw_signal = np.squeeze(raw_signal)

        n_samples = len(raw_signal)
        if n_samples < max(self.fps, 10):
            print(f'Error: Trial skipped. Samples ({n_samples}) < Framerate ({self.fps}).')
            features['extraction_status'] = 'failed'
            return features

        # robust filtering
        def _robust_lowpass(signal_data: np.ndarray, high_cutoff: float) -> np.ndarray:
            if len(signal_data) < 15:
                return signal_data
            sos = butter(4, high_cutoff, btype='lowpass', fs=self.fps, output='sos')
            return sosfiltfilt(sos, signal_data)

        # detrend the raw signal
        clean_signal: np.ndarray = _robust_lowpass(raw_signal, high_cutoff=cfg['lowpass_cutoff'])

        # peak & valley detection
        dist_frames: int = max(1, int(cfg['min_distance_sec'] * self.fps))
        prominence: float = cfg['min_prominence']

        peak_idc, _ = find_peaks(clean_signal, prominence=prominence, distance=dist_frames)
        valley_idc, _ = find_peaks(-clean_signal, prominence=prominence, distance=dist_frames)

        potential_peaks: list = [{'idx': i, 'amp': clean_signal[i], 'type': 'peak'} for i in peak_idc]
        potential_valleys: list = [{'idx': i, 'amp': abs(clean_signal[i]), 'type': 'valley'} for i in valley_idc]

        # enforce alternation (peak-valley-peak-valley)
        def _alternation_filter(peaks, valleys):
            events_lst = sorted(peaks + valleys, key=lambda x: x['idx'])
            if not events_lst:
                return [], []

            clean_events = [events_lst[0]]
            for current in events_lst[1:]:
                last = clean_events[-1]

                if current['type'] == last['type']:
                    # if two same types of events occur, keep the one with larger amplitude
                    if current['amp'] > last['amp']:
                        clean_events.pop()
                        clean_events.append(current)
                else:
                    clean_events.append(current)

            return [e for e in clean_events if e['type'] == 'peak'], [e for e in clean_events if e['type'] == 'valley']

        # get clean alternating peak and valley events
        valid_peaks, valid_valleys = _alternation_filter(potential_peaks, potential_valleys)

        features['valid_peaks_idx'] = [p['idx'] for p in valid_peaks]
        features['valid_valleys_idx'] = [v['idx'] for v in valid_valleys]

        if len(valid_peaks) < 2:
            features['extraction_status'] = 'failed_insufficient_reps'
            features['signal_original'] = clean_signal
            features['time_axis'] = np.arange(len(clean_signal)) / self.fps
            return features

        # 4) metrics calculation

        # period
        peak_indices = features['valid_peaks_idx']
        period_diffs = np.diff(peak_indices) / self.fps
        features.update(tb.get_descriptive_stats(np.array(period_diffs), 'period'))

        # repetitions & frequency
        avg_period = features['period_mean']
        if avg_period > 0:
            features['repetition_freq'] = round(1.0 / avg_period, 2)
            valid_duration = (peak_indices[-1] - peak_indices[0]) / self.fps
            features['repetition_num'] = round(valid_duration * features['repetition_freq'] + 1, 1)

        # amplitude (peak-to-peak)
        amplitudes = []
        for p in valid_peaks:
            p_idx = p['idx']
            next_valleys = [v for v in valid_valleys if v['idx'] > p_idx]
            if next_valleys:
                v = next_valleys[0]
                if (v['idx'] - p_idx) / self.fps < (1.5 * avg_period):
                    # Calculate true ptp amplitude
                    raw_p = clean_signal[p['idx']]
                    raw_v = clean_signal[v['idx']]
                    amplitudes.append(raw_p - raw_v)

        if amplitudes:
            features.update(tb.get_descriptive_stats(np.array(amplitudes), 'amplitude'))

        # velocities
        velocity_curve = np.gradient(clean_signal, 1 / self.fps)
        rise_vels, fall_vels = [], []
        all_events = sorted(valid_peaks + valid_valleys, key=lambda x: x['idx'])

        for i in range(len(all_events) - 1):
            curr, next_evt = all_events[i], all_events[i + 1]
            start, end = curr['idx'], next_evt['idx']

            start = max(0, start)
            end = min(len(velocity_curve), end)

            if start < end:
                vel_seg = velocity_curve[start:end]
                if len(vel_seg) > 0:
                    if next_evt['type'] == 'peak':
                        rise_vels.append(np.max(vel_seg))
                    else:
                        fall_vels.append(np.min(vel_seg))

        features.update(tb.get_descriptive_stats(np.array(rise_vels), 'velocity_pos'))
        features.update(tb.get_descriptive_stats(np.array(fall_vels), 'velocity_neg'))

        # extraction successful
        features['extraction_status'] = 'success'
        features['signal_original'] = clean_signal
        features['time_axis'] = np.arange(len(clean_signal)) / self.fps
        features['raw_amplitudes'] = np.array(amplitudes)
        return features

    @staticmethod
    def calc_flexion_angle(seg_vec_1: np.ndarray, seg_vec_2: np.ndarray) -> np.ndarray:
        """
        Calculates the interior flexion angle between two segments.
        Returns the angle in degrees (0.0: perfectly straight, 90.0: right angle flex).

        Args:
            seg_vec_1 (np.ndarray): Vector of 1st body segment
            seg_vec_2 (np.ndarray): Vector of 2nd body segment

        Returns:
            flexion_angle (np.ndarray): Interior flexion angle.
        """

        norm_v1: np.ndarray = np.linalg.norm(seg_vec_1, axis=1)
        norm_v2: np.ndarray = np.linalg.norm(seg_vec_2, axis=1)

        # avoid division by zero
        norm_v1[norm_v1 == 0] = 1e-8
        norm_v2[norm_v2 == 0] = 1e-8

        seg_vec_1_u: np.ndarray = seg_vec_1 / norm_v1[:, np.newaxis]
        seg_vec_2_u: np.ndarray = seg_vec_2 / norm_v2[:, np.newaxis]

        dot_product: np.ndarray = np.sum(seg_vec_1_u * seg_vec_2_u, axis=1)

        # clip to avoid NaN floating point inaccuracies
        flexion_angle: np.ndarray = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))

        return flexion_angle

    def extract_alt_tap_event_hysteresis(self, distance_signals: dict, tap_thresh: float = 0.25,
                                         release_thresh: float = 0.40, strict_cutoff: float = 0.10) -> list:
        """
        Extracts intended finger taps from Euclidean distances of multiple time series data using threshold hysteresis.

        Args:
            distance_signals (dict): Dictionary mapping finger pairs to 1D numpy distance arrays.
            tap_thresh (float): Distance threshold value the signal must drop below to activate a tap window.
            release_thresh (float): Distance threshold value the signal must rise above to deactivate the tap window.
            strict_cutoff (float): Events reaching a distance lower than this threshold are 'strict' taps, all other
            taps below 0.25 are classified as 'near_miss'.

        Returns:
            list: Time-sorted list of event dictionaries.
        """

        all_candidates = []

        # find all tap windows for all fingers independently
        for finger_key, signal in distance_signals.items():
            in_tap = False
            start_idx = 0
            min_val = float('inf')
            min_idx = 0

            for i, val in enumerate(signal):
                if not in_tap and val < tap_thresh:
                    # open a tap window
                    in_tap = True
                    start_idx = i
                    min_val = val
                    min_idx = i
                elif in_tap:
                    # track the local minimum within the tap window
                    if val < min_val:
                        min_val = val
                        min_idx = i

                    # close the tap window
                    if val > release_thresh:
                        in_tap = False
                        all_candidates.append({'finger_key': finger_key,
                                               'start_idx': start_idx,
                                               'min_idx': min_idx,
                                               'end_idx': i,
                                               'min_dist': min_val,
                                               'type': 'strict' if min_val <= strict_cutoff else 'near_miss'})

            # catch time windows not closed at the end of a video
            if in_tap:
                all_candidates.append({'finger_key': finger_key,
                                       'start_idx': start_idx,
                                       'min_idx': min_idx,
                                       'end_idx': len(signal) - 1,
                                       'min_dist': min_val,
                                       'type': 'strict' if min_val <= strict_cutoff else 'near_miss'})

        # sort all candidates chronologically by their minimum point
        all_candidates.sort(key=lambda x: x['min_idx'])

        if not all_candidates:
            return []

        # handle superpositions - multiple fingers tap within 0.25 seconds of each other
        min_separation_frames = int(self.fps * 0.25)

        resolve_events = []
        current_group = [all_candidates[0]]

        for event in all_candidates[1:]:
            last_event = current_group[-1]

            # current event happense to close to the last event
            if event['min_idx'] - last_event['min_idx'] < min_separation_frames:
                current_group.append(event)
            else:
                # intended target is always the digit with the lowest physical distance
                best_event = min(current_group, key=lambda x: x['min_dist'])
                resolve_events.append(best_event)
                # start a new group
                current_group = [event]

        # append the best event to the resolved events list
        if current_group:
            best_event = min(current_group, key=lambda x: x['min_dist'])
            resolve_events.append(best_event)

        return resolve_events

    @staticmethod
    def calc_levenshtein_accuracy(tap_list: list) -> tuple:

        # sequence accuracy
        tapping_sequence_digit_lst: list = [int(tap['finger_key'][-1]) for tap in tap_list]

        pat_len = len(tapping_sequence_digit_lst)

        # safety: for zero movement
        if pat_len == 0:
            return 0, 0.0, 0

        # template of possible target tapping sequences - one full repetition
        target_templates = [
            [2, 3, 4, 5],  # resetting
            [2, 3, 4, 5, 4, 3]  # looping
        ]

        # initialize rating variables
        best_accuracy = -1.0
        best_errors = 0
        best_target_len = 0

        # define a search window for the estimated taps intended
        min_search_len = max(1, pat_len // 2)  # x0.5 the taps if every finger was double-tapped
        max_search_len = int(pat_len * 1.5)  # x1.5 the taps if there are many skipped fingers

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

        return best_accuracy, best_errors, best_target_len

    def calc_sparc(self, velocity_profile: np.ndarray, pad_level: int = 4, fc: float = 10.0, amp_th: float = 0.05) -> float:
        """
        Calculates the Spectral Arc Length (SPARC) of a movement velocity profile.
        A value closer to 0 indicates a smoother movement. More negative values indicate jerky/interrupted movement.
        Ref: Balasubramanian et al. (2011), "A robust and sensitive metric for quantifying movement smoothness."

        Args:
            velocity_profile (np.ndarray): 1D array of velocity.
            pad_level (int): Zero-padding factor for the FFT to increase frequency resolution.
            fc (float): Maximum cutoff frequency (Hz) for human movement (default 10.0 Hz).
            amp_th (float): Amplitude threshold to determine the dynamic cutoff frequency.

        Returns:
            float: The SPARC smoothness metric (dimensionless, always negative).
        """

        # skip for profiles that are too short for spectral analysis
        if len(velocity_profile) < 5:
            return 0.0

        # number of points for FFT with zero-padding
        nfft = int(pow(2, np.ceil(np.log2(len(velocity_profile))) + pad_level))

        # frequency resolution axis
        f = np.arange(0, self.fps, self.fps / nfft)

        # normalize magnitude spectrum
        vf = np.abs(np.fft.fft(velocity_profile, nfft))
        vf_max = np.max(vf)
        if vf_max == 0:
            return 0.0
        vf = vf / vf_max

        # find dynamic cutoff frequency index (where amplitude drops below threshold)
        idx_fc = np.where((f <= fc) & (vf >= amp_th))[0]
        fc_idx = idx_fc[-1] if len(idx_fc) > 0 else 0

        if fc_idx == 0:
            return 0.0

        f_sel = f[:fc_idx + 1]
        vf_sel = vf[:fc_idx + 1]

        # calculate the arc length of the spectral curve
        dw = np.diff(f_sel) / f_sel[-1]
        dm = np.diff(vf_sel)

        arc_length = np.sum(np.sqrt(dw**2 + dm**2))

        return float(arc_length) * (-1)
