# src/kinematic_features.py

# libraries
import numpy as np
import pandas as pd
from scipy.stats import entropy
from scipy.signal import find_peaks, medfilt, butter, sosfiltfilt

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
        suppress tracking noise and isolate the complexity of the true biomechanical movement.

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
        global_ceiling = np.percentile(Sxx_power_active, config['parameter_extraction']['vmax_percentile'])
        n_order_mag: int = config['parameter_extraction']['vmin_factor']
        global_floor = global_ceiling * pow(1e1, (-1)*n_order_mag)

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
                Safely suppresses SciPy division warnings caused by empty/gated time windows.

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
        Extracts kinematic parameters from a 1D time-series signal using a robust, adaptive
        2-pass peak detection strategy. Identifies valid alternating peaks and valleys
        to calculate amplitude, period, velocity, and consistency (CoV) metrics.

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
        cfg: dict = {'min_segment_length': 0.00,
                     'min_peak_amp_diff': 0.00,
                     'min_peak_dur_diff': 0.00,
                     'min_valley_amp_diff': 0.00,
                     'min_valley_dur_diff': 0.00,
                     'max_peak_amp_diff': 0.00,
                     'max_peak_dur_diff': 0.0,
                     'max_valley_amp_diff': 0.0,
                     'prominence_factor': 0.35,
                     'distance_factor': 0.35}

        # update the peak detection config if a custom config exists
        if custom_cfg:
            cfg.update(custom_cfg)

        features = {
            'repetition_freq': 0.0, 'repetition_num': 0.0,
            'period_mean': 0.0, 'period_pct_90': 0.0, 'period_cov': 0.0,
            'amplitude_mean': 0.0, 'amplitude_pct_90': 0.0, 'amplitude_cov': 0.0,
            'velocity_pos_mean': 0.0, 'velocity_pos_pct_90': 0.0, 'velocity_pos_cov': 0.0,
            'velocity_neg_mean': 0.0, 'velocity_neg_pct_90': 0.0, 'velocity_neg_cov': 0.0,
            'extraction_status': 'failed',
            'valid_peaks_idx': [],
            'valid_valleys_idx': []
        }

        raw_signal = np.squeeze(raw_signal)

        n_samples = len(raw_signal)
        if n_samples < max(self.fps, 10):
            print(f'Error: Trial skipped. Samples ({n_samples}) < Framerate ({self.fps}).')
            features['extraction_status'] = 'failed'
            return features

        # 1) detrend
        def _robust_detrend(signal_data: np.ndarray, low_cutoff: float = 0.3, high_cutoff: float = 15) -> np.ndarray:
            """
            Removes non-linear baseline drifts (wandering baselines) and high frequency jitter
            using a zero-phase Bandpass filter.

            Args:
                signal_data (np.ndarray): The raw signal.
                low_cutoff (float): Frequency below which to remove data. defaults to 0.3 Hz.
                high_cutoff (float): Frequency above which to remove data. Defaults to 15 Hz.

            Returns:
                det_signal (np.ndarray): The detrended signal.
            """

            # requires at least 1 full period  of the lowest frequency
            min_samples: int = int(1.0 * (self.fps / low_cutoff))

            # fall back to 2nd order polynomial for a short signal
            if len(signal_data) < min_samples:
                x: np.ndarray = np.arange(len(signal_data))
                p: np.ndarray = np.polyfit(x, signal_data, 2)
                trend = np.polyval(p, x)
                return signal_data - trend

            # 4th order Butterworth Bandpass filter
            sos = butter(4, [low_cutoff, high_cutoff], btype='bandpass', fs=self.fps, output='sos')

            # apply filter forward and backward (zero phase distortion): centers the signal around 0
            det_signal: np.ndarray = sosfiltfilt(sos, signal_data)
            return det_signal

        # detrend the raw signal
        detrended_signal: np.ndarray = _robust_detrend(raw_signal, low_cutoff=0.3, high_cutoff=15.0)

        # 2) scouting for peaks and valleys using zero-crossings

        window_size: int = int(self.fps * 1.0)

        # ensure odd kernel size
        if window_size % 2 == 0:
            window_size += 1

        # subtract the rolling median (only for scouting)
        baseline_trend: np.ndarray = medfilt(volume=detrended_signal, kernel_size=window_size)
        scout_signal: np.ndarray = detrended_signal - baseline_trend

        # extract zero-crossings by from dynamically centered scout signal
        signs: np.ndarray = np.sign(scout_signal)
        signs[signs == 0] = 1
        zero_crossings: np.ndarray = np.where(np.diff(signs))[0]

        scout_peaks: list = []
        scout_valleys: list = []

        # only run if there are 2 or more zero_crossings (one period or more)
        if len(zero_crossings) >= 2:
            # move a window across the zero-crossings to capture all positive and negative area of the signal
            for i in range(len(zero_crossings) - 1):
                window_start, window_end = zero_crossings[i], zero_crossings[i + 1]
                if (window_end - window_start) < (self.fps * cfg['min_segment_length']):
                    continue
                # extract the signal segment between the two current zero crossings
                segment = detrended_signal[window_start:window_end]

                # extract the peak/valley amplitudes and its index for the current segment
                if np.mean(segment) > 0:
                    scout_peaks.append({'idx': window_start + int(np.argmax(segment)), 'amp': np.max(segment)})
                else:
                    scout_valleys.append({'idx': window_start + int(np.argmin(segment)), 'amp': abs(np.min(segment))})

        # abort if less than 2 peaks or no valleys were found
        if len(scout_peaks) < 2 or len(scout_valleys) < 1:
            features['extraction_status'] = 'failed_scout_pass'
            return features

        # calculate statistics from scouting (1-pass) the signal
        median_peak_amp: float = float(np.median([p['amp'] for p in scout_peaks]))
        median_valley_amp: float = float(np.median([v['amp'] for v in scout_valleys]))

        scout_peak_indices: list = sorted([p['idx'] for p in scout_peaks])
        median_period_samples: float = float(np.median(np.diff(scout_peak_indices)))

        # 3) refining the scouted peaks and valleys

        # dynamic tuning
        prominence_threshold_peak: float = cfg['prominence_factor'] * median_peak_amp
        prominence_threshold_valley = cfg['prominence_factor'] * median_valley_amp
        distance_threshold = int(cfg['distance_factor'] * median_period_samples)
        distance_threshold = max(distance_threshold, 1)

        # find peaks using prominence
        peak_idc, _ = find_peaks(detrended_signal,
                                 prominence=prominence_threshold_peak,
                                 distance=distance_threshold)

        # find valleys
        valley_idc, _ = find_peaks((-1)*detrended_signal,
                                   prominence=prominence_threshold_valley,
                                   distance=distance_threshold)

        potential_peaks = [{'idx': i, 'amp': detrended_signal[i], 'type': 'peak'} for i in peak_idc]
        potential_valleys = [{'idx': i, 'amp': abs(detrended_signal[i]), 'type': 'valley'} for i in valley_idc]

        # filter the peaks
        filtered_peaks = []
        for p in potential_peaks:
            # ceiling check
            if cfg['max_peak_amp_diff'] and p['amp'] > 0:
                if p['amp'] > (cfg['max_peak_amp_diff'] * median_peak_amp):
                    continue

            filtered_peaks.append(p)

        # filter the valleys
        filtered_valleys = []
        for v in potential_valleys:
            # ceiling check
            if cfg['max_valley_amp_diff'] and v['amp'] > (cfg['max_valley_amp_diff'] * median_valley_amp):
                continue

            filtered_valleys.append(v)

        # enforce alternation (peak-valley-peak-valley)
        def _alternation_filter(peaks, valleys):
            events_lst = sorted(peaks + valleys, key=lambda x: x['idx'])
            if not events_lst:
                return [], []

            clean_events = [events_lst[0]]
            for current in events_lst[1:]:
                last = clean_events[-1]

                if current['type'] == last['type']:

                    if current['amp'] > last['amp']:
                        clean_events.pop()
                        clean_events.append(current)
                else:
                    clean_events.append(current)

            return [e for e in clean_events if e['type'] == 'peak'], [e for e in clean_events if e['type'] == 'valley']

        # get clean alternating peak and valley events
        valid_peaks, valid_valleys = _alternation_filter(filtered_peaks, filtered_valleys)

        features['valid_peaks_idx'] = [p['idx'] for p in valid_peaks]
        features['valid_valleys_idx'] = [v['idx'] for v in valid_valleys]

        if len(valid_peaks) < 2:
            features['extraction_status'] = 'failed_insufficient_reps'
            return features

        # 4) metrics calculation

        # period
        peak_indices = [p['idx'] for p in valid_peaks]
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
                    raw_p = detrended_signal[p['idx']]
                    raw_v = detrended_signal[v['idx']]
                    amplitudes.append(raw_p - raw_v)

        if amplitudes:
            features.update(tb.get_descriptive_stats(np.array(amplitudes), 'amplitude'))

        # velocities
        velocity_curve = np.gradient(detrended_signal, 1 / self.fps)
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
        features['signal_detrended'] = detrended_signal
        features['time_axis'] = np.arange(len(detrended_signal)) / self.fps

        return features

    def calc_flexion_angle(self, seg_vec_1: np.ndarray, seg_vec_2: np.ndarray) -> np.ndarray:
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