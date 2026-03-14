# src/kinematic_features.py

# libraries
import os
import numpy as np
import pandas as pd
from scipy.stats import entropy


# modules
from src.config import config
from src.utils import ToolBox


class KinematicFeatures:
    def __init__(self, fps: float = config['camera_param']['fps']):
        self.fps = fps

    def calculate_spectral_entropy(self, sig_active: np.ndarray, sig_passive: np.ndarray) -> dict:
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

    def calculate_associated_reactions(self, segment_df: pd.DataFrame) -> dict:

        # calculate the velocity magnitudes of all features for both hands (active hand vs. passive hand)

        # calculate the cross-correlation

        pass

    def calculate_spastic_drift(self, segment_df: pd.DataFrame) -> dict:

        # calculate the average finger to palm distance

        # calculate the angular change between the hand and the lower arm (wrist flexion)

        # calculate the angular changes in the elbow joint and the shoulder joint

        pass
