"""
Module: baseline
Description: Provides algorithms for spectral preprocessing, including convex hull 
             baseline correction and peak detection for Raman spectra.
Author: Team Member
Date: August 2026
"""

import numpy as np


def baseline_correction(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Performs baseline correction on a spectrum using a lower convex hull approach.

    Parameters:
        x (np.ndarray): Array of Raman shift values (cm⁻¹).
        y (np.ndarray): Array of spectral intensity values.

    Returns:
        np.ndarray: Baseline-corrected spectral intensity values.
    """
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    hull = []
    for i in range(len(x_sorted)):
        while len(hull) >= 2:
            p1 = hull[-2]
            p2 = hull[-1]
            p3 = i

            cross = (
                (x_sorted[p2] - x_sorted[p1]) * (y_sorted[p3] - y_sorted[p1])
                - (y_sorted[p2] - y_sorted[p1]) * (x_sorted[p3] - x_sorted[p1])
            )

            if cross <= 0:
                hull.pop()
            else:
                break

        hull.append(i)

    baseline_x = x_sorted[hull]
    baseline_y = y_sorted[hull]

    baseline = np.interp(x_sorted, baseline_x, baseline_y)
    corrected = y_sorted - baseline

    return corrected


def peak_detection(x: np.ndarray, corrected: np.ndarray) -> tuple:
    """
    Identifies spectral peaks below a specified Raman shift threshold.

    Parameters:
        x (np.ndarray): Array of Raman shift values.
        corrected (np.ndarray): Baseline-corrected intensity values.

    Returns:
        tuple: A tuple (peak_positions, peak_intensities) containing NumPy arrays 
               of the detected peak positions and their corresponding intensities.
    """
    MAX_RAMAN_SHIFT = 3100
    valid = x < MAX_RAMAN_SHIFT

    x_clean = x[valid]
    corrected_clean = corrected[valid]

    # Peak detection logic
    peaks = np.where((corrected_clean[1:-1] > corrected_clean[:-2]) & 
                     (corrected_clean[1:-1] > corrected_clean[2:]))[0] + 1

    peak_positions = x_clean[peaks]
    peak_intensities = corrected_clean[peaks]

    return peak_positions, peak_intensities
