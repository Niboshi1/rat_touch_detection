import glob
import os

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd

import h5py
from scipy import signal

import deepethogram.postprocessing

import omegaconf


def calibrate_limb_positions(
    df_in: pd.DataFrame, set_reference: str = "x"
) -> pd.DataFrame:
    """
    Function to subtract shoulder position from elbow and foot coordinates.

    Parameters:
    df_in (pd.DataFrame): Input DataFrame.
    set_reference (str): Determines which coordinates (x or y) to calibrate. Default is "x".

    Returns:
    pd.DataFrame: Output DataFrame with calibrated limb positions.
    """

    # Funtion to subtract sholder position from elbow and foot coordinates
    # make empty DataFrame
    df_out = pd.DataFrame()

    df_out["foot_x"] = df_in.iloc[:, 0]
    df_out["foot_y"] = df_in.iloc[:, 1]
    df_out["elbow_x"] = df_in.iloc[:, 2]
    df_out["elbow_y"] = df_in.iloc[:, 3]

    # Subtract coordinates
    if "x" in set_reference:
        print("DLC: X-coordinate calibration")
        df_out["foot_x"] = df_in.iloc[:, 0] - df_in.iloc[:, 4]
        df_out["elbow_x"] = df_in.iloc[:, 2] - df_in.iloc[:, 4]
    if "y" in set_reference:
        print("DLC: Y-coordinate calibration")
        df_out["foot_y"] = df_in.iloc[:, 1] - df_in.iloc[:, 5]
        df_out["elbow_y"] = df_in.iloc[:, 3] - df_in.iloc[:, 5]

    return df_out


def apply_filter(
    df_in: pd.DataFrame,
    set_reference: str = "x",
    filter: str = "mean",
    low_pass: int = 30,
) -> pd.DataFrame:
    """
    Function to apply a filter to the DataFrame.

    Parameters:
    df_in (pd.DataFrame): Input DataFrame.
    set_reference (str): Determines which coordinates (x or y) to calibrate. Default is "x".
    filter (str): Determines the type of filter to apply. Default is "mean".
    low_pass (int): Sets the cutoff frequency for the low pass filter. Default is 30.

    Returns:
    pd.DataFrame: Output DataFrame with applied filter.
    """
    # make an empty DataFrame
    df_out = pd.DataFrame()

    # subtract and fix coordinates
    df_fixed = calibrate_limb_positions(df_in, set_reference)

    # low pass filter (7 Hz) with Butterworth filter 4th order
    if filter == "mean":
        for column_index in df_fixed.columns:
            df_out[column_index] = np.convolve(
                df_fixed[column_index], np.ones(3) / 3, mode="same"
            )
    elif filter == "bandpass":
        for column_index in df_fixed.columns:
            df_out[column_index] = butter_lowpass_filter(
                df_fixed[column_index], low_pass, 60, order=4
            )
    else:
        df_out = df_fixed.copy()

    return df_out


def butter_lowpass(lowcut: int, fs: int, order: int = 4) -> tuple:
    """Design the butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype="low")
    return b, a


def butter_lowpass_filter(
    x: np.ndarray, lowcut: int, fs: int, order: int = 4
) -> np.ndarray:
    """Low pass filter the data"""
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y


def extract_coordinates(
    dlc_file_path: str,
    set_reference: str = "",
    graph_preview=False,
    preview_range=[0, 2000],
    filter="mean",
    low_pass=30,
):
    # Read labeled h5 file
    print("loading:", os.path.basename(dlc_file_path))
    df_coords = pd.read_hdf(dlc_file_path, encoding="shift_jis")

    # extract forelimb positions and apply filter
    df_coords_forelimb = apply_filter(
        df_coords[df_coords.columns[[0, 1, 3, 4, 6, 7]]],
        set_reference,
        filter,
        low_pass,
    )

    if graph_preview:
        # Trimming for visualization
        fig = plt.figure(figsize=(10, 6), dpi=300)
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        ax1.plot(
            df_coords_forelimb["foot_x"].iloc[preview_range[0]
                : preview_range[1]],
            lw=1,
            c="k",
        )
        ax1.set_ylabel("x1")
        ax2.plot(
            df_coords_forelimb["foot_y"].iloc[preview_range[0]
                : preview_range[1]],
            lw=1,
            c="r",
        )
        ax2.set_ylabel("y1")

        ax3.plot(
            df_coords_forelimb["elbow_x"].iloc[preview_range[0]
                : preview_range[1]],
            lw=1,
            c="k",
        )
        ax3.set_ylabel("x2")
        ax4.plot(
            df_coords_forelimb["elbow_y"].iloc[preview_range[0]
                : preview_range[1]],
            lw=1,
            c="r",
        )
        ax4.set_ylabel("y2")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_yticks([])
            ax.set_xticks(
                np.arange(preview_range[0], preview_range[1], 60 * 5))
            ax.set_xticklabels(
                np.arange(0, (preview_range[1] - preview_range[0]) / 60, 5)
            )
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        plt.xlabel("Time (sec)")
        plt.tight_layout()

        plt.show()

    return df_coords, df_coords_forelimb


# Funtion to return the position of the maximum vale
def max_height_index(peak_idx, coords_y):
    return int(peak_idx + np.where(coords_y == np.max(coords_y))[0])


def adjust_peak_idx(filtered_peaks, y):
    y_data = y.values
    adjusted_peaks = []
    for i in range(len(filtered_peaks)):
        try:
            if y_data[filtered_peaks[i] + 1] - y_data[filtered_peaks[i]] > 1:
                adjusted_peaks.append(filtered_peaks[i] + 1)
            else:
                adjusted_peaks.append(filtered_peaks[i])
        except IndexError:
            adjusted_peaks.append(filtered_peaks[i])

    return adjusted_peaks


def extract_step_onsets(
    df_coords, df_coords_f, hight_cutoff=False, graph_preview=False
):
    # Automatically detect peaks from foot x coordinates
    x = df_coords_f["foot_x"].iloc[:] * -1
    y = df_coords_f["foot_y"].iloc[:]
    y_accel = np.gradient(np.gradient(
        df_coords_f["foot_y"].iloc[:])) * -1 * 3 + 200

    peaks, _ = signal.find_peaks(
        x, height=-10, prominence=(5, None), width=(None, 50))
    print("DLC: Detected onsets:", len(peaks))

    # Filter by confidence
    # df_coords.iloc[:, 4] is the confidence of foot
    peaks = peaks[np.where(df_coords.iloc[peaks, 4] > 0.99)]
    print("DLC: Filtered onsets by confidence:", len(peaks))

    # Get maximum y position of the foot inside 10 frames
    local_max_foot_height_idx = np.array(
        [max_height_index(peak, y_accel[peak: peak + 4]) for peak in peaks]
    )

    # Plot y coordinate of the foot on the detected onset
    if graph_preview:
        plt.figure(figsize=(5, 5))
        plt.hist(y[local_max_foot_height_idx])
        plt.show()

    # Filter detected onsets by foot height
    if hight_cutoff:
        filtered_peaks = local_max_foot_height_idx[
            np.intersect1d(
                np.where(df_coords_f.iloc[local_max_foot_height_idx, 1] > 200),
                np.where(df_coords_f.iloc[local_max_foot_height_idx, 1] < 300),
            )
        ]
    else:
        filtered_peaks = local_max_foot_height_idx

    # Fix peak index if the difference between peak idx and the next idx is above 2 pixels
    for i in range(4):
        filtered_peaks = adjust_peak_idx(filtered_peaks, y)

    # Plot the detected and filtered peaks
    if graph_preview:
        plt.figure(figsize=(60, 4))
        plt.plot(y, lw=1, c="r")
        plt.plot(filtered_peaks, y[filtered_peaks], "x", c="b")
        plt.show()
    filtered_peaks = np.array(filtered_peaks)

    return filtered_peaks


def read_ethogram_result(file_path, model="resnet18"):
    with h5py.File(file_path, "r") as f:
        data = f[model]
        probabilities = data["P"][()]
        thresholds = data["thresholds"][()]

        return probabilities, thresholds


def get_ethogram_predictions(percentiles, probabilities, thresholds):
    processor = deepethogram.postprocessing.MinBoutLengthPerBehaviorPostprocessor(
        thresholds, percentiles
    )
    predictions = processor(probabilities)

    return predictions


def thresh_step_confidence(
    dlc_filtered_steps,
    df_coords_f,
    probabilities,
    valid_threshold=0.7,
    graph_preview=False,
):
    # get step probabilities at each detected steps
    dlc_step_confidence = []
    for step_idx in dlc_filtered_steps:
        if len(probabilities[step_idx - 3: step_idx + 4, 3]) != 0:
            dlc_step_confidence.append(
                max(probabilities[step_idx - 3: step_idx + 4, 3])
            )
        else:
            dlc_step_confidence.append(probabilities[step_idx, 3])
    dlc_step_confidence = np.array(dlc_step_confidence)

    y = df_coords_f["foot_y"].iloc[:]
    confident_step_idx = dlc_step_confidence > valid_threshold

    if graph_preview:
        # plot histogram of step confidence
        plt.hist(dlc_step_confidence, bins=100)
        plt.show()

        # plot classified steps
        plt.figure(figsize=(60, 4))
        plt.plot(y, lw=1, c="k")
        plt.plot(
            dlc_filtered_steps[confident_step_idx],
            y[dlc_filtered_steps[confident_step_idx]],
            "x",
            c="g",
        )
        plt.plot(
            dlc_filtered_steps[~confident_step_idx],
            y[dlc_filtered_steps[~confident_step_idx]],
            "x",
            c="r",
        )
        plt.show()

    return confident_step_idx


def thresh_step_motion(
    dlc_filtered_steps, df_coords_f, predictions, graph_preview=False
):
    # detect whether the rat was in motion at detected step idx
    dlc_step_motion = [predictions[step_idx, 1]
                       for step_idx in dlc_filtered_steps]
    dlc_step_motion = np.array(dlc_step_motion)

    y = df_coords_f["foot_y"].iloc[:]
    motion_steps_idx = dlc_step_motion.astype(bool)

    if graph_preview:
        # plot classified steps
        plt.figure(figsize=(60, 4))
        plt.plot(y, lw=1, c="k")
        plt.plot(
            dlc_filtered_steps[motion_steps_idx],
            y[dlc_filtered_steps[motion_steps_idx]],
            "x",
            c="g",
        )
        plt.plot(
            dlc_filtered_steps[~motion_steps_idx],
            y[dlc_filtered_steps[~motion_steps_idx]],
            "x",
            c="r",
        )
        plt.show()

    return motion_steps_idx


def classify_floor_type(cfg, trial_name, step_idx):
    # DeepEthogram annotated files
    ethogram_results = get_ethogram_annotated_files(cfg)

    probabilities, thresholds = read_ethogram_result(
        ethogram_results[trial_name])

    # DeepEthogram =========================================================================
    bout_percentiles_file_path = cfg.paths.bout_percentiles

    with h5py.File(bout_percentiles_file_path, "r") as f:
        percentiles = f["percentiles"][()]

    probabilities, thresholds = read_ethogram_result(
        ethogram_results[trial_name])

    print(f"Ethogram: N-frames = {len(probabilities)}")

    predictions = get_ethogram_predictions(
        percentiles, probabilities, thresholds)

    # detect whether the rat was in motion at detected step idx
    step_floor = [predictions[step_idx, 2] for step_idx in step_idx]
    step_floor = np.array(step_floor, dtype=np.int32)

    return step_floor


def get_ethogram_annotated_files(cfg):
    ethogram_results = {}
    for dirs in cfg.paths.ethogram_labels:
        for root, dirs, files in os.walk(os.path.join(dirs, "DATA")):
            for file in files:
                if file.endswith(".h5"):
                    ethogram_results[file[:-11]] = os.path.join(root, file)
    return ethogram_results


def get_dlc_annotated_files(cfg: omegaconf.dictconfig.DictConfig):
    trial_names = [
        os.path.basename(x.split(".mp4")[0])
        for x in glob.glob(os.path.join(cfg.paths.dlc_labels, "*.mp4"))
    ]
    dlc_results = {}
    for trial_name in trial_names:
        dlc_results[trial_name] = os.path.join(
            cfg.paths.dlc_labels,
            f"{trial_name}DLC_effnet_b6_rat_stepsDec12shuffle1_1030000.h5",
        )

    return dlc_results


def get_valid_step_idx(cfg, trial_name, graph_preview):
    # DeepEthogram annotated files
    ethogram_results = get_ethogram_annotated_files(cfg)

    # DeepLabCut annocated files
    dlc_results = get_dlc_annotated_files(cfg)

    try:
        assert os.path.exists(dlc_results[trial_name])
        assert os.path.exists(ethogram_results[trial_name])
    except AssertionError:
        print("Missing file for trial:", trial_name)
        return np.array([])

    # DeepLabCut =========================================================================
    df_coords, df_coords_f = extract_coordinates(
        dlc_results[trial_name],
        set_reference="x",
        graph_preview=cfg.extraction.preview.raw_dlc_trace,
    )
    print(f"DLC: N-frames = {len(df_coords)}")
    dlc_filtered_steps = extract_step_onsets(
        df_coords, df_coords_f, graph_preview=cfg.extraction.preview.dlc_extracted_onset
    )

    # DeepEthogram =========================================================================
    bout_percentiles_file_path = cfg.paths.bout_percentiles

    with h5py.File(bout_percentiles_file_path, "r") as f:
        percentiles = f["percentiles"][()]

    probabilities, thresholds = read_ethogram_result(
        ethogram_results[trial_name])

    print(f"Ethogram: N-frames = {len(probabilities)}")

    predictions = get_ethogram_predictions(
        percentiles, probabilities, thresholds)

    # filter by confidence at the detected idx
    confident_step_idx = thresh_step_confidence(
        dlc_filtered_steps,
        df_coords_f,
        probabilities,
        valid_threshold=cfg.extraction.ethogram.step_confidence,
        graph_preview=cfg.extraction.preview.ethogram_step_confidence,
    )

    # detect whether the rat was in motion at detected step idx
    if cfg.extraction.filter.motion:
        motion_step_idx = thresh_step_motion(
            dlc_filtered_steps,
            df_coords_f,
            predictions,
            graph_preview=cfg.extraction.preview.ethogram_step_motion,
        )
        valid_step_idx = confident_step_idx * motion_step_idx
    else:
        valid_step_idx = confident_step_idx

    print(f"Ethogram: Valid onsets:{sum(valid_step_idx)}")

    # plot detected steps
    if graph_preview:
        # plot classified steps
        y = df_coords_f["foot_y"].iloc[:]
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        trans = mtransforms.blended_transform_factory(
            ax.transData, ax.transAxes)

        ax.plot(y, lw=1, c="k")
        ax.plot(
            dlc_filtered_steps[valid_step_idx],
            y[dlc_filtered_steps[valid_step_idx]],
            "x",
            c="g",
        )
        ax.plot(
            dlc_filtered_steps[~valid_step_idx],
            y[dlc_filtered_steps[~valid_step_idx]],
            "x",
            c="r",
        )
        ax.fill_between(
            np.arange(len(y)),
            0,
            1,
            where=predictions[:, 1] == 1,
            facecolor="grey",
            alpha=0.5,
            transform=trans,
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.show()

    return dlc_filtered_steps[valid_step_idx]
