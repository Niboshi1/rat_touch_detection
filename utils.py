
import glob
import os
import shutil
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
import h5py
from scipy import signal

import deepethogram.postprocessing

# Funtion to subtract sholder position from elbow and foot coordinates


def fix_limb_positions(df_in, set_reference):
    # make empty DataFrame
    df_out = pd.DataFrame()

    df_out["foot_x"] = df_in.iloc[:, 0]
    df_out["foot_y"] = df_in.iloc[:, 1]
    df_out["elbow_x"] = df_in.iloc[:, 2]
    df_out["elbow_y"] = df_in.iloc[:, 3]

    # Subtract coordinates
    if "x" in set_reference:
        print("filtering x")
        df_out["foot_x"] = df_in.iloc[:, 0] - df_in.iloc[:, 4]
        df_out["elbow_x"] = df_in.iloc[:, 2] - df_in.iloc[:, 4]
    if "y" in set_reference:
        print("filtering y")
        df_out["foot_y"] = df_in.iloc[:, 1] - df_in.iloc[:, 5]
        df_out["elbow_y"] = df_in.iloc[:, 3] - df_in.iloc[:, 5]

    return df_out


def apply_filter(
        df_in,
        set_reference="x",
        filter="mean",
        low_pass=30,
):
    # make an empty DataFrame
    df_out = pd.DataFrame()

    # subtract and fix coordinates
    df_fixed = fix_limb_positions(df_in, set_reference)

    # low pass filter (7 Hz) with Butterworth filter 4th order
    if filter == "mean":
        for column_index in df_fixed.columns:
            df_out[column_index] = np.convolve(
                df_fixed[column_index],
                np.ones(3)/3,
                mode="same"
            )
    elif filter == "bandpass":
        for column_index in df_fixed.columns:
            df_out[column_index] = butter_lowpass_filter(
                df_fixed[column_index], 7, 60, order=4
            )
    else:
        df_out = df_fixed.copy()

    return df_out


def butter_lowpass(lowcut, fs, order=4):
    """Design the butterworth filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = signal.butter(order, low, btype="low")
    return b, a


def butter_lowpass_filter(x, lowcut, fs, order=4):
    """Low pass filter the data"""
    b, a = butter_lowpass(lowcut, fs, order=order)
    y = signal.filtfilt(b, a, x)
    return y


def extract_coordinates(
        dlc_file_path,
        set_reference="",
        graph_preview=False,
        preview_range=[0, 200],
        filter="mean",
        low_pass=30,
):
    # Read labeled h5 file
    print("loading:", os.path.basename(dlc_file_path))
    df_coords = pd.read_hdf(dlc_file_path, encoding='shift_jis')

    # extract forelimb positions and apply filter
    df_coords_forelimb = apply_filter(
        df_coords[df_coords.columns[[0, 1, 3, 4, 6, 7]]],
        set_reference,
        filter,
        low_pass,
    )

    if graph_preview:
        # Trimming for visualization
        fig = plt.figure(figsize=(20, 8))
        ax1 = fig.add_subplot(411)
        ax2 = fig.add_subplot(412)
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)

        ax1.plot(
            df_coords_forelimb["foot_x"].iloc[preview_range[0]: preview_range[1]], lw=1, c="k")
        ax1.set_ylabel("x1")
        ax2.plot(
            df_coords_forelimb["foot_y"].iloc[preview_range[0]: preview_range[1]], lw=1, c="r")
        ax2.set_ylabel("y1")

        ax3.plot(
            df_coords_forelimb["elbow_x"].iloc[preview_range[0]: preview_range[1]], lw=1, c="k")
        ax3.set_ylabel("x2")
        ax4.plot(
            df_coords_forelimb["elbow_y"].iloc[preview_range[0]: preview_range[1]], lw=1, c="r")
        ax4.set_ylabel("y2")

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_yticks([])
            ax.set_xticks(np.arange(preview_range[0], preview_range[1], 60))
            ax.set_xticklabels(
                np.arange(0, (preview_range[1] - preview_range[0]) / 60, 1))
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

    adjusted_peaks = []
    for i in range(len(filtered_peaks)):
        try:
            if y[filtered_peaks[i]+1] - y[filtered_peaks[i]] > 1:
                adjusted_peaks.append(filtered_peaks[i]+1)
            else:
                adjusted_peaks.append(filtered_peaks[i])
        except IndexError:
            adjusted_peaks.append(filtered_peaks[i])

    return adjusted_peaks


def extract_footstrike_onsets_f(
        df_coords,
        df_coords_f,
        hight_cutoff=False,
        graph_preview=False
):
    # Automatically detect peaks from foot x coordinates
    x = df_coords_f["foot_x"].iloc[:] * -1
    y = df_coords_f["foot_y"].iloc[:]
    y_accel = np.gradient(np.gradient(df_coords_f["foot_y"].iloc[:]))*-1*3+200

    peaks, _ = signal.find_peaks(
        x, height=-10, prominence=(5, None), width=(None, 50))
    print("Detected onsets:", len(peaks))

    # Filter by confidence
    # df_coords.iloc[:, 4] is the confidence of foot
    peaks = peaks[np.where(df_coords.iloc[peaks, 4] > 0.99)]
    print("Filtered onsets #1:", len(peaks))

    # Get maximum y position of the foot inside 10 frames
    local_max_foot_height_idx = np.array(
        [max_height_index(peak, y_accel[peak:peak+4]) for peak in peaks]
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
    print("Refined onsets:", len(filtered_peaks))
    filtered_peaks = np.array(filtered_peaks)

    return filtered_peaks


def read_ethogram_result(file_path, model='resnet18'):
    with h5py.File(file_path, "r") as f:
        data = f[model]
        probabilities = data["P"][()]
        thresholds = data["thresholds"][()]

        return probabilities, thresholds


def get_ethogram_predictions(
    percentiles,
    probabilities,
    thresholds
):

    processor = deepethogram.postprocessing.MinBoutLengthPerBehaviorPostprocessor(
        thresholds, percentiles)
    predictions = processor(probabilities)

    return predictions


def thresh_step_confidence(
    dlc_filtered_steps,
    df_coords_f,
    probabilities,
    valid_threshold=0.7,
    graph_preview=False
):
    # get step probabilities at each detected steps
    dlc_step_confidence = []
    for step_idx in dlc_filtered_steps:
        if len(probabilities[step_idx-3:step_idx+4, 3]) != 0:
            dlc_step_confidence.append(
                max(probabilities[step_idx-3:step_idx+4, 3]))
        else:
            dlc_step_confidence.append(probabilities[step_idx, 3])
    dlc_step_confidence = np.array(dlc_step_confidence)

    y = df_coords_f["foot_y"].iloc[:]
    confident_step_idx = (dlc_step_confidence > valid_threshold)

    if graph_preview:
        # plot histogram of step confidence
        plt.hist(dlc_step_confidence, bins=100)
        plt.show()

        # plot classified steps
        plt.figure(figsize=(60, 4))
        plt.plot(y, lw=1, c="k")
        plt.plot(
            dlc_filtered_steps[confident_step_idx],
            y[dlc_filtered_steps[confident_step_idx]], "x", c="g")
        plt.plot(
            dlc_filtered_steps[~confident_step_idx],
            y[dlc_filtered_steps[~confident_step_idx]], "x", c="r")
        plt.show()

    return confident_step_idx


def thresh_step_motion(
    dlc_filtered_steps,
    df_coords_f,
    predictions,
    graph_preview=False
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
            y[dlc_filtered_steps[motion_steps_idx]], "x", c="g")
        plt.plot(
            dlc_filtered_steps[~motion_steps_idx],
            y[dlc_filtered_steps[~motion_steps_idx]], "x", c="r")
        plt.show()

    return motion_steps_idx


# ====================================================================================

# Load Images
def load_images_from_folder(folder, extention="*.jpg"):
    images = []
    filenames_path = glob.glob(os.path.join(folder, extention))
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return np.array(images), os.listdir(folder)


def extract_onset_images(video_path, filtered_peaks, df_coords, folder="./tmp_images/"):
    # Extract images corresponding to peak position

    # Clear destination folder
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))

    cap = cv2.VideoCapture(video_path)
    for frame_num in filtered_peaks:
        cap.set(1, frame_num)
        flag, frame = cap.read()
        cv2.imwrite(os.path.join(folder, f"{frame_num:07d}.jpg"), frame)

    # Cluster the images using kmeans

    # Perform UMAP clustering
    images, filenames = load_images_from_folder(folder)

    # Crop all images
    cropped_im_f = []
    crop_x = 50
    crop_y = 200

    for image, filename in zip(images, filenames):
        # Get image number
        im_index = int(os.path.basename(filename[:-4]))

        # Crop images according to x, y coordinates of the foot
        x_coor_f = round(df_coords.iloc[im_index, 0])
        y_coor_f = round(df_coords.iloc[im_index, 1])

        # Crop image
        cropped_im_f.append(
            image[y_coor_f - crop_y: y_coor_f + 20,
                  x_coor_f - crop_x: x_coor_f + crop_x]
        )
        # plt.imshow(image[y_coor_f+30:y_coor_f+crop_y, x_coor_f-crop_x:x_coor_f+crop_x])
        # plt.show()

    images = images.reshape(images.shape[0], -1)
    embedding = umap.UMAP().fit_transform(images[:, :])

    return embedding, filenames
