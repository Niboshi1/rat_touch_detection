import glob
import os
import h5py

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import utils

# Load config
cfg = OmegaConf.load('config.yaml')

# Video names
trial_names = [os.path.basename(x.split("DLC")[0]) for x in glob.glob(os.path.join(cfg.data.dlc_labels_path, "*labeled.mp4"))]

# DeepEthogram annotated files
deepethogram_dirs = cfg.data.ethogram_labels_path

ethogram_results = {}
for dirs in deepethogram_dirs:
    for root, dirs, files in os.walk(os.path.join(dirs, "DATA")):
        for file in files:
            if file.endswith(".h5"):
                ethogram_results[file[:-11]] = os.path.join(root, file)

# DeepLabCut annocated files
dlc_results = {}
for trial_name in trial_names:
    dlc_results[trial_name] = os.path.join(
        cfg.data.dlc_labels_path, f"{trial_name}DLC_effnet_b6_rat_stepsDec12shuffle1_1030000.h5")

trial_name = trial_names[4]

assert os.path.exists(dlc_results[trial_name])
assert os.path.exists(ethogram_results[trial_name])

# DeepLabCut =========================================================================
df_coords, df_coords_f = utils.extract_coordinates(
    dlc_results[trial_name],
    set_reference="x",
    graph_preview=cfg.extraction.preview.raw_dlc_trace
)
dlc_filtered_steps = utils.extract_footstrike_onsets_f(
    df_coords,
    df_coords_f,
    graph_preview=cfg.extraction.preview.dlc_extracted_onset)

# DeepEthogram =========================================================================
bout_percentiles_file_path = cfg.data.bout_percentiles_path

with h5py.File(bout_percentiles_file_path, "r") as f:
    percentiles = f['percentiles'][()]

probabilities, thresholds = utils.read_ethogram_result(
    ethogram_results[trial_name])

predictions = utils.get_ethogram_predictions(
    percentiles,
    probabilities,
    thresholds
)

confident_step_idx = utils.thresh_step_confidence(
    dlc_filtered_steps,
    df_coords_f,
    probabilities,
    valid_threshold=cfg.extraction.process.ethogram_confidence_thresh,
    graph_preview=cfg.extraction.preview.ethogram_step_confidence
)

motion_step_idx = utils.thresh_step_motion(
    dlc_filtered_steps,
    df_coords_f,
    predictions,
    graph_preview=cfg.extraction.preview.ethogram_step_motion
)

dlc_step_floor = utils.classify_floor_type(
    dlc_filtered_steps,
    predictions
)

valid_step_idx = confident_step_idx*motion_step_idx
print(f"Ethogram: Valid onsets:{sum(valid_step_idx)}")

# plot classified steps
y = df_coords_f["foot_y"].iloc[:]
plt.figure(figsize=(60, 4))
plt.plot(y, lw=1, c="k")
plt.plot(
    dlc_filtered_steps[valid_step_idx],
    y[dlc_filtered_steps[valid_step_idx]], "x", c="g")
plt.plot(
    dlc_filtered_steps[~valid_step_idx],
    y[dlc_filtered_steps[~valid_step_idx]], "x", c="r")
plt.show()