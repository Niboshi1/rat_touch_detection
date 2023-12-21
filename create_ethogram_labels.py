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
trial_names = [os.path.basename(x.split(".mp4")[0]) for x in glob.glob(os.path.join(cfg.data.dlc_labels_path, "*.mp4"))]

# DeepLabCut annocated files
dlc_results = {}
for trial_name in trial_names:
    dlc_results[trial_name] = os.path.join(
        cfg.data.dlc_labels_path, f"{trial_name}DLC_effnet_b6_rat_stepsDec12shuffle1_1030000.h5")
    
ethogram_results = utils.get_ethogram_annotated_files(cfg)



for trial_name in trial_names:

    # get valid step index
    valid_step_idx = utils.get_valid_step_idx(
        cfg, 
        trial_name, 
        graph_preview=cfg.extraction.preview.valid_step)
    
    # load predictions
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

    # Create pandas DataFrame
    df = pd.DataFrame(predictions, columns=cfg.data.class_names)

    # Update steps
    new_touch = np.zeros(len(predictions))
    new_touch[valid_step_idx] = 1
    df['touch'] = new_touch.astype(int)
    df['background'].iloc[valid_step_idx] = 0

    # If there are no labels, update the label
    label_file_path = ethogram_results[trial_name].split("outputs.h5")[0]+"labels.csv"
    if os.path.exists(label_file_path):
        print(f"File: {label_file_path} already exists!")
    else:
        print(f"Saving: {label_file_path}")
        df.to_csv(label_file_path)