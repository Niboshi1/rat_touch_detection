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
trial_names = [os.path.basename(x.split(".mp4")[0]) for x in glob.glob(
    os.path.join(cfg.paths.dlc_labels, "*.mp4"))]

ethogram_results = utils.get_ethogram_annotated_files(cfg)


error_trials = []
for trial_name in trial_names:
    try:
        ethogram_results[trial_name]
    except KeyError:
        error_trials.append(trial_name)
        print("Trial is not trained on DeepEthogram", trial_name)
        continue

    # If there are no labels, update the label
    label_file_path = ethogram_results[trial_name].split("outputs.h5")[
        0]+"labels.csv"
    if os.path.exists(label_file_path):
        print(f"Labels for {trial_name} already exists!")

    else:
        # get valid step index
        try:
            valid_step_idx = utils.get_valid_step_idx(
                cfg,
                trial_name,
                graph_preview=False)
        except KeyError:
            error_trials.append(trial_name)
            continue

        if len(valid_step_idx) > 0:
            # load predictions
            bout_percentiles_file_path = cfg.paths.bout_percentiles

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

            # Save labels
            print(f"Saving: {label_file_path}")
            df.to_csv(label_file_path)

print(error_trials)


# remove labels without small steps
for trial_name in trial_names:
    try:
        ethogram_results[trial_name]
    except KeyError:
        print("Trial is not trained on DeepEthogram", trial_name)
        continue

    # If there are no labels, update the label
    label_file_path = ethogram_results[trial_name].split("outputs.h5")[
        0]+"labels.csv"
    if os.path.exists(label_file_path):
        df = pd.read_csv(label_file_path)
        if sum(df["touch"].values) < 100:
            os.remove(label_file_path)
