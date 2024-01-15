import os
import h5py
import logging
import traceback

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

import utils
from dat_loader import DatFileLoader

# Initialize the log settings
logging.basicConfig(filename="main.log", level=logging.DEBUG)

# Load config
cfg = OmegaConf.load("config.yaml")
print(type(cfg))

# Get list of tasks
task_cfg = pd.read_csv(cfg.paths.task_config, header=0, index_col=0)
trial_names = task_cfg["trial"].drop_duplicates()

# iterate through all trials
error_trials = []
video_stopped_first_trials = []
for trial_name in trial_names:
    try:
        # Check .dat file existence
        trial_df = task_cfg[task_cfg["trial"] == trial_name]
        if str(trial_df["dat_path"].values[0]) == "nan":
            logging.info(f"No .dat file found for {trial_name}")
            continue
        dat_path = os.path.join(
            cfg.paths.dat_dir, trial_df["dat_path"].values[0], "continuous.dat"
        )

        # Check if already calculated
        onset_savedir = os.path.join(
            "/".join(dat_path.split("/")[:-3]),
            "events/Acquisition_Board-100.Rhythm Data/",
        )
        onset_h5file = os.path.join(onset_savedir, "step_onsets.h5")
        if os.path.exists(onset_h5file):
            print(f"{trial_name} already analyzed")
            print("overwriting...")
            # continue

        # ======================================================================
        # Load dat file

        dat_loader = DatFileLoader(
            dat_path,
            num_chs=cfg.extraction.dat.num_chs,
            ds_ratio=cfg.extraction.dat.downsampling_ratio,
            num_ads_chs=cfg.extraction.dat.num_ads_chs,
            verbose=False,
        )
        trigger = dat_loader.get_trigger(trigger_ch=0, thresh=0.4)

        # Split trigger if multiple
        samp_rate = int(
            cfg.extraction.dat.sampling_rate / cfg.extraction.dat.downsampling_ratio
        )
        # Data is considered discontinuous when trigger interval is larger than 1s
        segment_start_idx = np.where(np.diff(trigger) > samp_rate)[0] + 1
        trigger_segments = np.split(trigger, segment_start_idx)

        # ======================================================================
        # Detect onsets
        step_onset_segments = []
        step_texture_segments = []
        for video_fragment in trial_df["name_old"]:
            # Detect onsets
            valid_step_idx = utils.get_valid_step_idx(
                cfg, video_fragment, graph_preview=False
            )
            step_onset_segments.append(valid_step_idx)

            # Get annotation for floor textures
            step_texture = utils.classify_floor_type(
                cfg, video_fragment, valid_step_idx
            )
            step_texture_segments.append(step_texture)
        texture = [x for l in step_texture_segments for x in l]

        assert len(trigger_segments) == len(
            step_onset_segments
        ), f"dat-segments and video-numbers does not match ({len(trigger_segments)}, {len(step_onset_segments)})"

        onset_segments = []
        for trigger_seg, step_onset_seg in zip(trigger_segments, step_onset_segments):
            if len(trigger_seg) > max(step_onset_seg):
                onset_segments.append(trigger_seg[step_onset_seg])
            else:
                cut_step_onset_seg = step_onset_seg[
                    np.where(step_onset_seg < len(trigger_seg))
                ]
                onset_segments.append(trigger_seg[cut_step_onset_seg])
                video_stopped_first_trials.append(trial_name)

        onsets = [x for l in onset_segments for x in l]

        # Save to h5 file
        with h5py.File(onset_h5file, "w") as h5:
            h5.create_dataset(name="step_idx", data=onsets)
            h5.create_dataset(name="texture", data=texture)
    except Exception as e:
        logging.error(f"Error while processing {trial_name} {video_fragment}")
        logging.error(traceback.format_exc())
