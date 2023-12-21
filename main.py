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
trial_name = trial_names[4]

valid_step_idx = utils.get_valid_step_idx(
    cfg, 
    trial_name, 
    graph_preview=cfg.extraction.preview.valid_step)