import os
import struct
import sys
import mmap

import numpy as np


class DatFileLoader(object):
    def __init__(
        self,
        dat_path,
        num_chs,
        ds_ratio,
        num_ads_chs=8,
        verbose=False
    ):
        """
        Parameters:
            dat_path: string
                The path to the .dat file
            num_chs: int
                The number of channels in the .dat file
            ds_ratio: int
                The downsampling ratio of the output file
            num_ads: int
                Number of Analog Digital input channels, default is 8
            verbose: bool
                If True, returns parameters
        """
        self.dat_path = dat_path
        self.num_chs = num_chs
        self.ds_ratio = ds_ratio
        self.num_ads_chs = num_ads_chs
        self.verbose = verbose
        self.num_data_chs = int(num_chs-num_ads_chs)
        self.raw_ads = np.array([])
        self.data = np.array([])
        self.trigger = np.array([])
        self.trigger_ch = None
        self.thresh = None

        # extract data
        self.extract_data()
        self.report()

    def report(self):
        print("Data Path:", self.dat_path)
        print("Data Size:", self.data.shape)
        print("Signal Channels: 0 -", self.num_data_chs-1)
        print("Analog Input Channels:", self.num_data_chs, "-", self.num_chs-1)
        print("Downsampling Rate:", self.ds_ratio)

    def get_data(self):
        """Function that returns extracted data
        """
        if self.data.size == 0:
            self.extract_data()
            if self.verbose:
                self.report()
        return self.data

    def get_raw_trigger(self):
        """Function that returns extracted data
        """
        if self.raw_ads.size == 0:
            self.extract_data()
            if self.verbose:
                self.report()
        print("Warning: Raw trigger is not downsampled!")

        return self.raw_ads

    def get_trigger(self, trigger_ch, thresh):
        """Function that returns extracted trigger timepoint data
            Parameters:
            trigger_ch: int or list of ints
                Channel that stores trigger
            thresh: float
                Threshold to detect analog inputs
        """

        if self.trigger.size == 0:
            if trigger_ch != self.trigger_ch or thresh != self.thresh:
                self.extract_trigger(self.raw_ads[:, trigger_ch], thresh)
                self.trigger_ch = trigger_ch
                self.thresh = thresh
        return self.trigger

    def clear(self):
        """Clear memory
        """
        self.raw_ads = np.array([])
        self.data = np.array([])
        self.trigger = np.array([])

    def mmap_io(self):
        """Function to read binary file and return binary data.
        Returns:
            text: binary data
        """
        print(
            f"Loading: {self.dat_path.split('/')[-7]} {self.dat_path.split('/')[-4]}")
        with open(self.dat_path, mode="r", encoding="utf8") as file_obj:
            with mmap.mmap(file_obj.fileno(), length=0, access=mmap.ACCESS_READ) as mmap_obj:
                text = mmap_obj.read()
        return text

    def extract_trigger(self, input_array, thresh):
        """ A funtion to extract timepoints corssing the threshold

        Parameters:
            input_array: np.array (, channels)
                The input array to be analyzed
            thresh: float
                The threshold for the trigger
        """

        trigger_tp = []

        # Determine if input is multi-channel or not
        if len(input_array.shape) == 1:
            # Convert raw signal to 1s and 0s
            buf = np.zeros_like(input_array)
            over_thresh = np.where(input_array >= thresh)
            buf[over_thresh] = 1

            # Create array that reflect change
            # 1 is shutter begining, -1 is shutter ending, 0 is no change
            diff = np.diff(buf)

            trigger_tp = np.round(np.where(np.asarray(diff) == 1)[
                                  0]/self.ds_ratio).astype(np.int64)

        else:
            for ch in range(input_array.shape[1]):
                # Convert raw signal to 1s and 0s
                buf = np.zeros_like(input_array[:, ch])
                over_thresh = np.where(input_array[:, ch] >= thresh)
                buf[over_thresh] = 1

                # Create array that reflect change
                # 1 is shutter begining, -1 is shutter ending, 0 is no change
                diff = buf - np.insert(buf[:-1], 0, buf[0])

                trigger_tp.append(np.round(np.where(np.asarray(diff) == 1)[
                                  0]/self.ds_ratio).astype(np.int64))

            trigger_tp = np.asarray(trigger_tp)

        self.trigger = trigger_tp

    def extract_data(self):
        """Function to unpack dat files.
        Returns:
            out_array: np.array (, channels)
        """

        # read dat file
        bytes_data = self.mmap_io()
        data = np.frombuffer(bytes_data, dtype=np.int16)

        # Reshape to sample*channels
        out_array = data.reshape(-1, self.num_chs).astype(np.float64)

        # Adjust volts
        out_array = self.adjust_volts(out_array)

        self.raw_ads = out_array[:, self.num_data_chs:]

        # Downsample
        out_array = out_array[::self.ds_ratio]

        self.data = out_array[:, :self.num_data_chs]

    def adjust_volts(self, data):
        """Function to adjust recorded voltage

        Parameters:
            data: np.array (, channels)
                The input arrays
        Returns:
            input_array: np.array (, channels)

        """
        # Cast dtype to float64
        data = data.astype(np.float64)

        # Convert recording channels
        for i in range(self.num_data_chs):
            data[:, i] = data[:, i] * 0.19499999284744262695

        # Convert analog/digital in/out channels
        for j in range(self.num_data_chs, self.num_chs):
            data[:, j] = data[:, j] * 3.7400001019705086946e-05

        return data
