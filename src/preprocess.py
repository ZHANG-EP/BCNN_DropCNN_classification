# Author: Jesse Williams
# Company: Global Technology Connection
# Last updated: LBarama 2020 05 20

import numpy as np
from scipy.signal import butter, lfilter
from scipy import signal
from obspy import Trace

import defaults

stats = {"sampling_rate":defaults.SAMPLING_RATE}
def process_data(dataset, sampling_rate=20,
                 taper_fraction=0.01,  # taper function inputs
                 highpass_cutoff=1, highpass_order=4,  # high pass filter inputs
                 lowpass_cutoff=5, lowpass_order=4):  # low pass filter inputs
    """Processes a dataset of shape (n_observations, n_channels, n_readings)
    For each wave it applies:
    centering -> taper -> highpass -> lowpass"""

    # By copying the dataset, the orginal dataset will remain unaltered
    dataset = dataset.astype(np.float32).copy()
    # Process data by each observation and wave
    for idx, observation in enumerate(dataset):
        channel0 = observation[0]
        channel0 = center(channel0)
        channel0 = detrend(channel0)
        channel0 = soft_clip(channel0)
        channel0 = taper(channel0, taper_fraction)
        channel0 = normalize(channel0)
        # channel0 = Trace(data=channel0,header=stats).resample(defaults.NEW_SAMPLING_RATE).data
        channel0 = highpass_filter(channel0, highpass_cutoff, sampling_rate, highpass_order)
        channel0 = lowpass_filter(channel0, lowpass_cutoff, sampling_rate, lowpass_order)
        dataset[idx][0] = channel0

        # channel1 = observation[1]
        # channel1 = detrend(channel1)
        # channel1 = center(channel1)
        # channel1 = soft_clip(channel1)
        # channel1 = taper(channel1, taper_fraction)
        # channel1 = highpass_filter(channel1, highpass_cutoff, sampling_rate, highpass_order)
        # channel1 = lowpass_filter(channel1, lowpass_cutoff, sampling_rate, lowpass_order)
        # dataset[idx][1] = channel1
        #
        # channel2 = observation[2]
        # channel2 = center(channel2)
        # channel2 = detrend(channel2)
        # channel2 = soft_clip(channel2)
        # channel2 = taper(channel2, taper_fraction)
        # channel2 = highpass_filter(channel2, highpass_cutoff, sampling_rate, highpass_order)
        # channel2 = lowpass_filter(channel2, lowpass_cutoff, sampling_rate, lowpass_order)
        # dataset[idx][2] = channel2

    # Round the dataset to reduce the size of the array
    # dataset = np.around(dataset,decimals=decimals)
    return np.expand_dims(dataset[:, 0], axis=1)


def center(wave):
    """Centers the wave based on the mean"""
    wave = wave - wave.mean()
    return wave
def detrend(wave):
    wave = signal.detrend(wave,type="linear")
    return wave
def normalize(wave):
    wave = wave / (np.max(np.abs(wave)))
    return wave
def taper(wave, taper_fraction):
    taper_length = int(len(wave) * taper_fraction)
    # Build the start tapper
    taper_range = np.arange(taper_length)
    taper_start = np.sin(taper_range / taper_length * np.pi / 2)
    # Build the end tapper
    taper_end = np.sin(np.pi / 2 + taper_range / taper_length * np.pi / 2)
    # Build a center section of only 1s
    taper_center = np.ones(len(wave) - 2 * taper_length)
    # Concatenate the start, center, and end
    taper_function = np.concatenate([taper_start, taper_center, taper_end])
    # Multiply the wave by the taper function
    wave = wave * taper_function
    return wave


def highpass_filter(wave, highpass_cutoff, sampling_rate, highpass_order):
    """ High pass filter using a butter-highpass.
    The cutoff and sampling_rate parameters are in Hz.
    The order dictates the attenuation after the cutoff frequency.
    A low order has a long attenuation.
    Ref website: https://stackoverflow.com/questions/25191620/creating-lowpass-filter-in-scipy-understanding-methods-and-units"""

    nyq = 0.5 * sampling_rate
    normal_cutoff = highpass_cutoff / nyq
    b, a = butter(highpass_order, normal_cutoff, btype='high', analog=False)

    wave = lfilter(b, a, wave)

    return wave


def lowpass_filter(wave, lowpass_cutoff, sampling_rate, lowpass_order):
    """ Low pass filter using a butter-lowpass.
    The cutoff and sampling_rate parameters are in Hz.
    The order dictates the attenuation after the cutoff frequency.
    A low order has a long attenuation."""

    nyq = 0.5 * sampling_rate
    normal_cutoff = lowpass_cutoff / nyq
    b, a = butter(lowpass_order, normal_cutoff, btype='low', analog=False)

    wave = lfilter(b, a, wave)

    return wave


def soft_clip(wave):
    """Changed to just normalize the data"""
    import warnings
    # print(max(np.abs(wave)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")  # 将警告转为错误
            data = wave / max(np.abs(wave))
        return data
    except (RuntimeWarning, ValueError) as e:
        print(f"Error occurred! Wave data:\n{wave}")
        print(f"Details: {e}")
        return None  # 或处理无效数据


def scale_by_varience(wave):
    """Reduces sample by deviding by the varience of every wave"""

    return wave / (np.var(wave) ** 0.5)


def shuffle_data(dataset, labels):
    """Shuffles the dataset and labels to randomize them for training"""
    indexes = np.arange(len(labels))
    np.random.shuffle(indexes)  # in this case inplace=True

    # Make holding lists
    samples_shuffled = []
    labels_shuffled = []

    for index_shuffled in indexes:
        wave = dataset[index_shuffled]
        label = labels[index_shuffled]

        samples_shuffled.append(wave)
        labels_shuffled.append(label)

    samples_shuffled = np.array(samples_shuffled)
    labels_shuffled = np.array(labels_shuffled)

    return samples_shuffled, labels_shuffled


def reshape_data(dataset):
    """Takes dataset of (n_observations, n_channels, n_readings)
    and returns (n_observations, n_readings, n_channels)"""
    reshape_sample = np.array([sample.T for sample in dataset])
    return reshape_sample
