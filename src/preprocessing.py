import pandas as pd

from src import DATA
import os
import numpy as np
import matplotlib.pyplot as plt

SAMPLES_PER_SEC = 128
NUM_CHANNELS = 16
DESC_CHANNELS = ["F7", "F3", "F4", "F8", "T3", "C3", "Cz", "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]
TIME = 60


def process_file(path: str, file_name: str):
    activations = [set() for _ in range(TIME)]

    with open(os.path.join(path, file_name), 'r') as file:
        data = np.array([float(line) for line in file.readlines()])
        channels = np.split(data, NUM_CHANNELS)
        time = 0

        for i in range(len(channels)):
            channel = np.split(channels[i], len(channels[i]) / SAMPLES_PER_SEC)

            # reduce noise
            signal_per_sec = [np.median(sec) for sec in channel]
            time = len(signal_per_sec)

            #iqr = np.quantile(signal_per_sec, 0.75) - np.quantile(signal_per_sec, 0.25)
            upper_bar = 0

            for j in range(len(signal_per_sec)):
                if signal_per_sec[j] >= upper_bar:
                    activations[j].add(f"{DESC_CHANNELS[i]}")
            break

    return activations, time


def prepare_df(data: list, name: str):
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(DATA, f"{name}.csv"), header=False, index=False)


def get_data(path, description):
    data = list()
    for file in os.listdir(path):
        activation, time = process_file(path, file)
        for element in activation:
            # cleaning empty sets
            if len(element) > 0:
                data.append(element)

        prepare_df(data, description)


if __name__ == '__main__':
    # prepare df for normal data
    norm_path = os.path.join(DATA, "norm")
    get_data(norm_path, "norm_data")

    # prepare df for schizophrenia data
    schiz_path = os.path.join(DATA, "sch")
    get_data(schiz_path, "schiz_data")
