from cmath import nan
import os
import librosa
import json
import numpy as np
import parselmouth
import math

DATASET_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\SVD\\split_files'
JSON_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\data\\raw_data.json'
DURATION = 1
SAMPLE_RATE = 22050
period = 1/44


def save_mfcc(dataset_path, json_path):

    data = {
        'mapping':  [],
        'raw': [],
        'labels': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split('\\')
            label = dirpath_components[-1]
            data['mapping'].append(label)
            print('\nProccesing {}'.format(label))

            for f in filenames:
                raw_list = []
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                signal = signal[0:22049]

                for j in range(44):
                    signal_segment = signal[j * 501: (j+1) * 501]
                    raw_list.append(signal_segment.tolist())

                data['raw'].append(raw_list)
                data['labels'].append(i-1)
                print('{}'.format(file_path))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH)
