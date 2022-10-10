import os
import librosa
import json
import parselmouth
import math

DATASET_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\SVD\\split_files'
JSON_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\data\\jitter_data.json'

DURATION = 1
SAMPLE_RATE = 22050

period = 1/44


def save_mfcc(dataset_path, json_path):
    data = {
        'mapping':  [],
        'jitter': [],
        'labels': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split('\\')
            label = dirpath_components[-1]
            data['mapping'].append(label)
            print('\nProccesing {}'.format(label))

            for f in filenames:
                jitter_list = []
                file_path = os.path.join(dirpath, f)

                for j in range(44):
                    snd = parselmouth.Sound(file_path)
                    pitch = snd.to_pitch()
                    pulses = parselmouth.praat.call(
                        [snd, pitch], "To PointProcess (cc)")
                    pointProcess = parselmouth.praat.call(
                        snd, "To PointProcess (periodic, cc)", 75, 300)

                    jitter_local = parselmouth.praat.call(
                        pulses, "Get jitter (local)", j * period, (j+1) * period, 0.00001, 0.02, 1.3) * 100

                    if math.isnan(jitter_local):
                        jitter_local = 0

                    jitter_local = [jitter_local]
                    jitter_list.append(jitter_local)

                data['jitter'].append(jitter_list)
                data['labels'].append(i-1)
                print('{}'.format(file_path))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH)
