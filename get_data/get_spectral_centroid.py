import os
import librosa
import json


DATASET_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\SVD\\split_files'
JSON_PATH = 'C:\\Users\\semih\\OneDrive\\Uni\\Project\\Code\\data\\centroid_data.json'
DURATION = 1
SAMPLE_RATE = 22050


def save_mfcc(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=1):
    data = {
        'mapping':  [],
        'centroid': [],
        'labels': []
    }

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        if dirpath is not dataset_path:
            dirpath_components = dirpath.split('\\')
            label = dirpath_components[-1]
            data['mapping'].append(label)
            print('\nProccesing {}'.format(label))
            for f in filenames:
                file_path = os.path.join(dirpath, f)
                signal, sr = librosa.load(file_path, sr=SAMPLE_RATE)
                centroid = librosa.feature.spectral_centroid(
                    signal, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
                centroid = centroid.T
                data['centroid'].append([centroid.tolist()])
                data['labels'].append(i-1)
                print('{}'.format(file_path))

    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent=4)


if __name__ == '__main__':
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments=1)
