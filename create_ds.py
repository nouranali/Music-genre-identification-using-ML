import librosa
import numpy as np
import pandas as pd 
import os
from  sklearn.preprocessing import MinMaxScaler
from extract_features import extract_features

def main():
    samp_rate = 22050
    frame_size = 2048
    hop_size = 512
    dataset_dir:str=r"D:\AI_project\genres"
    sub_folders = get_subdirectories(dataset_dir)
    labels = []
    is_created = False
    print("Extracting features from audios...")
    for sub_folder in sub_folders:
        print(".....Working in folder:", sub_folder)
        sample_arrays = get_sample_arrays(dataset_dir, sub_folder, samp_rate)
        for sample_array in sample_arrays:
            row = extract_features(sample_array, samp_rate, frame_size, hop_size)
            if not is_created:
                dataset_numpy = np.array(row)
                is_created = True
            elif is_created:
                dataset_numpy = np.vstack((dataset_numpy, row))

            labels.append(sub_folder)
    print("Normalizing the data...")
    sc=MinMaxScaler(feature_range=(-1, 1))
    normalized_data = sc.fit_transform(dataset_numpy)
    Feature_Names = ['meanZCR', 'stdZCR', 'meanSpecCentroid', 'stdSpecCentroid', 'meanSpecContrast', 'stdSpecContrast',
                     'meanSpecBandwidth', 'stdSpecBandwidth', 'meanSpecRollof', 'stdSpecRollof',
                     'meanMFCC_1', 'stdMFCC_1', 'meanMFCC_2', 'stdMFCC_2', 'meanMFCC_3', 'stdMFCC_3',
                     'meanMFCC_4', 'stdMFCC_4', 'meanMFCC_5', 'stdMFCC_5', 'meanMFCC_6', 'stdMFCC_6',
                     'meanMFCC_7', 'stdMFCC_7', 'meanMFCC_8', 'stdMFCC_8', 'meanMFCC_9', 'stdMFCC_9',
                     'meanMFCC_10', 'stdMFCC_10', 'meanMFCC_11', 'stdMFCC_11', 'meanMFCC_12', 'stdMFCC_12',
                     'meanMFCC_13', 'stdMFCC_13'
                     ]
    dataset_pandas = pd.DataFrame(normalized_data , columns=Feature_Names)
    dataset_pandas["genre"] = labels
    dataset_pandas.to_csv("data.csv", index=False)
    print("Data set has been created and sent to the project folder!")

def get_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_sample_arrays(dataset_dir, folder_name, samp_rate):
    path_of_audios = librosa.util.find_files(dataset_dir + "/" + folder_name)
    audios = []
    for audio in path_of_audios:
        x, sr = librosa.load(audio, sr=samp_rate, duration=5.0)
        audios.append(x)
    audios_numpy = np.array(audios)
    return audios_numpy
if __name__ == '__main__':
    main()