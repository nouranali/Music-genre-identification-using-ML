import librosa
import numpy
import pandas
import os
import sklearn

def main():
    samp_rate = 22050
    frame_size = 2048
    hop_size = 512
    dataset_dir:str=r"../AI_project/genres"
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
                dataset_numpy = numpy.array(row)
                is_created = True
            elif is_created:
                dataset_numpy = numpy.vstack((dataset_numpy, row))
            labels.append(sub_folder)
    print("Normalizing the data...")
    scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
    dataset_numpy = scaler.fit_transform(dataset_numpy)
    all_Feature_Names = ['stft', 'cqt', 'cens', 'melspectogram', 
                     'rms', 'centroid',
                     'contrast', 'roloff',
                     'zcr', 'poly_features',
                     'tonnetz','bandwidth',
                     'MFCC_1', 'MFCC_2', 'MFCC_3', 
                     'MFCC_4',  'MFCC_5','MFCC_6', 
                     'MFCC_7', 'MFCC_8', 'MFCC_9', 
                     'MFCC_10',  'MFCC_11',  'MFCC_12', 
                     'MFCC_13',  'tempo'
                     ]
    dataset_pandas = pandas.DataFrame(dataset_numpy, columns=all_Feature_Names)

    dataset_pandas["genre"] = labels
    dataset_pandas.to_csv("features.csv", index=False)
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
    audios_numpy = numpy.array(audios)
    return audios_numpy


def extract_features(signal, sample_rate, frame_size, hop_size):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size,n_mfcc=13)
    stft = librosa.feature.chroma_stft(y=signal,sr=sample_rate)
    cqt= librosa.feature.chroma_cqt(y=signal,sr=sample_rate)
    cens=librosa.feature.chroma_cens(y=signal,sr=sample_rate)
    mel=librosa.feature.melspectrogram(y=signal,sr=sample_rate)
    rms=librosa.feature.rms(y=signal)
    poly= librosa.feature.poly_features(y=signal,sr=sample_rate)
    tonnetz=librosa.feature.tonnetz(y=signal,sr=sample_rate)
    tempo=librosa.feature.tempogram(y=signal,sr=sample_rate)[0]
    #f_tempo=librosa.feature.fourier_tempogram(y=signal,sr=sample_rate)
    ls=[]
    res=[]
    ls=[
        stft,
        cqt,
        cens,mel,rms,
        spectral_centroid,
        spectral_contrast,
        spectral_rolloff,
        zero_crossing_rate,
        poly,
        tonnetz,
        spectral_bandwidth,
        mfccs[0, :],       
        mfccs[1, :],
        mfccs[2, :],       
        mfccs[3, :],
        mfccs[4, :],
        mfccs[5, :],
        (mfccs[6, :]),
        (mfccs[7, :]),
        mfccs[8, :],
        mfccs[9, :],
        (mfccs[10, :]),
        (mfccs[11, :]),
       (mfccs[12, :]),
       tempo]
    for i in range(len(ls)):
        res.append(numpy.mean(ls[i]))
    return res

if __name__ == '__main__':
    main()