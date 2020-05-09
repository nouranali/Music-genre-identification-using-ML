import librosa
import numpy as np
def extract_features(signal, sample_rate, frame_size, hop_size):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                            hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    return [

        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_contrast),
        np.std(spectral_contrast),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),
        np.mean(mfccs[1, :]),
        np.std(mfccs[1, :]),
        np.mean(mfccs[2, :]),
        np.std(mfccs[2, :]),
        np.mean(mfccs[3, :]),
        np.std(mfccs[3, :]),
        np.mean(mfccs[4, :]),
        np.std(mfccs[4, :]),
        np.mean(mfccs[5, :]),
        np.std(mfccs[5, :]),
        np.mean(mfccs[6, :]),
        np.std(mfccs[6, :]),
        np.mean(mfccs[7, :]),
        np.std(mfccs[7, :]),
        np.mean(mfccs[8, :]),
        np.std(mfccs[8, :]),
        np.mean(mfccs[9, :]),
        np.std(mfccs[9, :]),
        np.mean(mfccs[10, :]),
        np.std(mfccs[10, :]),
        np.mean(mfccs[11, :]),
        np.std(mfccs[11, :]),
        np.mean(mfccs[12, :]),
        np.std(mfccs[12, :]),
        np.mean(mfccs[13, :]),
        np.std(mfccs[13, :]),
    ]
