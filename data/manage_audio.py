import librosa
import numpy as np
import scipy
import torch
from pydub import AudioSegment

windows = {'hamming': scipy.signal.hamming,
           'hann': scipy.signal.hann,
           'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def preprocess_audio(
        config, data, n_mels,
        dct_filters, in_feature="mfcc"):
    # Mel Spectrogram
    # n_fft(480) = 30ms, hop(160) = 10ms, overlap = 20ms
    # n_fft(400) = 25ms, hop(240) = 15ms, overlap = 10ms
    # fmax = 8000 for 16KHz sampling
    sr = config["sample_rate"]
    n_fft = int(sr * config["window_size"])
    stride_length = int(sr * config["window_stride"])
    hop_length = n_fft - stride_length
    data = librosa.feature.melspectrogram(
            data, sr=sr, n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft, fmin=20, fmax=sr//2)
    data[data > 0] = np.log(data[data > 0])
    if in_feature == "mfcc":
        ## DCT Part
        data = [np.matmul(dct_filters, x)
                for x in np.split(
                    data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        ## appending deltas
        # data_delta = librosa.feature.delta(data)
        # data_delta2 = librosa.feature.delta(data, order=2)
        # data = np.stack([data, data_delta, data_delta2], axis=0)
    elif in_feature == "fbank":
        data = data.astype(np.float32).transpose()
    else:
        raise NotImplementedError

    # Data Whitening
    # data: (time, mels)
    data = torch.from_numpy(data)
    data.add_(-data.mean(0))
    data.div_(data.std(0))
    return data  # dims:3, with no channel dimension.

def norm_strip_audio(wav_path):
    data = AudioSegment.from_wav(wav_path)
    data = data.normalize()
    data = data.strip_silence(silence_len=600,
            silence_thresh=-16, padding=600)
    data = (np.array(data.get_array_of_samples())
            / 32768.0).astype(np.float32)

    return data

