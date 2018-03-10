import librosa
import numpy as np
import scipy
import torch

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def set_speech_format(f):
    f.setnchannels(1)
    f.setsampwidth(2)
    f.setframerate(16000)

def preprocess_audio(data, n_mels, dct_filters, feature="mfcc"):
    ## DCT Part
    if feature == "mfcc":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        data = torch.from_numpy(data)
    elif feature == "fbank":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = data.astype(np.float32).transpose()
        data = torch.from_numpy(data)
    elif feature == "fft":
        data = fft_audio(data, 0.025, 0.010)
    else:
        raise NotImplementedError
    # return data.unsqueeze(0)
    return data

def fft_audio(data, window_size, window_stride):
    n_fft = 480
    # n_fft = int(16000*window_size)
    win_length = int(16000* window_size)
    hop_length = int(16000* window_stride)
    # STFT
    D = librosa.stft(data, n_fft=n_fft, hop_length=hop_length,
                     win_length=win_length, window=windows['hamming'])
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect) # (freq, time)
    spect = torch.FloatTensor(spect.T) # (time, freq)
    # normalization
    mean = spect.mean(0) # over time dim
    std = spect.std(0)
    spect.add_(-mean)
    spect.div_(std)
    return spect

def preprocess_from_path(config, audio_path, method="mfcc"):
    data = librosa.core.load(audio_path, sr=16000)[0]
    n_mels = config['n_mels']
    filters = librosa.filters.dct(config["n_dct_filters"], config["n_mels"])
    # in_len = config['input_length']
    # if len(data) > in_len:
        # # cliping the audio
        # start_sample = np.random.randint(0, len(data) - in_len)
        # data = data[start_sample:start_sample+in_len]
        # # data = data[:in_len]
    # else:
        # # zero-padding the audio
        # data = np.pad(data, (0, max(0, in_len - len(data))), "constant")
    data = preprocess_audio(data, n_mels, filters, method)
    return data
