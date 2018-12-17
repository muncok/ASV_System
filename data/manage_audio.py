import librosa
import numpy as np
import scipy
import torch
from pydub import AudioSegment

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}

def preprocess_audio(data, n_mels, dct_filters, in_feature="mfcc"):
    ## DCT Part
    if in_feature == "mfcc":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = [np.matmul(dct_filters, x) for x in np.split(data, data.shape[1], axis=1)]
        data = np.array(data, order="F").squeeze(2).astype(np.float32)
        ## appending deltas
        # data_delta = librosa.feature.delta(data)
        # data_delta2 = librosa.feature.delta(data, order=2)
        # data = np.stack([data, data_delta, data_delta2], axis=0)
        data = torch.from_numpy(data)
    elif in_feature == "fbank":
        data = librosa.feature.melspectrogram(data, sr=16000, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        data[data > 0] = np.log(data[data > 0])
        data = data.astype(np.float32).transpose()
        data = torch.from_numpy(data)
        mean = data.mean(0) # along time dimension
        data.add_(-mean)
        # std = data.std(0)
        # data.div_(std)
    else:
        raise NotImplementedError
    return data  # dims:3, with no channel dimension.

def norm_strip_audio(wav_path):
    data = AudioSegment.from_wav(wav_path)
    data = data.normalize()
    data = data.strip_silence(silence_len=600, silence_thresh=-16, padding=600)
    data = (np.array(data.get_array_of_samples())
            / 32768.0).astype(np.float32)

    return data

def strip_audio(x, frame_length=1024, hop_length=256, rms_ths=0.2):
    # compute energy
    rmse = librosa.feature.rmse(x, frame_length=frame_length, hop_length=hop_length)[0]
    rms_ratio = rmse/rmse.max()

    active_frames = np.nonzero(rms_ratio > rms_ths)[0]
    assert len(active_frames) > 0, "there is no voice part in the wav"

    # strip continous active part
    s_sample = librosa.frames_to_samples(active_frames[0], hop_length=hop_length)[0]
    e_sample = librosa.frames_to_samples(active_frames[-1], hop_length=hop_length)[0]

    return x[s_sample:e_sample]
