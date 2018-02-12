import os
import subprocess
from tempfile import NamedTemporaryFile

import librosa
import speechpy
import numpy as np
import scipy.signal
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

windows = {'hamming': scipy.signal.hamming, 'hann': scipy.signal.hann, 'blackman': scipy.signal.blackman,
           'bartlett': scipy.signal.bartlett}


def load_audio(path, norm=False):
    sound, _ = torchaudio.load(path, normalization=norm)
    sound = sound.numpy()
    if len(sound.shape) > 1:
        if sound.shape[1] == 1:
            sound = sound.squeeze()
        else:
            sound = sound.mean(axis=1)  # multiple channels, average
    return sound


class AudioParser(object):
    def parse_transcript(self, transcript_path):
        """
        :param transcript_path: Path where transcript is stored from the manifest file
        :return: Transcript in training/testing format
        """
        raise NotImplementedError

    def parse_audio(self, audio_path):
        """
        :param audio_path: Path where audio is stored from the manifest file
        :return: Audio in training/testing format
        """
        raise NotImplementedError


class NoiseInjection(object):
    def __init__(self,
                 path=None,
                 sample_rate=16000,
                 noise_levels=(0, 0.5)):
        """
        Adds noise to an input signal with specific SNR. Higher the noise level, the more noise added.
        Modified code from https://github.com/willfrey/audio/blob/master/torchaudio/transforms.py
        """
        if not os.path.exists(path):
            print("Directory doesn't exist: {}".format(path))
            raise IOError
        self.paths = path is not None and librosa.util.find_files(path)
        self.sample_rate = sample_rate
        self.noise_levels = noise_levels

    def inject_noise(self, data):
        noise_path = np.random.choice(self.paths)
        noise_level = np.random.uniform(*self.noise_levels)
        return self.inject_noise_sample(data, noise_path, noise_level)

    def inject_noise_sample(self, data, noise_path, noise_level):
        noise_len = get_audio_length(noise_path)
        data_len = len(data) / self.sample_rate
        noise_start = np.random.rand() * (noise_len - data_len)
        noise_end = noise_start + data_len
        noise_dst = audio_with_sox(noise_path, self.sample_rate, noise_start, noise_end)
        assert len(data) == len(noise_dst)
        noise_energy = np.sqrt(noise_dst.dot(noise_dst)/noise_dst.size)
        data_energy = np.sqrt(data.dot(data)/data.size)
        data += noise_level * noise_dst * data_energy / noise_energy
        return data


class SpectrogramParser(AudioParser):
    def __init__(self, audio_conf, normalize=False, augment=False):
        """
        Parses audio file into spectrogram with optional normalization and various augmentations
        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param normalize(default False):  Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        super().__init__()

        self.window_stride = audio_conf['window_stride']
        self.window_size = audio_conf['window_size']
        self.sample_rate = audio_conf['sample_rate']
        self.window = windows.get(audio_conf['window'], windows['hamming'])
        self.normalize = normalize
        self.augment = augment
        self.noiseInjector = NoiseInjection(audio_conf['noise_dir'], self.sample_rate,
                                            audio_conf['noise_levels']) if audio_conf.get(
            'noise_dir') is not None else None
        self.noise_prob = audio_conf.get('noise_prob')

    def parse_audio(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path, norm=True)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        #n_fft = int(self.sample_rate * self.window_size)
        #win_length = n_fft
        n_fft = 1023
        win_length = int(self.sample_rate * self.window_size)
        hop_length = int(self.sample_rate * self.window_stride)
        # STFT
        #print('Performing STFT with win_length: %d, hop_length: %d' % (win_length, hop_length))
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length,
                         win_length=win_length, window=self.window)
        spect, phase = librosa.magphase(D)
        # S = log(S+1)
        spect = np.log1p(spect)
        spect = torch.FloatTensor(spect)
        if self.normalize:
            mean = spect.mean()
            std = spect.std()
            spect.add_(-mean)
            spect.div_(std)

        return spect

    def parse_audio_ext(self, audio_path):
#         print(audio_path)
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path, True)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        signal = y.squeeze()
        signal = signal*(2**15)
        signal = scipy.signal.lfilter([1, -1], [1, -0.99], signal)
        dither = np.random.rand(*signal.shape) + np.random.rand(*signal.shape) - 1
        spow = np.std(signal)
        signal = signal + 1e-6 * spow * dither
        signal =  scipy.signal.lfilter([1, -0.97], 1, signal)
        n_fft = 512
        win_length = int(self.sample_rate * self.window_size)
        hop_length = int(self.sample_rate * self.window_stride)

        frames = vec2frames(signal, win_length, hop_length)
        # FFT
        D = np.fft.fft(frames, n=n_fft, axis = 0)
        spect = np.abs(D)
        spect = torch.FloatTensor(spect)
        mean = spect.mean(1)
        std = spect.std(1)
        for i in range(spect.size(1)):
            spect[:, i] -= mean
            spect[:, i] /= std

        return spect

    def parse_transcript(self, transcript_path):
        raise NotImplementedError

    def parse_audio_mfe(self, audio_path):
        if self.augment:
            y = load_randomly_augmented_audio(audio_path, self.sample_rate)
        else:
            y = load_audio(audio_path, True)
        if self.noiseInjector:
            add_noise = np.random.binomial(1, self.noise_prob)
            if add_noise:
                y = self.noiseInjector.inject_noise(y)
        signal = y.squeeze()
        mfe, _ = speechpy.mfe(signal, sampling_frequency=16000, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=512, low_frequency=0, high_frequency=None)
        mfe = torch.FloatTensor(mfe)

        return mfe



class SpectrogramDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, dataset='voxceleb', normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param dataset: Type of dataset (voxceleb or ASVspoof)
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.dataset = dataset
        super().__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path = sample[0]
        # if self.dataset == 'voxceleb':
            # speaker = [int(sample[1])]  # target
        # elif self.dataset == 'ASVspoof':
            # speaker = [int(sample[3])]
        speaker = [int(sample[1])]
        spect = self.parse_audio(audio_path)
        return spect, speaker

    def __len__(self):
        return self.size


class SpectrogramDatasetPair(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, dataset='voxceleb', normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        self.dataset = dataset
        super().__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path0 = sample[0]
        audio_path1 = sample[1]
        equity = [int(sample[2])]  # target
        spect0 = self.parse_audio(audio_path0)
        spect1 = self.parse_audio(audio_path1)
        return spect0, spect1, equity

    def __len__(self):
        return self.size


class MfeDataset(Dataset, SpectrogramParser):
    def __init__(self, audio_conf, manifest_filepath, labels, normalize=False, augment=False):
        """
        Dataset that loads tensors via a csv containing file paths to audio files and transcripts separated by
        a comma. Each new line is a different sample. Example below:

        /path/to/audio.wav,/path/to/audio.txt
        ...

        :param audio_conf: Dictionary containing the sample rate, window and the window length/stride in seconds
        :param manifest_filepath: Path to manifest csv as describe above
        :param labels: String containing all the possible characters to map to
        :param normalize: Apply standard mean and deviation normalization to audio tensor
        :param augment(default False):  Apply random tempo and gain perturbations
        """
        with open(manifest_filepath) as f:
            ids = f.readlines()
        ids = [x.strip().split(',') for x in ids]
        self.ids = ids
        self.size = len(ids)
        super().__init__(audio_conf, normalize, augment)

    def __getitem__(self, index):
        sample = self.ids[index]
        audio_path = sample[0]
        speaker = [int(sample[1])]  # target
        mfe = self.parse_audio_mfe(audio_path)
        if len(sample) > 2:
            sent = [int(sample[2])]
            return mfe, speaker, sent
        else:
            return mfe, speaker

    def __len__(self):
        return self.size

def _collate_fn(batch):
    def func(p):
        return p[0].size(1)

    longest_sample = max(batch, key=func)[0]
    freq_size = longest_sample.size(0)
    minibatch_size = len(batch)
    max_seqlength = longest_sample.size(1)
    container_legnth = 300
    inputs = torch.zeros(minibatch_size, 1, freq_size, container_legnth)
    input_percentages = torch.FloatTensor(minibatch_size)
    targets = []
    if minibatch_size == 1:
        sample = batch[0]
        tensor = sample[0]
        target = sample[1]
        frame_length = (tensor.size(1) // 100) * 100
        inputs = tensor.narrow(1, 0, frame_length)
        inputs.unsqueeze_(0)
        inputs.unsqueeze_(0)
        targets.extend(target)
    else:
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0]
            target = sample[1]
            seq_length = tensor.size(1)
            try:
                start = np.random.randint(0, seq_length-container_legnth)
                inputs[x][0].copy_(tensor.narrow(1, start, container_legnth))
            except:
                inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            input_percentages[x] = seq_length / float(max_seqlength)
            targets.extend(target)
    targets = torch.LongTensor(targets)
    return inputs, targets



def _collate_fn_pair(batch):
    freq_size = batch[0][0].size(0)
    minibatch_size = len(batch)
    targets = []
    if minibatch_size == 1:
        sample = batch[0]
        tensor = sample[0]
        tensor1 = sample[1]
        target = sample[2]
        frame_length = (tensor.size(1) // 100) * 100
        frame_length1 = (tensor1.size(1) // 100) * 100
        inputs = tensor.narrow(1, 0, frame_length)
        inputs.unsqueeze_(0)
        inputs.unsqueeze_(0)
        inputs1 = tensor1.narrow(1, 0, frame_length1)
        inputs1.unsqueeze_(0)
        inputs1.unsqueeze_(0)
        targets.extend(target)
    else:
        fixed_length = 300
        inputs = torch.zeros(minibatch_size, 1, freq_size, fixed_length)
        inputs1 = torch.zeros(minibatch_size, 1, freq_size, fixed_length)
        for x in range(minibatch_size):
            sample = batch[x]
            tensor = sample[0]
            tensor1 = sample[1]
            target = sample[2]
            try:
                start = np.random.randint(0, tensor.size(1)-fixed_length)
                inputs[x][0].copy_(tensor.narrow(1, start, fixed_length))
            except:
                seq_length = tensor.size(1)
                inputs[x][0].narrow(1, 0, seq_length).copy_(tensor)
            try:
                start1 = np.random.randint(0, tensor1.size(1)-fixed_length)
                inputs1[x][0].copy_(tensor1.narrow(1, start1, fixed_length))
            except:
                seq_length1 = tensor1.size(1)
                inputs1[x][0].narrow(1, 0, seq_length1).copy_(tensor1)

            targets.extend(target)
    targets = torch.FloatTensor(targets)
    return (inputs, inputs1), targets


def _collate_fn_mfe(batch):
    nb_uttrs = 1
    nb_frames = 100  # 1000ms
    minibatch_size = len(batch)
    inputs = torch.zeros(minibatch_size, nb_uttrs, nb_frames, 40)
    targets = []
    for x in range(minibatch_size):
        sample = batch[x]
        tensor = sample[0] # frames x 40
        target = sample[1] # spk
        seq_length = tensor.size(0)
        inputs[x][0].narrow(0,0,seq_length).copy_(tensor)
        #start = np.random.randint(0, seq_length-300)
        #inputs[x][0].copy_(tensor.narrow(1, start, 300))
        targets.extend(target)
    targets = torch.LongTensor(targets)
    return inputs, targets


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = _collate_fn


class AudioDataLoaderPair(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_pair


class AudioDataLoaderMfe(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super().__init__(*args, **kwargs)
        self.collate_fn = _collate_fn_mfe

def get_audio_length(path):
    output = subprocess.check_output(['soxi -D \"%s\"' % path.strip()], shell=True)
    return float(output)


def audio_with_sox(path, sample_rate, start_time, end_time):
    """
    crop and resample the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as tar_file:
        tar_filename = tar_file.name
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} trim {} ={} >/dev/null 2>&1".format(path, sample_rate,
                                                                                         tar_filename, start_time,
                                                                                         end_time)
        os.system(sox_params)
        y = load_audio(tar_filename)
        return y


def augment_audio_with_sox(path, sample_rate, tempo, gain):
    """
    Changes tempo and gain of the recording with sox and loads it.
    """
    with NamedTemporaryFile(suffix=".wav") as augmented_file:
        augmented_filename = augmented_file.name
        sox_augment_params = ["tempo", "{:.3f}".format(tempo), "gain", "{:.3f}".format(gain)]
        sox_params = "sox \"{}\" -r {} -c 1 -b 16 -e si {} {} >/dev/null 2>&1".format(path, sample_rate,
                                                                                      augmented_filename,
                                                                                      " ".join(sox_augment_params))
        os.system(sox_params)
        y = load_audio(augmented_filename)
        return y


def load_randomly_augmented_audio(path, sample_rate=16000, tempo_range=(0.85, 1.15),
                                  gain_range=(-6, 8)):
    """
    Picks tempo and gain uniformly, applies it to the utterance by using sox utility.
    Returns the augmented utterance.
    """
    low_tempo, high_tempo = tempo_range
    tempo_value = np.random.uniform(low=low_tempo, high=high_tempo)
    low_gain, high_gain = gain_range
    gain_value = np.random.uniform(low=low_gain, high=high_gain)
    audio = augment_audio_with_sox(path=path, sample_rate=sample_rate,
                                   tempo=tempo_value, gain=gain_value)
    return audio

def vec2frames(vec, Nw, Ns, direction='col', window='hamming', padding=False):
    L = len(vec)
    M = int(np.floor((L-Nw)/Ns + 1))

    E = (L - ((M-1) * Ns + Nw))
    if padding:
        if E > 0:
            P = Nw - E
            vec.append(np.zeros(P))
            M += 1

    if direction == 'col':
        indf = np.multiply(range(M), Ns)
        indf = np.expand_dims(indf, axis = 0)
        indf = np.repeat(indf, Nw, axis = 0)

        inds = np.transpose(np.arange(0, Nw))
        inds = np.expand_dims(inds, axis = 1)
        inds = np.repeat(inds, M, axis = 1)

        indices = np.transpose(np.add(indf, inds))

    frames = vec[indices]

    if window == 'hamming':
        window = np.hamming(Nw)
        frames = np.transpose(frames * window)

    return frames
