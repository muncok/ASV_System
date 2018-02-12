import speechpy
import scipy.io.wavfile as wav
from scipy.signal import stft
import os
import numpy as np
from sklearn.utils import shuffle

class vox_data():
    def __init__(self, feature = 'mag', nb_class = 100,
                 split_file = "../../dataset/voxceleb/Identification_split.txt"):
        speakers = {}
        trainX, valX, testX = [], [], []
        trainY, valY, testY = [], [], []

        with open(split_file, 'r') as f:
            lines = f.readlines()
            id = 0
            for line in lines:
                set_id, file_path = line.split()
                speaker, file_name = file_path.split('/')
                if speaker not in speakers:
                    speakers[speaker] = id
                    id += 1

        with open(split_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                set_id, file_path = line.split()
                set_id = int(set_id)
                speaker, file_name = file_path.split('/')
                if set_id == 1:
                    trainX.append(file_path)
                    trainY.append(speakers[speaker])
                elif set_id == 2:
                    valX.append(file_path)
                    valY.append(speakers[speaker])
                else:
                    testX.append(file_path)
                    testY.append(speakers[speaker])

        if nb_class != -1:
            #train_size = np.nonzero(trainY==(nb_class-1))[0][-1]  # last index with label 99
            train_size = trainY.index(nb_class)
            trainX = trainX[:train_size]
            self.trainY = trainY[:train_size]
            val_size = valY.index(nb_class)
            valX = valX[:val_size]
            self.valY = valY[:val_size]
            test_size = testY.index(nb_class)
            testX = testX[:test_size]
            self.testY = testY[:test_size]
            print("nb_class: %d" % nb_class)

        if feature == 'mag':
            self.trainX= self.compute_fft_mag(trainX)
            self.valX= self.compute_fft_mag(valX)
            self.testX= self.compute_fft_mag(testX)
        elif feature ==  'mfe':
            self.trainX= self.compute_mfe(trainX)
            self.valX= self.compute_mfe(valX)
            self.testX= self.compute_mfe(testX)
        else:
            raise NotImplementedError

        #trainX, trainY = np.array(trainX), np.array(trainY)
        #valX, valY = np.array(valX), np.array(valY)
        #testX, testY = np.array(testX), np.array(testY)



    def compute_fft_mag(self, X, nfft = 1023, base_dir = "../../dataset/voxceleb/"):
        fft_mags = []
        for x in X:
            file_name = os.path.join (base_dir, x)
            fs, signal = wav.read(file_name)
            emphasized_signal = self._pre_emphasis(signal)
            _, _, fft_signal = stft(emphasized_signal, fs, 'hamming', 0.025*fs, 0.015*fs, nfft=nfft)
            fft_mags.append(np.abs(fft_signal))
        return fft_mags


    def compute_mfe(self, X, nframes=80, nfft = 1023, base_dir = "../../dataset/voxceleb/"):
        mfes = []
        for x in X:
            file_name = os.path.join (base_dir, x)
            fs, signal = wav.read(file_name)
            emphasized_signal = self._pre_emphasis(signal)
            mfe, _ = speechpy.mfe(signal, sampling_frequency=fs, frame_length=0.020, frame_stride=0.01,
             num_filters=40, fft_length=nfft, low_frequency=0, high_frequency=None)
            # TODO: zero-padding
            #if mfe.shape[0] < nframes:
            #    mfe = np.pad(mfe, ((nframes - mfe.shape[0],0), (0,0)), mode='constant')
            #mfes.append(mfe[-nframes:])
            mfes.append(mfe)
        return mfes


    def _pre_emphasis(self, signal, alpha = 0.97):
            emphasized_signal = np.append(signal[0], signal[1:] - alpha * signal[:-1])
            return emphasized_signal


    def batchLoader(self, set_type, batch_size):
        if set_type == "train":
            x, y = self.trainX, self.trainY
        elif set_type == "val":
            x, y = self.valX, self.valY
        elif set_type == "test":
            x, y = self.testX, self.testY
        else:
            raise NotImplementedError

        nb_batches = int(np.ceil(len(x) / batch_size))
        x, y = shuffle(x, y)
        for i in range(nb_batches):
            if i == nb_batches:
                batch_x = x[i*batch_size:]
                batch_y = y[i*batch_size:]
            else:
                batch_x = x[i*batch_size:(i+1)*batch_size]
                batch_y = y[i*batch_size:(i+1)*batch_size]

            batch = []
            for fft_signal in batch_x:
                rstart = np.random.randint(0, high=fft_signal.shape[-1] - 300)
                fft = fft_signal[:, rstart:rstart+300]
                batch.append(fft)
            yield  (np.array(batch), np.array(batch_y))

    def batchLoader_mfe(self, set_type, nb_uttrs=20):
        if set_type == "train":
            x, y = self.trainX, self.trainY
        elif set_type == "val":
            x, y = self.valX, self.valY
        elif set_type == "test":
            x, y = self.testX, self.testY
        else:
            raise NotImplementedError

        arr_x = np.array(x)
        arr_y = np.array(y)
        print(nb_uttrs)
        unique_label = np.unique(arr_y)
        splits = []
        batch_x = []
        batch_y = []
        for unique in unique_label:
            splits = np.where(arr_y == unique)[0]
            nb_choice = (splits.shape[0] + (nb_uttrs - 1)) // nb_uttrs
            for i in range(nb_choice):
                batch_x.append(arr_x[np.random.choice(splits, size=(nb_uttrs,))])
                batch_y.append(unique)
        return np.reshape(batch_x, (-1, nb_uttrs, 80, 40)), np.array(batch_y).flatten()

def gen_manifest(nb_class = 10):
    voxc_dir = "/home/muncok/DL/dataset/voxceleb/"
    with open("voxc_train_manifest.csv", "w") as f:
        for wav_path in trainX:
            speaker = wav_path.split('/')[0]
            line = ','.join([voxc_dir+wav_path, str(speakers[speaker])])
            f.write(line)
            f.write('\n')

    with open("voxc_val_manifest.csv", "w") as f:
        for wav_path in valX:
            speaker = wav_path.split('/')[0]
            line = ','.join([voxc_dir+wav_path, str(speakers[speaker])])
            f.write(line)
            f.write('\n')

    with open("voxc_test_manifest.csv", "w") as f:
        for wav_path in testX:
            speaker = wav_path.split('/')[0]
            line = ','.join([voxc_dir+wav_path, str(speakers[speaker])])
            f.write(line)
            f.write('\n')
