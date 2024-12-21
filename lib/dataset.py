import os
import random
import librosa
import numpy as np
import torch
import torch.utils.data
import pathlib
from tqdm import tqdm

    
class VocalRemoverTrainingSet(torch.utils.data.Dataset):

    def __init__(self, 
            training_set, 
            sr, 
            hop_length, 
            cropsize, 
            copy_rate,
            mixup_rate, 
            mixup_alpha):
        self.training_set = training_set
        self.sr = sr
        self.hop_length = hop_length
        self.cropsize = cropsize
        self.copy_rate = copy_rate
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.waveform_sec = (cropsize - 1) * hop_length / sr

    def __len__(self):
        return len(self.training_set)

    def do_crop(self, X_path, y_path, duration):
        start_time = random.uniform(0, duration - self.waveform_sec - 0.1)
        X_crop, _ = librosa.load(
                    X_path,
                    sr = self.sr,
                    mono=False,
                    offset = start_time,
                    duration = self.waveform_sec)
        y_crop, _ = librosa.load(
                    y_path,
                    sr = self.sr,
                    mono=False,
                    offset = start_time,
                    duration = self.waveform_sec)
        if X_crop.ndim == 1:
            X_crop = np.asarray([X_crop])
        if y_crop.ndim == 1:
            y_crop = np.asarray([y_crop])
        return X_crop, y_crop

    def do_aug(self, X, y):
        max_amp = np.max([np.abs(X).max(), np.abs(y).max()])
        max_shift = min(1, np.log10(1 / max_amp))
        log10_shift = random.uniform(-1, max_shift)
        X =  X * (10 ** log10_shift)
        y =  y * (10 ** log10_shift)
        
        if np.random.uniform() < 0.5:
            # swap channel
            X = X[::-1].copy()
            y = y[::-1].copy()

        if np.random.uniform() < self.copy_rate:
            # inst
            X = y.copy()

        # if np.random.uniform() < 0.01:
        #     # mono
        #     X[:] = X.mean(axis=0, keepdims=True)
        #     y[:] = y.mean(axis=0, keepdims=True)

        return X, y

    def do_mixup(self, X, y):
        idx = np.random.randint(0, len(self))

        X_i, y_i = self.__getitem__(idx, skip_mix=True)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        X = lam * X + (1 - lam) * X_i
        y = lam * y + (1 - lam) * y_i

        return X, y

    def __getitem__(self, idx, skip_mix=False):
        X_path, y_path = self.training_set[idx]
        X_duration = librosa.get_duration(filename = X_path, sr = self.sr)
        y_duration = librosa.get_duration(filename = y_path, sr = self.sr)
        duration = min(X_duration, y_duration)
        if duration < (self.waveform_sec + 0.1):
            return self.__getitem__((idx + 1) % len(self), skip_mix=skip_mix)
        
        X, y = self.do_crop(X_path, y_path, duration)
        X, y = self.do_aug(X, y)
        if not skip_mix and np.random.uniform() < self.mixup_rate:
            X, y = self.do_mixup(X, y)
        return X, y


class VocalRemoverValidationSet(torch.utils.data.Dataset):
    def __init__(self, validation_set, sr):
        self.validation_set = validation_set
        self.sr = sr

    def __len__(self):
        return len(self.validation_set)

    def __getitem__(self, idx):
        X_path, y_path = self.validation_set[idx]
        X_duration = librosa.get_duration(filename = X_path, sr = self.sr)
        y_duration = librosa.get_duration(filename = y_path, sr = self.sr)
        duration = min(X_duration, y_duration)
        X_crop, _ = librosa.load(
                    X_path,
                    sr = self.sr,
                    mono=False,
                    offset = 0,
                    duration = duration)
        y_crop, _ = librosa.load(
                    y_path,
                    sr = self.sr,
                    mono=False,
                    offset = 0,
                    duration = duration)
        if X_crop.ndim == 1:
            X_crop = np.asarray([X_crop])
        if y_crop.ndim == 1:
            y_crop = np.asarray([y_crop])                    
        return X_crop, y_crop


def make_pair(mix_dir, inst_dir):
    input_exts = ['.wav', '.m4a', '.mp3', '.mp4', '.flac']
    
    mix_dir = pathlib.Path(mix_dir)
    inst_dir = pathlib.Path(inst_dir)
    X_list = sorted([
        fpath
        for fpath in mix_dir.glob('**/*')
        if fpath.suffix in input_exts
    ])
    y_list = [
        inst_dir / fpath.relative_to(mix_dir)
        for fpath in X_list
    ]

    filelist = list(zip(X_list, y_list))

    return filelist


def train_val_split(dataset_dir, split_mode, val_num, val_filelist):
    if split_mode == 'random':
        filelist = make_pair(
            os.path.join(dataset_dir, 'mixtures'),
            os.path.join(dataset_dir, 'instruments')
        )

        random.shuffle(filelist)

        if len(val_filelist) == 0:
            train_filelist = filelist[:-val_num]
            val_filelist = filelist[-val_num:]
        else:
            train_filelist = [
                pair for pair in filelist
                if list(pair) not in val_filelist
            ]
    elif split_mode == 'subdirs':
        if len(val_filelist) != 0:
            raise ValueError('`val_filelist` option is not available with `subdirs` mode')

        train_filelist = make_pair(
            os.path.join(dataset_dir, 'training/mixtures'),
            os.path.join(dataset_dir, 'training/instruments')
        )

        val_filelist = make_pair(
            os.path.join(dataset_dir, 'validation/mixtures'),
            os.path.join(dataset_dir, 'validation/instruments')
        )

    return train_filelist, val_filelist
