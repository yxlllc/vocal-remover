import os
import random
import librosa
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=True,
        is_sort=True,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


class VocalRemoverTrainingSet(torch.utils.data.Dataset):
    def __init__(self, 
            training_set, 
            sr, 
            hop_length, 
            cropsize, 
            mixup_rate, 
            mixup_alpha):
        self.training_set = training_set
        self.sr = sr
        self.hop_length = hop_length
        self.cropsize = cropsize
        self.mixup_rate = mixup_rate
        self.mixup_alpha = mixup_alpha
        self.waveform_sec = (cropsize - 1) * hop_length / sr

    def __len__(self):
        return len(self.training_set)

    def do_crop(self, X_path, y_path, y1_path, y2_path, duration):
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
        y1_crop, _ = librosa.load(
                    y1_path,
                    sr = self.sr,
                    mono=False,
                    offset = start_time,
                    duration = self.waveform_sec)
        y2_crop, _ = librosa.load(
                    y2_path,
                    sr = self.sr,
                    mono=False,
                    offset = start_time,
                    duration = self.waveform_sec)                    
        if X_crop.ndim == 1:
            X_crop = np.asarray([X_crop])
        if y_crop.ndim == 1:
            y_crop = np.asarray([y_crop])
        if y1_crop.ndim == 1:
            y1_crop = np.asarray([y1_crop])
        if y2_crop.ndim == 1:
            y2_crop = np.asarray([y2_crop])
        return X_crop, y_crop, y1_crop, y2_crop

    def do_aug(self, X, y, y1, y2):
        max_amp = np.max([np.abs(X).max(), np.abs(y).max(), np.abs(y1).max(), np.abs(y2).max()])
        max_shift = min(1, np.log10(1 / max_amp))
        log10_shift = random.uniform(-1, max_shift)
        X =  X * (10 ** log10_shift)
        y =  y * (10 ** log10_shift)
        y1 =  y1 * (10 ** log10_shift)
        y2 =  y2 * (10 ** log10_shift)
        
        if np.random.uniform() < 0.01:
            X = y.copy()

        return X, y, y1, y2

    def do_mixup(self, X, y, y1, y2):
        idx = np.random.randint(0, len(self))

        X_i, y_i, y1_i, y2_i = self.__getitem__(idx, skip_mix=True)

        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        X = lam * X + (1 - lam) * X_i
        y = lam * y + (1 - lam) * y_i
        y1 = lam * y1 + (1 - lam) * y1_i
        y2 = lam * y2 + (1 - lam) * y2_i
        return X, y, y1, y2

    def __getitem__(self, idx, skip_mix=False):
        X_path, y_path, y1_path, y2_path = self.training_set[idx]
        X_duration = librosa.get_duration(filename = X_path, sr = self.sr)
        y_duration = librosa.get_duration(filename = y_path, sr = self.sr)
        y1_duration = librosa.get_duration(filename = y1_path, sr = self.sr)
        y2_duration = librosa.get_duration(filename = y2_path, sr = self.sr)
        duration = min(X_duration, y_duration, y1_duration, y2_duration)
        if duration < (self.waveform_sec + 0.1):
            return self.__getitem__((idx + 1) % len(self), skip_mix=skip_mix)
        
        X, y, y1, y2 = self.do_crop(X_path, y_path, y1_path, y2_path, duration)
        X, y, y1, y2 = self.do_aug(X, y, y1, y2)
        if not skip_mix and np.random.uniform() < self.mixup_rate:
            X, y, y1, y2 = self.do_mixup(X, y, y1, y2)
        return X, y, y1, y2


class VocalRemoverValidationSet(torch.utils.data.Dataset):
    def __init__(self, validation_set, sr):
        self.validation_set = validation_set
        self.sr = sr

    def __len__(self):
        return len(self.validation_set)

    def __getitem__(self, idx):
        X_path, y_path, y1_path, y2_path = self.validation_set[idx]
        X_duration = librosa.get_duration(filename = X_path, sr = self.sr)
        y_duration = librosa.get_duration(filename = y_path, sr = self.sr)
        y1_duration = librosa.get_duration(filename = y1_path, sr = self.sr)
        y2_duration = librosa.get_duration(filename = y2_path, sr = self.sr)
        duration = min(X_duration, y_duration, y1_duration, y2_duration)
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
        y1_crop, _ = librosa.load(
                    y1_path,
                    sr = self.sr,
                    mono=False,
                    offset = 0,
                    duration = duration)
        y2_crop, _ = librosa.load(
                    y2_path,
                    sr = self.sr,
                    mono=False,
                    offset = 0,
                    duration = duration)
        if X_crop.ndim == 1:
            X_crop = np.asarray([X_crop])
        if y_crop.ndim == 1:
            y_crop = np.asarray([y_crop])
        if y1_crop.ndim == 1:
            y1_crop = np.asarray([y1_crop])
        if y2_crop.ndim == 1:
            y2_crop = np.asarray([y2_crop])
        return X_crop, y_crop, y1_crop, y2_crop


def make_pair(mix_dir, h_dir, hb_dir, he_dir):
    input_exts = ['wav', 'flac']
    X_list = traverse_dir(mix_dir, input_exts, is_pure=False, is_sort=True, is_ext=True)
    y_list = traverse_dir(h_dir, input_exts, is_pure=False, is_sort=True, is_ext=True)
    y1_list = traverse_dir(hb_dir, input_exts, is_pure=False, is_sort=True, is_ext=True)
    y2_list = traverse_dir(he_dir, input_exts, is_pure=False, is_sort=True, is_ext=True)
    filelist = list(zip(X_list, y_list, y1_list, y2_list))
    return filelist


def train_val_split(dataset_dir, split_mode, val_num, val_filelist):
    if split_mode == 'random':
        filelist = make_pair(
            os.path.join(dataset_dir, 'mixtures'),
            os.path.join(dataset_dir, 'harmonic'),
            os.path.join(dataset_dir, 'harmonic_base'),
            os.path.join(dataset_dir, 'harmonic_even')
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
            os.path.join(dataset_dir, 'training/harmonic'),
            os.path.join(dataset_dir, 'training/harmonic_base'),
            os.path.join(dataset_dir, 'training/harmonic_even')
        )

        val_filelist = make_pair(
            os.path.join(dataset_dir, 'validation/mixtures'),
            os.path.join(dataset_dir, 'validation/harmonic'),
            os.path.join(dataset_dir, 'validation/harmonic_base'),
            os.path.join(dataset_dir, 'validation/harmonic_even')
        )

    return train_filelist, val_filelist


