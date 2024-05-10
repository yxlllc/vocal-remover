import os
import random
import librosa
import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm


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

    def aggressively_remove_vocal(self, X, y):
        X_mag = np.abs(X)
        y_mag = np.abs(y)
        v_mag = X_mag - y_mag
        v_mag *= v_mag > y_mag

        y_mag = np.clip(y_mag - v_mag * self.reduction_weight, 0, np.inf)

        return y_mag * np.exp(1.j * np.angle(y))

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

        if np.random.uniform() < 0.01:
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

    X_list = sorted([
        os.path.join(mix_dir, fname)
        for fname in os.listdir(mix_dir)
        if os.path.splitext(fname)[1] in input_exts
    ])
    y_list = sorted([
        os.path.join(inst_dir, fname)
        for fname in os.listdir(inst_dir)
        if os.path.splitext(fname)[1] in input_exts
    ])

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


def make_padding(width, cropsize, offset):
    left = offset
    roi_size = cropsize - offset * 2
    if roi_size == 0:
        roi_size = cropsize
    right = roi_size - (width % roi_size) + left

    return left, right, roi_size


def make_training_set(filelist, sr, hop_length, n_fft, device):
    ret = []
    window = torch.hann_window(n_fft).to(device)
    for X_path, y_path in tqdm(filelist):
        X, y, X_cache_path, y_cache_path = spec_utils.cache_or_load(
            X_path, y_path, sr, hop_length, n_fft, window, read_cache=False,
        )
        ret.append([X_cache_path, y_cache_path])

    return ret


def make_validation_set(filelist, cropsize, sr, hop_length, n_fft, offset, device):
    patch_list = []
    patch_dir = 'cs{}_sr{}_hl{}_nf{}_of{}'.format(cropsize, sr, hop_length, n_fft, offset)
    os.makedirs(patch_dir, exist_ok=True)
    window = torch.hann_window(n_fft).to(device)
    
    for X_path, y_path in tqdm(filelist):
        basename = os.path.splitext(os.path.basename(X_path))[0]

        X, y, _, _ = spec_utils.cache_or_load(X_path, y_path, sr, hop_length, n_fft, window)

        l, r, roi_size = make_padding(X.shape[2], cropsize, offset)
        X_pad = np.pad(X, ((0, 0), (0, 0), (l, r)), mode='constant')
        y_pad = np.pad(y, ((0, 0), (0, 0), (l, r)), mode='constant')

        len_dataset = int(np.ceil(X.shape[2] / roi_size))
        for j in range(len_dataset):
            outpath = os.path.join(patch_dir, '{}_p{}.npz'.format(basename, j))
            start = j * roi_size
            if not os.path.exists(outpath):
                np.savez(
                    outpath,
                    X=X_pad[:, :, start:start + cropsize],
                    y=y_pad[:, :, start:start + cropsize]
                )
            patch_list.append(outpath)

    return patch_list


def get_oracle_data(X, y, oracle_loss, oracle_rate, oracle_drop_rate):
    k = int(len(X) * oracle_rate * (1 / (1 - oracle_drop_rate)))
    n = int(len(X) * oracle_rate)
    indices = np.argsort(oracle_loss)[::-1][:k]
    indices = np.random.choice(indices, n, replace=False)
    oracle_X = X[indices].copy()
    oracle_y = y[indices].copy()

    return oracle_X, oracle_y, indices


if __name__ == "__main__":
    import sys
    import utils

    mix_dir = sys.argv[1]
    inst_dir = sys.argv[2]
    outdir = sys.argv[3]

    os.makedirs(outdir, exist_ok=True)

    filelist = make_pair(mix_dir, inst_dir)
    for mix_path, inst_path in tqdm(filelist):
        mix_basename = os.path.splitext(os.path.basename(mix_path))[0]

        X_spec, y_spec, _, _ = spec_utils.cache_or_load(
            mix_path, inst_path, 44100, 1024, 2048
        )

        X_mag = np.abs(X_spec)
        y_mag = np.abs(y_spec)
        v_mag = X_mag - y_mag
        v_mag *= v_mag > y_mag

        outpath = '{}/{}_Vocal.jpg'.format(outdir, mix_basename)
        v_image = spec_utils.spectrogram_to_image(v_mag)
        utils.imwrite(outpath, v_image)
