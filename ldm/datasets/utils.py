import argparse
import os
import os.path as P
from copy import deepcopy
from functools import partial
from glob import glob
from multiprocessing import Pool
from pathlib import Path

import librosa
import numpy as np
import torchvision
from tqdm import tqdm

import torch
import torchlibrosa as tl
        
class MelSpectrogram(object):
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(
            sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels
        )

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x,
                sr=self.sr,
                n_fft=self.nfft,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.spec_power,
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = (
                np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen))
                ** self.spec_power
            )
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec


class LowerThresh(object):
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)


class Add(object):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val


class Subtract(Add):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val


class Multiply(object):
    def __init__(self, val, inverse=False):
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val


class Divide(Multiply):
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10(object):
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10**x
        else:
            return np.log10(x)


class Clip(object):
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)


class TrimSpec(object):
    def __init__(self, max_len, inverse=False):
        self.max_len = max_len
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x[:, : self.max_len]


class MaxNorm(object):
    def __init__(self, inverse=False):
        self.inverse = inverse
        self.eps = 1e-10

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return x / (x.max() + self.eps)


sr = 16000
# trim_len = 620

TRANSFORMS = torchvision.transforms.Compose(
    [
        MelSpectrogram(
            sr=sr, nfft=1024, fmin=125, fmax=7600, nmels=128, hoplen=250, spec_power=1
        ),
        LowerThresh(1e-5),
        Log10(),
        Multiply(20),
        Subtract(20),
        Add(100),
        Divide(100),
        Clip(0, 1.0),
        # TrimSpec(trim_len)
    ]
)


def inv_transforms(x, folder_name="melspec_10s_22050hz"):
    """relies on the GLOBAL contant TRANSFORMS which should be defined in this document"""
    if folder_name == "melspec_10s_22050hz":
        i_transforms = deepcopy(TRANSFORMS.transforms[::-1])
    else:
        raise NotImplementedError
    for t in i_transforms:
        t.inverse = True
    i_transforms = torchvision.transforms.Compose(i_transforms)
    return i_transforms(x)


def get_spectrogram(audio_path, length, sr=16000):
    # wav, _ = librosa.load(audio_path, sr=None)
    wav, sr_new = librosa.load(audio_path, sr=sr, mono=True)
    # wav = np.load(audio_path)
    wav = wav.reshape(-1)
    # print(sr)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[: len(wav)] = wav
    else:
        y = wav[:length]

    # # wav:
    y = y[: length - 1]  # ensure: 640 spec

    mel_spec = TRANSFORMS(y)

    return y, mel_spec

# exactly the same as above, but with hoplen==256 rather than 250
TRANSFORMS_LDM = torchvision.transforms.Compose(
    [
        MelSpectrogram(
            sr=sr, nfft=1024, fmin=125, fmax=7600, nmels=128, hoplen=256, spec_power=1
        ),
        LowerThresh(1e-5),
        Log10(),
        Multiply(20),
        Subtract(20),
        Add(100),
        Divide(100),
        Clip(0, 1.0),
        # TrimSpec(trim_len)
    ]
)


def inv_transforms_LDM(x, folder_name="melspec_10s_22050hz"):
    """relies on the GLOBAL contant TRANSFORMS which should be defined in this document"""
    if folder_name == "melspec_10s_22050hz":
        i_transforms = deepcopy(TRANSFORMS_LDM.transforms[::-1])
    else:
        raise NotImplementedError
    for t in i_transforms:
        t.inverse = True
    i_transforms = torchvision.transforms.Compose(i_transforms)
    return i_transforms(x)


def get_spectrogram_LDM(audio_path, length, sr=16000):
    # wav, _ = librosa.load(audio_path, sr=None)
    wav, sr_new = librosa.load(audio_path, sr=sr, mono=True)
    # wav = np.load(audio_path)
    wav = wav.reshape(-1)
    # print(sr)
    # this cannot be a transform without creating a huge overhead with inserting audio_name in each
    y = np.zeros(length)
    if wav.shape[0] < length:
        y[: len(wav)] = wav
    else:
        y = wav[:length]

    # # wav:
    y = y[: length - 1]  # ensure: 640 spec

    mel_spec = TRANSFORMS_LDM(y)

    return y, mel_spec


if __name__ == "__main__":
    input_wav_path = "/data/ego4d_chunked_audio_16k/002d2729-df71-438d-8396-5895b349e8fd_narration_pass_2_996.wav"
    sr = 16000
    time = 3
    length = sr * time
    y, mel_spec = get_spectrogram_LDM(input_wav_path, length, sr)
    print("Mel Spec Shape: {}".format(mel_spec.shape))
    print("Finished!")
