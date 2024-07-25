import os
import sys
import json
import pandas as pd
import random
import torch
import glob
from collections import defaultdict

from torch.utils.data import Dataset
from data_loader.transforms import *
import torch
import torchaudio
import decord
import librosa
import noisereduce as nr
from PIL import Image
from moviepy.editor import *
from decord import AudioReader, VideoReader
from decord import cpu, gpu
import matplotlib.pyplot as plt
import numpy as np
import torchaudio.functional as F
from ldm.datasets.utils import TRANSFORMS_LDM
from librosa.util import normalize
import logging
import types
from collections import defaultdict
def init_video_transform(input_res=224,
                        center_crop=256,
                        randcrop_scale=(0.5, 1.0),
                        color_jitter=(0, 0, 0),
                        norm_mean=(0.485, 0.456, 0.406),
                        norm_std=(0.229, 0.224, 0.225),
                        supress_message=False):
    return transforms.Compose([
            NormalizeVideo(mean=norm_mean, std=norm_std)
        ])


def find_segment_with_lowest_energy(waveform, segment_length, increment=1, lowest_k=1):
    """
    Find the segment with the lowest energy in an audio waveform.
    
    Parameters:
        waveform (numpy.ndarray): The audio waveform.
        segment_length (int): Length of the segments to consider.
        increment (int): Increment between segments.
    
    Returns:
        numpy.ndarray: The segment with the lowest energy.
    """
    num_segments = (len(waveform) - segment_length) // increment + 1
    segment_indices = np.arange(num_segments)[:, None] * increment + np.arange(segment_length)
    segments = waveform[segment_indices]
    energies = np.sum(segments**2, axis=1)
    # min_index = np.argmin(energies)
    lowest_k_indices = np.argpartition(energies, lowest_k)[:lowest_k]
    return lowest_k_indices * increment


class Ego4DSounds(Dataset):
    def __init__(self,
                split,
                dataset_name,
                video_params,
                audio_params,
                data_dir,
                metadata_file=None,
                seed=0,
                metadata_dir=None, # for backward compatibility
                args=None, # for backward compatibility
                ):
        self.dataset_name = dataset_name
        self.video_params = video_params
        self.audio_params = audio_params
        self.transforms = init_video_transform()
        # check for environment variables
        self.data_dir = os.path.expandvars(data_dir)
        self.split = split
        self.metadata = pd.read_csv(metadata_file, sep='\t', on_bad_lines='warn')
        self.seed = seed
            
    def set_args(self, args):
        self.args = args # getting args directly from main
        self.args.energy_param = getattr(self.args, 'energy_param', "1280_640") # framelength_hoplength
        self.energy_frame_length, self.energy_hop_length = [int(x) for x in self.args.energy_param.split("_")]
        self.args.energy_expected_len = getattr(self.args, 'energy_expected_len', 76) # if the target waveform is 3 sec long, the expected energy length is 76
        
        self.metadata = self.metadata.reset_index(drop=True)
        if self.split in ['train', 'validation']:
            all_video_uids = self.metadata['video_uid'].unique()
            random.Random(self.seed).shuffle(all_video_uids)
            train_uids = all_video_uids[:int(len(all_video_uids) * 0.95)]
            validation_uids = all_video_uids[int(len(all_video_uids) * 0.95):]
            # test_uids = all_video_uids[int(len(all_video_uids) * 0.9):]
            print(f"Total number of unique long videos: {len(all_video_uids)}")
            print(f"Number of unique long train videos: {len(train_uids)}")
            print(f"Number of unique long validation videos: {len(validation_uids)}")
            if self.split == 'train':
                self.metadata = self.metadata[self.metadata['video_uid'].isin(train_uids)]
            elif self.split == 'validation':
                self.metadata = self.metadata[self.metadata['video_uid'].isin(validation_uids)]
            self.oldid2newid = np.arange(len(self.metadata))
        elif self.split == 'test':
            self.metadata = self.metadata.sample(frac=1, random_state=self.seed).reset_index(drop=False)
            self.oldid2newid = np.ones(len(self.metadata), dtype=np.int32) * -1
            for i in range(len(self.metadata)):
                self.oldid2newid[self.metadata.iloc[i]['index']] = i
        else:
            raise NotImplementedError(f"Unknown split: {self.split}")
    
    def get_id(self, sample):
        if 'narration_source' in sample and 'narration_ind' in sample:
            return sample['video_uid'] + '_' + sample['narration_source'] + '_' + str(sample['narration_ind'])
        else:
            return sample['video_uid']

    def __len__(self):
        if self.args.num_test_samples != -1:
            return self.args.num_test_samples
        else: 
            return len(self.metadata)

    @property
    def video_size(self):
        return (self.args.num_frames, self.video_params['input_res'], self.video_params['input_res'], 3)

    @property
    def spec_size(self):
        return (self.audio_params['input_fdim'], self.audio_params['input_tdim'])

    @property
    def waveform_size(self):
        return (1, int(self.audio_params['sample_rate'] * self.audio_params['duration']))

    def __getitem__(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp = os.path.join(self.data_dir, sample['clip_file'])
        text = sample['clip_text']
        clip_id = self.get_id(sample)
        
        video, transformed_video = self.load_video(video_fp, num_frames=self.args.num_frames)
        video_32 = self.load_video(video_fp, num_frames=32)[1] if 'av_sync' in self.args.test_metrics else torch.zeros(32, 3, 224, 224)
        waveform = self.load_audio(video_fp)
        energy = librosa.feature.rms(y=waveform, frame_length=self.energy_frame_length, hop_length=self.energy_hop_length)

        spec, original_tdim = self.wav2spec(waveform)
        spec = torch.tensor(spec[None].repeat(3, axis=0))
        neighbor_waveform = np.zeros(self.waveform_size)[0]
        if self.args.neighbor_file != '':
            if self.args.neighbor_file == 'zero_spec':
                neighbor_spec = torch.tensor(np.zeros(self.spec_size, dtype=np.float32)[None].repeat(3, axis=0))
            else:
                neighbor_waveform = self.load_audio(self.args.neighbor_file)
                neighbor_spec, _ = self.wav2spec(neighbor_waveform)
                neighbor_spec = torch.tensor(neighbor_spec[None].repeat(3, axis=0))
        else:
            # get the nearest left and right k video
            if self.split == 'test':
                old_id = sample['index']
            else:
                old_id = item
            # NOTE assume nearby clips in the unshuffled metadata are the neighboring clips (in time), 
            # this seems to right by manual inspection of the metadata, but might need to be changed
            left_k_inds = [old_id-l for l in range(1, self.args.left_nearest_k+1) if old_id-l > 0 
                        and self.metadata.iloc[self.oldid2newid[old_id-l]]['video_uid'] == sample['video_uid']]
            right_k_inds = [old_id+r for r in range(1, self.args.right_nearest_k+1) if old_id+r < len(self.metadata) 
                            and self.metadata.iloc[self.oldid2newid[old_id+r]]['video_uid'] == sample['video_uid']]
            
            nearest_inds = left_k_inds + right_k_inds
            
            if len(nearest_inds) > 0:
                nearest_ind = np.random.choice(nearest_inds)
                neighbor_sample = self.metadata.iloc[self.oldid2newid[nearest_ind]]
                assert neighbor_sample['video_uid'] == sample['video_uid'], \
                    f"video uid not match: {neighbor_sample['video_uid']} vs {sample['video_uid']}"
                neighbor_video_fp = os.path.join(self.data_dir, neighbor_sample['clip_file'])
                neighbor_waveform = self.load_audio(neighbor_video_fp)
                neighbor_waveform = neighbor_waveform.numpy()
                if self.args.audio_cond == 'rand_neighbor':
                    neighbor_spec, _ = self.wav2spec(neighbor_waveform)
                    neighbor_spec = torch.tensor(neighbor_spec[None].repeat(3, axis=0))
                else:
                    raise NotImplementedError(f"Unknown audio condition: {self.args.audio_cond}")
            else:
                print("nearest_inds is empty, use black spec instead")
                neighbor_spec = torch.tensor(np.zeros(self.spec_size, dtype=np.float32)[None].repeat(3, axis=0))

        mix_info_dict = {
            "video_path1": video_fp,
            "video_time1": '0_0', # start frame and 
            "audio_name2": '',
        }
        
        return {'mix_spec': spec, 'mix_video_feat': video, 'mix_info_dict': mix_info_dict, 'mix_wav': waveform, 'original_tdim': original_tdim,
                'clip_id': clip_id, 'text': text, 'neighbor_spec': neighbor_spec, 'neighbor_wav': neighbor_waveform, "energy":energy, 'transformed_video': transformed_video,
                'video_32': video_32}

    def load_video(self, video_fp, num_frames):
        video_size= (num_frames, self.video_params['input_res'], self.video_params['input_res'], 3)
        try:
            vr = VideoReader(video_fp, ctx=cpu(0))
            frame_indices = np.linspace(0, len(vr) - 1, num_frames).astype(int)
            imgs = vr.get_batch(frame_indices).float()
        except Exception as e:
            print('failed to load video, use black image instead', e)
            imgs = torch.zeros(video_size)
        
        input_imgs = imgs.clone()
        imgs = (imgs / 255.0).permute(3, 0, 1, 2)  # [T, H, W, C] ---> [C, T, H, W]
        transformed_imgs = self.transforms(imgs)
        transformed_imgs = transformed_imgs.permute(1, 0, 2, 3)  # [C, T, H, W] ---> [T, C, H, W]

        output_imgs = transformed_imgs if self.video_params['transform'] else input_imgs.permute(0, 3, 1, 2)  # [T, H, W, C] ---> [T, C, H, W]

        return output_imgs, transformed_imgs

    def load_audio(self, audio_fp):
        try:
            ar = AudioReader(audio_fp, ctx=cpu(0), sample_rate=16000)
            waveform = ar[:]
            if waveform.shape[1] > self.waveform_size[1]:
                waveform = waveform[:, :self.waveform_size[1]]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, self.waveform_size[1] - waveform.shape[1]))
        except Exception as e:
            print(f'Exception while reading audio file {audio_fp} with {e}')
            waveform = torch.zeros(self.waveform_size)
    
        return waveform[0]
    
    def wav2spec(self, wav):
        wav = wav.numpy() if isinstance(wav, torch.Tensor) else wav
        spec = TRANSFORMS_LDM(wav)
        original_tdim = torch.tensor(spec.shape[1])
        if spec.shape[1] < self.spec_size[1]:
            spec = np.pad(spec, ((0, 0), (0, self.spec_size[1] - spec.shape[1])), 'constant', constant_values=0)
            
        return spec, original_tdim
    
class ego4dsounds_train(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="train", **dataset_cfg)

class ego4dsounds_validation(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="validation", **dataset_cfg)

class ego4dsounds_test(Ego4DSounds):
    def __init__(self, dataset_cfg):
        super().__init__(split="test", **dataset_cfg)


if __name__ == "__main__":
    from tqdm import tqdm
    import argparse
    import matplotlib.pyplot as plt
    import soundfile as sf
    import time
    import numpy as np
    import logging
    # faciliate logging/debugging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    np.random.seed(seed=int(time.time())) 
    
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.seed = 0
    args.transform = True
    args.left_nearest_k = 3
    args.right_nearest_k = 3
    args.seg_len = 16000
    args.audio_cond = "avsim6_aasim3"
    args.lowest_k = 1
    args.num_frames = 16
    args.noise_snr = "0,10"
    args.neighbor_file = ""
    
    kwargs = dict(
        split="train",
        dataset_name="ego4dsounds",
        data_dir="data/ego4dsounds_224p",
        metadata_file="data/meta/ego4dsounds/train_clips_0.4m.csv", # data/meta/ego4dsounds/test_clips_11k.csv
        video_params={
            "input_res": 224,
            "loading": "lax",
            "normalize": True,
            "transform": True,
        },
        audio_params={
            "sample_rate": 16000,
            "duration": 3,
            "input_fdim": 128,
            "input_tdim": 196,
        },
        args=args
    )
    
    dataset = Ego4DSounds(**kwargs)
    dataset.set_args(args)
    num_video = len(dataset)
    print('Total number of videos clips: {}'.format(num_video))
    
    # randomly sample 100 videos
    indices = np.random.choice(num_video, 10)
    output_dir = 'debug_audio'
    os.makedirs(output_dir, exist_ok=True)
    start = time.time()
    for i in tqdm(indices):
        item = dataset[i]
        # save audio and video to disk for debugging
        frames = item['mix_video_feat']
        audio = item['mix_spec']
        # video = item['video_copy']
        # print(audio.shape, video.shape)
        # sf.write(f"{output_dir}/{item['index']}.wav", audio.numpy()[0], 16000)
        # video = [img for img in video.permute(0, 2, 3, 1).numpy()]
        # # concate frames horizontally and save image
        # video = np.concatenate(video, axis=1)
        # plt.imsave(f'{output_dir}/{i}_video.png', video)
    
    print(f'Time taken: {time.time() - start}')
    print(f'Average time per video: {(time.time() - start) / len(indices)}')