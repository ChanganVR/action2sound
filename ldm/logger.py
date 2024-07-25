import os
import json
import subprocess
import shutil

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only
import torchvision
import pytorch_lightning as pl
import numpy as np
import soundfile as sf
from PIL import Image
import imageio
import matplotlib.pyplot as plt
import librosa
import noisereduce as nr
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip, concatenate_videoclips
from ldm.webify import webify, webify_with_energy
import logging
import random

VOCODER = None

# # griffin-lim version of spec-to-wav
# def spec_to_wav(spec, sr=16000):
#     n_fft = 1024
#     fmin = 125
#     fmax = 7600
#     nmels = 128
#     hoplen = 1024 // 4
#     spec_power = 1

#     # Inverse Transform
#     spec = spec * 100 - 100
#     spec = (spec + 20) / 20
#     spec = 10 ** spec
#     spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
#     wav = librosa.griffinlim(spec_out, hop_length=hoplen)
#     return wav


# hifigan version of spec-to-wav
@torch.no_grad()
def spec_to_wav(spec, sr=16000, generator=None):
    n_fft = 1024
    fmin = 125
    fmax = 7600
    nmels = 128
    hoplen = 1024 // 4
    spec_power = 1
    # load hifigan
    spec = spec * 100 - 100
    spec = (spec + 20) / 20
    spec = 10 ** spec
    if generator == 'griffinlim':
        if type(spec) == torch.Tensor:
            spec = spec.cpu().numpy()
        spec_out = librosa.feature.inverse.mel_to_stft(spec, sr = sr, n_fft = n_fft, fmin=fmin, fmax=fmax, power=spec_power)
        wav = librosa.griffinlim(spec_out, hop_length=hoplen)
    else:
        if type(spec) != torch.Tensor:
            spec = torch.from_numpy(spec).to(generator.conv_pre.weight.device)
        if spec.ndim == 2:
            spec = spec[None, :, :]
        if spec.shape[1] != 128:
            spec = spec.permute(0,2,1)
        spec = torch.log(spec) # inverse log10 -> log_e
        wav = generator(spec)
        wav = wav.squeeze(0).squeeze(0).cpu().numpy()
    return wav


def merge_audio_video(audio_file_path, video_file_path, output_file_path):
    command = [
        'ffmpeg', '-y',  # Overwrite output file if it exists
        '-an', '-i', video_file_path,  # Input video file
        '-i', audio_file_path,  # Input audio file
        '-c:v', 'copy',         # Copy the video stream
        '-c:a', 'aac',          # Encode the audio stream
        output_file_path        # Output file path
    ]

    subprocess.run(command)

def plot_spec(spec, save_path):
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    # flip y axis
    spec = np.flip(spec, axis=0)
    plt.imshow(spec)
    # tight layout
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_wav(wav, save_path, y_max=None, y_min=None):
    plt.figure()
    plt.plot(wav)
    if y_max == None:
        y_max = np.max(wav) + 0.2
    if y_min == None:
        y_min = np.min(wav) - 0.2
    plt.ylim(y_min, y_max)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return y_min, y_max


@torch.no_grad()
def generate_demo(model, batch, batch_idx, split, output_dir=None, args=None):
    # assert output_dir is None or output_postfix is None, "Only one of output_dir and output_postfix can be specified"
    
    global VOCODER
    if VOCODER is None:
        VOCODER = model.load_vocoder()

    with torch.no_grad():
        output = model.log_sound(batch, N=batch['mix_wav'].shape[0], split=split, size_len=24, guidance_scale=6.5)


    pred = output['samples'].detach().cpu().numpy()[:, :, :batch['original_tdim'][0]]
    gt_spec = batch['mix_spec'].detach().cpu().numpy()[:, 0, :, :batch['original_tdim'][0]]
    gt_wav = batch['mix_wav'].detach().cpu().numpy()
    for i in range(len(pred)):
        clip_id = batch['clip_id'][i]
        pred_wav = spec_to_wav(pred[i], sr=16000, generator=VOCODER) if not args.use_input_wav else gt_wav[i]
        pred_wav_file = os.path.join(output_dir, f'{clip_id}_pred.wav')
        sf.write(pred_wav_file, pred_wav, 16000)
        
        gt_video_file = output['mix_info_dict']['video_path1'][i]
        # softlink the original video to the sample folder
        gt_video_file = os.path.abspath(gt_video_file)
        os.symlink(gt_video_file, os.path.join(output_dir, f'tmp.mp4'))
        os.rename(os.path.join(output_dir, f'tmp.mp4'), os.path.join(output_dir, f'{clip_id}_gt.mp4'))
        
        pred_video_file = os.path.join(output_dir, f'{clip_id}_pred.mp4')
        merge_audio_video(pred_wav_file, gt_video_file, pred_video_file)
        # remove audio file
        os.remove(pred_wav_file)
        
        # save the pred and gt mel spec/wav
        plot_spec(gt_spec[i], os.path.join(output_dir, f'{clip_id}_gt_spec.png'))
        plot_spec(pred[i], os.path.join(output_dir, f'{clip_id}_pred_spec.png'))
        y_min, y_max = plot_wav(gt_wav[i], os.path.join(output_dir, f'{clip_id}_gt_wav.png'))
        _ = plot_wav(pred_wav, os.path.join(output_dir, f'{clip_id}_pred_wav.png'), y_max=y_max, y_min = y_min)

    
    print('Finish generating demo for batch {}'.format(batch_idx))
    webify(output_dir)


class SoundLogger_concat_fullset(Callback):
    def __init__(self, train_batch_frequency,val_batch_frequency, max_sound_num, sr=22050, clamp=True, increase_log_steps=False,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=False, ddim_step=250, size_len=64, 
                 guidance_scale=1.0, uncond_cond=None, fps=21.5):
        super().__init__()
        self.fps = fps
        self.rescale = rescale
        self.train_batch_freq = train_batch_frequency
        self.val_batch_freq = val_batch_frequency
        self.max_sound_num = max_sound_num
        self.log_steps = [2 ** n for n in range(int(np.log2(self.train_batch_freq)) + 1)]
        self.log_steps_val = [2 ** n for n in range(int(np.log2(self.val_batch_freq)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.train_batch_freq]
            self.log_steps_val = [self.val_batch_freq] 
        self.clamp = clamp
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.sr = sr
        self.ddim_step = ddim_step

        self.train_vis_frequency = train_batch_frequency
        self.val_vis_frequency = val_batch_frequency
        self.size_len = size_len
        self.guidance_scale = guidance_scale
        self.uncond_cond = uncond_cond


    @rank_zero_only
    def log_local(self, save_dir, split, log_dict,
                  global_step, current_epoch, batch_idx, show_curve=False, scale_factor=1, vocoder=None):

        # root:
        root = os.path.join(save_dir, "sound_eval", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        # gt_sound_list_numpy = log_dict['inputs_spec'].detach().cpu().numpy()
        # rec_sound_list_numpy = log_dict['reconstruction_spec'].detach().cpu().numpy()
        # # video_path_list= log_dict['video_frame_path']
        # # video_time_list = log_dict['video_time']

        # diff_sample_list_numpy = log_dict['samples'].detach().cpu().numpy()

        # for hifigan
        gt_sound_list = log_dict['inputs_spec'].detach()
        rec_sound_list = log_dict['reconstruction_spec'].detach()
        diff_sample_list = log_dict['samples'].detach()


        # root = os.path.join(save_dir, "sound", split, "epoch_{}_step_{}".format(current_epoch, global_step))

        os.makedirs(root,exist_ok=True)
        
        mix_info_dict = log_dict["mix_info_dict"]

        for i in range(len(gt_sound_list)):
            
            if mix_info_dict['audio_name2'][i] == "":
                video_path_list = mix_info_dict['video_path1']
                video_time_list = mix_info_dict['video_time1'] 
                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = spec_to_wav(gt_sound_list[i], self.sr, vocoder)
                rec_sound = spec_to_wav(rec_sound_list[i], self.sr, vocoder)
                sample = spec_to_wav(diff_sample_list[i], self.sr, vocoder)
                sf.write(os.path.join(sample_folder, "sample_{}_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_diff_sample_clamp.wav".format(i)), sample, self.sr) 
                
                if video_time_list == '0_0':
                    # softlink the original video to the sample folder
                    os.symlink(video_path_list[i], os.path.join(sample_folder, "origin.mp4"))
                else:
                    try:
                        video = self.extract_concat_frame_video(video_path_list[i], video_time_list[i], out_folder=sample_folder)
                    except Exception as e:
                        print(e)
                        pass

                with open(os.path.join(sample_folder, "video_path.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list[i]) + "    " + str(video_time_list[i])
                    f.writelines(txt)
            
            else:

                video_path_list1, video_path_list2 = mix_info_dict['video_path1'], mix_info_dict['video_path2']
                video_time_list1, video_time_list2 = mix_info_dict['video_time1'], mix_info_dict['video_time2'] 

                print('Gen examples ===========> {}'.format(i))
                sample_folder = os.path.join(root, "sample_{}".format(i))
                os.makedirs(sample_folder, exist_ok=True)
                gt_sound = spec_to_wav(gt_sound_list[i], self.sr, vocoder)
                rec_sound = spec_to_wav(rec_sound_list[i], self.sr, vocoder)
                sample = spec_to_wav(diff_sample_list[i], self.sr, vocoder)
                sf.write(os.path.join(sample_folder, "sample_{}_concat_gt.wav".format(i)), gt_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_stage1_concat_rec_clamp.wav".format(i)), rec_sound, self.sr)
                sf.write(os.path.join(sample_folder, "sample_{}_diff_concat_sample_clamp.wav".format(i)), sample, self.sr) 
                
                # video1 = self.concat_frame_video(video_path_list1[i], video_time_list1[i])
                # video2 = self.concat_frame_video(video_path_list2[i], video_time_list2[i])
                # video = video1 + video2
                # video_save_path = os.path.join(sample_folder, "origin_concat_video.mp4")
                # imageio.mimsave(video_save_path, video, fps=21.5)
                try:
                    video = self.extract_concat_frame_video(video_path_list1[i], video_time_list1[i], video_path_list2[i], video_time_list2[i], out_folder=sample_folder)
                except:
                    pass

                with open(os.path.join(sample_folder, "video_path_cat.txt"), "w") as f:
                    txt = "Video 1:" + '  ' + str(video_path_list1[i]) + "    " + str(video_time_list1[i]) + '\n' + "Video 2:" + '  ' + str(video_path_list2[i]) + "    " + str(video_time_list2[i])
                    f.writelines(txt)
        

    def extract_concat_frame_video(self, video_path1, video_time1, video_path2=None, video_time2=None, out_folder=None):
        # start_frame, end_frame = video_time[0], video_time[1]
        start_frame1, end_frame1 = int(video_time1.split('_')[0]), int(video_time1.split('_')[1])
        start_time1, end_time1 = start_frame1 / self.fps, end_frame1 / self.fps
        src_path1 = video_path1
        out_path = os.path.join(out_folder, "origin.mp4")

        video1 = VideoFileClip(src_path1).subclip(start_time1, end_time1)

        if video_path2 is not None:
            start_frame2, end_frame2 = int(video_time2.split('_')[0]), int(video_time2.split('_')[1])
            start_time2, end_time2 = start_frame2 / self.fps, end_frame2 / self.fps
            src_path2 = video_path2
            out_path = os.path.join(out_folder, "origin_cat.mp4") 
            video2 = VideoFileClip(src_path2).subclip(start_time2, end_time2)
            finalclip = concatenate_videoclips([video1, video2], method="compose")

            finalclip.write_videofile(out_path)
        else:
            video1.write_videofile(out_path)

        # ffmpeg_extract_subclip(src_path, start_time, end_time, out_path)
        return True

        
    @rank_zero_only
    def log_sound_steps(self, pl_module, batch, batch_idx, split="train"):
        global VOCODER
        if VOCODER is None:
            VOCODER = pl_module.load_vocoder()        
        is_train = pl_module.training
        if is_train:
            pl_module.eval()

        with torch.no_grad():
            log_dict = pl_module.log_sound(batch, N=self.max_sound_num, ddim_steps=self.ddim_step, split=split, size_len=self.size_len, guidance_scale=self.guidance_scale, uncond_cond=self.uncond_cond)

        # gt_sound_list = log_dict['inputs']
        # rec_sound_list = log_dict['reconstruction']
        # video_path_list= log_dict['video_frame_path']
        # video_time_list = log_dict['video_time']

        if pl_module.logger.save_dir is not None:
            self.log_local(pl_module.logger.save_dir, split, log_dict, pl_module.global_step, pl_module.current_epoch, batch_idx, vocoder=VOCODER)

        if is_train:
            pl_module.train()



    def check_frequency(self, check_idx, split):
        if split == "train":
            if ((check_idx % self.train_batch_freq) == 0 or (check_idx in self.log_steps)) and (
                    check_idx > 0 or self.log_first_step):
                try:
                    self.log_steps.pop(0)
                except IndexError as e:
                    print(e)
                    pass
                return True
            return False
        else:
            if (check_idx % self.val_batch_freq) == 0:
                return True
            else:
                return False


    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        # if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
        if not self.disabled and (pl_module.global_step % self.train_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="train")
            # pass

    @rank_zero_only
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step % self.val_vis_frequency == 0) and pl_module.global_step > 0:
            self.log_sound_steps(pl_module, batch, batch_idx, split="val")
            # pass

        if hasattr(pl_module, 'calibrate_grad_norm'):
            if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
                self.log_gradients(trainer, pl_module, batch_idx=batch_idx)
    
    # @rank_zero_only
    # def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
    #     self.log_sound_steps(pl_module, batch, batch_idx, split="test")

    #     if hasattr(pl_module, 'calibrate_grad_norm'):
    #         if (pl_module.calibrate_grad_norm and batch_idx % 25 == 0) and batch_idx > 0:
    #             self.log_gradients(trainer, pl_module, batch_idx=batch_idx)