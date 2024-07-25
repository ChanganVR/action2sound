"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""
from collections import defaultdict
import os
import json

import soundfile as sf
import librosa
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
import speechmetrics
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only
import noisereduce as nr

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.models.distribution import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder_img import AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.sd_ddim_scale import DDIMSampler
from ldm.logger import spec_to_wav
from ldm.util import print, print_all

# Add Other Sampler:
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

FEATURE_EXTRACTOR = None
VOCODER=None
import logging
__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


def calculate_envelope_sliding_window(signal, frame_size=1024, hop_size=512):
    return np.array([max(signal[i:i+frame_size]) for i in range(0, len(signal), hop_size)])


class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 video_encoder=None,
                 pretrained_video_extractor=None,
                 max_pool_patches=False, # back compatibility
                 mean_pool_patches=False, # back compatibility
                 vocoder=None, # back compatibility
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        self.video_encoder = video_encoder
        self.pretrained_video_extractor = pretrained_video_extractor
        
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        if self.global_rank == 0:
            print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
            if len(missing) > 0:
                print(f"Missing: {missing[:3]}, ...")
            if len(unexpected) > 0:
                print(f"Unexpected: {unexpected[:3]}, ...")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True, sample_weight=None):
        # logging.info(f"shape of target: {target.shape}, shape of pred: {pred.shape}")
        # logging.info(f"shape of sample_weight: {sample_weight.shape}")
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if sample_weight != None:
                loss = loss * sample_weight[:, None, None, None]
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
            if sample_weight != None:
                loss = loss * sample_weight[:, None, None, None]
            if mean:
                loss = loss.mean()
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean().item()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb.item()})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss.item()})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    # def get_input(self, batch, k):
    #     x = batch[k]
    #     if len(x.shape) == 3:
    #         x = x[..., None]
    #     x = rearrange(x, 'b h w c -> b c h w')
    #     x = x.to(memory_format=torch.contiguous_format).float()
    #     return x
    
    def get_input(self, batch, k):
        x = batch[k]
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch, train=False):
        x = self.get_input(batch, self.first_stage_key, train=train)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def inflate_positional_embeds(self, video_model, new_state_dict):
        num_frames = self.args.num_frames
        load_temporal_fix = self.args.load_temporal_fix
        
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(video_model.state_dict().keys())
        if 'temporal_embed' in new_state_dict and 'temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = num_frames
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded model has MORE frames than current...'
                            f'### loading weights, filling in the extras via {load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded model has FEWER frames than current...'
                            f'### loading weights, filling in the extras via {load_temporal_fix}')
                    if load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        align_corners = None
                        if load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                            align_corners = True
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed, (curr_num_frames, embed_dim), 
                                                           mode=mode, align_corners=align_corners).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'pos_embed' in new_state_dict and 'pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = video_model.state_dict()['pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')
    
        return new_state_dict
    
    def load_video_encoder(self):
        if self.video_encoder == 'timesformer':
            from ldm.models.video_transformer import SpaceTimeTransformer
            
            video_encoder = SpaceTimeTransformer(num_frames=self.args.num_frames, time_init='zeros').to(self.device)
            ckpt = torch.load(self.pretrained_video_extractor, map_location='cpu')
            state_dict = ckpt['state_dict']
            state_dict = {k.replace('module.video_model.', ''): v for k, v in state_dict.items() if k.startswith('module.video_model')}
            new_state_dict = self.inflate_positional_embeds(video_encoder, state_dict)
            video_encoder.load_state_dict(new_state_dict, strict=False)
            print(f'Loaded pretrained video extractor from {self.pretrained_video_extractor}')
        elif self.video_encoder == 'slow_only':
            from open_clip.audio_contrastive import ResNet3dSlowOnly
            
            video_encoder = ResNet3dSlowOnly(depth=50, pretrained=None).to(self.device)
            video_project_head = nn.Linear(2048, 512).to(self.device)
            ckpt = torch.load(self.pretrained_video_extractor, map_location='cpu')
            video_encoder.load_state_dict({k.replace('module.video_encoder.', ''): v for k, v in ckpt['state_dict'].items() 
                                           if k.startswith('module.video_encoder')}, strict=False)
            video_project_head.load_state_dict({k.replace('module.video_project_head.', ''): v for k, v in ckpt['state_dict'].items() 
                                                if k.startswith('module.video_project_head')}, strict=False)
            video_encoder = nn.Sequential(video_encoder, video_project_head)
            print(f'Loaded pretrained video extractor from {self.pretrained_video_extractor}')
        elif self.video_encoder == 'resnet':
            from torchvision.models import resnet50
            video_encoder = resnet50(pretrained=True).to(self.device)
            # remove last fc layer
            video_encoder = nn.Sequential(*list(video_encoder.children())[:-1])
            print(f'Loaded pretrained video extractor from torchvision')
        else:
            raise NotImplementedError(f'Video encoder {self.video_encoder} not implemented')
        # freeze the feature extractor
        for param in video_encoder.parameters():
            param.requires_grad = False
        
        return video_encoder
    
    @torch.no_grad()
    def load_vocoder(self):
        if self.use_vocoder == None:
            return "griffinlim"
        else:
            assert self.use_vocoder == 'hifigan', "current implementation only supports hifigan"
        class AttrDict(dict):
            def __init__(self, *args, **kwargs):
                super(AttrDict, self).__init__(*args, **kwargs)
                self.__dict__ = self
        from ..hifigan import Generator
        import json
        checkpoint_file = "data/pretrained/hifi-gan.pth"
        config_file = "data/pretrained/hifi-gan.json"
        with open(config_file) as f:
            data = f.read()

        json_config = json.loads(data)
        h = AttrDict(json_config)
        generator = Generator(h)

        state_dict_g = torch.load(checkpoint_file, map_location='cpu')
        generator.load_state_dict(state_dict_g['generator'])
        generator.eval()
        generator.remove_weight_norm()
        generator = generator.to(self.device)
        return generator

    @torch.no_grad()
    def extract_features(self, batch):
        global FEATURE_EXTRACTOR
        
        if FEATURE_EXTRACTOR is None:
            FEATURE_EXTRACTOR = self.load_video_encoder()
        
        # assume input video is of shape [B, T, C, H, W]
        video = batch[self.cond_stage_key].to(self.device)
        batch['transformed_video'] = batch['transformed_video'].to(self.device)
        batch['video_frames'] = video
        batch['video_32'] = batch['video_32'].to(self.device)
        if self.video_encoder == 'timesformer':
            video_feat = FEATURE_EXTRACTOR.forward_features(video, return_intermediate=True, max_pool_patches=self.max_pool_patches, 
                                                            mean_pool_patches=self.mean_pool_patches)
        elif self.video_encoder == 'slow_only':
            # permute to B C T H W
            video = video.permute(0, 2, 1, 3, 4)
            video_feat = FEATURE_EXTRACTOR[0](video)
            bs, c, t, _, _ = video_feat.shape
            video_feat = video_feat.reshape(bs, c, t).permute(0, 2, 1)
            video_feat = FEATURE_EXTRACTOR[1](video_feat)
            video_feat = F.normalize(video_feat, dim=-1)
        elif self.video_encoder == 'resnet':
            # reshape to images
            bs, t, c, h, w = video.shape
            video = video.reshape(bs * t, c, h, w)
            video_feat = FEATURE_EXTRACTOR(video)
            video_feat = video_feat.reshape(bs, t, -1)
            
        batch[self.cond_stage_key] = video_feat
        
        if self.args.retrieve_nn:
            batch['neighbor_spec'], batch['neighbor_wav'] = self.retrieve_nearest_neighbor(batch)
            batch['neighbor_spec'] = batch['neighbor_spec'].unsqueeze(1).repeat(1, 3, 1 ,1)
        
        return batch

    def training_step(self, batch, batch_idx):
        batch = self.extract_features(batch)
        loss, loss_dict = self.shared_step(batch, train=True)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch = self.extract_features(batch)
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)        

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 video_cond_len=785,
                 video_cond_dim=768,
                 scale_by_std=False,
                 test_metrics='',
                 audio_cond_config=None,
                 *args, **kwargs):
        self.conditioning_key = conditioning_key
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        self.sim_weighting_map = kwargs.pop("sim_weighting_map", None)
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.video_cond_len = video_cond_len
        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  
        
        self.neighbor_audio_cond_prob = audio_cond_config.neighbor_audio_cond_prob
        self.audio_cond_encoder = instantiate_from_config(audio_cond_config) if self.neighbor_audio_cond_prob >= 0.0 else None

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True

        # zero embedding:
        # self.zero_embed = nn.Parameter(torch.randn(1, 32, ))
        self.zero_embed = torch.zeros(1, video_cond_len, video_cond_dim)               # Zeros Embedding
        # for _ in range(20):
        #     print(video_cond_len)
        # assert False
        self.class_free_drop_prob = 0.2

        # audio zero embedding:
        if self.audio_cond_encoder is not None:
            if audio_cond_config.params.target_seq_len != -1:
                self.audio_zero_embed = nn.Parameter(torch.zeros(1, audio_cond_config.params.target_seq_len, audio_cond_config.params.embed_dim), requires_grad=False)
        self.test_metrics = test_metrics.split(',') if test_metrics != '' else []
    
    def set_args(self, args):
        self.args = args
        self.max_pool_patches = args.pool_patches == 'max'
        self.mean_pool_patches = args.pool_patches == 'mean'
        self.use_vocoder = args.vocoder
        
        if args.retrieve_nn and 'av_sim' not in self.test_metrics:
            self.test_metrics.append('av_sim')

        self.evaluators = {}
        for metric in self.test_metrics:
            if metric == 'fad':
                from frechet_audio_distance import FrechetAudioDistance
                
                self.evaluators['fad'] = FrechetAudioDistance(
                    model_name="vggish",
                    sample_rate=16000,
                    use_pca=False, 
                    use_activation=False,
                    verbose=False
                )
                print('Loaded FAD evaluator') 
            elif metric == 'al_sim':
                from model.ast_model import ASTModel
                from transformers import AutoModel
                import transformers
            
                ckpt = torch.load(self.args.pretrained_al_sim, map_location='cpu')
                state_dict = ckpt['state_dict']
                
                audio_encoder = ASTModel(label_dim=256, fstride=10, tstride=10, input_fdim=128, 
                                        input_tdim=self.args.ast_tdim, imagenet_pretrain=True)
                audio_state_dict = {k.replace('module.audio_model.', ''): v for k, v in state_dict.items() if k.startswith('module.audio_model')}
                audio_encoder.load_state_dict(audio_state_dict, strict=True)
                
                text_encoder = AutoModel.from_pretrained('distilbert-base-uncased',
                    cache_dir='pretrained/distilbert-base-uncased')
                text_encoder.load_state_dict({k.replace('module.text_model.', ''): v for k, v in state_dict.items() 
                                              if k.startswith('module.text_model')}, strict=True)
                txt_proj = nn.Sequential(nn.ReLU(), nn.Linear(768, 256))
                txt_proj.load_state_dict({k.replace('module.txt_proj.', ''): v for k, v in state_dict.items()
                                            if 'txt_proj' in k}, strict=True)
                self.tokenizer = transformers.AutoTokenizer.from_pretrained('distilbert-base-uncased',
                                                               TOKENIZERS_PARALLELISM=False)
                
                self.evaluators['al_sim'] = {'audio_encoder': audio_encoder, 'text_encoder': text_encoder, 'txt_proj': txt_proj}
                for model in [audio_encoder, text_encoder, txt_proj]:
                    model.eval()
                print('Loaded ALSim evaluator')
            elif metric == "av_sync":
                from model.ast_model import ASTModel
                from ldm.models.video_transformer import SpaceTimeTransformer
                save_root = "data/pretrained/av_sync_classifier"
                # load_from_model_name = "half_neg_easy_shiftmin1.5"
                # load_from_model_name = self.args.load_av_sync_from
                load_from_model_name = "half_neg_easy_shift1-1.5_weight_pos2_continue_continue"
                if "trm" in load_from_model_name:
                    self.transformer_head = True
                    transformer_head = True
                else:
                    self.transformer_head = False
                    transformer_head = False
                class Classifier(nn.Module):
                    def __init__(self, ftr_dim=512):
                        super(Classifier, self).__init__()
                        if transformer_head:
                            # use a 1D convolution to transformer audio feature shape of [B, 170, 768] to [B, 17, 768]
                            self.conv_audio = nn.Sequential(nn.Conv1d(kernel_size=10, in_channels=768, out_channels=ftr_dim, stride=5, padding=5), nn.GELU(), nn.Conv1d(kernel_size=5, in_channels=ftr_dim, out_channels=ftr_dim, stride=2, padding=0))
                            self.norm_audio = nn.LayerNorm(ftr_dim)
                            self.linear_video = nn.Linear(768, ftr_dim)
                            self.norm_video = nn.LayerNorm(ftr_dim)
                            # learned positional encoding
                            self.pos_audio = nn.Parameter(torch.randn(1, 16, ftr_dim))
                            self.pos_video = nn.Parameter(torch.randn(1, 16, ftr_dim))
                            trm_layer = nn.TransformerEncoderLayer(d_model=ftr_dim, nhead=8,dim_feedforward=2048, batch_first=True)
                            self.transformer_head = nn.TransformerEncoder(trm_layer, num_layers=2, norm=nn.LayerNorm(ftr_dim))
                            self.cls_token = nn.Parameter(torch.randn(1, 1, ftr_dim))
                            self.mlp = nn.Sequential(nn.Linear(ftr_dim, 1), nn.Sigmoid())
                        else:
                            self.mlp = nn.Sequential(nn.Linear(ftr_dim, ftr_dim*2), nn.ReLU(), nn.Linear(ftr_dim*2, ftr_dim), nn.ReLU(), nn.Linear(ftr_dim, 1), nn.Sigmoid())
                    def forward(self, x, x2=None):
                        if transformer_head:
                            assert x2 != None # x2 is video features
                            assert x2.shape[1:] == torch.Size([16, 768])
                            assert x.shape[1:] == torch.Size([170, 768])
                            x = self.conv_audio(x.transpose(1, 2)).transpose(1, 2)
                            x = self.norm_audio(x)
                            x2 = self.linear_video(x2)
                            x2 = self.norm_video(x2)
                            x = x + self.pos_audio.expand(x.shape[0], -1, -1)
                            x2 = x2 + self.pos_video.expand(x2.shape[0], -1, -1)
                            x_input = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x, x2], dim=1)
                            assert x_input.shape[1:] == torch.Size([33, 512]), x_input.shape
                            x = self.transformer_head(x_input)
                            x = x[:, 0, :]
                            return self.mlp(x)
                        else:
                            return self.mlp(x)
                ftr_dim = 256
                ast_tdim= 149
                classifier = Classifier(ftr_dim=ftr_dim*2)
                
                audio_encoder = ASTModel(label_dim=256, fstride=10, tstride=10, input_fdim=128, 
                                        input_tdim=ast_tdim, imagenet_pretrain=True)
                video_encoder = SpaceTimeTransformer(num_frames=16, time_init='zeros')
                video_encoder.head = nn.Identity()
                video_encoder.pre_logits = nn.Identity()
                video_encoder.fc = nn.Identity()
                


                logging.info(f"Loading model from {load_from_model_name}")
                classifier.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f'{save_root}/{load_from_model_name}_bestacc.pth', map_location='cpu').items()})
                video_encoder.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f'{save_root}/{load_from_model_name}_video_encoder_bestacc.pth', map_location='cpu').items()})
                audio_encoder.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f'{save_root}/{load_from_model_name}_audio_encoder_bestacc.pth', map_location='cpu').items()})
                if not self.transformer_head:
                    vid_proj = nn.Sequential(nn.Linear(768, 256))
                    vid_proj.load_state_dict({k.replace("module.", ""): v for k, v in torch.load(f'{save_root}/{load_from_model_name}_vid_proj_bestacc.pth', map_location='cpu').items()})
                else:
                    vid_proj = None

                self.evaluators['av_sync'] = {'audio_encoder': audio_encoder, 'video_encoder': video_encoder, 'vid_proj': vid_proj, 'classifier': classifier, "ast_tdim": ast_tdim}
                for m in [audio_encoder, video_encoder, vid_proj, classifier]:
                    if m != None:
                        m.eval()
                print('Loaded AVSync evaluator')
        
        if args.compute_retrieval or args.retrieve_nn:
            data_dir = 'data/ego4dsounds_audio_features/train'
            files = os.listdir(data_dir)
            self.train_audio_features = {}
            for file in tqdm(files):
                f = np.load(os.path.join(data_dir, file), allow_pickle=True)
                self.train_audio_features.update(f[()])
            print(f'Loaded {len(self.train_audio_features)} train audio features')
        
        self.test_stats = {}

    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")

    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution) or str(type(encoder_posterior)) == "<class 'ldm.modules.stage1_model.model_img.DiagonalGaussianDistribution'>":
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        input_list = False
        if 'hybrid' in self.conditioning_key:
            if type(c) == list: # that means it's during training
                input_list = True
                c_audio = None
                if len(c) == 1:
                    c = c[0]
                elif len(c) == 2:
                    c, c_audio = c
                else:
                    raise ValueError(f"len(c) == {len(c)}")
        else:
            if self.audio_cond_encoder is not None and c.shape[1] == self.video_cond_len + self.audio_cond_encoder.target_seq_len:
                c, c_audio = c.split([int(c.shape[1]-self.audio_cond_encoder.target_seq_len)], dim=1)
            else:
                assert c.shape[1]==self.video_cond_len, c.shape
                c_audio = None

        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        # assert c.shape[1] == self.video_cond_len, c.shape
        if 'hybrid' in self.conditioning_key:
            if input_list:
                return [c, c_audio]
            else:
                return c
        else:
            if self.audio_cond_encoder is not None and c_audio is not None:
                c = torch.cat([c, c_audio], dim=1)
            return c

    def meshgrid(self, h, w):
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        :param h: height
        :param w: width
        :return: normalized distance to image border,
         wtith min distance = 0 at border and max dist = 0.5 at image center
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1):  # todo load once not every time, shorten code
        """
        :param x: img of size (bs, c, h, w)
        :return: n img crops of size (n, bs, c, kernel_size[0], kernel_size[1])
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting
    
    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None, train=False):
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()
        # logging.info(f"shape of latent: {z.shape}")

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                # print(f"using {cond_key} as conditioning")
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device) # original conditional variables
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                    # print(f"line 904 shape of c: {c.shape}")
                else:
                    c = self.get_learned_conditioning(xc.to(self.device)) # encoded conditional variables
                    # print(f"during testing, c gets forwarded in the small video embedding model")
                    # print(f"line 907, shape of c: {c.shape}")
            else:
                c = xc
                # print("during training, c doesn't get forwarded in the small video embedding model")
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        
        # print(c.shape, z.shape, x.shape)
        # classifier-free Guidance Dropout:
        new_bs = z.shape[0]
        if train:
            video_feat_mask = self.prob_mask_like((new_bs, ), 1 - self.class_free_drop_prob, device=self.device)
            video_feat_mask = video_feat_mask.reshape(new_bs, 1, 1)
            null_embed = self.zero_embed.to(self.device)
            c = torch.where(video_feat_mask, c, null_embed)

        # when cond prob is 0, we set audio_neighbor to zero. applying to both training and test time
        if "hybrid" in self.conditioning_key:
            c = [c]
        if self.audio_cond_encoder is not None:
            neighbor_spec = super().get_input(batch, 'neighbor_spec')
            # logging.info(f"shape of neighbor_spec: {neighbor_spec.shape}")
            if bs is not None:
                neighbor_spec = neighbor_spec[:bs]
            neighbor_spec = neighbor_spec.to(self.device)
            encoder_posterior_neighbor = self.encode_first_stage(neighbor_spec)
            neighbor_spec_feat = self.get_first_stage_encoding(encoder_posterior_neighbor)
            # logging.info(f"shape of neighbor_spec_feat after VAE embedding: {neighbor_spec_feat.shape}")
            neighbor_spec_feat = self.audio_cond_encoder(neighbor_spec_feat)
            # logging.info(f"shape of neighbor_spec_feat after audio embedding: {neighbor_spec_feat.shape}")
            if "hybrid" in self.conditioning_key and "audio" in self.conditioning_key:
                audio_zero_embed = torch.zeros_like(neighbor_spec_feat)
            else:
                audio_zero_embed = self.audio_zero_embed.repeat(new_bs, 1, 1)
            neighbor_spec_feat_mask = self.prob_mask_like((new_bs, ), self.neighbor_audio_cond_prob, device=self.device)
            neighbor_spec_feat_mask = neighbor_spec_feat_mask.reshape(new_bs, 1, 1, 1) if "hybrid" in self.conditioning_key and "audio" in self.conditioning_key else neighbor_spec_feat_mask.reshape(new_bs, 1, 1)
            c_audio = torch.where(neighbor_spec_feat_mask, neighbor_spec_feat, audio_zero_embed)
            # logging.info(f"c_audio.shape: {c_audio.shape}")
            if "hybrid" in self.conditioning_key:
                c = c + [c_audio]
            else:
                c = torch.cat([c, c_audio], dim=1)

        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            # if isinstance(self.first_stage_model, VQModelInterface):
            #     return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            # else:
            return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, train=False, **kwargs):
        x, c = self.get_input(batch, self.first_stage_key, train=train)
        if self.sim_weighting_map.startswith("global_sim"):
            temperature = float(self.sim_weighting_map.split("_")[-1][len("temperature"):])
            assert 'mix_sim' in batch, batch.keys()
            sample_weight = batch['mix_sim']
            b = x.shape[0]
            sample_weight = b * torch.nn.functional.softmax(torch.tensor(sample_weight).to(self.device)/temperature, dim=0)
            # logging.info(f"global_sim: sample_weight: {sample_weight}")
        elif self.sim_weighting_map.startswith("xid"):
            _, omega, kappa, delta = self.sim_weighting_map.split("_")
            omega = float(omega[len("omega"):])
            kappa = float(kappa[len("kappa"):])
            delta = float(delta[len("delta"):])
            sample_weight = batch['mix_sim']
            mu = torch.mean(sample_weight)
            sigma = torch.std(sample_weight)
            dist = torch.distributions.normal.Normal(loc=mu+delta*sigma, scale=np.sqrt(kappa)*sigma)
            sample_weight = dist.cdf(sample_weight)
            sample_weight = x.shape[0] * sample_weight / torch.sum(sample_weight)
            # logging.info(f"xid: sample_weight: {sample_weight}")
        elif self.sim_weighting_map == None or self.sim_weighting_map == "None":
            sample_weight = None
        else:
            raise NotImplementedError(f"current implementation does not support {self.sim_weighting_map} for sample weight calculation")
        loss = self(x, c, sample_weight=sample_weight)
        return loss

    def forward(self, x, c, *args, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
                # print(f"this should only happen during *training* line 1052, shape of c: {c.shape}")
            if self.shorten_cond_schedule:  # TODO: drop this option
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))
        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  # TODO: move to dataset
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            # key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            # logging.info(f"self.model.conditioning_key: {self.model.conditioning_key}")
            if self.model.conditioning_key == 'concat':
                key = 'c_concat'
            elif 'hybrid' in self.model.conditioning_key:
                key = 'c_hybrid'
            else:
                key = 'c_crossattn'
            if key != "c_hybrid":
                if not isinstance(cond, list):
                    cond = [cond]
                cond = {key: cond}
            else:
                assert type(cond) == list, cond
                cond = {"c_concat": cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  # todo can only deal with one conditioning atm
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)

            h, w = x_noisy.shape[-2:]

            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)

            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element

                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]

            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'

                # assuming padding of unfold is always 0 and its dilation is always 1
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                # as we are operating on latents, we need the factor from the original image size to the
                # spatial latent size to properly rescale the crops for regenerating the bbox annotations
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)

                # get top left postions of patches as conforming for the bbbox tokenizer, therefore we
                # need to rescale the tl patch coordinates to be in between (0,1)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]

                # patch_limits are tl_coord, width and height coordinates as (x_tl, y_tl, h, w)
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # patch_values = [(np.arange(x_tl,min(x_tl+ks, 1.)),np.arange(y_tl,min(y_tl+ks, 1.))) for x_tl, y_tl in tl_patch_coordinates]

                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                print(patch_limits_tknzd[0].shape)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                print(cut_cond.shape)

                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                print(adapted_cond.shape)
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                print(adapted_cond.shape)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                print(adapted_cond.shape)

                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]

            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient

            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens

            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization

        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound, measured in
        bits-per-dim.
        This term can't be optimized, as it only depends on the encoder.
        :param x_start: the [N x C x ...] tensor of inputs.
        :return: a batch of [N] KL values (in bits), one per batch element.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None, sample_weight=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False, sample_weight=sample_weight).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean().item()})

        # Add: ??
        self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean().item()})
            loss_dict.update({'logvar': self.logvar.data.mean().item()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False, sample_weight=sample_weight).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb.item()})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss.item()})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()      # With Different Sampler
    def sample_log_diff_sampler(self, cond, batch_size, sampler_name, ddim_steps, size_len=64, unconditional_guidance_scale=1.0,unconditional_conditioning=None, **kwargs):

        if sampler_name == "DDIM":
            ddim_sampler = DDIMSampler(self)
            # shape = (self.channels, self.image_size, self.image_size)
            shape = (self.channels, 16, size_len)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,unconditional_guidance_scale=unconditional_guidance_scale,unconditional_conditioning=unconditional_conditioning, **kwargs)

        elif sampler_name == "DPM_Solver":
            dpm_solver_sampler = DPMSolverSampler(self)
            shape = (self.channels, 16, size_len)
            samples, intermediates = dpm_solver_sampler.sample(ddim_steps,batch_size, shape, cond, verbose=False, 
                                                               unconditional_guidance_scale=unconditional_guidance_scale, 
                                                               unconditional_conditioning=unconditional_conditioning,
                                                               seed=self.args.seed, **kwargs)

        elif sampler_name == "PLMS":
            plms_sampler = PLMSSampler(self)
            shape = (self.channels, 16, size_len)
            samples, intermediates = plms_sampler.sample(ddim_steps,batch_size,
                                            shape,cond,verbose=False,unconditional_guidance_scale=unconditional_guidance_scale,unconditional_conditioning=unconditional_conditioning, **kwargs)


        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    

    @torch.no_grad()
    def sample_log_with_classifier(self, embed_cond, origin_cond, batch_size, ddim, ddim_steps, size_len=64, unconditional_guidance_scale=1.0, unconditional_conditioning=None, classifier=None, classifier_guide_scale=0.0, **kwargs):

        if ddim:
            ddim_sampler = DDIMSampler(self)
            # shape = (self.channels, self.image_size, self.image_size)
            shape = (self.channels, 16, size_len)
            samples, intermediates = ddim_sampler.sample_with_classifier(ddim_steps, batch_size,
                                                        shape, embed_cond, origin_cond=origin_cond, verbose=False,unconditional_guidance_scale=unconditional_guidance_scale,unconditional_conditioning=unconditional_conditioning,classifier=classifier,classifier_guide_scale=classifier_guide_scale, **kwargs)

        else:
            samples, intermediates = self.sample(cond=embed_cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates


    @torch.no_grad()
    def sample_log_with_classifier_diff_sampler(self, embed_cond, origin_cond, batch_size, sampler_name="DDIM", ddim_steps=250, size_len=64, unconditional_guidance_scale=1.0, unconditional_conditioning=None, classifier=None, classifier_guide_scale=0.0, **kwargs):

        if sampler_name == "DDIM":
            ddim_sampler = DDIMSampler(self)
            # shape = (self.channels, self.image_size, self.image_size)
            shape = (self.channels, 16, size_len)
            samples, intermediates = ddim_sampler.sample_with_classifier(ddim_steps, batch_size,
                                                        shape, embed_cond, origin_cond=origin_cond, verbose=False,unconditional_guidance_scale=unconditional_guidance_scale,unconditional_conditioning=unconditional_conditioning,classifier=classifier,classifier_guide_scale=classifier_guide_scale, **kwargs)

        elif sampler_name == "DPM_Solver":
            dpm_solver_sampler = DPMSolverSampler(self)
            shape = (self.channels, 16, size_len)
            samples, intermediates = dpm_solver_sampler.sample_with_classifier(ddim_steps,batch_size,
                                            shape, embed_cond, origin_cond=origin_cond, verbose=False,unconditional_guidance_scale=unconditional_guidance_scale,unconditional_conditioning=unconditional_conditioning, classifier=classifier,classifier_guide_scale=classifier_guide_scale, **kwargs)


        else:
            samples, intermediates = self.sample(cond=embed_cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @staticmethod
    def wav2fbank(waveform, sample_rate=16000, input_fdim=128, input_tdim=149):
        # take the middle 1.5 * 16000 samples
        if input_tdim == 149:
            waveform = waveform[:, int(1 * sample_rate):int(2.5 * sample_rate)]
          
        waveform = waveform - waveform.mean()
        fbank = torchaudio.compliance.kaldi.fbank(waveform, htk_compat=True, sample_frequency=sample_rate, use_energy=False,
                                            window_type='hanning', num_mel_bins=input_fdim, dither=0.0, frame_shift=10)
        
        # normalize
        fbank = (fbank + 4.26) / (4.57 * 2)

        target_length = input_tdim
        n_frames = fbank.shape[0]

        p = target_length - n_frames

        # cut and pad
        if p > 0:
            m = torch.nn.ZeroPad2d((0, 0, 0, p))
            fbank = m(fbank)
        elif p < 0:
            pass
            # fbank = fbank[0:target_length, :]
            
        # pad waveform
        # target_length = int(self.args.sample_rate * 1.5)
        # if waveform.shape[1] < target_length:
        #     waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.shape[1]))
        
        return fbank

    @staticmethod
    def sim_matrix(a, b, eps=1e-8):
        """
        added eps for numerical stability
        """
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
        sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
        return sim_mt

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        global VOCODER
        if VOCODER is None:
            VOCODER = self.load_vocoder()  
        stats = dict()
        output = self.log_sound(batch, N=batch['mix_wav'].shape[0], size_len=24, guidance_scale=6.5)
        
        if 'fad' in self.test_metrics:
            # write all data to /tmp/v2a/ + the basename of logdir
            output_dir = os.path.join('/tmp/v2a', os.path.basename(self.trainer.logdir))
            gt_dir = os.path.join(output_dir, 'gt')
            pred_dir = os.path.join(output_dir, 'pred')
            os.makedirs(gt_dir, exist_ok=True)
            os.makedirs(pred_dir, exist_ok=True)
        
        metric_scores = defaultdict(list)
        pred = output['samples'].detach().cpu().numpy()[:, :, :batch['original_tdim'][0]]
        # logging.info(f"shape of pred spectrogram: {pred.shape}")
        for i in range(len(pred)):
            clip_id = batch['clip_id'][i]
            gt_wav = batch['mix_wav'][i].cpu().numpy()
            pred_wav = spec_to_wav(pred[i], sr=16000, generator=VOCODER) if not self.args.use_input_wav else gt_wav
            
            for metric in self.test_metrics:
                if metric == 'fad':
                    # write to disk
                    gt_path = os.path.join(gt_dir, f'{clip_id}.wav')
                    pred_path = os.path.join(pred_dir, f'{clip_id}.wav')
                    sf.write(gt_path, gt_wav, 16000)
                    sf.write(pred_path, pred_wav, 16000)
                elif metric == 'av_sim':
                    if next(self.evaluators['av_sim']['audio_encoder'].parameters()).device != self.device:
                        self.evaluators['av_sim']['audio_encoder'] = self.evaluators['av_sim']['audio_encoder'].to(self.device)
                        self.evaluators['av_sim']['video_encoder'] = self.evaluators['av_sim']['video_encoder'].to(self.device)
                        self.evaluators['av_sim']['vid_proj'] = self.evaluators['av_sim']['vid_proj'].to(self.device)
                    # calculate the similarity between visual frames and audio fbanks
                    fbanks = self.wav2fbank(torch.tensor(pred_wav).unsqueeze(0), input_tdim=self.args.ast_tdim)
                    audio_feat = self.evaluators['av_sim']['audio_encoder'](fbanks.unsqueeze(0).to(self.device))
                    video_feat = self.evaluators['av_sim']['video_encoder'](batch['transformed_video'][i:i+1].to(self.device))
                    video_feat = self.evaluators['av_sim']['vid_proj'](video_feat)
                    sim_score = self.sim_matrix(audio_feat, video_feat)[0, 0].cpu().numpy()
                    metric_scores['av_sim'].append(sim_score)
                elif metric == 'av_sync':
                    if next(self.evaluators['av_sync']['audio_encoder'].parameters()).device != self.device:
                        self.evaluators['av_sync']['audio_encoder'] = self.evaluators['av_sync']['audio_encoder'].to(self.device)
                        self.evaluators['av_sync']['video_encoder'] = self.evaluators['av_sync']['video_encoder'].to(self.device)
                        if not self.transformer_head:
                            self.evaluators['av_sync']['vid_proj'] = self.evaluators['av_sync']['vid_proj'].to(self.device)
                        self.evaluators['av_sync']['classifier'] = self.evaluators['av_sync']['classifier'].to(self.device)
                    # calculate the similarity between visual frames and audio fbanks
                    pred_wav_current = torch.tensor(pred_wav).unsqueeze(0)
                    assert pred_wav_current.ndim == 2 and (pred_wav_current.shape[1] == 16000*3 or pred_wav_current.shape[1] >= 16000*2.9), pred_wav_current.shape
                    assert batch['video_32'].ndim==5 and batch['video_32'].shape[1] == 32 and batch['video_32'].shape[2] == 3, batch['video_32'].shape
                    pred_wav1 = pred_wav_current[:, :pred_wav_current.shape[1]//2]
                    video_32_1 = batch['video_32'][i:i+1,:batch['video_32'].shape[1]//2]
                    fbanks = self.wav2fbank(pred_wav1, input_tdim=self.evaluators['av_sync']['ast_tdim'])
                    if not self.transformer_head:
                        audio_feat = self.evaluators['av_sync']['audio_encoder'](fbanks.unsqueeze(0).to(self.device))
                        video_feat = self.evaluators['av_sync']['video_encoder'](video_32_1.to(self.device))
                        video_feat = self.evaluators['av_sync']['vid_proj'](video_feat)
                        cls_input = torch.cat([audio_feat, video_feat], dim=1)
                        pred1 = self.evaluators['av_sync']['classifier'](cls_input)
                    else:
                        audio_feat = self.evaluators['av_sync']['audio_encoder'](x=fbanks.unsqueeze(0).to(self.device), return_features=True)
                        video_feat = self.evaluators['av_sync']['video_encoder'](x=video_32_1.to(self.device), return_intermediate=True, max_pool_patches=True, mean_pool_patches=False)
                        pred1 = self.evaluators['av_sync']['classifier'](audio_feat, video_feat)
                    assert pred1.shape == torch.Size([1,1]), pred1.shape

                    pred_wav2 = pred_wav_current[:, pred_wav_current.shape[1]//2:]
                    video_32_2 = batch['video_32'][i:i+1,batch['video_32'].shape[1]//2:]
                    fbanks = self.wav2fbank(pred_wav2, input_tdim=self.evaluators['av_sync']['ast_tdim'])
                    if not self.transformer_head:
                        audio_feat = self.evaluators['av_sync']['audio_encoder'](fbanks.unsqueeze(0).to(self.device))
                        video_feat = self.evaluators['av_sync']['video_encoder'](video_32_2.to(self.device))
                        video_feat = self.evaluators['av_sync']['vid_proj'](video_feat)
                        cls_input = torch.cat([audio_feat, video_feat], dim=1)
                        pred2 = self.evaluators['av_sync']['classifier'](cls_input)
                    else:
                        audio_feat = self.evaluators['av_sync']['audio_encoder'](x=fbanks.unsqueeze(0).to(self.device), return_features=True)
                        video_feat = self.evaluators['av_sync']['video_encoder'](x=video_32_2.to(self.device), return_intermediate=True, max_pool_patches=True, mean_pool_patches=False)
                        pred2 = self.evaluators['av_sync']['classifier'](audio_feat, video_feat)
                    # final_pred = ((pred1.squeeze(0)+pred2.squeeze(0))/2).round().cpu().item()
                    final_pred = max([(pred2 >= 0.5).int().cpu().item(), (pred1 >= 0.5).int().cpu().item()])
                    metric_scores['av_sync'].append(final_pred)
                elif metric == 'al_sim':
                    if next(self.evaluators['al_sim']['audio_encoder'].parameters()).device != self.device:
                        self.evaluators['al_sim']['audio_encoder'] = self.evaluators['al_sim']['audio_encoder'].to(self.device)
                        self.evaluators['al_sim']['text_encoder'] = self.evaluators['al_sim']['text_encoder'].to(self.device)
                        self.evaluators['al_sim']['txt_proj'] = self.evaluators['al_sim']['txt_proj'].to(self.device)
                    # calculate the similarity between text and audio fbanks
                    fbanks = self.wav2fbank(torch.tensor(pred_wav).unsqueeze(0), input_tdim=self.args.ast_tdim)
                    audio_feat = self.evaluators['al_sim']['audio_encoder'](fbanks.unsqueeze(0).to(self.device))
                    text = self.tokenizer(batch['text'][i:i+1], return_tensors='pt', padding=True, truncation=True)
                    text = {k: v.to(self.device) for k, v in text.items()}
                    text_feat = self.evaluators['al_sim']['text_encoder'](**text).last_hidden_state[:, 0, :]
                    text_feat = self.evaluators['al_sim']['txt_proj'](text_feat)
                    sim_score = self.sim_matrix(audio_feat, text_feat)[0, 0].cpu().numpy()
                    metric_scores['al_sim'].append(sim_score)
                elif metric == 'ambient':
                    from ldm.datasets.ego4dsounds_dataset import find_segment_with_lowest_energy
                    seg_len = 8000
                    index = find_segment_with_lowest_energy(pred_wav, seg_len, increment=1600, lowest_k=1)[0]
                    lowest_pred_energy = np.sum(pred_wav[index: index + seg_len] ** 2)
                    if self.args.save_test_stats:
                        neighbor_wav = batch['neighbor_wav'][i].cpu().numpy()
                        index = find_segment_with_lowest_energy(neighbor_wav, seg_len, increment=1600, lowest_k=1)[0]
                        lowest_neighbor_energy = np.sum(neighbor_wav[index: index + seg_len] ** 2)
                        self.test_stats[clip_id] = {'pred_energy': float(lowest_pred_energy), 'neighbor_energy': lowest_neighbor_energy}
                    metric_scores['ambient'].append(lowest_pred_energy)
        
        for metric_name, metric_value_list in metric_scores.items():
            stats[metric_name] = torch.tensor(np.mean(metric_value_list)).to(self.device)
        
        return stats
    
    def test_epoch_end(self, outputs):
        gathered_outputs = self.all_gather(outputs)

        metrics = gathered_outputs[0].keys()
        output_str = f'Test epoch {self.current_epoch}, '
        agg_values = dict()
        for metric in metrics:
            values = torch.concat([output[metric].unsqueeze(-1) for output in gathered_outputs], dim=1)
            output_str += f'{metric}: {values.mean():.4f}, '
            agg_values[metric] = values.mean().item()
        
        # comptue fad only on rank 0
        if 'fad' in self.test_metrics and self.trainer.global_rank == 0:
            gt_dir = os.path.join('/tmp/v2a', os.path.basename(self.trainer.logdir), 'gt')
            pred_dir = os.path.join('/tmp/v2a', os.path.basename(self.trainer.logdir), 'pred')
            fad = self.evaluators['fad'].score(gt_dir, pred_dir, dtype="float32")
            output_str += f'fad: {fad:.4f}, '
            agg_values['fad'] = fad
        
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
            
        if self.args.save_test_stats:
            # save self.test_stats as json file to the output folder
            with open(os.path.join(self.trainer.logdir, 'test_stats.json'), 'w') as f:
                json.dump(self.test_stats, f, cls=NpEncoder)
            print(f"Test stats saved to {os.path.join(self.trainer.logdir, 'test_stats.json')}")
        
        print(output_str[:-2])
        self.test_stats = agg_values
    
    def save_test_stats(self, logdir, epoch_num):
        # write metric value to local file
        try:
            metric_file = os.path.join(logdir, 'test_stats.json')
            metric_dict = dict()
            if os.path.exists(metric_file):
                with open(metric_file, 'r') as f:
                    metric_dict = json.load(f)
            if epoch_num in metric_dict:
                metric_dict[epoch_num].update(self.test_stats)
            else:
                metric_dict[epoch_num] = self.test_stats
            with open(metric_file, 'w') as f:
                json.dump(metric_dict, f)
        except Exception as e:
            print(e, metric_file)
    
    def load_retrieved_audio(self, fname):
        from decord import AudioReader, VideoReader
        from decord import cpu, gpu
        from ldm.datasets.utils import TRANSFORMS_LDM
        
        target_len = 48000
        input_dim = 196
        try:
            ar = AudioReader(fname, ctx=cpu(0), sample_rate=16000)
            waveform = ar[:]
            if waveform.shape[1] > target_len:
                waveform = waveform[:, :target_len]
            else:
                waveform = torch.nn.functional.pad(waveform, (0, target_len - waveform.shape[1]))
        except Exception as e:
            print(f'Exception while reading audio file {fname} with {e}')
            waveform = torch.zeros(1, target_len)
        
        waveform = waveform[0].numpy()
        spec = TRANSFORMS_LDM(waveform)
        if spec.shape[1] < input_dim:
            spec = np.pad(spec, ((0, 0), (0, input_dim - spec.shape[1])), 'constant', constant_values=0)
        
        return spec, waveform

    @torch.no_grad()
    def retrieve_nearest_neighbor(self, batch):
        if next(self.evaluators['av_sim']['audio_encoder'].parameters()).device != self.device:
            self.evaluators['av_sim']['audio_encoder'] = self.evaluators['av_sim']['audio_encoder'].to(self.device)
            self.evaluators['av_sim']['video_encoder'] = self.evaluators['av_sim']['video_encoder'].to(self.device)
            self.evaluators['av_sim']['vid_proj'] = self.evaluators['av_sim']['vid_proj'].to(self.device)
        video_feat = self.evaluators['av_sim']['video_encoder'](batch['transformed_video'].to(self.device))
        video_feat = self.evaluators['av_sim']['vid_proj'](video_feat)
        sim_scores = [{} for _ in range(video_feat.shape[0])]
        # compute similarity between video and audio across train audio features
        for j in range(0, len(self.train_audio_features), 100):
            train_audio_files = list(self.train_audio_features.keys())[j:j+100]
            audio_feat = torch.stack([torch.tensor(self.train_audio_features[fname], device=self.device) for fname in train_audio_files], dim=0)
            sim_score = self.sim_matrix(audio_feat, video_feat).cpu().numpy()
            for i in range(video_feat.shape[0]):
                sim_scores[i].update({fname: sim_score[k, i] for k, fname in enumerate(train_audio_files)})
        
        # find the top 1 audio file and load the audio as prediction
        samples = []
        wavs = []
        for i, sim_score_mapping in enumerate(sim_scores):
            # get the highest sim and the file 
            max_file = max(sim_score_mapping, key=sim_score_mapping.get)
            # load the audio file
            spec, wav = self.load_retrieved_audio(os.path.join('data/ego4dsounds_224p/', max_file))
            samples.append(torch.tensor(spec))
            wavs.append(torch.tensor(wav))
        samples = torch.stack(samples, dim=0)
        wavs = torch.stack(wavs, dim=0)
        
        return samples, wavs
    @torch.no_grad()
    def log_sound(self, batch, N=4, n_row=4, sample=True, ddim_steps=250, ddim_eta=0.0, return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, size_len=64, guidance_scale=1.0, uncond_cond=None, **kwargs):
        batch = self.extract_features(batch)
        use_ddim = ddim_steps is not False

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        # logging.info(f"shape of x after get_input: {x.shape}")

        # x: B x C x H x W
        log["inputs_spec"] = x[:, 0, :, :]
        log["reconstruction_spec"] = xrec[:, 0, :, :]
        try:
            log["video_frame_path"] = batch['video_frame_path']     # video path
            log["video_time"] = batch['video_time']                 # video start & end
        except:
            log['mix_info_dict'] = batch['mix_info_dict']

        if self.args.compute_retrieval:
            log["samples"] = self.retrieve_nearest_neighbor(batch)
        with self.ema_scope("Plotting"):
            if type(c) == list:
                uncond_cond = []
                for item in c:
                    # logging.info(f"shape of item: {item.shape}")
                    uncond_cond.append(torch.zeros(item.shape).to(item.device))
            else:
                uncond_cond = torch.zeros(c.shape).to(c.device)
            samples, intermediates = self.sample_log_diff_sampler(cond=c, batch_size=N, sampler_name='DPM_Solver', ddim_steps=25, eta=ddim_eta, 
                                                                    size_len=size_len, unconditional_guidance_scale=guidance_scale,unconditional_conditioning=uncond_cond)
        x_samples = self.decode_first_stage(samples)
        # clip:
        # x_samples = torch.clamp(x_samples, -1, 1)
        log["samples"] = x_samples[:, 0, :, :]
        log["intermediates"] = intermediates
        # if plot_denoise_rows:
        #     denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
        #     log["denoise_row"] = denoise_grid
        
        if self.args.use_rec_spec:
            log["samples"] = log["reconstruction_spec"]
        elif self.args.use_input_spec:
            log["samples"] = log["inputs_spec"]
            
        return log

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=True, plot_denoise_rows=False, plot_progressive_rows=True,
                   plot_diffusion_rows=True, **kwargs):
        batch = self.extract_features(batch)
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                    # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True,
                    #                                      quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.cond_stage_trainable:
            print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
            params = params + list(self.cond_stage_model.parameters())
        if self.learn_logvar:
            print('Diffusion model optimizing logvar')
            params.append(self.logvar)
        opt = torch.optim.AdamW(params, lr=lr)
        if self.use_scheduler:
            assert 'target' in self.scheduler_config
            scheduler = instantiate_from_config(self.scheduler_config)
            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                }]
            return [opt], scheduler
        return opt

    @torch.no_grad()
    def to_rgb(self, x):
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm', 'hybrid_video', 'hybrid_audio', 'hybrid_video_audio']
        # if self.conditioning_key is hybrid, will use both concat and crossattn
        # for concat, needs to have a linear layer that project the concatenated input to the same dimension as xx
        # concat_dim = diff_model_config.params.context_dim * 2
        # self.target_dim = diff_model_config.params.context_dim
        # self.proj = nn.Linear(concat_dim, self.target_dim) if self.conditioning_key == 'hybrid' else None
        if self.conditioning_key == 'hybrid_video':
            # have a projection that can do [B, 16, 768] -> [B, 4, 16, 24]
            # right now I;m using a very simple approach, i.e. mlps
            # might need better contextualization
            self.proj = nn.Linear(768, 64) # 64 = 4*16
            self.proj2 = nn.Linear(16, 24)
            # can concat it with x into [B, 8, 16, 24], and then project it to [B, 4, 16, 24]
            self.proj_down = nn.Linear(8, 4)
            # right now hard code the dimensions, because it can't be passed to the config easily
        elif self.conditioning_key == 'hybrid_audio':
            # have a projection that can do [B, 8, 16, 24] -> [B, 4, 16, 24]
            self.proj_down = nn.Linear(8, 4)
        
        elif self.conditioning_key == 'hybrid_video_audio':
            # have a projection that can do [B, 1, 16, 768] -> [B, 4, 16, 24]
            self.proj = nn.Linear(768, 64) # 64 = 4*16
            self.proj2 = nn.Linear(16, 24)
            # then concat the 3 inputs into [B, 12, 16, 24], and then project it to [B, 4, 16, 24]
            self.proj_down = nn.Linear(12, 4)
        

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif 'hybrid' in self.conditioning_key:
            assert type(c_concat) == list, c_concat
            # new_c_concat = []
            # bsz = x.shape[0]
            # logging.info(f"x.shape: {x.shape}")
            # for item in c_concat:
            #     if item != None:
            #         logging.info(f"item.shape: {item.shape}")
            #         item = item[:bsz]
            #         logging.info(f"item.shape after trimming: {item.shape}")
            #         new_c_concat.append(item)
            #     else:
            #         new_c_concat.append(None)
            # c_concat = new_c_concat
            cc, xc = [], []
            if self.conditioning_key == 'hybrid_video': # video is concated, other features, i.e. audio and energy (if exists) are crossattn
                c_v = c_concat[0]
                c_v = self.proj(c_v) # [B, 16, 768] -> [B, 16, 64]
                c_v = self.proj2(c_v.transpose(2,1)) # [B, 16, 64] -> [B, 64, 24]
                c_v = c_v.reshape(-1, 4, 16, 24) # [B, 64, 24] -> [B, 4, 16, 24]
                xc = torch.cat([x] + [c_v], dim=1)
                assert xc.shape == torch.Size([x.shape[0], 8, 16, 24]), f"Expected {torch.Size([x.shape[0], 8, 16, 24])} but got {xc.shape}"
                xc = self.proj_down(xc.permute(0,2,3,1)).permute(0,3,1,2) # [B, 8, 16, 24] -> [B, 4, 16, 24]
                cc = [item for item in c_concat[1:] if item != None]
                if len(cc) > 0:
                    cc = torch.cat(cc, 1)
                else:
                    cc = None
                assert cc == None or (cc.ndim == 3 and cc.shape[-1] == 768 and cc.shape[0] == c_v.shape[0]), cc.shape
                out = self.diffusion_model(xc, t) if cc == None else self.diffusion_model(xc, t, context=cc)
            elif self.conditioning_key == 'hybrid_audio':
                # for item in c_concat:
                #     logging.info(f"item.shape: {item.shape}")
                assert c_concat[1] != None, c_concat
                c_a = c_concat[1]
                assert c_a.shape == torch.Size([x.shape[0], 4, 16, 24]), f"Expected {torch.Size([x.shape[0], 4, 16, 24])} but got {c_a.shape}"
                xc = torch.cat([x] + [c_a], dim=1)
                assert xc.shape == torch.Size([x.shape[0], 8, 16, 24]), f"Expected {torch.Size([x.shape[0], 8, 16, 24])} but got {xc.shape}"
                xc = self.proj_down(xc.permute(0,2,3,1)).permute(0,3,1,2) # [B, 8, 16, 24] -> [B, 4, 16, 24]]
                if (len(c_concat) == 2 or (len(c_concat) == 3 and c_concat[2] == None)) and c_concat[0] != None:
                    cc = [c_concat[0]]
                elif len(c_concat) == 3 and c_concat[2] != None and c_concat[0] != None:
                    cc = [c_concat[0], c_concat[2]]
                else:
                    cc = []
                if len(cc) > 0:
                    cc = torch.cat(cc, 1)
                else:
                    cc = None
                assert cc == None or (cc.ndim == 3 and cc.shape[-1] == 768 and cc.shape[0] == c_a.shape[0]), cc.shape
                out = self.diffusion_model(xc, t) if cc == None else self.diffusion_model(xc, t, context=cc)
            elif self.conditioning_key == 'hybrid_video_audio':
                c_v = c_concat[0]
                c_v = self.proj(c_v) # [B, 16, 768] -> [B, 16, 64]
                c_v = self.proj2(c_v.transpose(2,1)) # [B, 16, 64] -> [B, 64, 24]
                c_v = c_v.reshape(-1, 4, 16, 24) # [B, 64, 24] -> [B, 4, 16, 24]
                c_a = c_concat[1]
                assert c_a.shape == torch.Size([x.shape[0], 4, 16, 24]), f"Expected {torch.Size([x.shape[0], 4, 16, 24])} but got {c_a.shape}"
                xc = torch.cat([x] + [c_v] + [c_a], dim=1)
                assert xc.shape == torch.Size([x.shape[0], 12, 16, 24]), f"Expected {torch.Size([x.shape[0], 12, 16, 24])} but got {xc.shape}"
                xc = self.proj_down(xc.permute(0,2,3,1)).permute(0,3,1,2)
                assert xc.shape == torch.Size([x.shape[0], 4, 16, 24]), f"Expected {torch.Size([x.shape[0], 4, 16, 24])} but got {xc.shape}"
                if len(c_concat) == 3 and c_concat[2] != None:
                    cc = c_concat[2]
                else:
                    cc = None
                assert cc == None or (cc.ndim == 3 and cc.shape[-1] == 768 and cc.shape[0] == c_v.shape[0]), cc.shape
                out = self.diffusion_model(xc, t) if cc == None else self.diffusion_model(xc, t, context=cc)
            else:
                raise NotImplementedError(f"{self.conditioning_key} not implemented")

        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    # TODO: move all layout-specific hacks to this class
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs