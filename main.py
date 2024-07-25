import importlib
import argparse, os, sys, datetime, glob
import builtins
import torch
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
import time
import shutil
from pytorch_lightning.utilities import rank_zero_info
import numpy as np
import logging
def print(*args, **kwargs):
    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        builtins.print(*args, **kwargs)


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2**20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            # print("Project config")
            # print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(
                self.config,
                os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)),
            )

            # print("Lightning config")
            # print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(
                OmegaConf.create({"lightning": self.lightning_config}),
                os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)),
            )


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(
        self,
        batch_size,
        train=None,
        validation=None,
        test=None,
        predict=None,
        wrap=False,
        num_workers=None,
        shuffle_test_loader=False,
        use_worker_init_fn=False,
        shuffle_val_dataloader=False,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
            # self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config_input_dict(data_cfg)  # !! input dict

    def setup(self, stage=None):  ## Init The Dataset !!
        self.datasets = dict(
            (
                k,
                instantiate_from_config_input_dict(self.dataset_configs[k]),
            )  # !! Input params dict
            for k in self.dataset_configs
        )
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    @staticmethod
    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    def _train_dataloader(self):
        return DataLoader(
            self.datasets["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            shuffle=True,
        )

    def _val_dataloader(self):
        return DataLoader(
            self.datasets["validation"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            shuffle=False,
        )

    def _test_dataloader(self):
        return DataLoader(
            self.datasets["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=self.worker_init_fn,
            shuffle=False,
        )


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
    def set_args(self, args):
        self.data.set_args(args)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit('.', 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(**config.get('params', dict()))

def instantiate_from_config_input_dict(config):
    if not 'target' in config:
        raise KeyError('Expected key `target` to instantiate.')
    return get_obj_from_str(config['target'])(config.get('params', dict()))


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    # print(vars(args))
    # print(opt.__dict__)
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))

        
def run(opt, unknown):
    seed_everything(opt.seed)

    # Init and Save Configs:  # DDP
    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    print("Overwritting: ", cli)
    config = OmegaConf.merge(*configs, cli)
    if opt.no_test:
        config.data.params.test = None
    print(OmegaConf.to_yaml(config))
    
    lightning_config = config.pop("lightning", OmegaConf.create())
    # merge trainer cli with config
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # default to ddp: !!!
    trainer_config['accelerator'] = 'ddp'
    # Non Default trainer Config:
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    print(OmegaConf.to_yaml(lightning_config))
        
    # Model:
    if 'energy_cond_config' in config.model['params']:  # legacy
        del config.model['params']['energy_cond_config']
    model = instantiate_from_config(config.model)
    model.set_args(opt)

    # trainer and callbacks:
    trainer_kwargs = dict()
    root_dir = os.getcwd()

    default_logger_cfg = {
        "target": "pytorch_lightning.loggers.WandbLogger",
        "params": {
            "entity": 'avlp',
            "project": opt.wandb_project,
            "name": nowname,
            "save_dir": os.path.join(root_dir, logdir),
            "id": nowname,
        }
    }
    
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    if not (opt.fast_dev_run or opt.test or opt.test_all or opt.visualize):
        trainer_kwargs['logger'] = instantiate_from_config(logger_cfg)

    default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": os.path.join(root_dir, ckptdir),
                # "monitor": "val/time_domain_loss",
                # "filename": "{val/loss_simple_ema:.4f}_{epoch:03d}",
                "filename": "{epoch:06d}",
                "verbose": True,
                "save_last": True,
                'save_top_k': -1,
                'period': config.checkpoint.save_every_n_epochs,
            }
    }
    
    if hasattr(model, "monitor"):
        print(f"Monitoring {model.monitor} as checkpoint metric.")
        print("config.checkpoint.save_every_n_epochs", config.checkpoint.save_every_n_epochs)
        default_modelckpt_cfg["params"]["monitor"] = model.monitor

    if "modelcheckpoint" in lightning_config:
        modelckpt_cfg = lightning_config.modelcheckpoint
    else:
        modelckpt_cfg = OmegaConf.create()
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)
    trainer_kwargs['checkpoint_callback'] = instantiate_from_config(modelckpt_cfg)
    default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            }
        }

    default_callbacks_cfg[config.callback.logger_name] = dict(config.callback)
    default_callbacks_cfg['cuda_callback'] = {"target": "main.CUDACallback"}

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
    trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]
    
    # Data:
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    for k in data.datasets:
        data.datasets[k].set_args(opt)
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # configure learning rate:
    bs, base_lr = config.data.params.batch_size, config.model.base_learning_rate
    ngpu = len(lightning_config.trainer.gpus.strip(",").split(','))
    if 'accumulate_grad_batches' in lightning_config.trainer:
        accumulate_grad_batches = lightning_config.trainer.accumulate_grad_batches
    else:
        accumulate_grad_batches = 1
    lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

    if opt.scale_lr:
        model.learning_rate = accumulate_grad_batches * ngpu * bs * base_lr
        print("Setting learning rate to {:.2e} = {} (accumulate_grad_batches) * {} (num_gpus) * {} (batchsize) * {:.2e} (base_lr)".format(
            model.learning_rate, accumulate_grad_batches, ngpu, bs, base_lr))
    else:
        model.learning_rate = base_lr
        print("+++ Not Using LR Scaling ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    if opt.visualize:
        from ldm.logger import generate_demo
        # only run inference for one epoch and generate demos
        dataloader = data.val_dataloader() if opt.test_val else data.test_dataloader()
        split = 'val' if opt.test_val else 'test'
        if not opt.compute_retrieval:
            ckpt = torch.load(opt.resume_from_checkpoint, map_location='cpu')
            print(f"Loaded model from {opt.resume_from_checkpoint}")
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
        epoch = int(opt.resume_from_checkpoint.split('=')[-1].split('.')[0])
        output_dir = os.path.join(logdir, "sound_eval", split, "epoch_{}".format(epoch)) if opt.output_dir is None else opt.output_dir
        if opt.output_postfix is not None:
            output_dir = output_dir + '_' + opt.output_postfix
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            print('Remove existing folder {}'.format(output_dir))
        os.makedirs(output_dir, exist_ok=True)
        
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= opt.num_vis_batch:
                break
            generate_demo(model, batch, batch_idx, split, output_dir, args=opt)
        exit()

    trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)
    trainer.logdir = logdir  ## logdir
    trainer.max_epochs = opt.epoch
    
    if opt.test:
        dataloader = data.val_dataloader() if opt.test_val else data.test_dataloader()
        ckpt = torch.load(opt.resume_from_checkpoint, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        trainer.test(model, dataloader)
        exit()
    
    if opt.test_all:
        dataloader = data.val_dataloader() if opt.test_val else data.test_dataloader()
        # test all checkpoints in ckptdir
        ckpts = glob.glob(os.path.join(ckptdir, 'epoch=*.ckpt'))
        for ckpt_file in sorted(ckpts):
            epoch_num = int(ckpt_file.split('=')[-1].split('.')[0])
            ckpt = torch.load(ckpt_file, map_location='cpu')
            model.load_state_dict(ckpt['state_dict'])
            print(f"Testing {ckpt_file}")
            trainer.current_epoch = epoch_num
            trainer.test(model, dataloader)
            model.save_test_stats(logdir, epoch_num)
        exit()

    # Checkpointing:
    def melk(*args, **kwargs):
        # run all checkponit hooks
        if trainer.global_rank == 0 and not opt.fast_dev_run:
            print("Summoning checkpoint")
            ckpt_path = os.path.join(ckptdir, "last.ckpt")
            trainer.save_checkpoint(ckpt_path)

    def divein(*args, **kwargs):
        if trainer.global_rank == 0:
            import pudb;
            pudb.set_trace()
    
    import signal
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # Run the Model:
    if opt.train:
        try:
            trainer.fit(model, data)
        except Exception:
            melk()
            raise

    if not opt.no_test and not trainer.interrupted:
        trainer.test(model, data.test_dataloader())
    print("Finishing Training !!")
    

def process_videos(video_dir):
    if "epic_kitchen_action_sounds" in video_dir:
        return video_dir
    processed_dir = os.path.join(video_dir, 'processed')
    if os.path.exists(processed_dir):
        return processed_dir
    else:
        os.makedirs(processed_dir, exist_ok=True)
    for video in os.listdir(video_dir):
        if not video.endswith('.mp4'):
            continue
        # process and keep the first 3 seconds
        os.system('''ffmpeg -i {} -vf "scale=224:224" -ar 16000 -ac 1 -t 3 {}'''
                .format(os.path.join(video_dir, video), os.path.join(processed_dir, video)))
    
    return processed_dir


def prepare_metadata_for_demo(demo_dir):
    import pandas as pd
    
    processed_dir = process_videos(demo_dir)
    # find all videos in demo_dir ending in mp4
    videos = [os.path.basename(fp) for fp in glob.glob(os.path.join(processed_dir, '*.mp4'))]
    csv_data = pd.DataFrame(columns=['video_uid', 'clip_file'])
    for video in videos:
        video_uid = video.split('.')[0]
        csv_data = csv_data._append({'video_uid': video_uid, 'clip_file': video, 'clip_text': ''}, ignore_index=True)
    
    print(f"Saving metadata for {len(videos)} videos in {demo_dir}")
    print(csv_data.head())
    csv_data.to_csv(os.path.join(demo_dir, 'metadata.csv'), index=False, sep='\t')
    return processed_dir


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p", "--project", help="name of new or path to existing project"
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=72,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="data/logs/ego4d",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=2000,
        help="epoch num",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="ego4d_diffusion",
        help="wandb project",
    )
    parser.add_argument("--test", default=False, action='store_true')
    parser.add_argument("--test-all", default=False, action='store_true')
    parser.add_argument("--num-nodes", default=1, type=int)
    parser.add_argument("--batch-size", default=150, type=int)
    parser.add_argument("--slurm", default=False, action='store_true')
    parser.add_argument("--part", default='learnfair', type=str)
    parser.add_argument("--pretrained-model", default='', type=str)
    parser.add_argument("--resume-from-checkpoint", default=None, type=str)
    parser.add_argument("--fast-dev-run", default=False, action='store_true')
    parser.add_argument("--comment", default='', type=str)
    parser.add_argument("--test-metrics", default='', type=str)
    parser.add_argument("--test-val", default=False, action='store_true')
    parser.add_argument("--use-rec-spec", default=False, action='store_true')
    parser.add_argument("--use-input-spec", default=False, action='store_true')
    parser.add_argument("--use-input-wav", default=False, action='store_true')
    ## args for dataloader
    parser.add_argument("--left-nearest-k", default=3, type=int)
    parser.add_argument("--right-nearest-k", default=3, type=int)
    parser.add_argument("--neighbor-file", default='', type=str)
    parser.add_argument("--audio-cond", default='rand_neighbor', type=str)
    parser.add_argument("--seg-len", default=8000, type=int)
    parser.add_argument("--lowest-k", default=1, type=int)
    parser.add_argument("--num-frames", default=16, type=int)
    parser.add_argument("--normalize", default=False, action='store_true', help="0: no normalization, 1: normalize waveform with librosa.utils.normalize")
    parser.add_argument("--num-test-samples", default=-1, type=int)
    ## args for model
    parser.add_argument("--pool-patches", default='mean', type=str)
    parser.add_argument("--load-temporal-fix", default='zeros', type=str)
    parser.add_argument("--pretrained-av-sim", default='data/pretrained/av.pth', type=str)
    parser.add_argument("--pretrained-al-sim", default='data/pretrained/al.pth', type=str)
    parser.add_argument("--ast-tdim", default=149, type=int)
    parser.add_argument("--compute-retrieval", default=False, action='store_true')
    ## args for evaluation
    parser.add_argument("--visualize", default=False, action='store_true')
    parser.add_argument("--demo-dir", default=None, type=str)  # given a video dir, generate examples for videos in that directory
    parser.add_argument("--output-dir", default=None, type=str)  # given a video dir, generate examples for videos in that directory
    parser.add_argument("--output-postfix", default=None, type=str)  # given a video dir, generate examples for videos in that directory
    parser.add_argument("--vocoder", default=None, type=str, help="if None, use griffinlim, otherwise currently only support hifigan")
    parser.add_argument("--cond-zero", default=False, action='store_true')
    parser.add_argument("--low-ambient", default=False, action='store_true')
    parser.add_argument("--mid-ambient", default=False, action='store_true')
    parser.add_argument("--eval-ckpt", default=-1, type=int)
    parser.add_argument("--num-vis-batch", default=2, type=int)
    parser.add_argument("--eval-last", default=False, action='store_true')
    parser.add_argument("--retrieve-nn", default=False, action='store_true')
    parser.add_argument("--no-hifigan", default=False, action='store_true')
    parser.add_argument("--save-test-stats", default=False, action='store_true')
    
    return parser


if __name__ == "__main__":
    # facilitate logging/debugging
    formatter = (
        "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d || %(message)s"
    )
    logging.basicConfig(format=formatter, level=logging.INFO)
    parser = get_parser()   # --base config_path.yaml --name exper1 --gpus 0, 1, 2
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()
    assert len([x for x in unknown if x.startswith('--') or x.startswith('-')]) == 0, f"Unknown args: {unknown}"
    
    if opt.fast_dev_run:
        opt.gpus = '0,'
        opt.no_test = True

    if opt.test or opt.test_all:
        opt.train = False
        opt.batch_size = 20
        opt.test_metrics = 'fad,av_sim,al_sim,ambient,av_sync' if opt.test_metrics == '' else opt.test_metrics
        opt.vocoder = 'hifigan' if not opt.no_hifigan else None
    
    if opt.visualize:
        # only generate demos and do not compute stats
        opt.train = False
        opt.gpus = '0,'
        opt.batch_size = 50
        opt.test_metrics = ''
        opt.no_test = False
        opt.vocoder = 'hifigan' if not opt.no_hifigan else None
        unknown += ["data.params.num_workers=1"] # ensure no stochasticity in dataloader
    
    if opt.retrieve_nn:
        opt.output_postfix = 'retrieve_ambient' if opt.output_postfix is None else f"{opt.output_postfix}_retrieve_ambient"
    
    if opt.low_ambient:
        opt.neighbor_file = 'data/ego4dsounds_224p_test/89193e37-1ed4-4a65-996b-6883f2e1cf61/89193e37-1ed4-4a65-996b-6883f2e1cf61_narration_pass_1_12636.mp4'
        opt.output_postfix = 'low_ambient' if opt.output_postfix is None else f"{opt.output_postfix}_low_ambient"
    
    if opt.mid_ambient:
        opt.neighbor_file = 'data/ego4dsounds_224p_test/82ace2ff-4a87-4abe-ba45-3965bbc13658/82ace2ff-4a87-4abe-ba45-3965bbc13658_narration_pass_1_4919.mp4'
        opt.output_postfix = 'mid_ambient' if opt.output_postfix is None else f"{opt.output_postfix}_mid_ambient"
    
    if opt.demo_dir is not None:
        processed_dir = prepare_metadata_for_demo(opt.demo_dir)
        opt.output_dir = os.path.join(opt.demo_dir, 'generated')
        unknown += ["data.params.test.params.data_dir={}".format(processed_dir)]
        unknown += ["data.params.test.params.metadata_file={}".format(os.path.join(opt.demo_dir, 'metadata.csv'))]
    
    if opt.eval_last:
        checkpoints = sorted(glob.glob(os.path.join(opt.logdir, opt.name, 'checkpoints', 'epoch=*.ckpt')))
        opt.resume = checkpoints[-1]
    
    if opt.eval_ckpt >= 0:
        # pad to 6 digits
        opt.resume = os.path.join(opt.logdir, opt.name, 'checkpoints', f'epoch={opt.eval_ckpt:06d}.ckpt')
    
    if opt.cond_zero:
        opt.output_postfix = 'cond_zero'
        unknown += ["model.params.audio_cond_config.neighbor_audio_cond_prob=0.0"]
        
    if opt.seed != 72:
        opt.output_postfix = f"seed{opt.seed}" if opt.output_postfix == '' else f"{opt.output_postfix}_seed{opt.seed}"
    
    if opt.compute_retrieval:
        opt.test_metrics = 'fad,av_sim,al_sim,ambient,av_sync'
    
    unknown += ["data.params.batch_size={}".format(opt.batch_size)]
    unknown += ["model.params.test_metrics={}".format(opt.test_metrics)] if opt.test_metrics != '' else []
    unknown += ["model.params.pretrained_video_extractor={}".format(opt.pretrained_model)] if opt.pretrained_model != '' else []
    unknown += ["model.params.video_cond_len={}".format(opt.num_frames)]
    unknown += ["model.params.cond_stage_config.params.seq_len={}".format(opt.num_frames)]
    
    now = datetime.datetime.now().strftime("%m-%dT%H")
    if opt.name and opt.resume:
        print(f"{opt.resume} exists. Ignoring name {opt.name}")
    
    if opt.resume:  # resume path
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            # logdir = "/".join(paths[:-2])
            logdir = "/".join(paths[:-2])   # /val paths
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume.rstrip("/")
            ckpt = os.path.join(logdir, "checkpoints", "last.ckpt")
        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))    # Find the Config File
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    elif opt.fast_dev_run:
        nowname = name = "debug"
        logdir = os.path.join(opt.logdir, name)
    else:
        if opt.name:
            name = opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = cfg_name
        else:
            name = ""
        nowname = now + '_' + name + opt.postfix # used for name log files
        logdir = os.path.join(opt.logdir, name)  # no longer use date for expt. keep name meaningful
        if os.path.exists(logdir):
            print('====================================================================')
            print(f"Warning: logdir {logdir} already exists!")
            print('====================================================================')
    
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")

    if opt.slurm:
        import submitit
        n_gpus = len(opt.gpus.split(','))
        executor = submitit.AutoExecutor(folder="data/logs/submitit/%j")
        executor.update_parameters(slurm_job_name=opt.name, timeout_min=60*48, slurm_partition=opt.part,
                                   nodes=opt.num_nodes, gpus_per_node=n_gpus, cpus_per_task=10,
                                   slurm_constraint='volta32gb', slurm_mem=100 * 1024,
                                   tasks_per_node=opt.n_gpus, comment=opt.comment
                                   )
        job = executor.submit(run, opt, unknown)
        print(job.job_id)
    else:
        run(opt, unknown)
