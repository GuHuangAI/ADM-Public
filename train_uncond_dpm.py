import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ddm.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from ddm.utils import *
import torchvision as tv

from ddm.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode


def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf


def main(args):
    cfg = CfgNode(args.cfg)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    unet_cfg = model_cfg.unet
    unet = construct_class_by_name(**unet_cfg)
    # model_cfg.cfg = model_cfg
    model_kwargs = {'model': unet, 'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    dpm = construct_class_by_name(**model_kwargs)
    model_kwargs.pop('model')

    data_cfg = cfg.data
    dataset = construct_class_by_name(**data_cfg)
    # if data_cfg.name == 'cifar10':
    #     dataset = CIFAR10(
    #         img_folder=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    # elif data_cfg.name == 'imagenet':
    #     dataset = ImageNetDataset(
    #         img_folder=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    # elif data_cfg['name'] == 'lsun':
    #     dataset = LSUNDataset(
    #         img_folder=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    # elif data_cfg['name'] == 'cityscapes':
    #     dataset = CityscapesDataset(
    #         data_root=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    # elif data_cfg['name'] == 'ade20k':
    #     dataset = ADE20KDataset(
    #         data_root=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    # else:
    #     dataset = ImageDataset(
    #         img_folder=data_cfg.img_folder,
    #         image_size=model_cfg.image_size,
    #         augment_horizontal_flip=data_cfg.augment_horizontal_flip
    #     )
    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    train_cfg = cfg.trainer
    trainer = Trainer(
        dpm, dl, train_batch_size=data_cfg.batch_size,
        gradient_accumulate_every=train_cfg.gradient_accumulate_every,
        train_lr=train_cfg.lr, train_num_steps=train_cfg.train_num_steps,
        save_and_sample_every=train_cfg.save_and_sample_every, results_folder=train_cfg.results_folder,
        amp=train_cfg.amp, fp16=train_cfg.fp16, log_freq=train_cfg.log_freq, cfg=cfg,
        resume_milestone=train_cfg.resume_milestone,
        train_wd=train_cfg.get('weight_decay', 1e-2),
    )
    if train_cfg.test_before:
        if trainer.accelerator.is_main_process:
            with torch.no_grad():
                for datatmp in dl:
                    break
                if isinstance(trainer.model, nn.parallel.DistributedDataParallel):
                    all_images = trainer.model.module.sample(batch_size=data_cfg.batch_size)
                elif isinstance(trainer.model, nn.Module):
                    all_images = trainer.model.sample(batch_size=data_cfg.batch_size)
                # all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

            nrow = 2 ** math.floor(math.log2(math.sqrt(data_cfg.batch_size)))
            tv.utils.save_image(all_images, str(trainer.results_folder / f'sample-{train_cfg.resume_milestone}_{model_cfg.sampling_timesteps}.png'), nrow=nrow)
            torch.cuda.empty_cache()
    trainer.train()
    pass


class Trainer(object):
    def __init__(
            self,
            model,
            data_loader,
            train_batch_size=16,
            gradient_accumulate_every=1,
            train_lr=1e-4,
            train_wd=1e-4,
            train_num_steps=100000,
            save_and_sample_every=1000,
            num_samples=25,
            results_folder='./results',
            amp=False,
            fp16=False,
            split_batches=True,
            log_freq=20,
            resume_milestone=0,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler]
        )

        self.accelerator.native_amp = amp

        self.model = model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.image_size
        self.sample_cfg = cfg.sampler
        self.test_in_train = self.sample_cfg.get('test_in_train', False)
        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        # optimizer
        self.opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=train_lr, weight_decay=train_wd)
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.96, cfg.trainer.min_lr/train_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda=lr_lambda)
        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok=True, parents=True)
            self.ema = EMA(model, ema_model=None, beta=0.9996,
                           update_after_step=cfg.trainer.ema_update_after_step,
                           update_every=cfg.trainer.ema_update_every)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.lr_scheduler = \
            self.accelerator.prepare(self.model, self.opt, self.lr_scheduler)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.lr_scheduler.load_state_dict(data['lr_scheduler'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:
                total_loss = 0.
                total_loss_dict = {'loss_simple': 0., 'loss_vlb': 0., 'total_loss': 0., 'lr': 5e-5,
                                    }
                for ga_ind in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    batch = next(self.dl)
                    for key in batch.keys():
                        if isinstance(batch[key], torch.Tensor):
                            batch[key].to(device)
                    # if self.step == 0 and ga_ind == 0:
                    #     if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    #         self.model.module.on_train_batch_start(batch)
                    #     else:
                    #         self.model.on_train_batch_start(batch)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict = self.model.module.training_step(batch)
                        else:
                            loss, log_dict = self.model.training_step(batch)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                        loss_simple = log_dict["train/loss_simple"].item() / self.gradient_accumulate_every
                        loss_vlb = log_dict["train/loss_vlb"].item() / self.gradient_accumulate_every
                        total_loss_dict['loss_simple'] += loss_simple
                        total_loss_dict['loss_vlb'] += loss_vlb
                        total_loss_dict['total_loss'] += total_loss
                        # total_loss_dict['s_fact'] = self.model.module.scale_factor
                        # total_loss_dict['s_bias'] = self.model.module.scale_bias

                    self.accelerator.backward(loss)
                total_loss_dict['lr'] = self.opt.param_groups[0]['lr']
                describtions = dict2str(total_loss_dict)
                describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                if accelerator.is_main_process:
                    pbar.desc = describtions

                if self.step % self.log_freq == 0:
                    if accelerator.is_main_process:
                        # pbar.desc = describtions
                        self.logger.info(describtions)

                accelerator.clip_grad_norm_(filter(lambda p: p.requires_grad, self.model.parameters()), 1.0)
                # pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()
                self.lr_scheduler.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('loss_simple', loss_simple, self.step)
                    self.writer.add_scalar('loss_vlb', loss_vlb, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        self.model.eval()
                        # self.ema.ema_model.eval()

                        with torch.no_grad():
                            # img = self.dl
                            # batches = num_to_groups(self.num_samples, self.batch_size)
                            # all_images_list = list(map(lambda n: self.model.module.validate_img(ns=self.batch_size), batches))
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                all_images = self.model.module.sample(batch_size=self.batch_size)
                            elif isinstance(self.model, nn.Module):
                                all_images = self.model.sample(batch_size=self.batch_size)
                            # all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

                        # all_images = torch.cat(all_images_list, dim = 0)
                        nrow = 2 ** math.floor(math.log2(math.sqrt(self.batch_size)))
                        tv.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow=nrow)
                        if self.test_in_train:
                            self.sample_test(milestone)
                        self.model.train()
                accelerator.wait_for_everyone()
                pbar.update(1)

        accelerator.print('training complete')

    def sample_test(self, milestone):
        results_folder = Path(self.sample_cfg.save_folder)
        results_folder.mkdir(exist_ok=True, parents=True)
        json_path = str(self.results_folder / f'result_{milestone}.json')
        target_path = self.sample_cfg.target_path
        sample_num = self.sample_cfg.sample_num
        batch_size = self.sample_cfg.batch_size
        batch_num = math.ceil(sample_num // batch_size)
        with torch.no_grad():
            for idx in tqdm(range(batch_num)):
                if idx == batch_num - 1:
                    real_batch_szie = sample_num - (batch_num - 1) * batch_size
                else:
                    real_batch_szie = batch_size
                if isinstance(self.ema.ema_model, nn.parallel.DistributedDataParallel):
                    batch_pred = self.ema.ema_model.module.sample(batch_size=real_batch_szie)
                elif isinstance(self.ema.ema_model, nn.Module):
                    batch_pred = self.ema.ema_model.sample(batch_size=real_batch_szie)
                for j in range(batch_pred.shape[0]):
                    img = batch_pred[j]
                    img_id = idx * batch_size + j
                    file_name = f'{img_id: 010d}.png'
                    file_name = results_folder / file_name
                    tv.utils.save_image(img, str(file_name))
        command = 'fidelity -g 0 -f -i -b {} --out_path {} --input1 {} --input2 {}' \
            .format(64, json_path, str(results_folder), target_path)
        os.system(command)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass