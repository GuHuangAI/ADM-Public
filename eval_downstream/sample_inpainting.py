import sys
import os
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from ddm.utils import *
import torchvision as tv
from ddm.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from ddm.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate


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

# Colors for all 20 parts
part_colors = [[0, 0, 0], [255, 85, 0],  [255, 170, 0],
               [255, 0, 85], [255, 0, 170],
               [0, 255, 0], [85, 255, 0], [170, 255, 0],
               [0, 255, 85], [0, 255, 170],
               [0, 0, 255], [85, 0, 255], [170, 0, 255],
               [0, 85, 255], [0, 170, 255],
               [255, 255, 0], [255, 255, 85], [255, 255, 170],
               [255, 0, 255], [255, 85, 255], [255, 170, 255],
               [0, 255, 255], [85, 255, 255], [170, 255, 255]]

def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    # random.seed(seed)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    assert model_cfg.ldm, 'This file is only used for ldm！'
    # model_cfg.cfg = model_cfg
    unet_cfg = model_cfg.unet
    unet_kwargs = {'cfg': unet_cfg}
    unet_kwargs.update(unet_cfg)
    unet = construct_class_by_name(**unet_kwargs)
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = construct_class_by_name(**first_stage_cfg)
    model_kwargs = {'model': unet, 'auto_encoder': first_stage_model, 'cfg': model_cfg}
    model_kwargs.update(model_cfg)
    ldm = construct_class_by_name(**model_kwargs)
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))

    data_cfg = cfg.data
    # data_cfg.augment_horizontal_flip = False
    # data_cfg.img_folder = cfg.sampler.target_path
    dataset = construct_class_by_name(**data_cfg)
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    # sample_num = model_cfg.sample_num
    # batch_size = sampler_cfg.sample_batch_size
    # batch_num = math.ceil(sample_num // batch_size)
    # save_dir = Path(cfg.save_folder)
    # save_dir.mkdir(exist_ok=True, parents=True)


    sampler_cfg = cfg.sampler
    sampler = Sampler(
        ldm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder,cfg=cfg,
    )
    sampler.sample()
    if data_cfg.name == 'cityscapes' or data_cfg.name == 'sr' or data_cfg.name == 'edge':
        exit()
    else:
        assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
        sampler.cal_fid(target_path=sampler_cfg.target_path)
    pass


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)
        self.whole_test = cfg.sampler.get('whole_test', True)

        self.image_size = model.image_size

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.cfg = cfg
        self.results_folder = Path(results_folder)
        # self.results_folder_cond = Path(results_folder+'_cond')
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)
            # self.results_folder_cond.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(cfg.sampler.ckpt_path,
                          map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        batch_num = self.batch_num
        with torch.no_grad():
            self.model.eval()
            psnr = 0.
            num = 0
            for idx, batch in tqdm(enumerate(self.dl)):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)
                image = batch["image"]
                image = unnormalize_to_zero_to_one(image)
                cond = batch['cond']
                mask = batch['ori_mask'] if 'ori_mask' in batch else None
                # raw_w = batch["raw_size"][0].item()  # default batch size = 1
                # raw_h = batch["raw_size"][1].item()
                # img_name = batch["img_name"][0]
                bs = cond.shape[0]
                # if self.rk45:
                #     batch_pred, nfe = self.rk45_sample(batch_size=bs, cond=cond, mask=mask)
                # else:
                #     if isinstance(self.model, nn.parallel.DistributedDataParallel):
                #         batch_pred = self.model.module.sample(batch_size=bs, cond=cond, mask=mask)
                #     elif isinstance(self.model, nn.Module):
                #         batch_pred = self.model.sample(batch_size=bs, cond=cond, mask=mask)
                if self.whole_test:
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        batch_pred = self.model.module.sample(batch_size=self.batch_size, cond=cond, mask=mask)
                        if self.cfg.sampler.get('flip_test', False):
                            batch_pred_flip = self.model.module.sample(batch_size=self.batch_size, cond=cond.flip(dims=[-1]),
                                                                           mask=mask)
                            batch_pred = 0.5 * batch_pred + 0.5 * batch_pred_flip.flip(dims=[-1])
                    elif isinstance(self.model, nn.Module):
                        batch_pred = self.model.sample(batch_size=self.batch_size, cond=cond, mask=mask)
                        if self.cfg.sampler.get('flip_test', False):
                            batch_pred_flip = self.model.sample(batch_size=self.batch_size, cond=cond.flip(dims=[-1]),
                                                                    mask=mask)
                            batch_pred = 0.5 * batch_pred + 0.5 * batch_pred_flip.flip(dims=[-1])
                    else:
                        raise NotImplementedError
                else:
                    if cond.shape[-2:] != image.shape[-2:]:
                        batch_pred = self.slide_sample_sr(cond, image, crop_size=self.cfg.sampler.crop_size,
                                                       stride=self.cfg.sampler.stride, mask=mask, out_channels=self.cfg.sampler.out_channels,
                                                          ori_size=batch['ori_size'])
                    else:
                        batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.crop_size,
                                                   stride=self.cfg.sampler.stride, mask=mask, out_channels=self.cfg.sampler.out_channels)

                for j, (img, c) in enumerate(zip(batch_pred, cond)):
                    img_name = batch["img_name"][j]
                    c = (c+1)/2
                    # psnr += -10. * torch.log10(F.mse_loss(batch_pred[j], image[j]))
                    # num += 1
                    # img = batch_pred[j]
                    # img_id = idx * self.batch_size + j
                    # file_name = f'{img_id: 010d}.png'
                    # file_name = self.results_folder / file_name
                    # tv.utils.save_image(img, str(file_name))
                    # file_name = f'{img_id: 010d}.png'
                    # file_name = self.results_folder_cond / file_name
                    cond_save_path = self.results_folder.parent / (self.results_folder.name + '_condition')
                    cond_save_path.mkdir(parents=True, exist_ok=True)
                    file_name = self.results_folder / img_name
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")
                    file_name_cond = cond_save_path / img_name
                    tv.utils.save_image(c, str(file_name_cond)[:-4] + ".png")
            # print('PSNR: ', psnr / num)
        accelerator.print('sampling complete')

    def slide_sample(self, inputs, crop_size, stride, mask=None, out_channels=1):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                # batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                # crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.module.sample(batch_size=1, cond=crop_img.flip(dims=[-1]),
                                                                       mask=mask)
                        crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.sample(batch_size=1, cond=crop_img.flip(dims=[-1]), mask=mask)
                        crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                else:
                    raise NotImplementedError
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits

    def slide_sample_sr(self, cond, image, crop_size, stride, mask=None, out_channels=1, **kwargs):
        ori_size = kwargs['ori_size']
        # print(ori_size)
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = image.size()
        _, _, h_cond, w_cond = cond.size()
        out_channels = out_channels
        h_grids = max(h_cond - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_cond - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = image.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = image.new_zeros((batch_size, out_channels, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_cond)
                x2 = min(x1 + w_crop, w_cond)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = cond[:, :, y1:y2, x1:x2]
                # change the image shape to patch shape
                # batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
                # the output of encode_decode is seg logits tensor map
                # with shape [N, C, H, W]
                # crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
                if isinstance(self.model, nn.parallel.DistributedDataParallel):
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.module.sample(batch_size=1, cond=crop_img.flip(dims=[-1]),
                                                                       mask=mask)
                        crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.sample(batch_size=1, cond=crop_img.flip(dims=[-1]), mask=mask)
                        crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                else:
                    raise NotImplementedError
                preds += F.pad(crop_seg_logit,
                               (int(x1*4), int(preds.shape[3] - x2*4), int(y1*4),
                                int(preds.shape[2] - y2*4)))

                count_mat[:, :, y1*4:y2*4, x1*4:x2*4] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        return seg_logits[:, :, :ori_size[0], :ori_size[1]]

    def cal_fid(self, target_path):
        command = 'fidelity -g 0 -f -i -b {} --input1 {} --input2 {}'\
            .format(self.batch_size, str(self.results_folder), target_path)
        os.system(command)

    def rk45_sample(self, batch_size):
        with torch.no_grad():
            # Initial sample
            # z = torch.randn(batch_size, 3, *(self.image_size))
            shape = (batch_size, 3, *(self.image_size))
            ode_sampler = get_ode_sampler(method='RK45')
            x, nfe = ode_sampler(model=self.model, shape=shape)
            x = unnormalize_to_zero_to_one(x)
            x.clamp_(0., 1.)
            return x, nfe

def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-3, device='cuda'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # Reverse diffusion predictor for denoising
    predictor_obj = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    _, x = predictor_obj.update_fn(x, vec_eps)
    return x

  def drift_fn(model, x, t, model_type='const'):
    """Get the drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # rsde = sde.reverse(score_fn, probability_flow=True)
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # vec_t = torch.ones(shape[0], device=x.device) * t
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      # if denoise:
      #   x = denoise_update_fn(model, x)

      # x = inverse_scaler(x)
      return x, nfe

  return ode_sampler

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))

if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass