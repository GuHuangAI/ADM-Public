import sys
import os
# print(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import argparse
import cv2
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
    # assert model_cfg.ldm, 'This file is only used for ldm！'
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

        self.image_size = model.image_size

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.cfg = cfg
        self.whole_test = cfg.sampler.get('whole_test', False)
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
            Eval_tool = Evaluation_metrics('DUTS', device)
            # gt = np.where(gt > 0.5, 1.0, 0.0)
            test_mae = AvgMeter()
            test_maxf = AvgMeter()
            test_avgf = AvgMeter()
            test_s_m = AvgMeter()
            num = 0
            results_dict = {}
            # silogs, log10s, rmss, log_rmss, abs_rels, sq_rels, d1s, d2s, d3s = [], [], [], [], [], [], [], [], []
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
                ori_size = batch['ori_size']
                # if self.rk45:
                #     batch_pred, nfe = self.rk45_sample(batch_size=bs, cond=cond, mask=mask)
                # else:
                #     if isinstance(self.model, nn.parallel.DistributedDataParallel):
                #         batch_pred = self.model.module.sample(batch_size=bs, cond=cond, mask=mask)
                #     elif isinstance(self.model, nn.Module):
                #         batch_pred = self.model.sample(batch_size=bs, cond=cond, mask=mask)
                # print(cond.shape[-2:])
                # print(image.shape[-2:])
                if self.whole_test:
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        crop_seg_logit = self.model.module.sample(batch_size=self.batch_size, cond=cond)
                        if self.cfg.sampler.get('flip_test', False):
                            crop_seg_logit_flip = self.model.module.sample(batch_size=self.batch_size, cond=cond.flip(dims=[-1]))
                            crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                    elif isinstance(self.model, nn.Module):
                        crop_seg_logit = self.model.sample(batch_size=self.batch_size, cond=cond)
                        if self.cfg.sampler.get('flip_test', False):
                            crop_seg_logit_flip = self.model.sample(batch_size=self.batch_size, cond=cond.flip(dims=[-1]))
                            crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                    else:
                        raise NotImplementedError
                    batch_pred = crop_seg_logit
                else:
                    batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.crop_size,
                                               stride=self.cfg.sampler.stride, mask=mask, out_channels=self.cfg.sampler.out_channels)
                # print(batch_pred.shape)
                for j, (img, c) in enumerate(zip(batch_pred, cond)):
                    img_name = batch["img_name"][j]
                    # img = 10 * img # [0, 10]
                    gt = image[j]
                    gt = torch.where(gt > 0.5, 1., 0.)
                    img = torch.where(img > 0.5, 1., 0.)
                    mae, max_f, avg_f, s_score = Eval_tool.cal_total_metrics(img, gt)
                    test_mae.update(mae, n=1)
                    test_maxf.update(max_f, n=1)
                    test_avgf.update(avg_f, n=1)
                    test_s_m.update(s_score, n=1)
                    num += 1

                    file_name = self.results_folder / img_name
                    img = F.interpolate(img.unsqueeze(0), size=(ori_size[0][j].item(), ori_size[1][j].item()))
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")
            results_dict['mae'] = test_mae.avg
            results_dict['max_f'] = test_maxf.avg
            results_dict['avg_f'] = test_avgf.avg
            results_dict['s_m'] = test_s_m.avg
            print('results: ', results_dict)
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
                    crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.module.sample(batch_size=1, cond=crop_img.flip(dims=[-1]))
                        crop_seg_logit = 0.5 * crop_seg_logit + 0.5 * crop_seg_logit_flip.flip(dims=[-1])
                elif isinstance(self.model, nn.Module):
                    crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img)
                    if self.cfg.sampler.get('flip_test', False):
                        crop_seg_logit_flip = self.model.sample(batch_size=1, cond=crop_img.flip(dims=[-1]))
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

def compute_errors(gt, pred):
    thresh = torch.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).to(torch.float32).mean()
    d2 = (thresh < 1.25 ** 2).to(torch.float32).mean()
    d3 = (thresh < 1.25 ** 3).to(torch.float32).mean()

    rmse = (gt - pred) ** 2
    rmse = torch.sqrt(rmse.mean())

    rmse_log = (torch.log(gt) - torch.log(pred)) ** 2
    rmse_log = torch.sqrt(rmse_log.mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred)**2) / gt)

    err = torch.log(pred) - torch.log(gt)
    silog = torch.sqrt(torch.mean(err ** 2) - torch.mean(err) ** 2) * 100

    err = torch.abs(torch.log10(pred) - torch.log10(gt))
    log10 = torch.mean(err)

    return silog, log10, abs_rel, sq_rel, rmse, rmse_log, d1, d2, d3


def get_ode_sampler(rtol=1e-5, atol=1e-5,
                    method='RK45', eps=1e-4, device='cuda'):
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
        C, noise = pred
        drift = -C - 1 / t.sqrt() * noise
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

class Evaluation_metrics():
    def __init__(self, dataset, device):
        self.dataset = dataset
        self.device = device

        print(f'Dataset:{self.dataset}')

    def cal_total_metrics(self, pred, mask):
        # MAE
        mae = torch.mean(torch.abs(pred - mask)).item()
        # MaxF measure
        beta2 = 0.3
        prec, recall = self._eval_pr(pred, mask, 255)
        f_score = (1 + beta2) * prec * recall / (beta2 * prec + recall)
        f_score[f_score != f_score] = 0  # for Nan
        max_f = f_score.max().item()
        # AvgF measure
        avg_f = f_score.mean().item()
        # S measure
        alpha = 0.5
        y = mask.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1:
            x = pred.mean()
            Q = x
        else:
            mask[mask >= 0.5] = 1
            mask[mask < 0.5] = 0
            Q = alpha * self._S_object(pred, mask) + (1 - alpha) * self._S_region(pred, mask)
            if Q.item() < 0:
                Q = torch.FloatTensor([0.0])
        s_score = Q.item()

        return mae, max_f, avg_f, s_score

    def _eval_pr(self, y_pred, y, num):
        if self.device:
            prec, recall = torch.zeros(num).to(self.device), torch.zeros(num).to(self.device)
            thlist = torch.linspace(0, 1 - 1e-10, num).to(self.device)
        else:
            prec, recall = torch.zeros(num), torch.zeros(num)
            thlist = torch.linspace(0, 1 - 1e-10, num)
        for i in range(num):
            y_temp = (y_pred >= thlist[i]).float()
            tp = (y_temp * y).sum()
            prec[i], recall[i] = tp / (y_temp.sum() + 1e-20), tp / (y.sum() + 1e-20)
        return prec, recall

    def _S_object(self, pred, mask):
        fg = torch.where(mask == 0, torch.zeros_like(pred), pred)
        bg = torch.where(mask == 1, torch.zeros_like(pred), 1 - pred)
        o_fg = self._object(fg, mask)
        o_bg = self._object(bg, 1 - mask)
        u = mask.mean()
        Q = u * o_fg + (1 - u) * o_bg
        return Q

    def _object(self, pred, mask):
        temp = pred[mask == 1]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)

        return score

    def _S_region(self, pred, mask):
        X, Y = self._centroid(mask)
        mask1, mask2, mask3, mask4, w1, w2, w3, w4 = self._divideGT(mask, X, Y)
        p1, p2, p3, p4 = self._dividePrediction(pred, X, Y)
        Q1 = self._ssim(p1, mask1)
        Q2 = self._ssim(p2, mask2)
        Q3 = self._ssim(p3, mask3)
        Q4 = self._ssim(p4, mask4)
        Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4
        # print(Q)
        return Q

    def _centroid(self, mask):
        rows, cols = mask.size()[-2:]
        mask = mask.view(rows, cols)
        if mask.sum() == 0:
            if self.device:
                X = torch.eye(1).to(self.device) * round(cols / 2)
                Y = torch.eye(1).to(self.device) * round(rows / 2)
            else:
                X = torch.eye(1) * round(cols / 2)
                Y = torch.eye(1) * round(rows / 2)
        else:
            total = mask.sum()
            if self.device:
                i = torch.from_numpy(np.arange(0, cols)).to(self.device).float()
                j = torch.from_numpy(np.arange(0, rows)).to(self.device).float()
            else:
                i = torch.from_numpy(np.arange(0, cols)).float()
                j = torch.from_numpy(np.arange(0, rows)).float()
            X = torch.round((mask.sum(dim=0) * i).sum() / total)
            Y = torch.round((mask.sum(dim=1) * j).sum() / total)
        return X.long(), Y.long()

    def _divideGT(self, mask, X, Y):
        h, w = mask.size()[-2:]
        area = h * w
        mask = mask.view(h, w)
        LT = mask[:Y, :X]
        RT = mask[:Y, X:w]
        LB = mask[Y:h, :X]
        RB = mask[Y:h, X:w]
        X = X.float()
        Y = Y.float()
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    def _dividePrediction(self, pred, X, Y):
        h, w = pred.size()[-2:]
        pred = pred.view(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    def _ssim(self, pred, mask):
        mask = mask.float()
        h, w = pred.size()[-2:]
        N = h * w
        x = pred.mean()
        y = mask.mean()
        sigma_x2 = ((pred - x) * (pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((mask - y) * (mask - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x) * (mask - y)).sum() / (N - 1 + 1e-20)

        aplha = 4 * x * y * sigma_xy
        beta = (x * x + y * y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

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