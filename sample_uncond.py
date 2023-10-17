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


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(cfg.get('seed', 42))
    np.random.seed(cfg.get('seed', 42))
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    # model_cfg.cfg = model_cfg
    unet_cfg = model_cfg.unet
    unet_kwargs = {'cfg': unet_cfg}
    unet_kwargs.update(unet_cfg)
    unet = construct_class_by_name(**unet_kwargs)
    if not model_cfg.ldm:
        # model_cfg.model = unet
        model_kwargs = {'model': unet, 'cfg': model_cfg}
        model_kwargs.update(model_cfg)
        dpm = construct_class_by_name(**model_kwargs)
    else:
        first_stage_cfg = model_cfg.first_stage
        first_stage_model = construct_class_by_name(**first_stage_cfg)
        # model_cfg.auto_encoder = first_stage_model
        # unet_cfg = model_cfg.unet
        # unet = construct_class_by_name(**unet_cfg)
        # model_cfg.model = unet
        model_kwargs = {'model': unet, 'auto_encoder': first_stage_model, 'cfg': model_cfg}
        model_kwargs.update(model_cfg)
        dpm = construct_class_by_name(**model_kwargs)

    data_cfg = cfg.data
    dataset = construct_class_by_name(**data_cfg)

    dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=False, pin_memory=True, num_workers=2)
    # sample_num = model_cfg.sample_num
    # batch_size = sampler_cfg.sample_batch_size
    # batch_num = math.ceil(sample_num // batch_size)
    # save_dir = Path(cfg.save_folder)
    # save_dir.mkdir(exist_ok=True, parents=True)


    sampler_cfg = cfg.sampler
    sampler = Sampler(
        dpm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder, cfg=cfg,
    )
    sampler.sample()
    assert len(os.listdir(sampler_cfg.target_path)) > 0, "{} have no image !".format(sampler_cfg.target_path)
    if sampler_cfg.get('cal_fid', False):
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
        self.accelerator.native_amp = False
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
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)


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
            # self.model.eval()
            for idx in tqdm(range(batch_num)):
                if idx == batch_num - 1:
                    real_batch_szie = self.sample_num - (batch_num - 1) * self.batch_size
                else:
                    real_batch_szie = self.batch_size
                if self.rk45:
                    batch_pred, nfe = self.rk45_sample(batch_size=real_batch_szie)
                else:
                    if isinstance(self.model, nn.parallel.DistributedDataParallel):
                        batch_pred = self.model.module.sample(batch_size=real_batch_szie)
                    elif isinstance(self.model, nn.Module):
                        batch_pred = self.model.sample(batch_size=real_batch_szie)
                for j in range(batch_pred.shape[0]):
                    img = batch_pred[j]
                    img_id = idx * self.batch_size + j
                    file_name = f'{img_id: 010d}.png'
                    file_name = self.results_folder / file_name
                    tv.utils.save_image(img, str(file_name))

        accelerator.print('sampling complete')

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
    if model_type == 'const_sde4':
        C, noise = pred
        drift = -1 * (C + noise / torch.sqrt(t.reshape(noise.shape[0], *((1,) * (len(C.shape) - 1)))))
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
        vec_t = torch.ones(shape[0], device=x.device) * t
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