import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
# .path.append()
from taming.modules.losses.vqperceptual import *

### used for VAE
class LPIPSWithDiscriminator(nn.Module):
    def __init__(self, *, disc_start, logvar_init=0.0, kl_weight=1.0, pixelloss_weight=1.0,
                 disc_num_layers=3, disc_in_channels=3, disc_factor=1.0, disc_weight=1.0,
                 perceptual_weight=1.0, use_actnorm=False, disc_conditional=False,
                 disc_loss="hinge"):

        super().__init__()
        assert disc_loss in ["hinge", "vanilla"]
        self.kl_weight = kl_weight
        self.pixel_weight = pixelloss_weight
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

        self.discriminator = NLayerDiscriminator(input_nc=disc_in_channels,
                                                 n_layers=disc_num_layers,
                                                 use_actnorm=use_actnorm
                                                 ).apply(weights_init)
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.disc_conditional = disc_conditional

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, inputs, reconstructions, posteriors, optimizer_idx,
                global_step, last_layer=None, cond=None, split="train",
                weights=None):
        rec_loss = torch.abs(inputs.contiguous() - reconstructions.contiguous()) + \
                    F.mse_loss(inputs, reconstructions, reduction="none")
        if self.perceptual_weight > 0:
            p_loss = self.perceptual_loss(inputs.contiguous(), reconstructions.contiguous())
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights*nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        kl_loss = posteriors.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            if cond is None:
                assert not self.disc_conditional
                logits_fake = self.discriminator(reconstructions.contiguous())
            else:
                assert self.disc_conditional
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous(), cond), dim=1))
            g_loss = -torch.mean(logits_fake)

            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(nll_loss, g_loss, last_layer=last_layer)
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            loss = weighted_nll_loss + self.kl_weight * kl_loss + d_weight * disc_factor * g_loss

            log = {"{}/total_loss".format(split): loss.clone().detach().mean(), "{}/logvar".format(split): self.logvar.detach(),
                   "{}/kl_loss".format(split): kl_loss.detach().mean(), "{}/nll_loss".format(split): nll_loss.detach().mean(),
                   "{}/rec_loss".format(split): rec_loss.detach().mean(),
                   "{}/d_weight".format(split): d_weight.detach(),
                   "{}/disc_factor".format(split): torch.tensor(disc_factor),
                   "{}/g_loss".format(split): g_loss.detach().mean(),
                   }
            return loss, log

        if optimizer_idx == 1:
            # second pass for discriminator update
            if cond is None:
                logits_real = self.discriminator(inputs.contiguous().detach())
                logits_fake = self.discriminator(reconstructions.contiguous().detach())
            else:
                logits_real = self.discriminator(torch.cat((inputs.contiguous().detach(), cond), dim=1))
                logits_fake = self.discriminator(torch.cat((reconstructions.contiguous().detach(), cond), dim=1))

            disc_factor = adopt_weight(self.disc_factor, global_step, threshold=self.discriminator_iter_start)
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {"{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                   "{}/logits_real".format(split): logits_real.detach().mean(),
                   "{}/logits_fake".format(split): logits_fake.detach().mean()
                   }
            return d_loss, log


# used for saliency detection
class API_Loss(nn.Module):
    def __init__(self, k1=3, k2=11, k3=23,
                        p1=1, p2=5, p3=11):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
    def forward(self, pred, mask): # B, C, H, W
        pred, mask = torch.sigmoid(pred), torch.sigmoid(mask)
        # w1 = torch.abs(F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) - mask)
        # w2 = torch.abs(F.avg_pool2d(mask, kernel_size=15, stride=1, padding=7) - mask)
        # w3 = torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
        w1 = torch.abs(F.avg_pool2d(mask, kernel_size=self.k1, stride=1, padding=self.p1) - mask)
        w2 = torch.abs(F.avg_pool2d(mask, kernel_size=self.k2, stride=1, padding=self.p2) - mask)
        w3 = torch.abs(F.avg_pool2d(mask, kernel_size=self.k3, stride=1, padding=self.p3) - mask)

        omega = 1 + 0.5 * (w1 + w2 + w3) * mask

        bce = F.binary_cross_entropy(pred, mask, reduce=None)
        abce = (omega * bce).sum(dim=(2, 3)) / (omega + 0.5).sum(dim=(2, 3))

        inter = ((pred * mask) * omega).sum(dim=(2, 3))
        union = ((pred + mask) * omega).sum(dim=(2, 3))
        aiou = 1 - (inter + 1) / (union - inter + 1)

        mae = F.l1_loss(pred, mask, reduce=None)
        amae = (omega * mae).sum(dim=(2, 3)) / (omega - 1).sum(dim=(2, 3))

        return (0.7 * abce + 0.7 * aiou + 0.7 * amae).mean(dim=1) # (B, )

# used for depth estimation
class MEADSTD_TANH_NORM_Loss(nn.Module):
    """
    loss = MAE((d-u)/s - d') + MAE(tanh(0.01*(d-u)/s) - tanh(0.01*d'))
    """
    def __init__(self, valid_threshold=1e-3, max_threshold=1, with_sigmoid=False):
        super(MEADSTD_TANH_NORM_Loss, self).__init__()
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        self.with_sigmoid = with_sigmoid
        #self.thres1 = 0.9

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).to(gt.device))
                data_std_dev.append(torch.tensor(1).to(gt.device))
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).to(gt.device)
        data_std_dev = torch.stack(data_std_dev, dim=0).to(gt.device)

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        if self.with_sigmoid:
            pred, gt = torch.sigmoid(pred), torch.sigmoid(gt)
        """
        Calculate loss.
        """
        mask = (gt > self.valid_threshold) & (gt < self.max_threshold)   # [b, c, h, w]
        mask_sum = torch.sum(mask, dim=(1, 2, 3))
        # mask invalid batches
        mask_batch = mask_sum > 100
        if True not in mask_batch:
            return torch.tensor(0.0, dtype=torch.float32, device=pred.device)
        mask_maskbatch = mask[mask_batch]
        pred_maskbatch = pred[mask_batch]
        gt_maskbatch = gt[mask_batch]

        gt_mean, gt_std = self.transform(gt_maskbatch)
        gt_trans = (gt_maskbatch - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)
        # gt_trans = gt_maskbatch

        B, C, H, W = gt_maskbatch.shape
        # loss = 0
        # loss_tanh = 0
        loss_out = []
        for i in range(B):
            mask_i = mask_maskbatch[i, ...]
            pred_depth_i = pred_maskbatch[i, ...][mask_i]
            gt_trans_i = gt_trans[i, ...][mask_i]

            depth_diff = torch.abs(gt_trans_i - pred_depth_i)
            # loss += torch.mean(depth_diff)
            loss = torch.mean(depth_diff)

            tanh_norm_gt = torch.tanh(0.1*gt_trans_i)
            tanh_norm_pred = torch.tanh(0.1*pred_depth_i)
            # tanh_norm_gt = torch.tanh(gt_trans_i)
            # tanh_norm_pred = torch.tanh(pred_depth_i)
            # loss_tanh += torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))
            loss_tanh = torch.mean(torch.abs(tanh_norm_gt - tanh_norm_pred))
            loss_out.append(loss + loss_tanh)
        # loss_out = loss/B + loss_tanh/B
        loss_out = torch.stack(loss_out, dim=0)
        return loss_out
class MSGIL_NORM_Loss(nn.Module):
    """
    Our proposed GT normalized Multi-scale Gradient Loss Function.
    """
    def __init__(self, scale=4, valid_threshold=-1e-8, max_threshold=1e8):
        super(MSGIL_NORM_Loss, self).__init__()
        self.scales_num = scale
        self.valid_threshold = valid_threshold
        self.max_threshold = max_threshold
        self.EPSILON = 1e-6

    def one_scale_gradient_loss(self, pred_scale, gt, mask):
        mask_float = mask.to(dtype=pred_scale.dtype, device=pred_scale.device)

        d_diff = pred_scale - gt

        v_mask = torch.mul(mask_float[:, :, :-2, :], mask_float[:, :, 2:, :])
        v_gradient = torch.abs(d_diff[:, :, :-2, :] - d_diff[:, :, 2:, :])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(d_diff[:, :, :, :-2] - d_diff[:, :, :, 2:])
        h_mask = torch.mul(mask_float[:, :, :, :-2], mask_float[:, :, :, 2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        valid_num = torch.sum(h_mask) + torch.sum(v_mask)

        gradient_loss = torch.sum(h_gradient) + torch.sum(v_gradient)
        gradient_loss = gradient_loss / (valid_num + 1e-8)

        return gradient_loss

    def transform(self, gt):
        # Get mean and standard deviation
        data_mean = []
        data_std_dev = []
        for i in range(gt.shape[0]):
            gt_i = gt[i]
            mask = gt_i > 0
            depth_valid = gt_i[mask]
            if depth_valid.shape[0] < 10:
                data_mean.append(torch.tensor(0).cuda())
                data_std_dev.append(torch.tensor(1).cuda())
                continue
            size = depth_valid.shape[0]
            depth_valid_sort, _ = torch.sort(depth_valid, 0)
            depth_valid_mask = depth_valid_sort[int(size*0.1): -int(size*0.1)]
            data_mean.append(depth_valid_mask.mean())
            data_std_dev.append(depth_valid_mask.std())
        data_mean = torch.stack(data_mean, dim=0).cuda()
        data_std_dev = torch.stack(data_std_dev, dim=0).cuda()

        return data_mean, data_std_dev

    def forward(self, pred, gt):
        mask = gt > self.valid_threshold
        grad_term = 0.0
        gt_mean, gt_std = self.transform(gt)
        gt_trans = (gt - gt_mean[:, None, None, None]) / (gt_std[:, None, None, None] + 1e-8)
        for i in range(self.scales_num):
            step = pow(2, i)
            d_gt = gt_trans[:, :, ::step, ::step]
            d_pred = pred[:, :, ::step, ::step]
            d_mask = mask[:, :, ::step, ::step]
            grad_term += self.one_scale_gradient_loss(d_pred, d_gt, d_mask)
        return grad_term

class MSE_Loss(nn.Module):
    def __init__(self, thresh_min=0, thresh_max=1, mask=False, with_sigmoid=False):
        super(MSE_Loss, self).__init__()
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.mask = mask
        self.with_sigmoid = with_sigmoid

    def forward(self, pred, gt, reduce_dims=[1, 2, 3], mask=None, reduction='mean'):
        if self.with_sigmoid:
            pred, gt = torch.sigmoid(pred), torch.sigmoid(gt)
        B = pred.shape[0]
        if not self.mask:
            if reduction == 'mean':
                loss = F.mse_loss(pred, gt, reduction='none').mean(dim=reduce_dims)
            elif reduction == 'sum':
                loss = F.mse_loss(pred, gt, reduction='none').sum(dim=reduce_dims)
            else:
                raise NotImplementedError("")
        else:
            loss = []
            mask = (gt > self.thresh_min) & (gt < self.thresh_max)  # [b, c, h, w]
            mask_sum = torch.sum(mask, dim=(1, 2, 3))
            # mask invalid batches
            mask_batch = mask_sum > 100
            if True not in mask_batch:
                return torch.tensor(0.0, dtype=torch.float32, device=pred.device)
            mask_maskbatch = mask[mask_batch]
            pred_maskbatch = pred[mask_batch]
            gt_maskbatch = gt[mask_batch]
            for i in range(B):
                try:
                    mask_i = mask_maskbatch[i, ...]
                    pred_depth_i = pred_maskbatch[i, ...][mask_i]
                    gt_depth_i = gt_maskbatch[i, ...][mask_i]
                    loss.append(F.mse_loss(pred_depth_i, gt_depth_i, reduction=reduction))
                except:
                    loss.append(torch.tensor(0.0, dtype=torch.float32, device=pred.device))
            loss = torch.stack(loss, dim=0)
        return loss

class MAE_Loss(nn.Module):
    def __init__(self, thresh_min=0, thresh_max=1, mask=False, with_sigmoid=False):
        super(MAE_Loss, self).__init__()
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max
        self.mask = mask
        self.with_sigmoid = with_sigmoid

    def forward(self, pred, gt, reduce_dims=[1, 2, 3], mask_gt=None, reduction='mean'):
        if self.with_sigmoid:
            pred, gt = torch.sigmoid(pred), torch.sigmoid(gt)
        B = pred.shape[0]
        if not self.mask:
            if reduction == 'mean':
                loss = torch.abs(pred - gt).mean(dim=reduce_dims)
            elif reduction == 'sum':
                loss = torch.abs(pred - gt).sum(dim=reduce_dims)
            else:
                raise NotImplementedError
        else:
            loss = []
            if mask_gt is not None:
                mask = (mask_gt > self.thresh_min) & (mask_gt < self.thresh_max)
            else:
                mask = (gt > self.thresh_min) & (gt < self.thresh_max)  # [b, c, h, w]
            mask_sum = torch.sum(mask, dim=(1, 2, 3))
            # mask invalid batches
            mask_batch = mask_sum > 100
            if True not in mask_batch:
                return torch.tensor(0.0, dtype=torch.float32, device=pred.device)
            mask_maskbatch = mask[mask_batch]
            pred_maskbatch = pred[mask_batch]
            gt_maskbatch = gt[mask_batch]
            for i in range(B):
                try:
                    mask_i = mask_maskbatch[i, ...]
                    pred_depth_i = pred_maskbatch[i, ...][mask_i]
                    gt_depth_i = gt_maskbatch[i, ...][mask_i]
                    if reduction == 'mean':
                        loss.append(torch.abs(pred_depth_i - gt_depth_i).mean(dim=reduce_dims))
                    elif reduction == 'sum':
                        loss.append(torch.abs(pred_depth_i - gt_depth_i).sum(dim=reduce_dims))
                    else:
                        raise NotImplementedError
                except:
                    loss.append(torch.tensor(0.0, dtype=torch.float32, device=pred.device))
            loss = torch.stack(loss, dim=0)
        return loss

class silog_loss(nn.Module):
    def __init__(self, variance_focus):
        super(silog_loss, self).__init__()
        self.variance_focus = variance_focus

    def forward(self, depth_est, depth_gt, mask):
        d = torch.log(depth_est[mask]) - torch.log(depth_gt[mask])
        return torch.sqrt((d ** 2).mean() - self.variance_focus * (d.mean() ** 2)) * 10.0

if __name__ == '__main__':
    x = torch.rand(10, 1, 50, 50) * 2 - 1
    y = torch.rand(10, 1, 50, 50) * 2 - 1
    pass