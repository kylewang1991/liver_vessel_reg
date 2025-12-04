import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable

import abc
import torch.nn as nn

class PixelBasedLoss(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __call__(self, true, pred, weight):
        pass

class StructureLoss(abc.ABC):
    def __init__(self):
        pass
    
    @abc.abstractmethod
    def __call__(self, true, pred, distance):
        pass


class NCC(PixelBasedLoss):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, inshape, device, use_weight=False, win=None):
        super(NCC, self).__init__()

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(inshape)
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        self.win = [9] * ndims if win is None else win

        # compute filters
        self.sum_filt = torch.ones([1, 1, *self.win]).to(device)

        pad_no = math.floor(self.win[0] / 2)

        if ndims == 1:
            self.stride = (1)
            self.padding = (pad_no)
        elif ndims == 2:
            self.stride = (1, 1)
            self.padding = (pad_no, pad_no)
        else:
            self.stride = (1, 1, 1)
            self.padding = (pad_no, pad_no, pad_no)

        # get convolution function
        self.conv_fn = getattr(F, 'conv%dd' % ndims)

        self.win_size = np.prod(self.win)

        self.use_weight = use_weight

    def __call__(self, y_true, y_pred, weight):

        Ii = y_true
        Ji = y_pred

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = self.conv_fn(Ii, self.sum_filt, stride=self.stride, padding=self.padding)
        J_sum = self.conv_fn(Ji, self.sum_filt, stride=self.stride, padding=self.padding)
        I2_sum = self.conv_fn(I2, self.sum_filt, stride=self.stride, padding=self.padding)
        J2_sum = self.conv_fn(J2, self.sum_filt, stride=self.stride, padding=self.padding)
        IJ_sum = self.conv_fn(IJ, self.sum_filt, stride=self.stride, padding=self.padding)


        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        if self.use_weight:
            if weight is not None:
                weight = weight.to(cc.device)
                cc_weighted = cc * weight
            else:
                # raise error   
                raise ValueError("weight is not set")
        else:
            cc_weighted = cc

        return -torch.mean(cc_weighted)


class MSE(PixelBasedLoss):
    """
    Mean squared error loss.
    """
    def __init__(self, use_weight=False):
        super(MSE, self).__init__()
        self.use_weight = use_weight

    def __call__(self, y_true, y_pred, weight):
        mse = (y_true - y_pred) ** 2

        if self.use_weight:
            if weight is not None:
                weight = weight.to(mse.device)
                mse_weighted = mse * weight
            else:
                # raise error
                raise ValueError("weight is not set")
        else:
            mse_weighted = mse

        return torch.mean(mse_weighted)


class Dice(StructureLoss):
    """
    N-D dice for segmentation
    """
    def __init__(self, inshape):
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))

    def __call__(self, y_true, y_pred, distance):
        y_true = y_true.to(y_pred.device)
        top = 2 * (y_true * y_pred).sum(dim=self.vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=self.vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return 1-dice


# class Grad:
#     """
#     N-D gradient loss.
#     """

#     def __init__(self, inshape, penalty='l1', loss_mult=None):
#         self.penalty = penalty
#         self.loss_mult = loss_mult
#         self.ndims = len(inshape)

#     def _diffs(self, y):

#         df = [None] * self.ndims
#         for i in range(self.ndims):
#             d = i + 2
#             # permute dimensions
#             r = [d, *range(0, d), *range(d + 1, self.ndims + 2)]
#             y = y.permute(r)
#             dfi = y[1:, ...] - y[:-1, ...]

#             # permute back
#             # note: this might not be necessary for this loss specifically,
#             # since the results are just summed over anyway.
#             r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, self.ndims + 2)]
#             df[i] = dfi.permute(r)

#         return df

#     def loss(self, _, y_pred):
#         if self.penalty == 'l1':
#             dif = [torch.abs(f) for f in self._diffs(y_pred)]
#         else:
#             assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
#             dif = [f * f for f in self._diffs(y_pred)]

#         df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
#         grad = sum(df) / len(df)

#         if self.loss_mult is not None:
#             grad *= self.loss_mult

#         return grad.mean()


class Grad3d(PixelBasedLoss):
    """
    N-D gradient loss.
    """

    def __init__(self, use_weight=False, penalty='l1'):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.use_weight = use_weight

    def __call__(self, y_true, y_pred, weight):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        if self.use_weight:
            if weight is not None:
                weight = weight.to(dy.device)
                dy = dy * weight[:, :, :-1, :, :]
                dx = dx * weight[:, :, :, :-1, :]
                dz = dz * weight[:, :, :, :, :-1]
            else:
                # raise error
                raise ValueError("weight is not set")
        else:
            dy = dy
            dx = dx
            dz = dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        return grad


class mind_ssc(PixelBasedLoss):
    def __init__(self, device, use_weight=False, win=None, radius=2, dilation=2):
        super(mind_ssc, self).__init__()

        # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor

        # kernel size
        kernel_size = radius * 2 + 1

        # define start and end locations for self-similarity pattern
        six_neighbourhood = torch.Tensor([[0, 1, 1],
                                          [1, 1, 0],
                                          [1, 0, 1],
                                          [1, 1, 2],
                                          [2, 1, 1],
                                          [1, 2, 1]]).long()

        # squared distances
        dist = self.pdist_squared(six_neighbourhood.t().unsqueeze(0)).squeeze(0)

        # define comparison mask
        x, y = torch.meshgrid(torch.arange(6), torch.arange(6))
        mask = ((x > y).view(-1) & (dist == 2).view(-1))

        # build kernel
        idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1, 6, 1).view(-1, 3)[mask, :]
        idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6, 1, 1).view(-1, 3)[mask, :]
        mshift1 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift1.view(-1)[torch.arange(12) * 27 + idx_shift1[:, 0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
        mshift2 = torch.zeros(12, 1, 3, 3, 3).to(device)
        mshift2.view(-1)[torch.arange(12) * 27 + idx_shift2[:, 0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
        
        self.mshift1 = mshift1
        self.mshift2 = mshift2
        self.rpad1 = nn.ReplicationPad3d(dilation)
        self.rpad2 = nn.ReplicationPad3d(radius)

        self.win = win
        self.dilation = dilation
        self.radius = radius
        self.kernel_size = kernel_size

        self.use_weight = use_weight

    def pdist_squared(self, x):
        xx = (x ** 2).sum(dim=1).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x.permute(0, 2, 1), x)
        dist[dist != dist] = 0
        dist = torch.clamp(dist, 0.0, np.inf)
        return dist

    def MINDSSC(self, img):
        # compute patch-ssd
        ssd = F.avg_pool3d(self.rpad2(
            (F.conv3d(self.rpad1(img), self.mshift1, dilation=self.dilation) 
             - F.conv3d(self.rpad1(img), self.mshift2, dilation=self.dilation)) ** 2),
                           self.kernel_size, stride=1)

        # MIND equation
        mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
        mind_var = torch.mean(mind, 1, keepdim=True)
        mind_var = torch.clamp(mind_var, (mind_var.mean() * 0.001).item(), (mind_var.mean() * 1000).item())
        mind = mind / mind_var
        mind = torch.exp(-mind)

        # permute to have same ordering as C++ code
        mind = mind[:, torch.Tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3]).long(), :, :, :]

        return mind

    def __call__(self, y_true, y_pred, weight):
        describe_pred = self.MINDSSC(y_pred)
        describe_true = self.MINDSSC(y_true)
        describe_error = (describe_pred - describe_true) ** 2

        if self.use_weight:
            if weight is not None:
                weight = weight.to(describe_error.device)
                describe_error_weighted = describe_error * weight
            else:
                # raise error
                raise ValueError("weight is not set")
        else:
            describe_error_weighted = describe_error

        return torch.mean(describe_error_weighted)


class boundary_loss(StructureLoss):
    def __init__(self, inshape):
        super(boundary_loss, self).__init__()
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))

    def __call__(self, y_true, y_pred, distance):

        if distance is not None:
            distance = distance.to(y_pred.device)
            multiplied = (distance * y_pred).sum(dim=self.vol_axes)
        else:
            # rasie error
            raise ValueError("distance map is not set")
 
        loss = torch.mean(multiplied)
 
        return loss
    
class dice_boundary_comb(StructureLoss):
    def __init__(self, inshape, epoch_start=0, total_epoch=3000):
        super(dice_boundary_comb, self).__init__()
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))
        self.epoch = epoch_start
        self.total_epoch = total_epoch

    def update_balance(self):
        self.epoch += 1
        if(self.epoch > self.total_epoch):
            self.epoch = self.total_epoch
        return self.epoch / self.total_epoch



    def __call__(self,  y_true, y_pred, distance):
        balance = self.update_balance()

        if balance != 1 :
            y_true = y_true.to(y_pred.device)
            top = 2 * (y_true * y_pred).sum(dim=self.vol_axes)
            bottom = torch.clamp((y_true + y_pred).sum(dim=self.vol_axes), min=1e-5)
            dice = 1 - torch.mean(top / bottom)
        else:
            dice = 0

        if balance != 0 :
            if distance is None:
                raise ValueError("distance map is not set")

            distance = distance.to(y_pred.device)
            multiplied = (distance * y_pred).sum(dim=self.vol_axes)
            multiplied = torch.mean(multiplied)
        else:
            multiplied = 0
 
        loss = (1 - balance) * dice + balance * multiplied
 
        return loss
    
class liver_vessel_comb(StructureLoss):
    def __init__(self, inshape, 
                 liver_label_index=0, portalvein_label_index=1, venacava_label_index=2, 
                 liver_boundary_weight=0.5, vessel_dice_weight=0.5, vessel_unoverlap_weight=0.5,
                 epoch_start=0, total_epoch=3000):
        super(liver_vessel_comb, self).__init__()
        ndims = len(inshape)
        self.vol_axes = list(range(2, ndims + 2))
        self.epoch = epoch_start
        self.total_epoch = total_epoch
        self.liver_label_index = liver_label_index
        self.portalvein_label_index = portalvein_label_index
        self.vencava_label_index = venacava_label_index
        self.liver_boundary_weight = liver_boundary_weight
        self.vessel_dice_weight = vessel_dice_weight
        self.vessel_unoverlap_weight = vessel_unoverlap_weight

    def update_balance(self):
        self.epoch += 1
        if(self.epoch > self.total_epoch):
            self.epoch = self.total_epoch
        return self.epoch / self.total_epoch

    def __call__(self, y_true, y_pred, distance):

        liver_loss = 0
        vessel_dice_loss = 0
        vessel_unoverlap_loss = 0

        if self.liver_boundary_weight != 0 :
            balance = self.update_balance()
            true_liver_label = y_true[:, (self.liver_label_index,), ...]
            pred_liver_label = y_pred[:, (self.liver_label_index,), ...]

            if balance != 1 :
                true_liver_label = true_liver_label.to(pred_liver_label.device)
                top = 2 * (true_liver_label * pred_liver_label).sum(dim=self.vol_axes)
                bottom = torch.clamp((true_liver_label + pred_liver_label).sum(dim=self.vol_axes), min=1e-5)
                liver_dice = 1 - torch.mean(top / bottom)
            else:
                liver_dice = 0

            if balance != 0 :
                if distance is None:
                    raise ValueError("distance map is not set")

                distance = distance.to(pred_liver_label.device)
                liver_boundary = (distance * pred_liver_label)
                liver_boundary = torch.mean(liver_boundary)
            else:
                liver_boundary = 0

            liver_loss = (1 - balance) * liver_dice + balance * liver_boundary

        if self.vessel_dice_weight != 0 :
            true_vessel_label = y_true[:, [self.portalvein_label_index, self.vencava_label_index], ...]
            pred_vessel_label = y_pred[:, [self.portalvein_label_index, self.vencava_label_index], ...]

            true_vessel_label = true_vessel_label.to(pred_vessel_label.device)
            top = 2 * (true_vessel_label * pred_vessel_label).sum(dim=self.vol_axes)
            bottom = torch.clamp((true_vessel_label + pred_vessel_label).sum(dim=self.vol_axes), min=1e-5)
            
            vessel_dice_loss = 1 - torch.mean(top / bottom)

        if self.vessel_unoverlap_weight != 0 :
            pred_portalvein_label = y_pred[:, (self.portalvein_label_index,), ...]
            pred_venacava_label = y_pred[:, (self.vencava_label_index,), ...]

            top = 2 * (pred_portalvein_label * pred_venacava_label).sum(dim=self.vol_axes)
            bottom = torch.clamp((pred_portalvein_label + pred_venacava_label).sum(dim=self.vol_axes), min=1e-5)
            
            vessel_unoverlap_loss = torch.mean(top / bottom)

            
        loss = self.liver_boundary_weight * liver_loss + \
               self.vessel_dice_weight * vessel_dice_loss + \
               self.vessel_unoverlap_weight * vessel_unoverlap_loss
            
        return loss


class MutualInformation(PixelBasedLoss):
    """
    Mutual Information
    """

    def __init__(self, device, use_weight=False, sigma_ratio=1, minval=0., maxval=1., num_bin=32):
        super(MutualInformation, self).__init__()

        """Create bin centers"""
        bin_centers = np.linspace(minval, maxval, num=num_bin)
        vol_bin_centers = Variable(torch.linspace(minval, maxval, num_bin), requires_grad=False)
        num_bins = len(bin_centers)

        """Sigma for Gaussian approx."""
        sigma = np.mean(np.diff(bin_centers)) * sigma_ratio

        self.preterm = 1 / (2 * sigma ** 2)
        self.bin_centers = bin_centers
        self.max_clip = maxval
        self.num_bins = num_bins

        """Reshape bin centers"""
        self.vbc = torch.reshape(vol_bin_centers, (1, 1, -1)).to(device)

        self.use_weight = use_weight

    def mi(self, y_true, y_pred):
        y_pred = torch.clamp(y_pred, 0., self.max_clip)
        y_true = torch.clamp(y_true, 0, self.max_clip)

        y_true = y_true.view(y_true.shape[0], -1)
        y_true = torch.unsqueeze(y_true, 2)
        y_pred = y_pred.view(y_pred.shape[0], -1)
        y_pred = torch.unsqueeze(y_pred, 2)

        nb_voxels = y_pred.shape[1]  # total num of voxels

        """compute image terms by approx. Gaussian dist."""
        I_a = torch.exp(- self.preterm * torch.square(y_true - self.vbc))
        I_a = I_a / torch.sum(I_a, dim=-1, keepdim=True)

        I_b = torch.exp(- self.preterm * torch.square(y_pred - self.vbc))
        I_b = I_b / torch.sum(I_b, dim=-1, keepdim=True)

        # compute probabilities
        pab = torch.bmm(I_a.permute(0, 2, 1), I_b)
        pab = pab / nb_voxels
        pa = torch.mean(I_a, dim=1, keepdim=True)
        pb = torch.mean(I_b, dim=1, keepdim=True)

        papb = torch.bmm(pa.permute(0, 2, 1), pb) + 1e-6
        mi = torch.sum(torch.sum(pab * torch.log(pab / papb + 1e-6), dim=1), dim=1)
        return mi.mean()  # average across batch

    def __call__(self, y_true, y_pred, weight):
        mi = self.mi(y_true, y_pred)

        if self.use_weight:
            if weight is not None:
                weight = weight.to(mi.device)
                mi_weighted = mi * weight
            else:
                # raise error
                raise ValueError("weight is not set")
        else:
            mi_weighted = mi

        return -mi_weighted
    
class MINE(nn.Module):
    def __init__(self, hidden_input = 30, hidden_output = 30, shuffle_mode = "global"):
        super().__init__()

        self.T = nn.Sequential(

            nn.Conv3d(2, hidden_input, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_input, hidden_output, 1, 1, 0),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(hidden_output, 1, 1, 1, 0),
        )
        self.shuffle_mode = shuffle_mode

    def forward(self, y_true, y_pred, weight=None,  mask=None):
        if self.shuffle_mode == "global":
            batch_size = y_true.shape[0]
            idx = torch.randperm(y_true[0].nelement())
            target_shuffle = y_true.view(batch_size, -1)
            target_shuffle = target_shuffle[:, idx]
            target_shuffle = target_shuffle.view(y_true.size())
        elif self.shuffle_mode == "mask":
            B, C, D, H, W = y_true.size()
            assert((B==1) and (C == 1))

            y_true = y_true.flatten()
            y_pred = y_pred.flatten()
            mask = y_true if mask is None else mask.flatten()
            y_true = y_true[mask != 0]
            y_pred = y_pred[mask != 0]

            idx = torch.randperm(y_true.nelement())
            target_shuffle = y_true[idx]

            y_true = y_true.unsqueeze(0).unsqueeze(1)
            y_pred = y_pred.unsqueeze(0).unsqueeze(1)
            target_shuffle = target_shuffle.unsqueeze(0).unsqueeze(1)
        else:
            raise ValueError()


        t = self.T(torch.cat((y_pred, y_true), dim=1))
        t_marg = self.T(torch.cat((y_pred, target_shuffle), dim=1))

        second_term = torch.exp(t_marg - 1)
            
        mi_result = t - second_term

        if mi_result.ndim == 5:
            mi_result = mi_result.mean((1,2,3,4))
        else:
            mi_result = mi_result.mean((1,2,3))

        return mi_result
    
    def loss(self, y_true, y_pred, weight=None):
        mi = self(y_true, y_pred)

        return mi.mean() * (-1)

    # def weights_init(self):
    #     for m in self.modules():
    #         classname = m.__class__.__name__
    #         if classname.find('Conv') != -1:
    #             if not m.weight is None:
    #                 nn.init.xavier_normal_(m.weight.data)
    #             if not m.bias is None:
    #                 m.bias.data.zero_()


