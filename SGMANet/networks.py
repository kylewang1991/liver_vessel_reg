import Core_Conv as vxm

import Core_Trans as trans  # nopep8

import importlib.util

# Check if mamba_ssm module is installed
mamba_spec = importlib.util.find_spec("mamba_ssm")
if mamba_spec is not None:
    import Core_Mamba as mamba
    import Core_Vmamba as vmamba

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

import layers


class sgmanet(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 config,
                 inshape,
                 int_steps=7,
                 int_downsize=2,
                 bidir=False):
        """ 
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            nb_unet_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the unet class documentation.
            nb_unet_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            unet_feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_unet_conv_per_level: Number of convolutions per unet level. Default is 1.
            int_steps: Number of flow integration steps. The warp is non-diffeomorphic when this 
                value is 0.
            int_downsize: Integer specifying the flow downsample factor for vector integration. 
                The flow field is not downsampled when this value is 1.
            bidir: Enable bidirectional cost function. Default is False.

            src_feats: Number of source image features. Default is 1.
            trg_feats: Number of target image features. Default is 1.
        """
        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # configure core unet model
        if config["core"] == "conv" :
            self.core_model = vxm.Unet(
                inshape,
                infeats=config["in_chans"],
                nb_features=[config["enc"], config["dec"]],
                nb_conv_per_level=1,
                half_res=False,
            )
        elif config["core"] == "mamba":
            self.core_model = mamba.MambaMorph(config)
        elif config["core"] == "trans":
            self.core_model = trans.TransMorph(config)
        # elif config["core"] == "vmamba":
        #     self.core_model = vmamba.Mamba_VSS(                
        #         inshape,
        #         infeats=config["in_chans"],
        #         nb_features=[config["enc"], config["dec"]],
        #         nb_conv_per_level=1,
        #         half_res=False)
        elif config["core"] == "vmamba":
            self.core_model = vmamba.VmambaNet(inshape, config)
        else:
            raise ValueError("Core U-Net arctecture is not supported!")



        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.core_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        # configure optional resize layers (downsize)
        if int_steps > 0 and int_downsize > 1:
            self.resize = layers.ResizeTransform(int_downsize, ndims)
        else:
            self.resize = None

        # resize to full res
        if int_steps > 0 and int_downsize > 1:
            self.fullsize = layers.ResizeTransform(1 / int_downsize, ndims)
        else:
            self.fullsize = None

        # configure bidirectional training
        self.bidir = bidir

        # configure optional integration layer for diffeomorphic warp
        down_shape = [int(dim / int_downsize) for dim in inshape]
        self.integrate = layers.VecInt(down_shape, int_steps) if int_steps > 0 else None

        # configure transformer
        self.transformer = layers.SpatialTransformerDouble(inshape)

    def forward(self, source, target, source_label, target_label, need_moved_label=True, need_deformation_filed=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = self.core_model(x)

        # transform into flow field
        flow_field = self.flow(x)

        # resize flow for integration
        pos_flow = flow_field
        if self.resize:
            pos_flow = self.resize(pos_flow)

        preint_flow = pos_flow

        # negate flow for bidirectional model
        neg_flow = -pos_flow if self.bidir else None

        # integrate to produce diffeomorphic warp
        if self.integrate:
            pos_flow = self.integrate(pos_flow)
            neg_flow = self.integrate(neg_flow) if self.bidir else None

            # resize to final resolution
            if self.fullsize:
                pos_flow = self.fullsize(pos_flow)
                neg_flow = self.fullsize(neg_flow) if self.bidir else None

        # Move source_label and target_label to the same device as the model
        source_label = source_label.to(next(self.parameters()).device) if need_moved_label else None
        target_label = target_label.to(next(self.parameters()).device) if self.bidir and need_moved_label else None

        # warp image with flow field
        y_source, y_source_label = self.transformer(source[:, (0,), ...], source_label, pos_flow, need_moved_label)
        y_target, y_target_label = self.transformer(target[:, (0,), ...], target_label, neg_flow, need_moved_label) if self.bidir else (None, None)

        return_dict={}
        return_dict["moved_image"] = y_source
        return_dict["preint_flow"] = preint_flow
        return_dict["displacement_filed"] = pos_flow
        if need_moved_label:
            return_dict["moved_label"] = y_source_label
        if need_deformation_filed:
            return_dict["deformation_filed"] = self.transformer.grid + pos_flow

        if self.bidir:
            return_reverse_dict={}
            return_reverse_dict["moved_image"] = y_target
            return_reverse_dict["preint_flow"] = -preint_flow
            return_reverse_dict["displacement_filed"] = neg_flow
            if need_moved_label:
                return_reverse_dict["moved_label"] = y_target_label
            if need_deformation_filed:
                return_reverse_dict["deformation_filed"] = self.transformer.grid + neg_flow

        # return non-integrated flow field if training
        return (return_dict, return_reverse_dict) if self.bidir else return_dict


class sgmanet_affine(nn.Module):
    """
    VoxelMorph network for (unsupervised) nonlinear registration between two images.
    """
    def __init__(self,
                 config,
                 shape,
                 transform_type,
                 bidir=False):

        super().__init__()

        # configure core unet model
        if config["core"] == "trans" :
            self.core_model = trans.TransMorphAffine(config, transform_type)
        elif config["core"] == "mamba":
            self.core_model = mamba.MambaAffine(config, transform_type)
        else:
            raise ValueError("sgmanet_affine only support transmorph archtecture.")

        self.transformer = layers.AffineTransform()

        self.bidir = bidir
        self.shape = shape



    def forward(self, source, target, source_label, target_label, need_moved_label=True, need_deformation_filed=False):
        '''
        Parameters:
            source: Source image tensor.
            target: Target image tensor.
            registration: Return transformed image and flow. Default is False.
        '''

        # concatenate inputs and propagate unet
        x = torch.cat([source, target], dim=1)
        x = x.to(next(self.parameters()).device)
        affine, scale, translate, shear = self.core_model(x)

        theta_x = affine[:, 0]
        theta_y = affine[:, 1]
        theta_z = affine[:, 2]
        scale_x = scale[:, 0]
        scale_y = scale[:, 1]
        scale_z = scale[:, 2]
        trans_x = translate[:, 0]
        trans_y = translate[:, 1]
        trans_z = translate[:, 2]
        shear_xy = shear[:, 0]
        shear_xz = shear[:, 1]
        shear_yx = shear[:, 2]
        shear_yz = shear[:, 3]
        shear_zx = shear[:, 4]
        shear_zy = shear[:, 5]

        rot_mat_x = torch.stack([torch.stack([torch.ones_like(theta_x), torch.zeros_like(theta_x), torch.zeros_like(theta_x)], dim=1), 
                                 torch.stack([torch.zeros_like(theta_x), torch.cos(theta_x), -torch.sin(theta_x)], dim=1), 
                                 torch.stack([torch.zeros_like(theta_x), torch.sin(theta_x), torch.cos(theta_x)], dim=1)], dim=2)
        
        rot_mat_y = torch.stack([torch.stack([torch.cos(theta_y), torch.zeros_like(theta_y), torch.sin(theta_y)], dim=1), 
                                 torch.stack([torch.zeros_like(theta_y), torch.ones_like(theta_x), torch.zeros_like(theta_x)], dim=1), 
                                 torch.stack([-torch.sin(theta_y), torch.zeros_like(theta_y), torch.cos(theta_y)], dim=1)], dim=2)
        
        rot_mat_z = torch.stack([torch.stack([torch.cos(theta_z), -torch.sin(theta_z), torch.zeros_like(theta_y)], dim=1), 
                                 torch.stack([torch.sin(theta_z), torch.cos(theta_z), torch.zeros_like(theta_y)], dim=1), 
                                 torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), torch.ones_like(theta_x)], dim=1)], dim=2)
        
        scale_mat = torch.stack([torch.stack([scale_x, torch.zeros_like(theta_z), torch.zeros_like(theta_y)], dim=1),
                                 torch.stack([torch.zeros_like(theta_z), scale_y, torch.zeros_like(theta_y)], dim=1),
                                 torch.stack([torch.zeros_like(theta_y), torch.zeros_like(theta_y), scale_z], dim=1)], dim=2)
        
        shear_mat = torch.stack([torch.stack([torch.ones_like(theta_x), torch.tan(shear_xy), torch.tan(shear_xz)], dim=1),
                                 torch.stack([torch.tan(shear_yx), torch.ones_like(theta_x), torch.tan(shear_yz)], dim=1),
                                 torch.stack([torch.tan(shear_zx), torch.tan(shear_zy), torch.ones_like(theta_x)], dim=1)], dim=2)
        
        trans = torch.stack([trans_x, trans_y, trans_z], dim=1).unsqueeze(dim=2)
        
        mat = torch.bmm(shear_mat, torch.bmm(scale_mat, torch.bmm(rot_mat_z, torch.matmul(rot_mat_y, rot_mat_x))))
        inv_mat = torch.inverse(mat)
        mat = torch.cat([mat, trans], dim=-1)
        inv_trans = torch.bmm(-inv_mat, trans)
        inv_mat = torch.cat([inv_mat, inv_trans], dim=-1)

        # warp image with flow field
        source_label = source_label.to(next(self.parameters()).device)
        target_label = target_label.to(next(self.parameters()).device)
        y_source, y_source_label = self.transformer(source[:, (0,), ...], source_label, mat, need_moved_label)
        y_target, y_target_label = self.transformer(target[:, (0,), ...], target_label, inv_mat, need_moved_label) if self.bidir else (None, None)

        return_dict={}
        return_dict["moved_image"] = y_source
        return_dict["matrix"] = mat
        if need_moved_label:
            return_dict["moved_label"] = y_source_label

        if self.bidir:
            return_reverse_dict={}
            return_reverse_dict["moved_image"] = y_target
            return_reverse_dict["matrix"] = inv_mat
            if need_moved_label:
                return_reverse_dict["moved_label"] = y_target_label

        # return non-integrated flow field if training
        return (return_dict, return_reverse_dict) if self.bidir else return_dict
