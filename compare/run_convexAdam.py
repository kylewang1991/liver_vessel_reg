from convexAdam.convex_adam_MIND import convex_adam_pt

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../SGMANet'))
import layers
import torch
import time
import numpy as np

class convexadam_reg():
    def __init__(self, config) -> None:
       self.transform = layers.SpatialTransformerDouble(config["dataset_param"]["image_shape"])
       self.transform_single = layers.SpatialTransformer(config["dataset_param"]["image_shape"], mode='bilinear')
    
    def registration(self, moving, fix):
        moving_path = moving[-1]

        # Check if the moving_path contains 'portalvein' or 'venacava'
        has_portalvein = 'portalvein' in moving_path
        has_venacava = 'venacava' in moving_path

        # check if the moving[0] is a tensor and convert it to numpy array if needed    
        if isinstance(moving[0], torch.Tensor):
            moving_image_numpy = moving[0].squeeze().numpy()
        else:
            moving_image_numpy = moving[0].squeeze()

        # check if the fix[0] is a tensor and convert it to numpy array if needed
        if isinstance(fix[0], torch.Tensor):
            fix_image_numpy = fix[0].squeeze().numpy()
        else:
            fix_image_numpy = fix[0].squeeze()

        # Check if the images are 4D, and extract the first slice if true
        if moving_image_numpy.ndim == 4:
            moving_image_numpy = moving_image_numpy[0]
        if fix_image_numpy.ndim == 4:
            fix_image_numpy = fix_image_numpy[0]

        start_time = time.perf_counter()

        displacements = convex_adam_pt(
            img_fixed=fix_image_numpy,
            img_moving=moving_image_numpy,
        )

        disp = torch.from_numpy(displacements).float()
        disp = disp.permute(3, 0, 1, 2).unsqueeze(0)

        # Check if moving_image_numpy and moving[2] are numpy arrays and convert them to tensors if needed
        moving_image = torch.from_numpy(moving_image_numpy).unsqueeze(0).unsqueeze(0)

        if isinstance(moving[2], np.ndarray):
            moving_label_onehot = torch.from_numpy(moving[2]).unsqueeze(0)
        else:
            moving_label_onehot = moving[2].unsqueeze(0)

        moved_image, moved_label = self.transform(moving_image, moving_label_onehot, disp, need_moved_label=True)

        # Check if moved_label has enough channels before extracting portalvein and venacava
        moved_portalvein = None
        moved_venacava = None
        if has_portalvein:
            moving_portalvein = torch.tensor(moving[2][(1,), ...]).unsqueeze(0).float()
            #print(moving_portalvein.shape)
            moved_portalvein = self.transform_single(moving_portalvein, disp)
        if has_venacava:
            moving_venacava = torch.tensor(moving[2][(2,), ...]).unsqueeze(0).float()
            #print(moving_venacava.shape)
            moved_venacava = self.transform_single(moving_venacava, disp)
            
        if has_venacava or has_portalvein:
            moved_label = moved_label[:, (0,), ...]
            # cast moved_label to uint8
            moved_label = moved_label.type(torch.uint8)
            # print(moved_label.dtype)

        end_time = time.perf_counter()

        result = {}
        result["moved_image"] = moved_image
        result["moved_label"] = moved_label
        result["moved_portalvein"] = moved_portalvein
        result["moved_venacava"] = moved_venacava
        result["affine_matrix"] = None
        result["svf"] = disp
        result["deformation"] = self.transform.grid + disp
        result["run_time"] = end_time - start_time
        return result