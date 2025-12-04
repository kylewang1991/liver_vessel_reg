import SimpleITK as sitk
import os
import sys
import collections
import errno

import torch
from torch.utils.data import Dataset

import json
import numpy
import torch.nn.functional as F

from scipy.ndimage import distance_transform_edt as distance
from scipy.spatial.transform import Rotation as R

import ants

import matplotlib.pyplot as plt
import edt

import visual
import torch.nn.functional as nnf

def read_image_segmentation_list(yaml_files, data_type="train",  n_samples=None):    
    if not os.path.exists(yaml_files):
       raise ValueError(yaml_files + ' not exist!')
        
    with open(yaml_files) as f:
        config = json.load(f)

    data_dir = os.path.join(os.path.dirname(__file__), "../", config["root_dir"])

    fixed_flag = str(config["registration_direction"]["fixed"])
    moving_flag = str(config["registration_direction"]["moving"])

    if data_type == "train":
        num_key = "numTraining"
        list_key = "training"
    elif data_type == "valid":
        num_key = "numValid"
        list_key = "valid"
    elif data_type == "test": 
        num_key = "numTest" 
        list_key = "test"       

    num_fixed_image  = config[num_key][fixed_flag]
    num_moving_image  = config[num_key][moving_flag]

    fixed_list = config[list_key][fixed_flag]
    moving_list = config[list_key][moving_flag]

    if isinstance(n_samples, collections.abc.Sequence):
        if max(n_samples) > min(num_fixed_image, num_moving_image) : 
            raise ValueError("n_smaple is too big!")
        fixed_list = [fixed_list[i] for i in n_samples]
        moving_list = [moving_list[i] for i in n_samples]
    elif type(n_samples) is int:
        if n_samples > min(num_fixed_image, num_moving_image) : 
            raise ValueError("n_smaple is too big!")
        fixed_list = fixed_list[0 : n_samples]
        moving_list = moving_list[0 : n_samples]
    elif n_samples is not None:
        raise TypeError(
            "n_samples should be None, or int, or a sequence of int bug get {}".format(type(n_samples)))
             
    return data_dir, fixed_list, moving_list

def load_image(path, path_ref = None, path_mat_dir = None, is_mask=False):
    if not os.path.exists(path):
        raise ValueError(path + ' not exist!')
    
    if path_mat_dir:
        assert os.path.exists(path_ref)
        name_fixed = os.path.basename(path_ref).split('.')[0]
        name_moving = os.path.basename(path).split('.')[0]

        name_fixed_split = name_fixed.split('_')
        if "reduce" in name_fixed_split:
            name_fixed_split.remove("reduce")
            name_fixed = '_'.join(name_fixed_split)

        name_moving_split = name_moving.split('_')
        if "reduce" in name_moving_split:
            name_moving_split.remove("reduce")
            name_moving = '_'.join(name_moving_split)       


        name_mat = name_fixed + '_' + name_moving + ".mat"
        path_mat = os.path.join(path_mat_dir, name_mat)
        
        if not os.path.exists(path_mat):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path_mat)
    
    image = ants.image_read(path)

    if path_mat_dir:
        head,tail = os.path.split(path_mat_dir)
        if tail in ["affine_mi", "affine_mine", "affine_struct"]:
            image = image.numpy().transpose((2,1,0))
            image = torch.from_numpy(image)
            image = image.unsqueeze(0).unsqueeze(0).float()

            mat = torch.load(path_mat)
            grid = nnf.affine_grid(mat, [image.shape[0], 3, image.shape[2], image.shape[3], image.shape[4]], align_corners=True)
            itp = 'nearest' if is_mask else 'bilinear'
            image = nnf.grid_sample(image, grid, align_corners=True, mode=itp)

            image = image.squeeze()

        else:
            image_fixed = ants.image_read(path_ref)
            itp = 'genericLabel' if is_mask else 'linear'
            image = ants.apply_transforms(image_fixed, image, 
                                            transformlist= [path_mat],
                                            interpolator = itp)
            
            image = image.numpy().transpose((2,1,0))
            image = torch.from_numpy(image)
    else:
        image = image.numpy().transpose((2,1,0))
        image = torch.from_numpy(image)       

    return image

def onehot_encode(seg):
    # input: z * y * x
    seg_one_hot = F.one_hot(seg.long(), num_classes=5).to(torch.uint8)
    seg_one_hot = seg_one_hot.permute(3, 0, 1, 2)[1:, :, :, :]
    return seg_one_hot   


# def onehot2dist(seg_onehot):
#     #input: c*z*y*x
#     if type(seg_onehot) == torch.Tensor:
#         seg_onehot= seg_onehot.numpy()

#     res_list = []
    
#     for channel in seg_onehot[:]:
#         res = numpy.zeros_like(channel)

#         posmask = channel > 0
#         negmask = ~posmask

#         dis_background = distance(negmask)
#         dis_foreground = distance(posmask) - 1
#         dis_background_norm = (dis_background - dis_background.min()) / (dis_background.max() - dis_background.min())
#         dis_foreground_norm = (dis_foreground - dis_foreground.min()) / (dis_foreground.max() - dis_foreground.min())
        
#         res = dis_background_norm * negmask - dis_foreground_norm * posmask

#         res_list.append(res)

#     dist = numpy.stack(res_list)

#     return dist 

# def onehot2dist_edt(seg_onehot):
#     #input: c*z*y*x
#     if type(seg_onehot) == torch.Tensor:
#         seg_onehot= seg_onehot.numpy()

#     res_list = []
    
#     for channel in seg_onehot[:]:
#         res = numpy.zeros_like(channel)

#         posmask = channel > 0
#         negmask = ~posmask

#         dis_background = edt.edt(negmask)
#         dis_foreground = edt.edt(posmask) - 1
#         dis_background_norm = (dis_background - dis_background.min()) / (dis_background.max() - dis_background.min())
#         dis_foreground_norm = (dis_foreground - dis_foreground.min()) / (dis_foreground.max() - dis_foreground.min())
        
#         res = dis_background_norm * negmask - dis_foreground_norm * posmask

#         res_list.append(res)

#     dist = numpy.stack(res_list)

#     return dist    

# def onehot2weight(seg_onehot, roi_index:list, limit_ratio):

#     if type(seg_onehot) == torch.Tensor:
#         seg_onehot= seg_onehot.numpy()
  
#     seg_onehot_new = seg_onehot[roi_index]
#     seg_onehot_new = seg_onehot_new.sum(axis=0)
#     foreground_mask = seg_onehot_new > 0
#     background_mask = ~foreground_mask

#     dis = edt.edt(background_mask)

#     dis = 1 - (dis - dis.min()) / (dis.max() - dis.min())
#     dis = dis.clip(limit_ratio, 1)

#     return dis


# def onehot2weight(seg_onehot, roi_index:list, distance_outer=20, distance_inner = 3, gama=2):
#     #input: c*z*y*x
#     if type(seg_onehot) == torch.Tensor:
#         seg_onehot= seg_onehot.numpy()

#     seg_onehot_new = seg_onehot[roi_index]
#     seg_onehot_new = seg_onehot_new.sum(axis=0)

#     posmask = seg_onehot_new > 0
#     negmask = ~posmask

#     dis_background = edt.edt(negmask)
#     dis_foreground = edt.edt(posmask) - 1

#     dis_background = dis_background.clip(max=distance_outer)
#     dis_foreground = dis_foreground.clip(max=distance_inner)

#     dis_background_norm = (dis_background - dis_background.min()) / (distance_outer - dis_background.min())
#     dis_foreground_norm = (dis_foreground - dis_foreground.min()) / (distance_inner - dis_foreground.min())
        
#     dist = dis_background_norm * negmask - dis_foreground_norm * posmask

#     # outline = (dist > -0.04) & (dist < 0.04)

#     dist = numpy.exp(-(dist*dist)/(2*gama)) + 1

#     return torch.from_numpy(dist)

def label2distance(seg):
    #input: c*z*y*x
    if type(seg) == torch.Tensor:
        seg= seg.numpy()

    posmask = seg> 0
    negmask = ~posmask

    dis_background = edt.edt(negmask)
    dis_foreground = edt.edt(posmask) - 1
        
    dist = dis_background * negmask - dis_foreground * posmask

    return torch.from_numpy(dist)

def distance2weight(dist, gama=2):
    weight = torch.exp(-(dist**2)/(2*gama)) + 1

    return weight

# def comput_fig(img, min=0, max=1):
#     if len(img.shape) == 4:
#         img = img[0]

#     img = img[[10, 20, 30, 40, 50, 60]]
#     fig = plt.figure(figsize=(12,12))
#     for i in range(img.shape[0]):
#         plt.subplot(2, 3, i + 1)
#         plt.axis('off')
#         plt.imshow(img[i, :, :], cmap='gray', vmin=min, vmax=max)
#     fig.subplots_adjust(wspace=0, hspace=0)
#     return fig




def collate(list):
    fix = [x[1] for x in list]
    moving = [x[0] for x in list]

    fix_image = [x[0] for x in fix]
    fix_label_onehot = [x[2] for x in fix]
    fix_label_weight = [x[4] for x in fix if x[4] is not None]


    moving_image = [x[0] for x in moving]
    moving_label_onehot = [x[2] for x in moving]
  

    fix_image = torch.stack(fix_image, 0).unsqueeze(0)
    fix_label_one_hot = torch.stack(fix_label_onehot, 0)
    fix_label_weight = torch.stack(fix_label_weight, 0).unsqueeze(0) if len(fix_label_weight) > 0 else None

    moving_image = torch.stack(moving_image, 0).unsqueeze(0)
    moving_label_one_hot = torch.stack(moving_label_onehot, 0)


    return moving_image, fix_image, moving_label_one_hot, fix_label_one_hot, fix_label_weight, None

def compose_linear_transform(angle, scale, translation):
    # Create rotation object
    rotation_matrix = numpy.eye(4)
    rotation_matrix[:3, :3] = R.from_euler('xyz', angle, degrees=True).as_matrix()

    # Create scaling matrix
    scaling_matrix = numpy.diag(numpy.append(scale, 1))
        
    # Create translation matrix
    translation_matrix = numpy.eye(4)
    translation_matrix[:3, 3] = translation
        
    # Compose the transformation matrix
    transformation_matrix = translation_matrix @ rotation_matrix @ scaling_matrix
        
    return transformation_matrix

def random_linear_transform(image, label, label_onehot, transform_type):

    image = image.unsqueeze(0).unsqueeze(0)
    label = label.unsqueeze(0).unsqueeze(0)
    label_onehot = label_onehot.unsqueeze(0)

    # angle = numpy.random.uniform(-5, 5, size=3)
    # scale  = numpy.random.uniform(0.9, 1.1, size=3)
    # translation = numpy.random.uniform(-0.05, 0.05, size=3)

    # angle = numpy.random.uniform(-2.5, 2.5, size=3)
    # scale  = numpy.random.uniform(0.975, 1.025, size=3)
    # translation = numpy.random.uniform(-0.025, 0.025, size=3)

    # random = numpy.random.normal(0, 0.24, size=9).clip(-1, 1)
    # angle = random[0:3] * 10
    # scale  = 1 + random[3:6] * 0.1
    # translation = random[6:9] * 0.1

    if transform_type == "Rigid":    
        angle = numpy.random.uniform(-15, 15, size=3)
        scale = numpy.ones(3)
        translation = numpy.random.uniform(-0.3, 0.3, size=3)
    elif transform_type == "Similarity":
        angle = numpy.random.uniform(-10, 10, size=3)
        scale  = numpy.random.uniform(0.9, 1.1, size=3)
        translation = numpy.random.uniform(-0.1, 0.1, size=3)
    elif transform_type == "Nonlinear":
        random = numpy.random.normal(0, 0.24, size=9).clip(-1, 1)
        angle = random[0:3] * 10
        scale  = 1 + random[3:6] * 0.1
        translation = random[6:9] * 0.1
    else:
        raise ValueError("Not supported random transform type")
    
    mat = compose_linear_transform(angle, scale, translation)
    mat = torch.from_numpy(mat).unsqueeze(0)
    mat = mat[:, :3, :]

    grid = F.affine_grid(mat, [1, 3, image.shape[2], image.shape[3], image.shape[4]], align_corners=True)
    
    transformed_image = F.grid_sample(image.float(), grid.float(), align_corners=True, mode='bilinear')
    transformed_label = F.grid_sample(label.float(), grid.float(), align_corners=True, mode='nearest')
    transformed_label_onehot = F.grid_sample(label_onehot.float(), grid.float(), align_corners=True, mode='nearest')

    return transformed_image.squeeze().to(torch.float), transformed_label.squeeze().to(torch.uint8), transformed_label_onehot.squeeze().to(torch.uint8)

class RegDataSet(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, yaml_files, intra_patient=False, affine_mat=None, type="train", 
                 n_samples=None, shuffle=False, 
                 use_distance=False, use_weight=False, 
                 roi_index=[0], distance_outer=10, distance_inner = 30, segma=2, 
                 do_random_affine_data_enhancement=False, random_type = "Rigid"):

        self.n_samples = n_samples
        self.shuffle = shuffle
        self.type = type
        self.intra_patient = intra_patient
        self.affine_mat = affine_mat
        self.roi_index = roi_index
        self.use_distance = use_distance
        self.use_weight = use_weight
        self.distance_outer = distance_outer
        self.distance_inner = distance_inner
        self.segma = segma
        self.do_random_affine_data_enhancement = do_random_affine_data_enhancement
        self.random_type = random_type

        self.data_dir, self.fixed_list, self.moving_list  = read_image_segmentation_list(yaml_files,
                                                           self.type,
                                                           self.n_samples)

        self.length_fixed = len(self.fixed_list)
        self.length_moving = len(self.moving_list)

        if self.intra_patient:
            assert(self.length_fixed == self.length_moving)
            self.total_num = self.length_fixed
        else:
            self.total_num = self.length_fixed * self.length_moving

        if self.shuffle:
            self.shuffle_id = torch.randperm(self.total_num)

    def __len__(self):
        return self.total_num

    def __getitem__(self, id):
        if self.shuffle:
            id = self.shuffle_id[id]

        if self.intra_patient:
            fixed_ind = id
            moving_ind = id
        else:
            fixed_ind = id // (self.length_moving)
            moving_ind = id % (self.length_moving)

        

        fixed_image_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["image"])
        fixed_image = load_image(fixed_image_path)

        moving_image_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["image"])
        moving_image = load_image(moving_image_path, fixed_image_path, self.affine_mat) if self.affine_mat else load_image(moving_image_path)

        fixed_label_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["label"])
        fixed_label = load_image(fixed_label_path).to(torch.uint8)

        moving_label_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["label"])
        moving_label = load_image(moving_label_path, fixed_label_path, self.affine_mat, True) if self.affine_mat else load_image(moving_label_path)
        moving_label = moving_label.to(torch.uint8)

        fixed_label_onehot = onehot_encode(fixed_label)

        moving_label_onehot = onehot_encode(moving_label)

        if self.use_weight:
            fixed_label_distance_map = label2distance(fixed_label)
            fixed_label_weight = distance2weight(fixed_label_distance_map, self.segma)
        else:
            fixed_label_distance_map = None
            fixed_label_weight = None

        if self.use_distance:
            fixed_label_distance_map = label2distance(fixed_label)
            moving_label_distance_map = label2distance(moving_label)
        else:
            fixed_label_distance_map = None
            moving_label_distance_map = None

        if self.do_random_affine_data_enhancement and self.type=="train":
            moving_image, moving_label, moving_label_onehot = random_linear_transform(moving_image, moving_label, moving_label_onehot, self.random_type)


        # if moving_label_onehot.shape[0] != 4 or fixed_label_onehot.shape[0] != 4:
        #     visual.save_image(fixed_image.unsqueeze(0).unsqueeze(0), "./save_fix_image.nii.gz",  fixed_image_path)
        #     visual.save_image(fixed_label.unsqueeze(0).unsqueeze(0), "./save_fix_label.nii.gz",  fixed_image_path)            
        #     visual.save_image(moving_image.unsqueeze(0).unsqueeze(0), "./save_moving_image.nii.gz",  fixed_image_path)
        #     visual.save_image(moving_label.unsqueeze(0).unsqueeze(0), "./save_moving_label.nii.gz",  fixed_image_path)
        #     assert False



        # in order image, seg, name
        return [moving_image, moving_label, moving_label_onehot, moving_label_distance_map,         None,    {"image" : moving_image_path, "label" : moving_label_path}], \
               [fixed_image, fixed_label, fixed_label_onehot, fixed_label_distance_map, fixed_label_weight, {"image" : fixed_image_path, "label" : fixed_label_path}]

