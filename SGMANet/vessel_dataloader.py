import SimpleITK as sitk
import os
import sys
import collections
import errno

import torch
from torch.utils.data import Dataset

import json
import numpy

import edt

def read_image_segmentation_list(yaml_files, data_type="train",  n_samples=None):    
    if not os.path.exists(yaml_files):
       raise ValueError(yaml_files + ' not exist!')
    
    print(f"Loading data from {yaml_files}")
        
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

    if isinstance(config[list_key], list):
        fixed_list = config[list_key]
        moving_list = config[list_key]
    elif isinstance(config[list_key], dict):
        fixed_list = config[list_key][fixed_flag]
        moving_list = config[list_key][moving_flag]
    else:
        raise TypeError("config[list_key] should be either a list or a dictionary")
             
    return data_dir, fixed_list, moving_list

def load_image(path, path_ref = None, path_mat_dir = None, is_mask=False):
    if not os.path.exists(path):
        raise ValueError(path + ' not exist!')
    
    image = sitk.ReadImage(path)
    image = sitk.GetArrayFromImage(image)     

    return image


def label2distance(seg):
    #input: c*z*y*x
    if type(seg) == torch.Tensor:
        seg= seg.numpy()

    posmask = seg> 0
    negmask = ~posmask

    dis_background = edt.edt(negmask)
    dis_foreground = edt.edt(posmask) - 1
        
    dist = dis_background * negmask - dis_foreground * posmask

    return dist

def distance2weight(dist, gama=2):
    if type(dist) == torch.Tensor:
        dist= dist.numpy()

    weight = numpy.exp(-(dist**2)/(2*gama)) + 1

    return weight




def collate(list):
    fix = [x[1] for x in list]
    moving = [x[0] for x in list]

    fix_image = [torch.tensor(x[0]) for x in fix]
    fix_label = [torch.tensor(x[1]) for x in fix]
    
    
    fix_distance = [torch.tensor(x[3]).unsqueeze(0) for x in fix if x[3] is not None]
    fix_weight = [torch.tensor(x[4]).unsqueeze(0) for x in fix if x[4] is not None]

    if len(fix_distance) == 0:
        fix_distance = None

    if len(fix_weight) == 0:
        fix_weight = None
        
    moving_image = [torch.tensor(x[0]) for x in moving]
    moving_label = [torch.tensor(x[1]) for x in moving]
  

    fix_image = torch.stack(fix_image, 0)
    fix_label = torch.stack(fix_label, 0)
    fix_distance = torch.stack(fix_distance, 0) if fix_distance is not None else None
    fix_weight = torch.stack(fix_weight, 0) if fix_weight is not None else None

    moving_image = torch.stack(moving_image, 0)
    moving_label = torch.stack(moving_label, 0)


    return moving_image, fix_image, moving_label, fix_label, fix_weight, fix_distance


class vessel_dataset(Dataset):
    """
    a dataset class to load medical image data in Nifti format using simpleITK
    """

    def __init__(self, yaml_files, intra_patient=False, type="train", segma=2,
                 use_distance=False, use_weight=False):

        self.type = type
        self.use_distance = use_distance
        self.use_weight = use_weight
        self.segma = segma

        self.data_dir, self.fixed_list, self.moving_list  = read_image_segmentation_list(yaml_files,
                                                           self.type,
                                                           None)

        self.length_fixed = len(self.fixed_list)
        self.length_moving = len(self.moving_list)

        self.intra_patient = intra_patient



        if self.fixed_list == self.moving_list:
            self.total_num = self.length_fixed * (self.length_moving - 1)
        else:
            if self.intra_patient:
                assert(self.length_fixed == self.length_moving)
                self.total_num = self.length_fixed
            else:
                self.total_num = self.length_fixed * self.length_moving

    def __len__(self):
        return self.total_num

    def __getitem__(self, id):
        """
        Retrieves a pair of fixed and moving images along with their corresponding labels and metadata.
        Args:
            id (int): The index of the data item to retrieve.
        Returns:
            tuple: A tuple containing two lists:
                - The first list contains the moving image, moving label, moving label one-hot encoded, moving label distance map, and metadata.
                - The second list contains the fixed image, fixed label, fixed label one-hot encoded, fixed label distance map, fixed label weight, and metadata.
        The function performs the following steps:
            1. Adjusts the index if shuffling is enabled.
            2. Determines the indices for fixed and moving images based on the provided index.
            3. Loads the fixed and moving images and their corresponding labels.
            4. Converts the labels to one-hot encoding.
            5. Optionally loads additional labels (portalvein, venacava, sato) if available.
            6. Computes the distance map and weight for the fixed label if required.
            7. Applies random affine data enhancement to the moving image and label if enabled.
            8. Returns the processed images, labels, and metadata.
        """

        if self.fixed_list == self.moving_list:
            fixed_ind = id//(self.length_moving - 1)
            moving_ind = id%(self.length_moving - 1)
            if moving_ind >= fixed_ind:
                moving_ind += 1
        else:
            if self.intra_patient:
                fixed_ind = id
                moving_ind = id
            else:
                fixed_ind = id // (self.length_moving)
                moving_ind = id % (self.length_moving)

        

        fixed_image_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["image"])
        fixed_image = load_image(fixed_image_path).astype(numpy.float32)

        moving_image_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["image"])
        moving_image = load_image(moving_image_path).astype(numpy.float32)

        fixed_label_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["label"])
        fixed_label = load_image(fixed_label_path).astype(numpy.uint8)

        moving_label_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["label"])
        moving_label = load_image(moving_label_path).astype(numpy.uint8)


        fixed_label_portalvein_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["portalvein"])
        fixed_label_portalvein = load_image(fixed_label_portalvein_path).astype(numpy.uint8)

        fixed_label_venacava_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["venacava"])
        fixed_label_venacava = load_image(fixed_label_venacava_path).astype(numpy.uint8)

        fixed_image_sato_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["sato"])
        fixed_image_sato = load_image(fixed_image_sato_path).astype(numpy.float32)

        moving_label_portalvein_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["portalvein"])
        moving_label_portalvein = load_image(moving_label_portalvein_path).astype(numpy.uint8)


        moving_label_venacava_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["venacava"])
        moving_label_venacava = load_image(moving_label_venacava_path).astype(numpy.uint8)

        moving_image_sato_path = os.path.join(self.data_dir, self.moving_list[moving_ind]["sato"])
        moving_image_sato = load_image(moving_image_sato_path).astype(numpy.float32)

        # # Combine fixed_label_portalvein and fixed_label_venacava into one label
        # fixed_label_vessel = numpy.logical_or(fixed_label_portalvein, fixed_label_venacava)


        if self.use_weight:
            fixed_label_vessel_distance_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["vessel_contour_distance"])
            fixed_label_vessel_distance_map = load_image(fixed_label_vessel_distance_path).astype(numpy.float32)
            fixed_label_vessel_weight = distance2weight(fixed_label_vessel_distance_map, self.segma)
        else:
            fixed_label_vessel_weight = None

        if self.use_distance:
            fixed_label_distance_path = os.path.join(self.data_dir, self.fixed_list[fixed_ind]["distance"])
            fixed_label_distance_map = load_image(fixed_label_distance_path).astype(numpy.float32)
        else:
            fixed_label_distance_map = None

        fixed_image_with_sato = numpy.stack((fixed_image, fixed_image_sato), axis=0)
        moving_image_with_sato = numpy.stack((moving_image, moving_image_sato), axis=0)

        fixed_label_combined = numpy.stack((fixed_label, fixed_label_portalvein, fixed_label_venacava), axis=0)
        moving_label_combined = numpy.stack((moving_label, moving_label_portalvein, moving_label_venacava), axis=0) 

        path_dict_moving = {
            "image": moving_image_path,
            "label": moving_label_path,
            "portalvein": moving_label_portalvein_path,
            "venacava": moving_label_venacava_path,
        }

        path_dict_fixed = {
            "image": fixed_image_path,
            "label": fixed_label_path,
            "portalvein": fixed_label_portalvein_path,
            "venacava": fixed_label_venacava_path,
        }

        # in order image, seg, name
        return [moving_image_with_sato, moving_label_combined, moving_label_combined, None,                     None,                      path_dict_moving], \
               [fixed_image_with_sato,  fixed_label_combined, fixed_label_combined, fixed_label_distance_map, fixed_label_vessel_weight, path_dict_fixed]
