import torch
from torch.utils.data import Dataset
import torch.nn.functional as F

import ants
import numpy
import SimpleITK as sitk
import os
import shutil

def save_image(result, path, template=None):
    if type(result) == ants.core.ants_image.ANTsImage:
        ants.image_write(result, path)
    elif type(result) == sitk.SimpleITK.Image :
        sitk.WriteImage(result, path)
    elif  isinstance(result, torch.Tensor):

        result = result.squeeze().numpy()

        if result.ndim == 4:
            composed_onehot = torch.zeros(result.shape[1:])
            for index,tensor in enumerate(result):
                composed_onehot = composed_onehot + tensor * (index + 1)
            result = composed_onehot


        if result.ndim != 3 :
            raise ValueError(f"Input tensor has {result.ndim} dimensions, expected 3")

        save_image_from_array(result, path, template)
    elif  isinstance(result, numpy.ndarray):
        assert(result.ndim == 3)
        save_image_from_array(result, path, template)
    else:
        raise TypeError('Work with ants image, tensor, numpy array Only')
    

def save_image_from_array(array, path, template):
    assert(os.path.exists(template))

    template = sitk.ReadImage(template)
    image = sitk.GetImageFromArray(array)
    image.CopyInformation(template)
    sitk.WriteImage(image, path)

def save_deformation(result, path, template=None):
    if type(result) == ants.core.ants_image.ANTsImage:
        ants.image_write(result, path)
    elif type(result) == sitk.SimpleITK.Image :
        sitk.WriteImage(result, path)
    elif isinstance(result, str):
        if os.path.isfile(result):
            shutil.copy(result, path)
        else:
            raise ValueError("file doesn't exist!")
    elif  isinstance(result, torch.Tensor):

        result = result.squeeze().numpy()
        assert((result.ndim == 4) and (result.shape[0]==3))

        result = result.transpose((1,2,3, 0))

        save_image_from_array(result, path, template)
    else:
        raise TypeError('Work with ants image, tensor, numpy array Only')
