import torch
import numpy as np

# local/our imports
import pystrum.pynd.ndutils as nd

import ants
import os
import SimpleITK as sitk

import data_loader
from skimage import measure


def label_to_onehot(label):
    if isinstance(label, str):
        assert os.path.exists(label)

        label = sitk.ReadImage(label)
        label = sitk.GetArrayFromImage(label)
        label = data_loader.onehot_encode(torch.from_numpy(label))
    elif  isinstance(label, torch.Tensor):
        label = label.squeeze()

        if label.ndim == 3:
            label = data_loader.onehot_encode(label)
        elif label.ndim == 4:
            assert(label.shape[0] == 4)
        else:
            raise TypeError('format not support!')
    elif type(label) == ants.core.ants_image.ANTsImage:
        label = label.numpy().transpose((2,1,0))
        label = data_loader.onehot_encode(torch.from_numpy(label))
    elif type(label) == sitk.SimpleITK.Image :
        label = sitk.GetArrayFromImage(label)
        label = data_loader.onehot_encode(torch.from_numpy(label))
    else:
        raise TypeError('format not support!')  
  
    return label 

def label_to_tensor(label):
    if isinstance(label, str):
        assert os.path.exists(label)

        label = sitk.ReadImage(label)
        label = sitk.GetArrayFromImage(label)
        label = torch.from_numpy(label)
    elif  isinstance(label, torch.Tensor):
        return label
    elif type(label) == ants.core.ants_image.ANTsImage:
        label = label.numpy().transpose((2,1,0))
        label = torch.from_numpy(label)
    elif type(label) == sitk.SimpleITK.Image :
        label = sitk.GetArrayFromImage(label)
        label = torch.from_numpy(label)
    else:
        raise TypeError('format not support!')  
  
    return label 

def disp_def_to_tensor(image):
    if isinstance(image, str):
        assert os.path.exists(image)

        image = sitk.ReadImage(image)
        image = sitk.GetArrayFromImage(image)
        image = image.transpose([3, 0, 1, 2])
        image = torch.from_numpy(image)
    elif  isinstance(image, torch.Tensor):
        image = image.squeeze()
        assert(image.ndim == 4)
        assert(image.shape[0] == 3)
    elif type(image) == ants.core.ants_image.ANTsImage:
        image = image.numpy().transpose((3, 2, 1, 0))
        image = torch.from_numpy(image)
    elif type(image) == sitk.SimpleITK.Image :
        image = sitk.GetArrayFromImage(image).transpose([3, 0, 1, 2])
        image = torch.from_numpy(image)
    else:
        raise TypeError('format not support!') 
    
    return image 


def dsc_one_hot(fixed_label, moved_label):
    fixed_label = fixed_label.to(moved_label.device)
    ndims = len(list(moved_label.size())) - 2
    vol_axes = list(range(2, ndims + 2))
    top = 2 * (fixed_label * moved_label).sum(dim=vol_axes)
    bottom = torch.clamp((fixed_label + moved_label).sum(dim=vol_axes), min=1e-5)
    dice = torch.mean(top / bottom, dim=0)
    return dice

def jacobian_determinant_tensor(disp):

    # check inputs
    disp = disp.permute((1,2,3,0))
    volshape = disp.shape[:-1]

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    return np.sum(jcb <= 0)/np.prod(jcb.shape)

def jacobian_determinant_pytorch(deformation_filed ):
    size = deformation_filed.shape[1:]

    vectors = [torch.arange(0, s) for s in size]
    grids = torch.meshgrid(vectors)
    grid = torch.stack(grids)

    deformation_filed = grid + deformation_filed

    deformation_filed = deformation_filed.permute((1,2,3,0))

    # compute gradients
    J = torch.gradient(deformation_filed)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    jcb_percent = torch.sum(jcb <= 0).item()/np.prod(size)

    return jcb_percent

def jacobian_determinant_from_deformation(deformation_filed ):
    size = deformation_filed.shape[1:]

    deformation_filed = deformation_filed.permute((1,2,3,0))

    # compute gradients
    J = torch.gradient(deformation_filed)

    # 3D glow
    dx = J[0]
    dy = J[1]
    dz = J[2]

        # compute jacobian components
    Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] - dy[..., 2] * dz[..., 1])
    Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] - dy[..., 2] * dz[..., 0])
    Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] - dy[..., 1] * dz[..., 0])

    
    jcb = Jdet0 - Jdet1 + Jdet2

    jcb_percent = torch.sum(jcb <= 0).item()/np.prod(size)

    return jcb_percent


# def jacobian_determinant(disp, use_torch=False):
#     if use_torch :
#         jcb_batch = [jacobian_determinant_pytorch(x.cpu()) for x in disp[:]]
#     else:
#         jcb_batch = [jacobian_determinant_tensor(x.cpu()) for x in disp[:]]

#     return np.mean(jcb_batch)

def jacobian_determinant(disp, use_torch=False):
    jcb_batch = [jacobian_determinant_from_deformation(x) for x in disp[:]]

    return np.mean(jcb_batch)

def preprocess_segmentation(seg):
    # Check if the input is a tensor and convert to numpy array if necessary
    if isinstance(seg, torch.Tensor):
        seg = seg.cpu().numpy().squeeze()

    # Check if the input is 4D or 5D and squeeze to 3D if necessary
    if seg.ndim != 3:
        raise TypeError('seg must be 3D, or 4D with the first dimension of length 1, or 5D with the first two dimensions of length 1.')
    
    return seg

def compute_ravd(true_label, pred_label):
    """
    Compute Relative Absolute Volume Difference (RAVD)
    
    Parameters:
    - pred_label: numpy array, the predicted binary segmentation label (0, 1)
    - true_label: numpy array, the ground truth binary segmentation label (0, 1)
    
    Returns:
    - RAVD value
    """
    pred_label = preprocess_segmentation(pred_label)
    true_label = preprocess_segmentation(true_label)

    # Calculate the foreground volume of the predicted segmentation label (i.e., the number of 1s)
    V_pred = np.sum(pred_label == 1)
    
    # Calculate the foreground volume of the ground truth segmentation label (i.e., the number of 1s)
    V_true = np.sum(true_label == 1)
    
    # Calculate RAVD
    ravd = np.abs(V_pred - V_true) / V_true
    
    return ravd


def get_maps(binary_seg):
    dist_map = sitk.Abs(sitk.SignedMaurerDistanceMap(binary_seg,
                                                     squaredDistance=False,
                                                     useImageSpacing=True))
    surface_map = sitk.LabelContour(binary_seg)

    return (dist_map, surface_map)


def get_surface_metrics(ref_seg, pred_seg, template_path):

    # Preprocess the segmentations
    pred_seg = preprocess_segmentation(pred_seg)
    ref_seg = preprocess_segmentation(ref_seg)

    # Load the template image
    template = sitk.ReadImage(template_path)

    pred_seg_image = sitk.GetImageFromArray(pred_seg)
    pred_seg_image.CopyInformation(template)

    ref_seg_image = sitk.GetImageFromArray(ref_seg)
    ref_seg_image.CopyInformation(template)

    # Get surface & distance maps
    pred_dist_map, pred_surface = get_maps(pred_seg_image)
    ref_dist_map, ref_surface = get_maps(ref_seg_image)

    # Get surf2surf distance maps
    pred2ref_dist_map = ref_dist_map * sitk.Cast(pred_surface, sitk.sitkFloat32)
    ref2pred_dist_map = pred_dist_map * sitk.Cast(ref_surface, sitk.sitkFloat32)

    # Get the surface distances
    pred2ref_dist_map_arr = sitk.GetArrayViewFromImage(pred2ref_dist_map)
    pred_surface_arr = sitk.GetArrayViewFromImage(pred_surface)
    pred2ref_distances = list(pred2ref_dist_map_arr[pred_surface_arr == 1])

    ref2pred_dist_map_arr = sitk.GetArrayViewFromImage(ref2pred_dist_map)
    ref_surface_arr = sitk.GetArrayViewFromImage(ref_surface)
    ref2pred_distances = list(ref2pred_dist_map_arr[ref_surface_arr == 1])

    # Calculate outcomes
    ASSD = np.mean(ref2pred_distances + pred2ref_distances)
    RSSD = np.sqrt(np.mean(np.square(ref2pred_distances + pred2ref_distances)))
    MSSD = np.max(ref2pred_distances + pred2ref_distances)

    return ASSD, RSSD, MSSD

def count_disconnected_regions(image):
    """
    Count the number of disconnected regions in a numpy array or a tensor.
    
    Parameters:
    - image: numpy array or torch tensor, the input image
    
    Returns:
    - num_regions: int, the number of disconnected regions
    """
    # Check if the input is a tensor and convert to numpy array if necessary
    image = preprocess_segmentation(image)
    
    # Label the connected regions
    _, num_regions = measure.label(image, return_num=True, connectivity=3)
    
    return num_regions