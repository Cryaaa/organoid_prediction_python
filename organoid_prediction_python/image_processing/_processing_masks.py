import numpy as np
from scipy.ndimage import map_coordinates
from scipy.ndimage import label
from skimage.morphology import binary_dilation, disk
from skimage.measure import regionprops
from morgana.ImageTools.morphology import (
    anchorpoints, 
    spline, 
    midline, 
    meshgrid
)

# TODO docstring
def keep_label_closest_to_avg_size(mask, avg_size = 12000):
    areas=[]
    for prop in regionprops(mask):
        areas.append(prop["area"])

    idx = (np.abs(np.array(areas) - avg_size)).argmin()
    
    flag_list = [1 if i==idx else 0 for i in range(len(areas))]
    
    
    return np.take(np.array([0] + flag_list), mask)

# Code modified from Morgana: 
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/morphology/computemorphology.py
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/straightmorphology/computestraightmorphology.py
# TODO docstring
def straighten_mask_and_image(mask,intensity_image,image_to_reshape = None,margin = 3):
    
    strele = disk(margin)
    dilated_mask = binary_dilation(mask,strele)
    dilated_mask = label(dilated_mask)[0]
    
    props = regionprops(dilated_mask)
    slice_prop = props[0]["slice"]

    bf = intensity_image
    if len(bf.shape) == 2:
        bf = np.expand_dims(bf,0)
    if bf.shape[-1] == np.min(bf.shape):
        bf = np.moveaxis(bf, -1, 0)
    bf= bf[0][slice_prop]
    ma= dilated_mask[slice_prop]
    original_mask_cropped = mask[slice_prop]

    anch = anchorpoints.compute_anchor_points(dilated_mask,slice_prop,1)
    N_points, tck = spline.compute_spline_coeff(ma,bf,anch)

    diagonal = int(np.sqrt(ma.shape[0]**2+ma.shape[1]**2)/2)
    mid, tangent, width = midline.compute_midline_and_tangent(anch,N_points,tck,diagonal)

    mesh = meshgrid.compute_meshgrid(mid, tangent, width)

    out_image = bf
    if image_to_reshape is not None:
        out_image = image_to_reshape[slice_prop]

    # straighten the mask
    ma_straight = np.reshape(map_coordinates(original_mask_cropped,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))
    bf_straight = np.reshape(map_coordinates(out_image,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))
    
    return ma_straight, bf_straight