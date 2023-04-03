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

# Code modified from Morgana: 
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/morphology/computemorphology.py
# https://github.com/LabTrivedi/MOrgAna/blob/master/morgana/ImageTools/straightmorphology/computestraightmorphology.py
# TODO docstring and License
def straighten_mask_and_image(mask,intensity_image,image_to_reshape = None,margin = 3):
    """
    Straighten a binary mask and an intensity image using a midline-based approach.

    Parameters:
    -----------
    mask : ndarray
        A 2D binary array representing the object of interest.
    intensity_image : ndarray
        A 2D or 3D array representing the intensity values of the object of interest.
    image_to_reshape : ndarray, optional
        A 2D or 3D array representing the intensity values of the object of interest to reshape to the size of the straightened mask and image.
    margin : int, optional
        The margin around the object to include in the straightened mask and image.

    Returns:
    --------
    ma_straight : ndarray
        The straightened binary mask.
    bf_straight : ndarray
        The straightened intensity image.
    """
    # The code first checks if the maximum value of the mask array is 0. If it is, 
    # the function returns mask and intensity_image as is. If not, it proceeds to 
    # create a disk-shaped structuring element using skimage.morphology.disk function 
    # with the radius of margin. It then performs binary dilation of the mask array 
    # using the created structuring element and labels the resulting dilated mask 
    # using skimage.measure.label. The function then extracts the properties of 
    # the labeled regions using skimage.measure.regionprops. It assumes that the 
    # dilated mask contains only one labeled region and retrieves the slice 
    # corresponding to that region.
    if np.max(mask)==0:
        return mask, intensity_image
    strele = disk(margin)
    dilated_mask = binary_dilation(mask,strele)
    dilated_mask = label(dilated_mask)[0]
    
    props = regionprops(dilated_mask)
    slice_prop = props[0]["slice"]

    # Next, the function checks the shape of the intensity_image array. If the array 
    # is 2D, it expands its dimensions to 3D with the additional dimension having a 
    # size of 1. If the last dimension of the intensity_image array is the smallest, 
    # it moves that dimension to the front using numpy.moveaxis. It then extracts the 
    # slice corresponding to the labeled region from the 3D intensity_image array 
    # and the dilated mask.
    bf = intensity_image
    if len(bf.shape) == 2:
        bf = np.expand_dims(bf,0)
    if bf.shape[-1] == np.min(bf.shape):
        bf = np.moveaxis(bf, -1, 0)
    bf= bf[0][slice_prop]
    ma= dilated_mask[slice_prop]
    original_mask_cropped = mask[slice_prop]

    # The function then computes the anchor points of the dilated mask using anchorpoints.
    # compute_anchor_points function and the spline coefficients of the mask and intensity 
    # values using spline.compute_spline_coeff function. It then computes the midline and 
    # tangent of the mask using midline.compute_midline_and_tangent function. Finally, it 
    # generates a meshgrid using meshgrid.compute_meshgrid function.
    anch = anchorpoints.compute_anchor_points(dilated_mask,slice_prop,1)
    N_points, tck = spline.compute_spline_coeff(ma,bf,anch)

    diagonal = int(np.sqrt(ma.shape[0]**2+ma.shape[1]**2)/2)
    mid, tangent, width = midline.compute_midline_and_tangent(anch,N_points,tck,diagonal)

    mesh = meshgrid.compute_meshgrid(mid, tangent, width)

    # The function then generates an output image by setting out_image to intensity_image by 
    # default or to the image_to_reshape slice corresponding to the labeled region if 
    # image_to_reshape is provided. It then uses scipy.ndimage.map_coordinates function to map 
    # the original mask and intensity values to the straightened mask and intensity arrays 
    # using the computed meshgrid. It reshapes the resulting arrays to have the same shape as 
    # the meshgrid and returns them as ma_straight and bf_straight.
    out_image = bf
    if image_to_reshape is not None:
        out_image = image_to_reshape[slice_prop]

    # straighten the mask
    ma_straight = np.reshape(map_coordinates(original_mask_cropped,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))
    bf_straight = np.reshape(map_coordinates(out_image,np.reshape(mesh,(mesh.shape[0]*mesh.shape[1],2)).T,order=0,mode='constant',cval=0).T,(mesh.shape[0],mesh.shape[1]))
    
    return ma_straight, bf_straight