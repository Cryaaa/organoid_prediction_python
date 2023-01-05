import pandas as pd
import numpy as np
import pandas as pd
import os

from scipy.ndimage import label
from skimage import measure

from morgana.ImageTools.locoefa import computecoeff

def calculate_morgana_shapes(
    masks: np.ndarray, 
    mask_paths: list,
):
    """
    function calculates regionproperties of all images specified by masks 
    and mask_paths

    Parameters
    ----------
    masks: np.ndarray
        1st dimension represents the list of images
    mask_paths: list
        list of strings specifying the full path where the 
        masks are located. Order must match image order in
        masks
    """
    props = []
    for mask, mask_path in zip(masks,mask_paths):
        prop = compute_morphological_info_no_mesh(
            mask,
            mask_path,
        )
        reform_props(prop)
        props.append(prop)
    df = pd.concat(props,axis=1)
    return df.transpose()

# Taken from morgana and modified to work with straightened images
def compute_morphological_info_no_mesh(
    mask: np.ndarray, 
    f_ma: str,
    keys: list = [
        'centroid',
        'slice',
        'area',
        'eccentricity',
        'major_axis_length',
        'minor_axis_length',
        'equivalent_diameter',
        'perimeter',
        'extent',
        'inertia_tensor',
        'inertia_tensor_eigvals',
        'moments_hu',
        'orientation',
    ]
):
    """
    Function adapted from the Morgana library to be easier to handle with
    a more regular data structure. Computes shape features from skimage
    as well as LOCO-EFA shape features.

    Parameters
    ----------

    mask: np.ndarray
        labeled binary image specifying objects to be quantified
    f_ma: str
        path to mask image which is used to return a column with the file name
    keys: list
        list of strings that specify which properties to measure. Must match 
        property names found in skimage.measure.regionprops
    """
    # label mask
    labeled_mask, _ = label(mask)
    # compute morphological info
    props = measure.regionprops(labeled_mask)
    dict_ = {}
    for key in keys:
        dict_[key] = props[0][key]

    dict_['form_factor'] = dict_['perimeter']**2/(4*np.pi*dict_['area'])

    dict_['input_file'] = os.path.join('result_segmentation', os.path.split(f_ma)[1])

    dict_['locoefa_coeff'] = computecoeff.compute_LOCOEFA_Lcoeff(mask, 1).locoefa_coeff.values

    return pd.Series(dict_)

def reform_props(region_props: dict) -> None:
    """
    Function that takes regionproperties computed by morgana
    (eg. compute_morphological_info_no_mesh function) and reforms it to
    be usable in a machine learning setting (each column only has 1 value
    per row). 

    Parameters
    ----------

    region_props: dict
        Regionproperties to be reformed
    """
    region_props.pop('centroid')
    region_props.pop('slice')
    
    min_ax,maj_ax = (region_props['minor_axis_length'],region_props['major_axis_length'])
    try:
        region_props['aspect_ratio'] = maj_ax/min_ax
    except:
        print("error during AR calculation")
        region_props['aspect_ratio'] = np.nan
    
    hu_moms = region_props.pop('moments_hu')
    for i, mom in enumerate(hu_moms):
        region_props[f'moments_hu_{i}'] = mom
    
    inertia_tensor = region_props.pop('inertia_tensor')
    for i,tens_i in enumerate(inertia_tensor):
        for j,tens_ij in enumerate(tens_i):
            region_props[f'inertia_tensor_{i}_{j}'] = tens_ij
            
    inertia_tens_eigen = region_props.pop('inertia_tensor_eigvals')
    for i,tens_eigen in enumerate(inertia_tens_eigen):
        region_props[f'inertia_tensor_eigvals_{i}'] = tens_eigen
        
    coeffs = region_props.pop('locoefa_coeff')
    for i, coeff in enumerate(coeffs):
        region_props[f'locoefa_coeff_{i}'] = coeff