import numpy as np
from skimage.measure import regionprops,label

def keep_label_closest_to_avg_size(mask, avg_size = 12000,label_mask=True):
    """
    This function takes in a binary mask image and returns the mask with only the label closest to the average size
    (in terms of area) of all the labels in the mask.

    -----------
    Parameters:
        mask: a binary numpy array representing the input mask image.
        avg_size: an integer representing the average size (in terms of area) of all labels in the mask.

    Returns:
        A binary numpy array representing the mask with only the label closest to the average size.
    """
    if np.max(mask) ==0:
        return mask
    if label_mask:
        mask = label(mask)
    areas=[]
    for prop in regionprops(mask):
        areas.append(prop["area"])

    idx = (np.abs(np.array(areas) - avg_size)).argmin()
    
    flag_list = [1 if i==idx else 0 for i in range(len(areas))]
    
    
    return np.take(np.array([0] + flag_list), mask)


def keep_labels_closest_to_stack_median(mask_stack):
    """
    This function takes in a stack of binary mask images and returns a new stack with each image containing only the 
    label closest to the median size (in terms of area) of all the labels in the corresponding input mask.

    -----------
    Parameters:
        mask_stack: a 3D numpy array representing the stack of binary mask images.

    Returns:
        A 3D numpy array representing the new stack of binary mask images with each image containing only the label closest 
        to the median size of all the labels in the corresponding input mask.
    """
    areas=[]
    for mask in mask_stack:
        for prop in regionprops(mask):
            areas.append(prop["area"])

    area_median = np.median(areas)
    out = np.array([
        keep_label_closest_to_avg_size(mask,avg_size=area_median) 
        for mask in mask_stack
    ])

    return out