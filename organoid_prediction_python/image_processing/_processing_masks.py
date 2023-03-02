import numpy as np
from skimage.measure import regionprops

# TODO docstring
def keep_label_closest_to_avg_size(mask, avg_size = 12000):
    areas=[]
    for prop in regionprops(mask):
        areas.append(prop["area"])

    idx = (np.abs(np.array(areas) - avg_size)).argmin()
    
    flag_list = [1 if i==idx else 0 for i in range(len(areas))]
    
    
    return np.take(np.array([0] + flag_list), mask)

# TODO docstring
def keep_labels_closest_to_stack_median(mask_stack):
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