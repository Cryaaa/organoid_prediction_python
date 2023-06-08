import numpy as np
from skimage.measure import regionprops

def crop_image_or_mask_around_CM(image, mask, return_image = True, crop_height = 350, crop_width = 350):
    half_wid = int(crop_width/2)
    half_hgt = int(crop_height/2)
    
    center_of_mass = np.array(image.shape)/2
    if mask.max() >0:
        properties = regionprops(mask, image)
        center_of_mass = (properties[0].centroid)
    c_0, c_1 = [int(c) for c in center_of_mass]
    if c_0-half_wid < 0:
        x_borders = np.amax(np.array([
            [c_0-half_wid,0],
            [c_0+half_wid,crop_width]]),axis = 1)
    else:
        x_borders = np.amin(np.array([
            [c_0-half_wid,image.shape[0]-crop_width],
            [c_0+half_wid,image.shape[0]]]),axis = 1)
    if c_1-half_wid < 0:
        y_borders = np.amax(np.array([
            [c_1-half_hgt,0],
            [c_1+half_hgt,crop_height]]),axis = 1)
    else:
        y_borders = np.amin(np.array([
            [c_1-half_hgt,image.shape[1]-crop_height],
            [c_1+half_hgt,image.shape[1]]]),axis = 1)
    cropped_img = image[x_borders[0]:x_borders[1],y_borders[0]:y_borders[1]]
    cropped_mask = mask[x_borders[0]:x_borders[1],y_borders[0]:y_borders[1]]
    
    if return_image:
        return cropped_img
    return cropped_mask

def crop_around_centroid(image, centroid, crop_height = 800, crop_width = 800):
    half_wid = int(crop_width/2)
    half_hgt = int(crop_height/2)
    c_0, c_1 = [int(c) for c in centroid]
    
    if c_0-half_wid < 0:
        x_borders = np.amax(np.array([
            [c_0-half_wid,0],
            [c_0+half_wid,crop_width]]),axis = 1)
    else:
        x_borders = np.amin(np.array([
            [c_0-half_wid,image.shape[0]-crop_width],
            [c_0+half_wid,image.shape[0]]]),axis = 1)
    if c_1-half_wid < 0:
        y_borders = np.amax(np.array([
            [c_1-half_hgt,0],
            [c_1+half_hgt,crop_height]]),axis = 1)
    else:
        y_borders = np.amin(np.array([
            [c_1-half_hgt,image.shape[1]-crop_height],
            [c_1+half_hgt,image.shape[1]]]),axis = 1)
    cropped_img = image[x_borders[0]:x_borders[1],y_borders[0]:y_borders[1]]

    return cropped_img