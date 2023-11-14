from skimage import measure
import numpy as np
import pandas as pd

def percentile_centroid(image, percentile = 0.95):
    """
    Compute the centroid of an image above a certain percentile.

    Parameters:
    - image: A 2D numpy array representing an image.
    - percentile: A float representing the percentile above which to consider pixels.

    Returns:
    - A tuple of floats representing the x and y coordinates of the centroid.
    """
    cutoff = np.percentile(image,percentile)
    percentile_image = image * (image>cutoff)
    M = measure.moments(percentile_image)
    return (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

def measure_brachyury_polarisation(bra_images, masks,filenames) -> pd.DataFrame:
    """
    Measure the polarisation of Brachyury expression in a set of images.

    Parameters:
    - bra_images: A list of 2D numpy arrays representing images of Brachyury expression.
    - masks: A list of 2D numpy arrays representing masks of the organoids in the images.
    - filenames: A list of strings representing the filenames of the images.

    Returns:
    - A pandas DataFrame with columns "first_axis_polarisation", "second_axis_polarisation", and "filenames".
    """
    first_axis_bra_polarisation = []
    second_axis_bra_polarisation = []
    for mask, bra_image in zip(masks,bra_images):
        if mask.max() < 1:
            first_axis_bra_polarisation.append(np.nan)
            second_axis_bra_polarisation.append(np.nan)
            continue
        props = measure.regionprops(mask)
        wc = percentile_centroid(bra_image)
        cent = props[0]["centroid"]
        bbox = props[0]["bbox"]
        height = bbox[2]-bbox[0]
        width = bbox[3]-bbox[1]

        first_axis_bra_polarisation.append(np.abs(cent[0]-wc[0])/height)
        second_axis_bra_polarisation.append(np.abs(cent[1]-wc[1])/width)
    
    out = {
        "first_axis_polarisation":first_axis_bra_polarisation,
        "second_axis_polarisation":second_axis_bra_polarisation,
        "filenames":filenames
    }
    
    return pd.DataFrame(out)