from skimage import measure
import numpy as np
import pandas as pd

# TODO docstring
def percentile_centroid(image, percentile = 0.95):
    cutoff = np.percentile(image,percentile)
    percentile_image = image * (image>cutoff)
    M = measure.moments(percentile_image)
    return (M[1, 0] / M[0, 0], M[0, 1] / M[0, 0])

# TODO docstring
def measure_brachyury_polarisation(bra_images, masks,filenames) -> pd.DataFrame:
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