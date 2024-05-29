from skimage.filters import gaussian, threshold_otsu
from ._processing_masks import keep_labels_closest_to_stack_median
from numpy import array

def segment_bra(bra_img,bf_mask,thresh,sigma=2):
    smooth = gaussian(bra_img,sigma=sigma)
    segment = smooth > thresh
    masked = segment*bf_mask
    return masked

def segment_brachyury_stack(bra_images, bf_masks, thresh=None, sigma = 2,only_1_label = True):
    if thresh is None:
        thresh = threshold_otsu(bra_images)
    segmented = array([segment_bra(b_img, b_mask,thresh,sigma) for (b_img, b_mask) in zip(bra_images,bf_masks)])
    if only_1_label:
        return keep_labels_closest_to_stack_median(segmented.astype(int))
    return segmented.astype(int)
