from skimage.measure import regionprops_table
import numpy as np
import pandas as pd

# TODO docstring
def simple_brightfield_regionprops(
    bf_mask,
    bf_image,
    bf_prop_names = tuple([
        "area",
        "perimeter",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "feret_diameter_max",
        "solidity"])
):
    bf_props_table = regionprops_table(bf_mask,bf_image,properties=bf_prop_names)
    bf_props_table["aspect_ratio"] = np.array([bf_props_table["axis_minor_length"][0]/bf_props_table["axis_major_length"][0]])

    return bf_props_table

# TODO docstring
def simple_brachyury_regionprops(
    bra_mask, 
    bra_image,
    bf_regionprops_table = None,
    bra_prop_names = tuple([
        "area",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "solidity",
        "intensity_mean"])
):

    def image_stdev(region, intensities):
            return np.std(intensities[region], ddof=1)
    
    bra_props_table = regionprops_table(bra_mask,bra_image,properties=bra_prop_names, extra_properties=[image_stdev])
    bra_props_table["aspect_ratio"] = np.array([bra_props_table["axis_minor_length"][0]/bra_props_table["axis_major_length"][0]])
    if bf_regionprops_table is not None:
        bra_props_table["area_fraction"] = np.array([bra_props_table["area"][0]/bf_regionprops_table["area"][0]])

    return bra_props_table

# TODO docstring
def extract_simple_features(
    bf_mask,
    bf_image,
    bra_mask,
    bra_image,
    bf_prop_names = tuple([
        "area",
        "perimeter",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "feret_diameter_max",
        "solidity"]),
    bra_prop_names = tuple([
        "area",
        "axis_major_length",
        "axis_minor_length",
        "eccentricity",
        "solidity",
        "intensity_mean"])
):

    bf_props_table = simple_brightfield_regionprops(bf_mask,bf_image,bf_prop_names)
    bra_props_table = simple_brachyury_regionprops(bra_mask,bra_image,bf_props_table,bra_prop_names)

    bf_bra_combined_props = {}
    for k, v in bf_props_table.items():
        bf_bra_combined_props[f"BF_{k}"] = v
    for k, v in bra_props_table.items():
        bf_bra_combined_props[f"BRA_{k}"] = v 
        
    return bf_bra_combined_props

def extract_simple_features_image_series(bf_masks, bf_images, bra_masks, bra_images):
    dfs = []
    for i,(bf_image,bf_mask, bra_image, bra_mask) in enumerate(zip(bf_images,bf_masks, bra_images, bra_masks)):
        print(i)
        dfs.append(pd.DataFrame(extract_simple_features(bf_mask, bf_image, bra_mask, bra_image)))
    
    return pd.concat(dfs,axis =0,ignore_index=True)
