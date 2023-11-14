from skimage.measure import regionprops_table
import numpy as np
import pandas as pd

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
    """
    Calculate region properties for brightfield images.

    This function computes various properties of regions in a brightfield image, 
    such as area, perimeter, major and minor axis lengths, eccentricity, etc.

    Parameters
    ----------
    bf_mask : array_like
        Binary mask of the brightfield image, where regions of interest are non-zero.
    bf_image : array_like
        The brightfield image itself.
    bf_prop_names : tuple of str, optional
        Names of the properties to be calculated. 
        Default is ("area", "perimeter", "axis_major_length", "axis_minor_length", 
        "eccentricity", "feret_diameter_max", "solidity").

    Returns
    -------
    dict
        A dictionary with keys as property names and values as property values. 
        Includes an additional key "aspect_ratio" calculated from the major and minor axis lengths.

    Notes
    -----
    If the mask is empty (all zero values), the function returns NaN for all properties.
    """
    if np.max(bf_mask) == 0:
        out = {k:[np.nan] for k in bf_prop_names}
        out["aspect_ratio"] = [np.nan]
        return out

    # Calculate region properties using skimage's regionprops_table
    bf_props_table = regionprops_table(bf_mask, bf_image, properties=bf_prop_names)
    # Calculate and add aspect ratio
    bf_props_table["aspect_ratio"] = np.array([bf_props_table["axis_major_length"][0]/bf_props_table["axis_minor_length"][0]])

    return bf_props_table


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
    """
    Calculate region properties for brachyury-stained images.

    Computes properties of regions in brachyury-stained images, similar to the brightfield images,
    with additional support for intensity mean and standard deviation.

    Parameters
    ----------
    bra_mask : array_like
        Binary mask of the brachyury-stained image, where regions of interest are non-zero.
    bra_image : array_like
        The brachyury-stained image.
    bf_regionprops_table : dict, optional
        Region properties of the corresponding brightfield image. Used to calculate area fraction.
    bra_prop_names : tuple of str, optional
        Names of the properties to be calculated.
        Default is ("area", "axis_major_length", "axis_minor_length", "eccentricity", 
        "solidity", "intensity_mean").

    Returns
    -------
    dict
        A dictionary with keys as property names and values as property values.
        Additional keys include "aspect_ratio" and, if bf_regionprops_table is provided, "area_fraction".

    Notes
    -----
    If the mask is empty (all zero values), the function returns NaN for all properties.
    The function defines and uses `image_stdev` to compute the standard deviation of intensity within the regions.
    """
    if np.max(bra_mask) == 0:
        out = {k: [np.nan] for k in bra_prop_names}
        out["aspect_ratio"] = [np.nan]
        out["area_fraction"] = [np.nan]
        return out

    def image_stdev(region, intensities):
        # Custom function to calculate standard deviation of intensities
        return np.std(intensities[region], ddof=1)
    
    # Calculate region properties using skimage's regionprops_table with extra_properties
    bra_props_table = regionprops_table(bra_mask, bra_image, properties=bra_prop_names, extra_properties=[image_stdev])
    # Calculate and add aspect ratio
    bra_props_table["aspect_ratio"] = np.array([bra_props_table["axis_major_length"][0]/bra_props_table["axis_minor_length"][0]])
    # Calculate and add area fraction if brightfield properties are available
    if bf_regionprops_table is not None:
        bra_props_table["area_fraction"] = np.array([bra_props_table["area"][0]/bf_regionprops_table["area"][0]])

    return bra_props_table


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
    """
    Extract simple features from both brightfield and brachyury-stained images.

    Combines the region properties calculated separately for brightfield and 
    brachyury-stained images.

    Parameters
    ----------
    bf_mask : array_like
        Binary mask of the brightfield image.
    bf_image : array_like
        The brightfield image.
    bra_mask : array_like
        Binary mask of the brachyury-stained image.
    bra_image : array_like
        The brachyury-stained image.
    bf_prop_names : tuple of str, optional
        Names of the properties to be calculated for the brightfield image.
    bra_prop_names : tuple of str, optional
        Names of the properties to be calculated for the brachyury-stained image.

    Returns
    -------
    dict
        A combined dictionary of region properties from both image types. 
        Keys are prefixed with "BF_" for brightfield and "BRA_" for brachyury-stained properties.

    Notes
    -----
    This function utilizes `simple_brightfield_regionprops` and `simple_brachyury_regionprops` 
    for computing the individual properties.
    """

    bf_props_table = simple_brightfield_regionprops(bf_mask,bf_image,bf_prop_names)
    bra_props_table = simple_brachyury_regionprops(bra_mask,bra_image,bf_props_table,bra_prop_names)

    bf_bra_combined_props = {}
    for k, v in bf_props_table.items():
        bf_bra_combined_props[f"BF_{k}"] = v
    for k, v in bra_props_table.items():
        bf_bra_combined_props[f"BRA_{k}"] = v 
        
    return bf_bra_combined_props

def extract_simple_features_image_series(
        bf_masks, 
        bf_images, 
        bra_masks, 
        bra_images
):
    """
    Extract simple features from series of brightfield and brachyury-stained images.

    Processes multiple image pairs (brightfield and brachyury-stained) to extract features
    and compile them into a single DataFrame.

    Parameters
    ----------
    bf_masks : list or array_like
        A list or array of binary masks for brightfield images.
    bf_images : list or array_like
        A list or array of brightfield images.
    bra_masks : list or array_like
        A list or array of binary masks for brachyury-stained images.
    bra_images : list or array_like
        A list or array of brachyury-stained images.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the combined features extracted from all image pairs.
        Each row corresponds to the features extracted from one pair of images.

    Notes
    -----
    This function iterates over each pair of images (brightfield and brachyury-stained),
    extracts their features using `extract_simple_features`, and compiles them into a DataFrame.
    It's useful for batch processing of image data.
    """
    dfs = []
    for bf_image, bf_mask, bra_image, bra_mask in zip(bf_images, bf_masks, bra_images, bra_masks):
        
        # Extract features for each image pair and append to the list
        dfs.append(pd.DataFrame(extract_simple_features(bf_mask, bf_image, bra_mask, bra_image)))
    
    # Concatenate all DataFrames into one, ignoring the index to avoid duplicate indices
    return pd.concat(dfs, axis=0, ignore_index=True)