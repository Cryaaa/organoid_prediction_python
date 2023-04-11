import pyclesperanto_prototype as cle
import numpy as np

def analysis(image,min_size_threshold:int = 8000) -> np.ndarray:
    from skimage.morphology import remove_small_objects
    from skimage.morphology import reconstruction

    blurred = cle.invert(cle.gaussian_blur(image,sigma_x=2,sigma_y=2))
    thresholded = cle.threshold_otsu(blurred)
    only_large_objects = remove_small_objects(cle.pull(thresholded).astype(bool),min_size=min_size_threshold)
    
    seed = np.copy(only_large_objects)
    seed[1:-1, 1:-1] = only_large_objects.max()

    filled = reconstruction(seed, only_large_objects, method='erosion')
    
    return filled

def exclude_labels_based_on_property(label_image,property_name:str,threshold:float,exclude_large:bool=True) -> np.ndarray:
    from skimage.measure import regionprops_table
    regprops = regionprops_table(label_image,properties=["label",property_name])
    exclude_list = np.array([0 if regprop > threshold else 1 for regprop in regprops[property_name]])
    if not exclude_large:
        exclude_list = 1 - exclude_list
        
    return np.array(cle.replace_intensities(label_image,np.insert(exclude_list,0,0)))

def sequentially_modifying_mask(mask,property_name:str,threshold:float,exclude_large:bool=True) -> np.ndarray:
    import warnings
    label = cle.connected_components_labeling_diamond(mask)
    if np.max(np.array(label)) > 1:
        label = cle.exclude_labels_on_edges(label)
    if np.max(np.array(label)) > 1:
        label = exclude_labels_based_on_property(label,property_name,threshold,exclude_large=True)
    if np.max(np.array(label)) > 1:
        warnings.warn("More than one label in image")
    return np.array(label)

def workflow_2D_organoids(
    image,
    min_size_threshold:int = 8000,
    property_name:str = 'eccentricity',
    threshold:float = 0.9,
    exclude_large:bool=True
):
    mask = analysis(image,min_size_threshold)
    labels = sequentially_modifying_mask(mask,property_name,threshold)
    return labels

