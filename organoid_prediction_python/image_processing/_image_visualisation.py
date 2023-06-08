from skimage import color

def make_composite_image(
        image_1, 
        image_2, 
        rgb_color_factor_1 = [1,1,1], 
        rgb_color_factor_2 = [1,0,0], 
        max_value_for_scaling = 1.1
    ):
    """
    Creates a composite image by combining two grayscale images.

    Args:
        image_1: The first grayscale image.
        image_2: The second grayscale image.
        rgb_color_factor_1: A list of RGB color factors for image_1. Defaults to [1, 1, 1].
        rgb_color_factor_2: A list of RGB color factors for image_2. Defaults to [1, 0, 0].
        max_value_for_scaling: The maximum value used for scaling the output image. Defaults to 1.1.

    Returns:
        The composite image obtained by combining image_1 and image_2 using the specified color factors and scaling.
    """
    image_1 = color.gray2rgb(image_1)
    image_2 = color.gray2rgb(image_2)
    
    out = image_1*rgb_color_factor_1+rgb_color_factor_2*image_2
    scaling = [max_value_for_scaling,1,1]
    return out /scaling