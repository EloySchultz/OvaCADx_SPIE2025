import numpy as np
import SimpleITK as sitk
import os
def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
def process_image_clip(image):
    image=np.clip(image, -200, 300)
    image=image.astype('float32')
    image=(image-np.min(image))/(np.max(image)-np.min(image))
    return image

def process_image(image):
#     image=np.clip(image, -200, 300)
    image=image.astype('float32')
    image=(image-np.min(image))/(np.max(image)-np.min(image))
    return image

def class_to_index(cls):
    classes = ['B', 'M']
    if isinstance(cls,list):
        indices = []
        for _class in cls:
            if _class in classes:
                indices.append(classes.index(_class))
            else:
                print("Invalid class: " + str(_class))
                raise ValueError("Invalid class: " + str(_class))
        return indices




    if cls in classes:
        return classes.index(cls)
    else:
        print("Invalid class: "+str(cls))
        raise ValueError("Invalid class: "+str(cls))
    return -1

def check_files_in_folder(file_list, folder_path):
    # Get the list of files in the folder
    folder_contents = os.listdir(folder_path)

    # Check if each file in the file list is present in the folder
    missing_files = []
    for file_name in file_list:
        if file_name not in folder_contents:
            missing_files.append(file_name)

    return missing_files
def downsample_xy(image):
    # Get the original size and spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()

    target_size = [256, 256, original_size[-1]]
    min_spacing = np.min(np.divide(np.multiply(original_spacing,original_size),target_size))
    target_spacing = [min_spacing,min_spacing,original_spacing[-1]]
    # Create the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
#     resampler.SetDefaultPixelValue(-1024)
    resampler.SetOutputOrigin(original_origin) #mage.GetOrigin()
    resampler.SetOutputDirection(image.GetDirection())

    # Resample the image to the cube
    resampled_image = resampler.Execute(image)
    resampled_image=sitk.Flip(resampled_image, [False, False, True])
    return resampled_image


def resample_to_cubic_voxels(image):
    original_image = image

    # Calculate the new size (dimensions) for the output image
    original_spacing = original_image.GetSpacing()
    desired_spacing = tuple(i * original_spacing[0] for i in
                            (1.0, 1.0, 1.0))  # This makes sure that spacing in X, Y stays the same and in Z changes.
    original_size = original_image.GetSize()
    new_size = [int(round(s * os / ds)) for s, os, ds in zip(original_spacing, original_size, desired_spacing)]

    # Create an affine transformation to preserve the srow information (some samples may have SROW transform going on, and this applies the proper transform)
    affine_transform = sitk.AffineTransform(3)
    affine_transform.SetMatrix(original_image.GetDirection())
    affine_transform.SetTranslation(original_image.GetOrigin())

    # Resample the original image to the new size and spacing
    resampled_image = sitk.Resample(original_image, new_size, affine_transform, sitk.sitkLinear, (0.0, 0.0, 0.0),
                                    desired_spacing)

    if original_image.GetDirection()[-1] == 1:  # Some samples, such as B026, have a direction matrix that ends in -1 which means they do not need a flip.... This is currently just a hardcoded edge case, but should be looked after...
        resampled_image = sitk.Flip(resampled_image, [False, False, True])

    return resampled_image
def resample_to_cube(image):
    #Idea is that this function resamples the image such that you always have a 512x512x512 cube. Since the phyisical size of the scan
    #cuboid is often not a cube, some parts of the scan are cropped of. So essentailly, it makes the image even spacing and then scales it
    #such that the entire 512x512x512 cube is filled with data, and then it crops off the remaining data. This cropping is kinda random right now I think (just "bottom" half).

    # Get the original size and spacing
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()

    target_size = [512, 512, 512]
    min_spacing = np.min(np.divide(np.multiply(original_spacing,original_size),target_size))
    target_spacing = [min_spacing,min_spacing,min_spacing]
    # Create the resampling filter
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
#     resampler.SetDefaultPixelValue(-1024)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetOutputDirection(image.GetDirection())

    # Resample the image to the cube
    resampled_image = resampler.Execute(image)
    resampled_image=sitk.Flip(resampled_image, [False, False, True])
    return resampled_image


def move_spot_to_middle(tensor, im):
    ##roll will cause repetition artifacts around the edges of the image! Beware!
    # Get the indices of the nonzero elements (spot)
    spot_indices = np.argwhere(tensor)

    # Calculate the centroid of the spot
    centroid = np.mean(spot_indices.astype(float), axis=0)

    # Calculate the desired shift to the middle
    shift = tuple(np.array(tensor.shape) // 2 - centroid.astype(int))

    # Create a new tensor with the desired shift
    shifted_tensor = np.roll(tensor, shift, axis=tuple(range(tensor.ndim)))
    shifted_im = np.roll(im, shift, axis=tuple(range(tensor.ndim)))

    return shifted_tensor, shifted_im