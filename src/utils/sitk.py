import os

from pathlib import Path
from typing import List, Union

import numpy as np

from SimpleITK import (
    DICOMOrientImageFilter_GetOrientationFromDirectionCosines,
    DICOMOrient,
    Image,
    ReadImage,
    ResampleImageFilter,
    sitkNearestNeighbor,
    sitkLinear,
    sitkBSpline3,
    ConstantPadImageFilter,
    Minimum,
    Maximum,
    Threshold,
    RescaleIntensity,
    ExtractImageFilter,
)


def reorient_volume(image: Image, orientation: str = "LPS") -> Image:
    """
    Reorient the input image to the specified orientation.

    Args:
        image (sitk.Image): The input image to be reoriented.
        orientation (str): The target orientation for the image. The default is "LPS".

    Returns:
        sitk.Image: The reoriented image.

    Notes:
        The function checks the current orientation of the input image and reorients it to the specified orientation.
        If the input image is already in the specified orientation, it is returned as is.
    """
    _orient = DICOMOrientImageFilter_GetOrientationFromDirectionCosines(
        direction=image.GetDirection()
    )
    if _orient != orientation:
        return DICOMOrient(image, orientation)
    return image


def load_image_data(image_path: Path, labels: List[Path]) -> Union[Image, Image, None]:
    """
    Load image and label data from the specified paths.

    Args:
        image_path (AnyStr): Path to the image data file.
        labels (List[AnyStr]): List of paths to the label data files.

    Returns:
        Tuple[sitk.Image, sitk.Image]: A tuple containing the loaded image and label data.

    Raises:
        FileNotFoundError: If the image file or any of the label files are not found.
    """

    basename = os.path.basename(image_path).split("_image")[0]
    label_path = next(lb_path for lb_path in labels if basename in lb_path.name)
    try:
        image = reorient_volume(ReadImage(image_path))
        label = reorient_volume(ReadImage(label_path)) if label_path else None
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e.filename}")

    return image, label


def resample_image(image: Image, new_spacing: List[float], interp: int = 1) -> Image:
    """
    Resamples the input image to the specified spacing.

    Args:
        image (sitk.Image): The input image to be resampled.
        new_spacing (List[float]): The target spacing after resampling.
        interp (int): The interpolation method to use:
                      - 0: Nearest neighbor interpolation.
                      - 1: Linear interpolation (default).
                      - 2: Cubic spline interpolation.

    Returns:
        sitk.Image: The resampled image.

    Raises:
        ValueError: If an invalid interpolation method is provided.
    """

    # Calculate new size after resampling
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()

    new_size = tuple(
        [
            int(round(osize * (ospc / nspc)))
            for osize, ospc, nspc in zip(original_size, original_spacing, new_spacing)
        ]
    )

    # Set image's new spatial properties
    sampler = ResampleImageFilter()
    sampler.SetSize(new_size)
    sampler.SetOutputSpacing(new_spacing)
    sampler.SetOutputOrigin(image.GetOrigin())
    sampler.SetOutputDirection(image.GetDirection())

    # Interpolate according the method
    if interp == 0:
        sampler.SetInterpolator(sitkNearestNeighbor)
    elif interp == 1:
        sampler.SetInterpolator(sitkLinear)
    elif interp == 2:
        sampler.SetInterpolator(sitkBSpline3)
    else:
        raise ValueError(f"Unknown interpolation method: {interp}")

    return sampler.Execute(image)


def pad_image(image: Image, patch_size: List[int], voxel_value: int = 0) -> Image:
    """
    Pad the input image so that it becomes divisible into patches of the specified size.
    This function also ensures that the patches cover the entire image by adding padding if necessary.

    Args:
        image (sitk.Image): The input image to be padded.
        patch_size (List[int]): The size of each patch in each dimension.
        voxel_value (int): The value to use for padding (default is 0).

    Returns:
        sitk.Image: The padded image.

    Raises:
        ValueError: If the dimensions of the input image do not match the specified patch size.
    """

    size = image.GetSize()
    if len(size) != len(patch_size):
        raise ValueError(
            f"Expected image dimensions to equal {patch_size} but got {size}"
        )

    # Calculate the number of patches in each dimension
    num_patches = [int(np.ceil(size / patch)) for size, patch in zip(size, patch_size)]
    # Calculate the size needed to accommodate the patches
    new_size = [patch * num_patch for patch, num_patch in zip(patch_size, num_patches)]
    # Calculate the amount of padding needed for each dimension
    padding_size = [new - old for new, old in zip(new_size, size)]

    flt = ConstantPadImageFilter()
    flt.SetConstant(voxel_value)
    flt.SetPadUpperBound(padding_size)

    return flt.Execute(image)


def normalize_intensity(image: Image, value_range: List[int]) -> Image:
    """
    Normalize the input image within the specified value range.

    Args:
        image (sitk.Image): The input image to be normalized.
        value_range (List[int]): A list containing the lower and upper bounds for normalization.

    Returns:
        sitk.Image: The normalized image.

    Notes:
        If value_range is None or has length not equal to 2, the minimum and maximum intensity values of the image
        will be used as the value range for normalization.
    """

    if value_range is None or len(value_range) != 2:
        value_range = [Minimum(image), Maximum(image)]

    # Thresholding and normalization
    img_thresholded = Threshold(
        image, lower=value_range[0], upper=value_range[1], outsideValue=value_range[0]
    )
    img_normalized = RescaleIntensity(img_thresholded, outputMinimum=0, outputMaximum=1)

    # Convert normalized array back to SimpleITK image
    img_normalized.CopyInformation(image)

    return img_normalized


def get_patch_coordinates(image: Image, patch_size: List[int]) -> List[List[List[int]]]:
    """
    Generate coordinates for dividing a SimpleITK image into patches of the specified size with minimal overlap.

    Args:
        image (sitk.Image): The input image from which to generate patch coordinates.
        patch_size (List[int]): The size of each patch along the x, y, and z axes.

    Returns:
        List[List[List[int]]]: A list of coordinates representing the patches. Each coordinate triplet
        corresponds to the (x, y, z) coordinates of a patch's starting voxel.

    Notes:
        The function calculates the coordinates of patches based on the input image size and patch size.
        It ensures minimal overlap between patches by adjusting the patch coordinates.
    """

    image_size = image.GetSize()
    min_overlap = [patch_size[0] / 2, patch_size[1] / 2, patch_size[2] / 2]

    n_x = 2
    while (n_x * patch_size[0] - image_size[0]) / (n_x - 1) < min_overlap[0]:
        n_x += 1

    n_y = 2
    while (n_y * patch_size[1] - image_size[1]) / (n_y - 1) < min_overlap[1]:
        n_y += 1

    n_z = 2
    while (n_z * patch_size[2] - image_size[2]) / (n_z - 1) < min_overlap[2]:
        n_z += 1

    overlap = [
        int((n_x * patch_size[0] - image_size[0]) / (n_x - 1)),
        int((n_y * patch_size[1] - image_size[1]) / (n_y - 1)),
        int((n_z * patch_size[2] - image_size[2]) / (n_z - 1)),
    ]

    coordinates = []

    # Iterate over the range of patches along the x-axis
    for x in range(n_x):
        coordinates.append([])
        # Iterate over the range of patches along the y-axis
        for y in range(n_y):
            coordinates[x].append([])
            # Iterate over the range of patches along the z-axis
            for z in range(n_z):
                # Calculate the raw coordinates of the current patch
                raw_values = [
                    x * patch_size[0] - x * overlap[0],
                    y * patch_size[1] - y * overlap[1],
                    z * patch_size[2] - z * overlap[2],
                ]

                # Adjust coordinates to ensure the patch fits within the image boundaries
                for i, v in enumerate(raw_values):
                    if v + patch_size[i] >= image_size[i]:
                        raw_values[i] -= v + patch_size[i] - image_size[i]

                # Append the raw coordinates of the current patch to the list of coordinates
                coordinates[x][y].append(raw_values)

    return coordinates


def get_patches(
    image: Image, patch_size: List[int], coordinates: List[List[List[int]]]
) -> List[List[Image]]:
    """
    Extract patches from a SimpleITK image based on the provided coordinates and patch size.

    Args:
        image (sitk.Image): The input image from which to extract patches.
        patch_size (List[int]): The size of each patch along the x, y, and z axes.
        coordinates (List[List[List[int]]]): A list of coordinates representing the starting voxel of each patch.
            The coordinates should be organized as a three-dimensional array (x, y, z).

    Returns:
        List[List[sitk.Image]]: A list of extracted patches. Each patch is represented as a SimpleITK image.

    Notes:
        The function iterates over the provided coordinates and extracts patches from the input image
        using the specified patch size. It returns a list of extracted patches corresponding to the
        provided coordinates.
    """

    patches = []
    for x in range(len(coordinates)):
        patches.append([])
        for y in range(len(coordinates[0])):
            patches[x].append([])
            for z in range(len(coordinates[0][0])):
                origin = coordinates[x][y][z]
                bbox = [
                    origin[0],
                    origin[0] + patch_size[0],
                    origin[1],
                    origin[1] + patch_size[1],
                    origin[2],
                    origin[2] + patch_size[2],
                ]

                extract_filter = ExtractImageFilter()
                extract_filter.SetIndex([int(bbox[0]), int(bbox[2]), int(bbox[4])])
                extract_filter.SetSize(
                    [
                        int(bbox[1] - bbox[0]),
                        int(bbox[3] - bbox[2]),
                        int(bbox[5] - bbox[4]),
                    ]
                )
                patches[x][y].append(extract_filter.Execute(image))

    return patches


def channels_first_3d(x):
    """
    Rearranges the dimensions of a 3D array to adhere to the "channels-first" convention.

    Args:
        x (numpy.ndarray): Input array representing a 3D volume or a batch of 3D volumes.
                           For a single 3D volume, the shape should be (channels, depth, height, width).
                           For a batch of 3D volumes, the shape should be (batch_size, channels, depth,
                           height, width).

    Returns:
        numpy.ndarray: Rearranged array with dimensions rearranged to adhere to the "channels-first" convention.

    Raises:
        ValueError: If the input array has an unsupported number of dimensions.

    Note:
        - For a single 3D volume, the channel dimension is expected to be the first dimension after
          the batch dimension.
        - For a batch of 3D volumes, the channel dimension is expected to be the second dimension.
    """

    if len(x.shape) == 5:
        x = np.transpose(x, (0, 4, 1, 2, 3))
    elif len(x.shape) == 4:
        x = np.transpose(x, (3, 0, 1, 2))
    else:
        raise ValueError(f"Expected 4 or 5 dimensions, got {x.shape}.")
    return x


def channels_last_3d(x):
    """
    Rearranges the dimensions of a 3D array to adhere to the "channels-last" convention.

    Args:
        x (numpy.ndarray): Input array representing a 3D volume or a batch of 3D volumes.
                           For a single 3D volume, the shape should be (channels, depth, height, width).
                           For a batch of 3D volumes, the shape should be (batch_size, channels, depth,
                           height, width).

    Returns:
        numpy.ndarray: Rearranged array with dimensions rearranged to adhere to the "channels-last" convention.

    Raises:
        ValueError: If the input array has an unsupported number of dimensions.

    Note:
        - For a single 3D volume, the channel dimension is expected to be the first dimension.
        - For a batch of 3D volumes, the channel dimension is expected to be the second dimension after
          the batch dimension.
    """

    if len(x.shape) == 5:
        x = np.transpose(x, (0, 2, 3, 4, 1))
    elif len(x.shape) == 4:
        x = np.transpose(x, (1, 2, 3, 0))
    else:
        raise ValueError(f"Expected 4 or 5 dimensions, got {x.shape}.")
    return x
