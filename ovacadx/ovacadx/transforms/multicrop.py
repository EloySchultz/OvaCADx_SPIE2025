from typing import Callable, Optional

from ovacadx.utils import ensure_list


class MultiCrop:
    """
    Multi-Crop augmentation.
    """

    def __init__(self, high_resolution_transforms: Callable, low_resolution_transforms: Optional[Callable]):
        """
        Initialize an instance of a class with transformations for high-resolution and low-resolution images.

        Args:
            high_resolution_transforms (list): A list of Callable objects representing the transformations to be
                                               applied to high-resolution images.
            low_resolution_transforms (list, optional): A list of Callable objects representing the transformations to
                                                        be applied to low-resolution images. Default is None.
        """
        self.high_resolution_transforms = high_resolution_transforms
        self.low_resolution_transforms = low_resolution_transforms

    def __call__(self, input):
        """
        This function applies a set of transformations to an input image and returns high and low-resolution crops.

        Args:
            input (image): The input image to be transformed.

        Returns:
            list: A list containing the high and low-resolution crops.
        """

        high_resolution_crops = ensure_list(self.high_resolution_transforms(input))
        low_resolution_crops = ensure_list(self.low_resolution_transforms(input))

        return high_resolution_crops + low_resolution_crops
