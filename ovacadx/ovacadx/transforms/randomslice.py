from monai.transforms import Transform, Randomizable


class RandomSlice(Randomizable, Transform):
    def __init__(self, num_slices):
        self.num_slices = num_slices

    def __call__(self, data):

        if len(data.shape) != 4:
            raise ValueError("Input data must be 3D and have a feature channel in dimension 0 (4D).")

        num_slices = data.shape[1]
        if self.num_slices > num_slices:
            selected_slices = self.R.choice(num_slices, self.num_slices, replace=True)
        else:
            selected_slices = self.R.choice(num_slices, self.num_slices, replace=False)

        selected_data = data[selected_slices]

        return selected_data
