import random
from typing import Optional, Callable, List
from pathlib import Path

import monai
import torch
import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch.utils
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import StratifiedGroupKFold

####SAME AS mil_dataset.py, but the outliers are always included in the external test set.
class MILDataset_temp(Dataset):
    """
    Dataset class for Ovarian Tumor dataset.

    Args:
        path (str): The path to the dataset.
        mode (str, optional): The mode of the dataset. Default is "train".
        label (str, optional): The label column name in the dataset annotations. Default is None.
        orient (bool, optional): Whether to orient the images to LPI orientation. Default is False.
        mask_tumor (bool, optional): Whether to mask the tumor in the images. Default is False.
        transform (callable, optional): A function/transform to apply on the images. Default is None.
    """

    def __init__(
            self,
            path,
            label=None,
            orient=False,
            mask_tumor=False,
            crop_tumor=False,
            transform=None,
            num_instances=-1,
            sampling_method="random",
            remove_outliers=True,
    ):
        """
        Creates an instance of the SSLRadiomicsDataset class with the given parameters.

        Args:
            path (str): The path to the dataset.
            mode (str): The mode of the dataset. Defaults to "train".
            label (Optional[str]): The label to use for the dataset. Defaults to None.
            orient (bool): True if the dataset should be oriented, False otherwise. Defaults to False.
            mask_tumor (bool): True if the tumor should be masked, False otherwise. Defaults to False.
            transform: The transformation to apply to the dataset. Defaults to None.

        Raises:
            None.

        Returns:
            None.
        """

        assert remove_outliers==False #remove outliers must be false, as this module is only
        monai.data.set_track_meta(False)
        sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
        super().__init__()
        self._path = Path(path)

        self.label = label
        self.orient = orient
        self.mask_tumor = mask_tumor
        self.crop_tumor = crop_tumor
        self.transform = transform
        self.num_instances = num_instances

        assert sampling_method in ["random", "uniform"], "Invalid sampling method. Must be one of 'random' or 'uniform'"
        self.sampling_method = sampling_method
        self.annotations = pd.read_csv(self._path)

        self.annotations = self.annotations.loc[
            (self.annotations[self.label] == 0) | (self.annotations[self.label] == 1)]
        self.annotations = self.annotations.loc[
            self.annotations["outlier"] == 0] if remove_outliers else self.annotations
        self.annotations = self.annotations.reset_index(drop=True)

    def get_rows(self):
        """
        Get the rows of the annotations as a list of dictionaries.

        Returns:
            list of dict: The rows of the annotations as dictionaries.
        """
        return self.annotations.to_dict(orient="records")

    def get_labels(self):
        """
        Function to get labels for when they are available in the dataset.

        Args:
            None

        Returns:
            None
        """

        labels = self.annotations[self.label].values
        assert not np.any(labels == -1), "All labels must be specified"
        return labels

    def __len__(self):
        """
        Size of the dataset.
        """
        return len(self.annotations)

    def __getitem__(self, idx: int):
        """
        Implement how to load the data corresponding to the idx element in the dataset from your data source.
        """

        # Get a row from the CSV file
        row = self.annotations.iloc[idx]
        image_path = row["image_path"]
        image = sitk.ReadImage(str(image_path))

        # Mask the tumor if specified
        if "annot_path" in row and self.mask_tumor:
            annot_path = row["annot_path"]
            annot = sitk.ReadImage(str(annot_path), sitk.sitkUInt8)
            image = sitk.Mask(image, annot)
            del annot

        # Extract the relevant tumor slices from the image
        if not self.crop_tumor:
            image = sitk.Extract(
                image,
                (image.GetSize()[0], image.GetSize()[1], int(row["tumor_zend_idx"] - row["tumor_zstart_idx"])),
                (0, 0, int(row["tumor_zstart_idx"])),
            )
        else:
            image = sitk.Extract(
                image,
                (int(row["tumor_xend_idx"] - row["tumor_xstart_idx"]),
                 int(row["tumor_yend_idx"] - row["tumor_ystart_idx"]),
                 int(row["tumor_zend_idx"] - row["tumor_zstart_idx"])),
                (int(row["tumor_xstart_idx"]), int(row["tumor_ystart_idx"]), int(row["tumor_zstart_idx"]))
            )

        # Orient all images to LPI orientation
        image = sitk.DICOMOrient(image, "LPI") if self.orient else image

        array = sitk.GetArrayFromImage(image)

        if self.num_instances > 0:
            # Randomly select num_instances instances from the image
            if self.sampling_method == "random":
                idxs = random.choices(range(array.shape[0]), k=self.num_instances)
                array = array[idxs]
            elif self.sampling_method == "uniform":
                idxs = np.linspace(0, array.shape[0] - 1, self.num_instances, dtype=int)
                array = array[idxs]

        tensor = array if self.transform is None else self.transform(array)

        # Put the slices dimension first for the MIL model
        tensor = np.einsum("C H W D -> D C H W", tensor) if isinstance(tensor, np.ndarray) else torch.einsum(
            "C H W D -> D C H W", tensor)

        # Get the target if available and add channel dimension
        if self.label is not None:
            target = int(row[self.label])
        else:
            target = None

        target = torch.tensor(target).unsqueeze(-1).float()

        return [tensor, target]


class MILNestedKFoldModule_temp(LightningDataModule):
    """
    Dataset class for Ovarian Tumor dataset. This class is used for k-fold cross-validation.
    """

    def __init__(
            self,
            path: str = "data/",
            label: Optional[str] = None,
            orient: bool = False,
            mask_tumor: bool = False,
            crop_tumor: bool = False,
            train_transform: Optional[Callable] = None,
            val_transform: Optional[Callable] = None,
            test_transform: Optional[Callable] = None,
            num_instances: int = -1,
            key_external_testset: str = "is_external_testset",
            remove_outliers: bool = True,
            k: int = 0,  # fold number
            num_inner_splits: int = 5,
            num_outer_splits: int = 5,
            seed: int = 0,
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            collate_fn: Optional[Callable] = torch.utils.data._utils.collate.default_collate,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        # num_splits = 10 means our dataset will be split to 10 parts
        # so we train on 90% of the data and validate on 10%
        assert 0 <= self.hparams.k < self.hparams.num_inner_splits * self.hparams.num_outer_splits, \
            "incorrect fold number, should be in [0, num_outer_splits * num_inner_splits)"

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_samples: List[str] = []
        self.val_samples: List[str] = []
        self.test_samples: List[str] = []

    def setup(self, stage: str):
        # Create full dataset with training and validation samples
        full_dataset = MILDataset_temp(
            path=self.hparams.path,
            label=self.hparams.label,
            orient=self.hparams.orient,
            mask_tumor=self.hparams.mask_tumor,
            crop_tumor=self.hparams.crop_tumor,
            transform=self.hparams.train_transform,
            num_instances=self.hparams.num_instances,
            sampling_method="random",
            remove_outliers=self.hparams.remove_outliers,
        )

        # Extract patients to make patient-wise split using GroupKFold
        indices = np.arange(len(full_dataset))
        patients = [sample['patient_id'] for sample in full_dataset.get_rows()]
        labels = full_dataset.get_labels()

        # Take apart the external test set
        external_indices = full_dataset.annotations.index[
            full_dataset.annotations[self.hparams.key_external_testset] == 1].tolist()
        indices = [idx for idx in indices if idx not in external_indices]
        patients = [patients[idx] for idx in indices]
        labels = [labels[idx] for idx in indices]

        # Split the dataset into train/val and test sets for the current fold
        kf_outer = StratifiedGroupKFold(n_splits=self.hparams.num_outer_splits, random_state=self.hparams.seed,
                                        shuffle=True)
        all_outer_splits = [k for k in kf_outer.split(indices, y=labels, groups=patients)]
        train_val_indeces, test_indeces = all_outer_splits[self.hparams.k // self.hparams.num_inner_splits]  # change k
        train_val_indeces, test_indeces = train_val_indeces.tolist(), test_indeces.tolist()

        test_indeces = [indices[idx] for idx in test_indeces] + external_indices

        # Create subsets of the dataset for testing
        self.test_dataset = MILDataset_temp(
            path=self.hparams.path,
            label=self.hparams.label,
            orient=self.hparams.orient,
            mask_tumor=self.hparams.mask_tumor,
            crop_tumor=self.hparams.crop_tumor,
            transform=self.hparams.test_transform,
            num_instances=-1,
            sampling_method="uniform",
            remove_outliers=self.hparams.remove_outliers,
        )
        self.test_dataset.annotations = self.test_dataset.annotations.iloc[test_indeces]

        # Set test samples
        self.test_samples = [sample['tumor_id'] for sample in self.test_dataset.get_rows()]

        if stage == "fit":
            # Extract patients to make patient-wise split using GroupKFold
            patients = [patients[idx] for idx in train_val_indeces]
            labels = [labels[idx] for idx in train_val_indeces]

            train_val_indeces = [indices[idx] for idx in train_val_indeces]

            # Split the dataset into training and validation sets for the current fold
            kf_inner = StratifiedGroupKFold(n_splits=self.hparams.num_inner_splits, random_state=self.hparams.seed,
                                            shuffle=True)
            all_inner_splits = [k for k in kf_inner.split(train_val_indeces, y=labels, groups=patients)]
            train_indeces, val_indeces = all_inner_splits[self.hparams.k % self.hparams.num_inner_splits]  # change k
            train_indeces, val_indeces = train_indeces.tolist(), val_indeces.tolist()
            train_indices = [train_val_indeces[idx] for idx in train_indeces]
            val_indices = [train_val_indeces[idx] for idx in val_indeces]

            # Create subsets of the dataset for training and validation
            self.train_dataset = MILDataset_temp(
                path=self.hparams.path,
                label=self.hparams.label,
                orient=self.hparams.orient,
                mask_tumor=self.hparams.mask_tumor,
                crop_tumor=self.hparams.crop_tumor,
                transform=self.hparams.train_transform,
                num_instances=self.hparams.num_instances,
                sampling_method="random",
                remove_outliers=self.hparams.remove_outliers,
            )
            self.train_dataset.annotations = self.train_dataset.annotations.iloc[train_indices]

            self.val_dataset = MILDataset_temp(
                path=self.hparams.path,
                label=self.hparams.label,
                orient=self.hparams.orient,
                mask_tumor=self.hparams.mask_tumor,
                crop_tumor=self.hparams.crop_tumor,
                transform=self.hparams.val_transform,
                num_instances=self.hparams.num_instances,
                sampling_method="uniform",
                remove_outliers=self.hparams.remove_outliers,
            )
            self.val_dataset.annotations = self.val_dataset.annotations.iloc[val_indices]

            # Set training and validation samples
            self.train_samples = [sample['tumor_id'] for sample in self.train_dataset.get_rows()]
            self.val_samples = [sample['tumor_id'] for sample in self.val_dataset.get_rows()]

    def train_dataloader(self):
        return DataLoader(dataset=self.train_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, shuffle=True, collate_fn=self.hparams.collate_fn)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_dataset, batch_size=self.hparams.batch_size,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, collate_fn=self.hparams.collate_fn)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_dataset, batch_size=1, num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory, collate_fn=self.hparams.collate_fn)


if __name__ == "__main__":
    from pathlib import Path

    import monai.transforms as transforms

    test_transforms = transforms.Compose([
        transforms.EnsureChannelFirst(channel_dim="no_channel"),
        transforms.ScaleIntensityRange(
            a_min=-100,
            a_max=300,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        transforms.Resize((512, 512, -1)),
        # transforms.SpatialPad((512, 512, 20)),
        transforms.ToTensor(),
    ])

    # Test pytorch lightning datamodule
    module = MILNestedKFoldModule_temp(
        path="/share/colon/cclaessens/datasets/Data_gyn_3.5-Processed_new/preprocessed_data.csv",
        label="label",
        orient=True,
        mask_tumor=False,
        train_transform=test_transforms,
        val_transform=test_transforms,
        k=0,
        num_inner_splits=5,
        num_outer_splits=5,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )

    module.setup("fit")

    print('train samples:', module.train_samples)
    print('val samples:', module.val_samples)
    print('test samples:', module.test_samples)

    # check for data leakage
    train_set = set(module.train_samples)
    val_set = set(module.val_samples)
    test_set = set(module.test_samples)
    print('intersection train and val:', train_set.intersection(val_set))
    print('intersection train and test:', train_set.intersection(test_set))
    print('intersection val and test:', val_set.intersection(test_set))
