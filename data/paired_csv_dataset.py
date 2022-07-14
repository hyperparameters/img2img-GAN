import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import is_image_file  # make_dataset
from PIL import Image
import numpy as np
import cv2
import albumentations as A
import torch
import pandas as pd


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class PairedCSVDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        csv_path = os.path.join(
            opt.dataroot, opt.phase+".csv")  # get csv file path
        df = pd.read_csv(csv_path)

        if self.opt.direction == "AtoB":
            f = "A"
        else:
            f = "B"
        # self.AB_paths = sorted(make_dataset(os.path.join(self.dir_AB,f), opt.max_dataset_size))  # get image paths
        self.A_paths = list(df.A)
        self.B_paths = list(df.B)
        self.shadow_paths = list(df.shadow_only_mask)
        assert (len(self.A_paths) == len(self.B_paths) == len(
            self.shadow_paths), "len A, B shadow must be same")
        self.len = len(self.A_paths)

        # crop_size should be smaller than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        A_path = self.A_paths[index]
        A = Image.open(A_path)
        B_path = self.B_paths[index]
        B = Image.open(B_path)
        shadow_path = self.shadow_paths[index]
        shadow = Image.open(shadow_path)
        if self.opt.input_nc == self.opt.output_nc == 1:
            A = np.array(A)[:, :, -1]
            B = np.array(B)[:, :, -1]

        # apply the same transform to both A and B
        if self.opt.albumentations:
            alb = get_albumentations(self.opt)
            transformed = alb(image=np.array(
                A), B=np.array(B), shadow=np.array(shadow))
            A = Image.fromarray(transformed["image"])
            B = Image.fromarray(transformed["B"])
            shadow = Image.fromarray(transformed["shadow"])

        transform_params = get_params(self.opt, A.size)
        self.opt.no_flip = False
        self.opt.preprocess = ""

        transform = get_transform(self.opt, transform_params, grayscale=(
            self.input_nc == 1), channel=self.input_nc)
        transform_shadow = get_transform(
            self.opt, transform_params, grayscale=(True), channel=1, normalize=False)
        # B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), channel=self.output_nc)

        A = transform(A)
        B = transform(B)
        shadow = (transform_shadow(shadow) > 0.008).float()
        return {'A': A, 'B': B, 'shadow': shadow, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.len


def get_albumentations(opt):
    transforms = A.Compose([
        A.ShiftScaleRotate(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=80,
                           interpolation=0, border_mode=0, value=(0, 0, 0, 0), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.Resize(opt.load_size, opt.load_size, p=1),
        A.RandomCrop(opt.crop_size, opt.crop_size, p=1)
    ], additional_targets={"B": "image", "shadow": "image"})

    return transforms
