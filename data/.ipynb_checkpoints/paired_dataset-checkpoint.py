import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import is_image_file #make_dataset
from PIL import Image
import numpy as np
import cv2
import albumentations as A
import torch

def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


class PairedDataset(BaseDataset):
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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        if self.opt.direction =="AtoB":
            f = "A"
        else:
            f = "B"
        self.AB_paths = sorted(make_dataset(os.path.join(self.dir_AB,f), opt.max_dataset_size))  # get image paths
        
        
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
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
        if self.opt.direction=="AtoB":
            A_path = self.AB_paths[index]
            A = Image.open(A_path)
            B_path = A_path.replace("train/A","train/B")
            B = Image.open(B_path)
        else:
            B_path = self.AB_paths[index]
            B = Image.open(B_path)
            A_path = B_path.replace("train/B","train/A")
            A = Image.open(A_path)
            
        # apply the same transform to both A and B
        if self.opt.albumentations:
            alb = get_albumentations()
            transformed = alb(image=np.array(A), B=np.array(B))
            A = Image.fromarray(transformed["image"])
            B = Image.fromarray(transformed["B"])
            
        
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1),channel=self.input_nc)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), channel=self.output_nc)
        
        A = A_transform(A)
        B = B_transform(B)
        
        if self.output_nc==1:
            B = B[-1:,:,:]
#             B = torch.unsqueeze(B, dim=0)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)

    
def get_albumentations():
    transforms = A.Compose([
        A.ShiftScaleRotate(p=0.3),
        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=80, interpolation=0, border_mode=0,value=(0,0,0,0), p=0.7),
        A.HorizontalFlip(p=0.5),
    ],additional_targets={"B":"image"})
    
    return transforms
    