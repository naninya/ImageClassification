import os
import torch
import numpy as np
import cv2
from glob import glob
from PIL import Image
import torchvision.datasets as dsets

class MnistDataset(dsets.MNIST):
    def __init__(
        self, 
        root_dir, 
        step, 
        transform, 
    ):
        train = step == "train"
        
        super().__init__(
            root=root_dir, 
            train=train, 
            transform=transform, 
            download=True
        )

class Dataset(torch.utils.data.Dataset): 
    def __init__(
        self,
        root_dir,
        step,
        transform=None
    ):
        self.root_dir = root_dir
        self.step = step
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._split()
    def __len__(self):
        return len(self.image_paths)
    
    def _split(self):
        for index, (image_path, annotation_path) in enumerate(zip(
            sorted(glob(f"{self.root_dir}/images/*.*")),
            sorted(glob(f"{self.root_dir}/annotations/*.*")),
        )):
            image_id = os.path.basename(image_path).split(".")[0]
            annotation_id = os.path.basename(annotation_path).split(".")[0]
            assert image_id == annotation_id
            
            if self.step == "train" :
                if index % 10 == 0:
                    continue

            elif self.step == "valid":
                if index % 10 != 0:
                    continue
            self.image_paths.append(image_path)
            self.labels.append((cv2.imread(annotation_path).sum() > 0)*1.0)
    
    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        image = np.array(Image.open(image_path).convert("RGB"))/255
        label = self.labels[idx]
        sample = dict(
            image=image,
            image_path=image_path,
            label=label,
        )

        if self.transform is not None:
            sample = self.transform(**sample)
#             return self.transform(image), label
            return sample["image"], sample["label"]
        return sample