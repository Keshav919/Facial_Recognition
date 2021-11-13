import os, sys
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import copy


class FaceLoader(Dataset):

    def __init__(self, dataset_path = "./data/", loader_type = "Train") -> None:
        super(FaceLoader, self).__init__()

        self.loader_type = loader_type
        self.path = dataset_path

        # Read the training and test datasets
        if self.loader_type == "Train":
            with open(dataset_path+"train_lab.txt", 'r') as f:
                self.labels = f.read().split('\n')
                f.close()
            with open(dataset_path+"train_im.txt", 'r') as f:
                self.images = f.read().split('\n')
                f.close()
        else:
            with open(dataset_path+"test_lab.txt", 'r') as f:
                self.labels = f.read().split('\n')
                f.close()
            with open(dataset_path+"test_im.txt", 'r') as f:
                self.images = f.read().split('\n')
                f.close()
        
        print("------- There are ", len(self.images), " samples! -------")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        
        # Get the corresponding image and label
        image = self.images[idx]
        label = int(self.labels[idx])
        
        # Read the image
        img = copy.deepcopy(plt.imread(self.path+image+".jpg"))

        # Return data
        data = {
            'img': img,
            'lab': label
        }

        return data


