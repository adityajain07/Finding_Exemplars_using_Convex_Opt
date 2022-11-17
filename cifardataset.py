"""
Author: Aditya Jain, Safwan Jamal
Date  : November 16, 2022
About : A custom class for cifar-10 dataset for ece1505 project
"""
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
import json
import torch

class CIFARDataset(Dataset):
	def __init__(self, root_dir, data_list, label_list, transform=None):      
		"""
		Args:
			root_dir (string)  : root directory path that contains all the data
			data_list (string) : contains the list of data points for a particular set (train/val/test)
			label_list (list)  : list of labels for conversion to numerics
			transform (callable, optional): Optional transform to be applied
                on a sample.
        """
		self.root_dir   = root_dir
		self.data_list  = pd.read_csv(data_list)
		self.transform  = transform
		self.label_list = label_list

	def __len__(self):
		return len(self.data_list)
	
	def __getitem__(self, idx):
		image_data = self.data_list.iloc[idx, :]
		image_path = self.root_dir + image_data['category'] + '/' + image_data['image']
		image      = Image.open(image_path)
		if self.transform:
			image = self.transform(image)
		
		label = image_data['category']
		label = self.label_list.index(label)
		label = torch.LongTensor([label])

		return image, label