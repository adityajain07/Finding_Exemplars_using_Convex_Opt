#!/usr/bin/env python
# coding: utf-8

# In[37]:


"""
Authors      : Aditya Jain and Safwan Jamal
Date started : November 16, 2022
About        : Convex Optimization project; training script
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import argparse

from cifardataset import CIFARDataset
from custom_cnn_one import CustomCNN
from custom_cnn_two import CustomCNNTwo

def train_model(args):
	"""main traninig function"""

	num_classes = 10
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Available device is {device}')
	model = CustomCNNTwo(num_classes).to(device)
	# model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)

	train_set      = args.train_set
	num_epochs     = args.num_epochs
	early_stopping = args.early_stopping

	transform = transforms.Compose(
		[transforms.ToTensor(),
		transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
	batch_size     = args.batch_size
	class_list     = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                  'dog', 'frog', 'horse', 'ship', 'truck']
	train_root_dir = args.train_root_dir
	test_root_dir  = args.test_root_dir
	test_set       = args.test_set

	train_data       = CIFARDataset(train_root_dir, train_set, class_list, transform)
	train_dataloader = DataLoader(train_data,batch_size=batch_size, shuffle=True, num_workers=2)

	test_data        = CIFARDataset(test_root_dir, test_set, class_list, transform)
	test_dataloader  = DataLoader(test_data,batch_size=batch_size, shuffle=True, num_workers=2)

	optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
	loss_func = nn.CrossEntropyLoss()

	best_test_accuracy = 0.0
	early_stop_count   = 0

	for epoch in range(num_epochs): 
		# Model Training
		model.train()
		train_epoch_loss = 0.0
		for image_batch, label_batch in train_dataloader:    
			image_batch, label_batch = image_batch.to(device), label_batch.to(device)
			label_batch = label_batch.squeeze_()
        
			# Compute and apply gradients
			optimizer.zero_grad()
			outputs   = model(image_batch)  
			t_loss    = loss_func(outputs, label_batch)
			t_loss.backward()
			optimizer.step()
			train_epoch_loss += t_loss.item()      
		print(f'Training loss for epoch {epoch+1} is {train_epoch_loss/len(train_dataloader)}')
    
		# Model Evaluation
		model.eval()
		total_samples   = 0.0
		total_correct   = 0.0
		for image_batch, label_batch in test_dataloader:    
			image_batch, label_batch = image_batch.to(device), label_batch.to(device)  
			label_batch = label_batch.squeeze_()
			outputs = model(image_batch)
        
			# Calculate batch accuracy
			_, predicted = torch.max(outputs.data, 1)
			total_samples += label_batch.size(0)
			total_correct += (predicted == label_batch).sum().item()
		curr_accuracy = (total_correct/total_samples)*100
		print(f'Test accuracy for epoch {epoch+1} is {curr_accuracy}%')
    
		if curr_accuracy > best_test_accuracy:
			best_test_accuracy = curr_accuracy
			print(f'Best test accuracy improved to {best_test_accuracy}%')
			early_stop_count = 0
		else:
			early_stop_count += 1
        
		if early_stop_count==early_stopping:
			print(f'The best test accuracy for {args.run_name} achieved is {best_test_accuracy}')
			break


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--train_set', help = 'file containing the list of training points', required=True)
	parser.add_argument('--test_set', help = 'file containing the list of testing points', required=True)
	parser.add_argument('--train_root_dir', help = 'directory containing the training data', required=True)
	parser.add_argument('--test_root_dir', help = 'directory containing the testing data', required=True)
	parser.add_argument('--num_epochs', help = 'number of epochs to train for', required=True, type=int)
	parser.add_argument('--early_stopping', help = 'number of epochs to stop training after test loss does not improve', required=True, type=int)
	parser.add_argument('--batch_size', help = 'batch size for training', required=True, type=int)
	parser.add_argument('--dataloader_num_workers', help = 'number of cpus available', required=True, type=int)
	parser.add_argument('--run_name', help = 'name of the training instance', required=True)
	args   = parser.parse_args()

	train_model(args)




