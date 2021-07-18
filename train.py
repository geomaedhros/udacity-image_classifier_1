

""" 

1. Train
Train a new network on a data set with train.py

Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains

Options:
Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
Choose architecture: python train.py data_dir --arch "vgg13"   
Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
Use GPU for training: python train.py data_dir --gpu

"""

# Import necessary libraries and external data
import argparse

import matplotlib.pyplot as plt # to plot graphs 
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F  
from torchvision import datasets, transforms, models
from torch.autograd import Variable

from time import time, sleep # to measure and set time-based functions
import os, sys # to include some os functions in order to work with files, etc.

from PIL import Image  # to import/transform images
from collections import OrderedDict

import json
with open('cat_to_name.json', 'r') as f:
    catg_to_name = json.load(f)
    
# Imports functions created for this program
from get_input_parse_train import get_input_parse_train

start_time = time()

# Defines different models to use
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)

#create dictionary with all selectable options
mods_pretr = {'resnet': resnet18, 'alexnet': alexnet, 'vgg': vgg16}

# loading args
args_train = get_input_parse_train()

data_dir = args_train.data_dir
print('data_dir value:', data_dir)

arch = args_train.arch

# Use GPU if selected and available
if args_train.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

else:
    device = torch.device('cpu')

    
                                            # In order for this to work, the image-files need to be
                                            # divided within this folder as well, categorized by train, test and valid:
train_dir = data_dir + '/train'             # subdir "train"
test_dir = data_dir + '/test'               # subdir "test"
valid_dir = data_dir + '/valid'             # subdir "valid"



# Define transforms for the training, testing and validation data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose( [transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# transforms and loads images
train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)

# Use the pretrained network from torchvision.models selected
model = mods_pretr[arch]

### BUILDING AND TRAINING

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

classifier_input = model.classifier[0].in_features  # uses the "in_features" attribute to define the number of inputs of our set. 
                                                    # Source https://towardsdatascience.com/a-beginners-tutorial-on-building-an-ai-image-classifier-using-pytorch-6f85cb69cba7
   
   
# Build a feed-forward network replacing the classifier within the model
hidden_units = args_train.huts

classifier = nn.Sequential( nn.Linear(classifier_input, 4096),
                                               nn.ReLU(),
                                               nn.Linear(4096, hidden_units),
                                               nn.Dropout(p=0.15),            # set dropout 15%
                                               nn.ReLU(),
                                               nn.Linear(hidden_units, 102),
                                               nn.Dropout(p=0.15),            # set dropout 15%
                                               nn.LogSoftmax(dim=1))


# Replace default classifier with new classifier
model.classifier = classifier
model = model.to(device)

# Load images via available device
images, labels = next(iter(testloader))
images, labels = images.to(device), labels.to(device)

# Flatten images
images = images.view(images.shape[0], -1)  # "flatten()"

# Define the loss
criterion = nn.CrossEntropyLoss() # Multiclass

# Define optimizer 
learning_rate = args_train.lr
optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

# Train network

epochs = args_train.eps
steps = 0
train_loss = 0
print_every = 10

for epoch in range(epochs):
    for images, labels in trainloader:
        # use GPU if available, CPU if not
        images, labels = images.to(device), labels.to(device)
        
        steps += 1
        
        # Train model
        model.train()
        
        # Set gradients back to zero
        optimizer.zero_grad()
        
        # Forward pass
        log_ps = model.forward(images)     # calculate model output
        loss = criterion(log_ps, labels)   # calculate loss 
        loss.backward()                    # calculate gradients
        optimizer.step()                   # adjust parameters

        # Add the loss to the training set's running loss
        train_loss += loss.item()  
        
        # Print results of the training steps
        if steps % print_every == 0:
            valid_loss = 0
            accuracy = 0
            
            # Evaluate model
            model.eval()
            
            # turn off gradiant calculations
            with torch.no_grad():    
                for images, labels in validloader:
                    
                    # use GPU if available, CPU if not
                    images, labels = images.to(device), labels.to(device)
                    
                    # Forward pass 
                    log_ps = model.forward(images)  # calculate output
                    batch_loss = criterion(log_ps, labels)  # calculate validation batch loss
                    
                    # add batch loss to running test loss
                    valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(log_ps)   # transform log-prob to regular probability
                    top_p, top_class = ps.topk(1, dim=1)  # create top prob class 
                    equals = top_class == labels.view_as(top_class)  # compare predicted to actual classes
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item() # calculate accuracy
            
                       
            print("Epoch: {}/{}.. ".format(epoch+1, epochs),                     # Print training status
                      f"Train loss: {train_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
            
            train_loss = 0
            model.train()

# Test model
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        log_ps = model.forward(images)
        
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()
                    
        # Calculate accuracy
        ps = torch.exp(log_ps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view_as(top_class)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
print(  f"Test loss: {test_loss/len(testloader):.3f}.. "
        f"Test accuracy: {accuracy/len(testloader):.3f}")

test_loss = 0


# Save checkpoint 

model.class_to_idx = train_data.class_to_idx  # trainset 

# Create checkpoint dictionary 
ckpt_save_dir = args_train.save_dir
checkpoint = {    'model': model, # vgg16(pretrained)
                  'state_dict': model.state_dict(), 
                  'classifier': classifier, 
                  'epochs': epochs, # 5
                  'learning_rate': learning_rate, # 0.003
                  'input_size': classifier_input, # 25088
                  'output_size': 102, # 102
                  'batch_size': 64,
                  'class_to_idx': model.class_to_idx,
                  'optimizer_dict': optimizer.state_dict()}

torch.save(checkpoint, os.path.join(ckpt_save_dir, '/checkpoint.pth'))
print("Checkpoint saved:", os.path.join(ckpt_save_dir, '/checkpoint.pth'))
print("\n")
end_time = time()
tot_time = end_time - start_time #calculate difference between end time and start time
print('Training completed.')
print('Total time elapsed:' + str(int((tot_time/3600)))+":"+str(int((tot_time%3600)/60))+":"+str(int((tot_time%3600)%60)) )


