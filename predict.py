import matplotlib.pyplot as plt # to plot graphs 
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F  
from torchvision import datasets, transforms, models
from torch.autograd import Variable

import time # to measure and set time-based functions
import os, sys # to include some os functions in order to work with files, etc.

from PIL import Image  # to import/transform images
from collections import OrderedDict
import random
import glob

from get_input_parse_predict import get_input_parse_predict

import json

with open('cat_to_name.json', 'r') as f:
    catg_to_name = json.load(f)

args_pred = get_input_parse_predict()

ckpt_path = args_pred.checkpoint
print('Loading checkpoint:', ckpt_path)
top_k = args_pred.top_k

# Use GPU if selected and available
if args_pred.gpu:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define function: load dictionary based checkpoint 

def load_checkpoint(filepath = ckpt_path):
    
    checkpoint =  torch.load(filepath, map_location = "cpu")
    model = models.vgg16(pretrained=True)
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.003) 
    
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.epochs = checkpoint['epochs']
    model.learning_rate = checkpoint['learning_rate']
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    model.batch_size = checkpoint['batch_size']
    model.class_to_idx = checkpoint['class_to_idx']
    optimizer.load_state_dict(checkpoint['optimizer_dict']) 

    for param in model.parameters():
        param.requires_grad = False
    
    return model, optimizer, checkpoint['class_to_idx']


# Load checkpoint
model, optimizer, class_to_idx = load_checkpoint(ckpt_path)

def process_image(img_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Process a PIL image for use in a PyTorch model
      
    minsize = 256
    crop_h = 224
    crop_v = 224
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = Image.open(img_path)  # open image via given path
    
    width, height = img.size # extract size of the image 
    
    if width > height:  # choose the smallest/biggest side as a reference to calculate
        w2 = int((width/height) * minsize)  # adapt sizes to minsize
        h2 = int(minsize)
        
    else:
        h2 = int((height/width) * minsize) 
        w2 = int(minsize)
    
    img = img.resize((w2, h2))
       
    left = int((img.size[0]/2)-(crop_h/2))    # code adapted from StackOverflow
    upper = int((img.size[1])/(2-crop_v/2))   # divides size by 2 to get the center of the image and defines  
    right = left +crop_h                      # the crop sizes depending on the crop_ size variable
    lower = upper + crop_v
    
    img_cropped = img.crop((left, upper, right, lower))  #crop
    
    np_img = np.array(img_cropped)/255   # convert to array and divide by RGB
    
    np_img = ((np_img - mean) / std)  # normalization
    np_img = np_img.transpose(2, 0, 1)  # reorder dimensions
    
    np_img = torch.from_numpy(np_img) # transforms img into a tensor
            
    return np_img

def imshow(image, ax=None, title=None):
    ### Imshow for Tensor
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=top_k):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''       
   
    # Use GPU if it is available
    model = model.to(device)
    
    # Set mode to evaluation
    model.eval() 
    
    # Load image and convert to tensor
    image = process_image(image_path)
    
    if device == 'cpu':  
            
        img_tensor = image.type(torch.FloatTensor)
        
    else:
        img_tensor = image.type(torch.cuda.FloatTensor)
    
    # avoiding RuntimeError: expected stride to be a single integer value or a list of 1 values to match the convolution dimensions, but got stride=[1, 1]
    # source: https://discuss.pytorch.org/t/expected-stride-to-be-a-single-integer-value-or-a-list/17612 
    # taken out of the discussion forum @Udacity, https://knowledge.udacity.com/questions/53053

    img_tensor.unsqueeze_(0)
       
    
    # Iterate through images in testloader and calculate probabilities without gradients
    with torch.no_grad():
        
        # use model and convert log to linear probabilities
        ps = torch.exp(model.forward(img_tensor.to(device)))
                       
        top_ps, top_labels = ps.topk(topk, dim=1)

    
    return top_ps, top_labels
        
# Predict classes for loaded image 
img_path = args_pred.img_path
ps, classes = predict(img_path, model)

# Convert results to lists
results_probs = ps.tolist()[0]
results_classes = classes.tolist()[0]
results = zip(results_classes, results_probs)

# Iterate through lists to lookup the names and probs and assign them to list catg
ind = []

# Display the top_k classes

for i in range(len(model.class_to_idx.items())):
    ind.append(list(model.class_to_idx.items())[i])

for result in results:
    print("{}: {:.2%}".format(catg_to_name[ind[result[0]][0]].capitalize(), result[1]))

