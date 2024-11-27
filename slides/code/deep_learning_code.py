import mglearn
import json
import numpy as np
import pandas as pd
import os, sys
from collections import OrderedDict
import torch
import torchvision
from torch import nn, optim
from torchvision import transforms, models, datasets
from PIL import Image
from sklearn.linear_model import LogisticRegression
from torchvision import datasets, models, transforms, utils
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import vgg16
torch.manual_seed(42)

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def classify_image(img, topn = 4):
    clf = vgg16(weights='VGG16_Weights.DEFAULT') # initialize the classifier with VGG16 weights
    preprocess = transforms.Compose([
                 transforms.Resize(299),
                 transforms.CenterCrop(299),
                 transforms.ToTensor(),
                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225]),])

    with open('data/imagenet_classes.txt') as f:
        classes = [line.strip() for line in f.readlines()]
    
    img_t = preprocess(img)
    batch_t = torch.unsqueeze(img_t, 0)
    clf.eval()
    output = clf(batch_t)
    _, indices = torch.sort(output, descending=True)
    probabilities = torch.nn.functional.softmax(output, dim=1)
    d = {'Class': [classes[idx] for idx in indices[0][:topn]], 
         'Probability score': [np.round(probabilities[0, idx].item(),3) for idx in indices[0][:topn]]}
    df = pd.DataFrame(d, columns = ['Class','Probability score'])
    return df


# Attribution: [Code from PyTorch docs](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html?highlight=transfer%20learning)


IMAGE_SIZE = 200
BATCH_SIZE = 64

def read_data(data_dir): 
    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),     
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),            
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),                        
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                        
            ]
        ),
    }

    import os 
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ["train", "valid"]
    }
    
    dataloaders = {}
    
    dataloaders["train"] = torch.utils.data.DataLoader(
            image_datasets["train"], batch_size=BATCH_SIZE, shuffle=True
        )
    
    dataloaders["valid"] = torch.utils.data.DataLoader(
            image_datasets["valid"], batch_size=BATCH_SIZE, shuffle=False
        )
    
    return image_datasets, dataloaders

def get_features(model, train_loader, valid_loader):
    """Extract output of squeezenet model"""
    with torch.no_grad():  # turn off computational graph stuff
        Z_train = torch.empty((0, 1024))  # Initialize empty tensors
        y_train = torch.empty((0))
        Z_valid = torch.empty((0, 1024))
        y_valid = torch.empty((0))
        for X, y in train_loader:
            Z_train = torch.cat((Z_train, model(X)), dim=0)
            y_train = torch.cat((y_train, y))
        for X, y in valid_loader:
            Z_valid = torch.cat((Z_valid, model(X)), dim=0)
            y_valid = torch.cat((y_valid, y))
    return Z_train.detach(), y_train.detach(), Z_valid.detach(), y_valid.detach()


def show_predictions(pipe, Z_valid, y_valid, dataloader, class_names, num_images=20):
    """Display images from the validation set and their predicted labels."""
    images_so_far = 0
    fig = plt.figure(figsize=(15, 25))  # Adjust the figure size for better visualization

    # Convert the features and labels to numpy arrays
    Z_valid = Z_valid.numpy()
    y_valid = y_valid.numpy()

    # Make predictions using the trained logistic regression model
    preds = pipe.predict(Z_valid)

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.cpu()
            for j in range(inputs.size()[0]):
                if images_so_far >= num_images:
                    return
                ax = plt.subplot(num_images // 5, 5, images_so_far + 1)  # 5 images per row
                ax.axis('off')
                ax.set_title(f'Predicted: {class_names[int(preds[images_so_far])]}'
                             f'\nTrue: {class_names[int(y_valid[images_so_far])]}')
                inp = inputs.data[j].numpy().transpose((1, 2, 0))
                mean = np.array([0.5, 0.5, 0.5])
                std = np.array([0.5, 0.5, 0.5])
                inp = std * inp + mean
                inp = np.clip(inp, 0, 1)
                ax.imshow(inp)
                #imshow(inputs.data[j])
                images_so_far += 1
