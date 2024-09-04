# basic imports
import numpy as np

# DL library imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.psp import PSPNet

# libraries for loading image, plotting
import cv2
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp



output_path = 'dataset'


##########################
# Data Loading
##########################

# utility functions to get BDD100k Pytorch dataset and dataloaders
from utils import get_datasets, get_dataloaders
images = np.load("dataset/image_180_320.npy")
labels = np.load("dataset/label_180_320.npy")

train_set, val_set, test_set= get_datasets(images, labels)
sample_image, sample_label = train_set[0]
print(f"There are {len(train_set)} train images, {len(val_set)} validation images, {len(test_set)} test Images")
print(f"Input shape = {sample_image.shape}, output label shape = {sample_label.shape}")

train_dataloader, val_dataloader, test_dataloader = get_dataloaders(train_set, val_set, test_set)


from collections import namedtuple

# Each label is a tuple with name, class id and color
Label = namedtuple( "Label", [ "name", "train_id", "color"])
drivables = [
             Label("direct", 0, (206, 44, 40)),        # red
             Label("alternative", 1, (44, 108, 239)),  # cyan
             Label("background", 2, (0, 0, 0)),        # black
            ]

train_id_to_color = [c.color for c in drivables if (c.train_id != -1 and c.train_id != 255)]
train_id_to_color = np.array(train_id_to_color)


##############
# Training
##############


from utils import meanIoU                  # metric class
from utils import plot_training_results    # function to plot training curves
from utils import evaluate_model           # evaluation function
from utils import train_validate_model     # train validate function
from utils import pspnet_loss



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# MODEL HYPERPARAMETERS
N_EPOCHS = 50
NUM_CLASSES = 3
MAX_LR = 3e-4
MODEL_NAME = 'PSPNet_resnet50_aux'

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
criterion = pspnet_loss(num_classes = NUM_CLASSES, aux_weight=0.4)

# create model, optimizer, lr_scheduler and pass to training function
model = PSPNet(in_channels=3, num_classes=NUM_CLASSES, use_aux=True).to(device)
optimizer = optim.Adam(model.parameters(), lr=MAX_LR)
scheduler = OneCycleLR(optimizer, max_lr= MAX_LR, epochs = N_EPOCHS,steps_per_epoch = len(train_dataloader),
                       pct_start=0.3, div_factor=10, anneal_strategy='cos')

## Run Train/Evaluate Function
_ = train_validate_model(model, N_EPOCHS, MODEL_NAME, criterion, optimizer,
                         device, train_dataloader, val_dataloader, meanIoU, 'meanIoU',
                         NUM_CLASSES, lr_scheduler = scheduler, output_path = output_path)
