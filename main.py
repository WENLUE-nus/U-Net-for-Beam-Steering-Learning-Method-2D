import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from math import ceil
import numpy as np
from model import UNet
from loss import ComprehensiveLoss
from data import CustomDataset_Train, CustomDataset_Test
from test_model import test_model
from train_model import train_model

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 32
learning_rate = 3e-4
num_epochs = 50

# Create dataset and DataLoader
dataset = CustomDataset_Train("/media/nus/Disk2/WenLue/Beam-steering-2D/beam_unet/data")
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
model1 = UNet(in_channels=16, out_channels=1)
model2 = UNet(in_channels=5, out_channels=1)

# Path to saved model parameters
model1_path = "/media/nus/Disk2/WenLue/Beam-steering-2D-github/beam_unet/model_para/model_unet_16_1.pth"
model2_path = "/media/nus/Disk2/WenLue/Beam-steering-2D-github/beam_unet/model_para/model_unet_5_1.pth"

model1.load_state_dict(torch.load(model1_path))
model2.load_state_dict(torch.load(model2_path))
model1.to(device)
model2.to(device)

# Define loss function and optimizer
criterion = ComprehensiveLoss(alpha=0.7, beta=0.1, gamma=0.1, delta=0.1)
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)

# Define learning rate scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Save training loss for visualization
train_losses = []
best_loss_model1 = float('inf')
best_loss_model2 = float('inf')

for epoch in range(6):
    dataset_test = CustomDataset_Test("/media/nus/Disk2/WenLue/Beam-steering-2D/beam_unet/data")
    test_model(dataset_test, model1, model2, device, train_losses=train_losses)