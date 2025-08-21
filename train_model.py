import torch
import torch.nn as nn
import torch.nn.functional as F


def train_model(dataloader, model1, model2, criterion, optimizer, device):
    model1.to(device)
    model2.to(device)
    model1.train()
    model2.train()

    epoch_loss = 0
    num_batches = 0
    
    for inputs, fusion_data, targets in dataloader:
        # Move data to device
        inputs, targets = inputs.to(device), targets.to(device)  # inputs: [batch, 5, 16, 150, 200]
    
        # Initialize list to store outputs of each algorithm
        batch_outputs = []
    
        # Step 1: For each algorithm, send 16×150×200 input into model1
        for algo_idx in range(inputs.size(1)):  # Loop over 5 algorithms
            algo_input = inputs[:, algo_idx]  # [batch, 16, 150, 200]
            
            # Resize each algorithm's input to 160×224 and send into model1
            algo_input_resized = F.interpolate(algo_input, size=(160, 224), mode='bilinear', align_corners=False)  # [batch, 16, 160, 224]
            algo_output = model1(algo_input_resized)  # [batch, 1, 160, 224]
            
            # Resize output back to target size [batch, 1, 150, 200]
            algo_output_resized = F.interpolate(algo_output, size=(150, 200), mode='bilinear', align_corners=False)
            batch_outputs.append(algo_output_resized)
    
        # Stack outputs of all algorithms -> [batch, 5, 150, 200]
        outputs_1 = torch.cat(batch_outputs, dim=1)
    
        # Step 2: Send [batch, 5, 150, 200] into model2
        outputs_1_resized = F.interpolate(outputs_1, size=(160, 224), mode='bilinear', align_corners=False)
        final_output = model2(outputs_1_resized)  # Output: [batch, 1, 160, 224]
        final_output_resized = F.interpolate(final_output, size=(150, 200), mode='bilinear', align_corners=False)
    
        # Compute loss
        loss = criterion(final_output_resized, targets)  # Target shape: [batch, 150, 200]
    
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        # Accumulate loss and count batches
        epoch_loss += loss.item()
        num_batches += 1

    # Compute and return average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    return avg_epoch_loss
