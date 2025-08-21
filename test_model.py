import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch.nn.functional as F
import random
import torch
import scipy.io as sio


def test_model(dataset_test, model1, model2, device, train_losses=None):
    # Randomly select an index
    random_idx = random.randint(0, len(dataset_test) - 1)
    
    # Get a random sample
    input_data, fusion_data, ground_truth = dataset_test[random_idx]  # Note: dataset returns fusion_data
    input_data = input_data.unsqueeze(0)  # Add batch dimension
    ground_truth = ground_truth.unsqueeze(0)
    
    # Move data to device
    input_data, fusion_data, ground_truth = input_data.to(device), fusion_data.to(device), ground_truth.to(device)
    
    # Set models to evaluation mode
    model1.eval()
    model2.eval()
    
    # Forward pass through model1 and model2
    with torch.no_grad():  # Disable gradient computation
        # Step 1: Send 16×150×200 input into model1 for each of the 5 algorithms
        batch_outputs = []
        for algo_idx in range(input_data.size(1)):  # Loop over 5 algorithms
            algo_input = input_data[:, algo_idx]  # [1, 16, 150, 200]
            algo_input_resized = F.interpolate(algo_input, size=(160, 224), mode='bilinear', align_corners=False)  # [1, 16, 160, 224]
            algo_output = model1(algo_input_resized)  # [1, 1, 160, 224]
            algo_output_resized = F.interpolate(algo_output, size=(150, 200), mode='bilinear', align_corners=False)  # [1, 1, 150, 200]
            batch_outputs.append(algo_output_resized)
        
        # Stack all algorithm outputs: [1, 5, 150, 200]
        outputs_1 = torch.cat(batch_outputs, dim=1)
    
        # Step 2: Feed [1, 5, 150, 200] into model2
        outputs_1_resized = F.interpolate(outputs_1, size=(160, 224), mode='bilinear', align_corners=False)  # [1, 5, 160, 224]
        final_output = model2(outputs_1_resized)  # [1, 1, 160, 224]
        final_output_resized = F.interpolate(final_output, size=(150, 200), mode='bilinear', align_corners=False)  # [1, 150, 200]
    
    # Plot visualizations
    fig = plt.figure(figsize=(20, 12))  # Set figure size
    gs = gridspec.GridSpec(4, 8, figure=fig)  # Create 4x8 grid layout
    
    # 1. Plot Loss curve
    ax_loss = fig.add_subplot(gs[0, 4:])  # First row, columns 4 to 7
    if train_losses is not None:
        ax_loss.plot(range(1, len(train_losses) + 1), train_losses, marker='o', linestyle='-', color='r', label='Train Loss')
    ax_loss.set_title("Loss Curve", fontsize=12)
    ax_loss.set_xlabel("Epochs", fontsize=10)
    ax_loss.set_ylabel("Loss", fontsize=10)
    # ax_loss.legend()
    
    # 2. Left: plot 16 angle images (4x4 subplots)
    input_data = input_data.squeeze(0)
    num_angles = input_data.shape[1]
    for i in range(num_angles):
        ax = fig.add_subplot(gs[i // 4, i % 4])  # Subplots in the 4x4 grid
        ax.imshow(input_data[0, i].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        ax.set_title(f"Angle {i+1}", fontsize=10)
        ax.axis('off')
    
    # 3. Show fusion images of each algorithm (5 images) and ground truth
    algo_names = ["AIHT", "AMP", "FISTA", "Matched_filter", "ROMP"]
    
    # Right: draw 5 fusion images + 1 ground truth (in 3x3 layout)
    for idx, algo in enumerate(algo_names):
        row = idx % 3 + 1  # Row index
        col = (idx // 3) * 2 + 4  # Column index
        ax_fusion = fig.add_subplot(gs[row, col:col+1])  # Fusion image subplot
        ax_fusion.imshow(fusion_data[idx].cpu().numpy(), cmap='hot', vmin=0, vmax=1)
        ax_fusion.set_title(f"{algo_names[idx]}_Input", fontsize=10)
        ax_fusion.axis('off')
        plt.colorbar(ax_fusion.images[0], ax=ax_fusion, fraction=0.046, pad=0.04)
    
    # Plot outputs from model1 for 5 algorithms
    for i in range(5):
        row = i % 3 + 1
        col = (i // 3) * 2 + 5
        ax_fusion = fig.add_subplot(gs[row, col:col + 1])
        ax_fusion.imshow(outputs_1[0, i].cpu().numpy(), cmap="hot", vmin=0, vmax=1)
        ax_fusion.set_title(f"{algo_names[i]}_Unet1", fontsize=10)
        ax_fusion.axis("off")
        plt.colorbar(ax_fusion.images[0], ax=ax_fusion, fraction=0.046, pad=0.04)
    
    # 4. Plot ground truth
    ax_gt = fig.add_subplot(gs[3, 6])
    ax_gt.imshow(ground_truth.squeeze().cpu().numpy(), cmap="hot", vmin=0, vmax=1)
    ax_gt.set_title("Ground Truth", fontsize=10)
    ax_gt.axis("off")
    plt.colorbar(ax_gt.images[0], ax=ax_gt, fraction=0.046, pad=0.04)
    
    # 5. Plot final prediction
    ax_pred = fig.add_subplot(gs[3, 7])
    ax_pred.imshow(final_output_resized.squeeze().cpu().numpy(), cmap="hot", vmin=0, vmax=1)
    ax_pred.set_title("Final Prediction", fontsize=10)
    ax_pred.axis("off")
    plt.colorbar(ax_pred.images[0], ax=ax_pred, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(f"visualization_result_idx_{random_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save intermediate results to .mat
    save_dict = {
        'input_data': input_data.cpu().numpy(),            # [5, 16, 150, 200]
        'unet1_outputs': outputs_1.squeeze(0).cpu().numpy(),        # [5, 150, 200]
        'final_output': final_output_resized.squeeze(0).cpu().numpy(),  # [150, 200]
        'ground_truth': ground_truth.squeeze(0).cpu().numpy(),      # [150, 200]
    }

    save_path = f"visualization_result_idx_{random_idx}.mat"
    sio.savemat(save_path, save_dict)
    print(f"Saved visualization results to: {save_path}")
