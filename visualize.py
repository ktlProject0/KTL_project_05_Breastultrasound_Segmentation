import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from model import Net
from dataset import CustomDataset

def plot_sample(input_image, gt_image, pred_image, output_path):
    """
    Combines input, ground truth, prediction, input+ground truth, and input+prediction into one visualization.
    """
    # Permute input image to (H, W, C) for visualization if it's RGB
    if input_image.ndim == 3 and input_image.shape[0] == 3:
        input_image = input_image.transpose(1, 2, 0)  # Convert (C, H, W) to (H, W, C)

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    axes[0].imshow(input_image, cmap='gray' if input_image.ndim == 2 else None)
    axes[0].set_title("Input")
    axes[0].axis('off')

    axes[1].imshow(gt_image, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_image, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    axes[3].imshow(input_image, cmap='gray' if input_image.ndim == 2 else None)
    axes[3].imshow(gt_image, cmap='gray', alpha=0.5)
    axes[3].set_title("Input + GT")
    axes[3].axis('off')

    axes[4].imshow(input_image, cmap='gray' if input_image.ndim == 2 else None)
    axes[4].imshow(pred_image, cmap='gray', alpha=0.5)
    axes[4].set_title("Input + Prediction")
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualization of Segmentation')
    parser.add_argument('--data_direc', type=str, default='./data/test', help="Path to test dataset")
    parser.add_argument('--n_classes', type=int, default=1, help="Number of classes")
    parser.add_argument('--cuda', action='store_true', help='Use CUDA')
    parser.add_argument('--seed', type=int, default=42, help='Random seed to use. Default=123')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='Path to save/load the model')
    parser.add_argument('--output_dir', type=str, default='./test_results', help='Directory to save visualization results')
    opt = parser.parse_args()

    if not os.path.isdir(opt.model_save_path):
        raise Exception("Checkpoints not found, please run train.py first")

    os.makedirs(opt.output_dir, exist_ok=True)

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if opt.cuda and torch.cuda.is_available() else "cpu")

    # Load model
    model = Net(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    # Load dataset
    test_dataset = CustomDataset(direc=opt.data_direc, mode='eval')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"[INFO] Total test samples: {len(test_dataset)}")

    # Process 5 samples
    for idx, data in enumerate(test_loader):
        if idx >= 5:
            break

        input_image = data['input'].squeeze().numpy()
        gt_image = data['target'].squeeze().numpy()

        # Prediction
        with torch.no_grad():
            pred_logit = model(data['input'].to(device).float())
            pred_image = (pred_logit > 0.5).float().squeeze().cpu().numpy()

        # Save visualization
        output_path = os.path.join(opt.output_dir, f'sample_{idx + 1}.png')
        plot_sample(input_image, gt_image, pred_image, output_path)
        print(f"[INFO] Saved visualization for sample {idx + 1} at {output_path}")
