import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from model import Net
from torchvision.datasets.folder import default_loader
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Malignant Segmentation')
    parser.add_argument('--data_path', type=str, default='/home/pwrai/userarea/hansung3/KTL_project_05_Breastultrasound_Segmentation/data/test/images/malignant (124).png', help="data path")
    parser.add_argument('--n_classes', type=int, default=1, help="number of classes (set to 1 if model was trained for binary segmentation)")
    parser.add_argument('--cuda', action='store_true', help='use cuda?')
    parser.add_argument('--seed', type=int, default=42, help='random seed to use. Default=123')
    parser.add_argument('--model_save_path', type=str, default='./checkpoints', help='path to load the model')
    opt = parser.parse_args()

    if not os.path.isdir(opt.model_save_path):
        raise Exception("Checkpoints not found, please run train.py first")

    os.makedirs("Visualize_result", exist_ok=True)

    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")

    torch.manual_seed(opt.seed)
    device = 'cuda'

    # Load model and set to eval mode
    model = Net(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device), strict=False)
    model.eval()

    if not os.path.isdir(opt.model_save_path):
        raise Exception("checkpoints not found, please run train.py first")

    os.makedirs("Visualize_result", exist_ok=True)
    
    if opt.cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run without --cuda")
    
    torch.manual_seed(opt.seed)
    
    if opt.cuda:
        device = torch.device("cuda:1")
    else:
        device = torch.device("cpu")
    
    model = Net(n_classes=opt.n_classes).to(device)
    model.load_state_dict(torch.load(os.path.join(opt.model_save_path, 'model_statedict.pth'), map_location=device))
    model.eval()

    transform = A.Compose([
        A.Resize(width=128, height=128),
        A.Normalize(mean=0.5, std=0.5, max_pixel_value=1.0, always_apply=True, p=1.0),
        ToTensorV2(transpose_mask=True)
    ])

    image = (np.array(default_loader(opt.data_path)) / 255.).astype(np.float32)
    sample_origin_shape = image.shape
    transformed = transform(image=image)
    sample_input = transformed['image']

    with torch.no_grad():
        pred_logit = model(sample_input[None].to(device))
        pred = (pred_logit > 0.5).float().squeeze().detach().cpu()

    origin_size_pred = torch.nn.functional.interpolate(pred[None, None].float(), size=sample_origin_shape[:2], mode='nearest').squeeze().numpy()

    cbct = sample_input.clone()
    cbct_visualize = cbct.permute(1, 2, 0).squeeze().detach().cpu().clone()
    cbct_visualize = ((cbct_visualize * 0.5 + 0.5) * 255).numpy().astype(np.uint8)
    cbct_visualize_contour = cbct_visualize.copy()
    contours, hier = cv2.findContours((pred.numpy().astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(cbct_visualize_contour, [largest_contour], -1, (0, 0, 255), 2, cv2.LINE_8)

    cv2.imwrite(f"Visualize_result/{opt.data_path.split('/')[-1]}", cbct_visualize[..., ::-1])
    cv2.imwrite(f"Visualize_result/{opt.data_path.split('/')[-1].split('.')[0]}_visualize.{opt.data_path.split('/')[-1].split('.')[-1]}", cbct_visualize_contour[..., ::-1])
