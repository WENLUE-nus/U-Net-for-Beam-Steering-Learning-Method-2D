from torchmetrics.functional import structural_similarity_index_measure as ssim
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.filters import threshold_otsu

class ComprehensiveLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2, gamma=0.2, delta=0.1):
        """
        Comprehensive loss function
        Parameters:
            alpha: Weight for MAE loss
            beta: Weight for TBR (Total Variation) loss
            gamma: Weight for SSIM loss
            delta: Weight for PSNR loss
        """
        super(ComprehensiveLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def mae_loss(self, y_pred, y_true):
        """Mean Absolute Error (MAE) loss"""
        return F.l1_loss(y_pred, y_true)

    def tbr_loss(self, y_pred, y_true,):
        device = y_pred.device
        dtype  = y_pred.dtype
        gt_np = y_true.detach().cpu().numpy()
        thr   = threshold_otsu(gt_np)
        Mt = torch.from_numpy((gt_np >= thr).astype('float32')).to(device=device, dtype=dtype)
        Mb = (1.0 - Mt) 
        x = y_pred.abs()
        x = x ** 2
        Nt = Mt.sum().clamp_min(1.0)
        Nb = Mb.sum().clamp_min(1.0)    
        mu_t = (x * Mt).sum() / Nt
        mu_b = (x * Mb).sum() / Nb
        tbr_loss = 10*torch.log10(mu_t/mu_b + 0.259)  # avoid division by zero
        return 1/tbr_loss

    def ssim_loss(self, y_pred, y_true):
        """Structural Similarity Index (SSIM) loss"""
        if y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(1)
            y_true = y_true.unsqueeze(1)
        ssim_value = ssim(y_pred, y_true, data_range=1.0)
        if isinstance(ssim_value, tuple):  # torchmetrics >= 1.3
            ssim_value = ssim_value[0]
        return 1 - ssim_value

    def psnr_loss(self, y_pred, y_true):
        """Peak Signal-to-Noise Ratio (PSNR) loss (reciprocal form for stability)"""
        mse = F.mse_loss(y_pred, y_true)
        psnr_value = 10 * torch.log10(1 / (mse + 1e-8))  # avoid division by zero
        return 1 / psnr_value  # return reciprocal form as loss

    def forward(self, y_pred, y_true):
        """Compute the combined loss"""
        mae = self.mae_loss(y_pred, y_true)
        tbr_l = self.tbr_loss(y_pred, y_true)
        ssim_l = self.ssim_loss(y_pred, y_true)
        psnr_l = self.psnr_loss(y_pred, y_true)

        # Total combined loss
        total_loss = self.alpha * mae + self.beta * tbr_l + self.gamma * ssim_l + self.delta * psnr_l
        print(f"Loss components - MAE: {mae.item():.4f}, TBR: {tbr_l.item():.4f}, SSIM: {ssim_l.item():.4f}, PSNR: {psnr_l.item():.4f}")
        return total_loss
