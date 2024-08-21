import torch
from utils.metrics import calculate_correlation_fc, calculate_fc_diff, calculate_ssim_fc

class CombinedLoss(torch.nn.Module):
    def __init__(self, lambda_mse=1.0, lambda_corrcoef=1.0, lambda_diff=1.0,
                 lambda_ssim=1.0, lambda_diversity=1.0):
        super(CombinedLoss, self).__init__()
        self.mse = torch.nn.MSELoss()

        self.lambda_mse = lambda_mse
        self.lambda_corr = lambda_corrcoef
        self.lambda_diff = lambda_diff
        self.lambda_diversity = lambda_diversity
        self.lambda_ssim = lambda_ssim

    def forward(self, outputs, targets, fc_outputs, fc_inputs):
        loss_mse = self.mse(outputs, targets)
        loss_diff = torch.nn.functional.mse_loss(fc_outputs, fc_inputs)
        loss_corrcoef = torch.corrcoef(torch.stack((fc_outputs.flatten(), fc_inputs.flatten())))[0, 1]
        loss_ssim = calculate_ssim_fc(fc_outputs, fc_inputs)
        loss_diversity = self.diversity_loss(outputs)
        
        total_loss = (
            self.lambda_mse * loss_mse +
            self.lambda_diff * loss_diff +
            self.lambda_corr * (1 - loss_corrcoef) +
            self.lambda_ssim * (1 - loss_ssim) +
            self.lambda_diversity * loss_diversity
        )
        
        return total_loss
    
    def diversity_loss(self, outputs):
        if outputs.size(0) > 1:
            return -torch.std(outputs)
        else:
            return 0