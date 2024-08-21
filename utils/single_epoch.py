from matplotlib import pyplot as plt
import numpy as np
import torch
from utils.metrics import calculate_correlation_fc, calculate_fc_diff, calculate_ssim_fc, compute_fc_matrix_regular
import seaborn as sns
from skimage.metrics import structural_similarity as ssim

def train_epoch(model, train_loader, criterion, optimizer, scheduler, device):
    model.train()
    lat_dim = model.latent_dim

    running_loss = 0.0
    last_fc_inputs, last_fc_outputs = None, None

    for inputs in train_loader:
        optimizer.zero_grad()
        
        inputs = inputs[0].to(device)  # Ensure inputs are on the correct device
        outputs, latent = model(inputs)

        # Ensure inputs are properly shaped for correlation computation
        last_fc_inputs = compute_fc_matrix_regular(inputs)
        last_fc_outputs = compute_fc_matrix_regular(outputs)
        # last_fc_latent = compute_fc_matrix_regular(latent, num_brain_nodes=lat_dim)

        fc_diff = torch.nn.functional.mse_loss(last_fc_outputs, last_fc_inputs).item()
        fc_corr = torch.corrcoef(torch.stack((last_fc_outputs.flatten(), last_fc_inputs.flatten())))[0, 1].item()

        metrics = [fc_diff, fc_corr]
            
        loss = criterion(outputs, inputs, last_fc_outputs, last_fc_inputs)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    average_loss = running_loss / len(train_loader)

    if scheduler is not None:
        scheduler.step(average_loss)

    return average_loss, metrics, last_fc_inputs, last_fc_outputs

def validate_epoch(model, val_loader, criterion, device):
    model.eval()
    lat_dim = model.latent_dim

    running_loss = 0.0
    last_fc_inputs, last_fc_outputs = None, None

    with torch.no_grad():
        for inputs in val_loader:
            inputs = inputs[0].to(device)  # Ensure inputs are on the correct device
            outputs, latent = model(inputs)

            # Ensure inputs are properly shaped for correlation computation
            last_fc_inputs = compute_fc_matrix_regular(inputs)
            last_fc_outputs = compute_fc_matrix_regular(outputs)

            fc_diff = torch.nn.functional.mse_loss(last_fc_outputs, last_fc_inputs).item()
            fc_corr = torch.corrcoef(torch.stack((last_fc_outputs.flatten(), last_fc_inputs.flatten())))[0, 1].item()

            metrics = [fc_diff, fc_corr]

            loss = criterion(outputs, inputs, last_fc_outputs, last_fc_inputs)

            running_loss += loss.item()
            
    average_loss = running_loss / len(val_loader)

    return average_loss, metrics, last_fc_inputs, last_fc_outputs