import os
import random
import copy, sys
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import torch, csv

from tqdm import tqdm
from utils.Preprocessing import Preprocessing
from utils.single_epoch import train_epoch, validate_epoch
from utils.metrics import calculate_fc_diff, calculate_ssim_fc, calculate_correlation_fc
from networks.Autoencoder import Autoencoder
import numpy as np
from networks.LossCriterion import CombinedLoss
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from skimage.metrics import structural_similarity as ssim
import seaborn as sns

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def single_fold_validated_training(train_dataset, val_dataset, csv_writer, fold=0,
                                   batch_size=200, num_epochs=100,
                                   lbd_mse=1.0, lbd_corr=1.0, lbd_ssim=1.0,
                                   lbd_diff=1.0, lbd_div=1.0,
                                   lr=1e-3, latent_dim=10,
                                   keyword="results"):

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers = 0, worker_init_fn=seed_worker)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                                    num_workers = 0, worker_init_fn=seed_worker)

        best_results = []
        best_val_loss = np.inf
        best_model_wts = None
        wait = 0

        model = Autoencoder(latent_dim=latent_dim)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model.to(device)

        criterion = CombinedLoss(lambda_mse=lbd_mse, lambda_corrcoef=lbd_corr, lambda_ssim=lbd_ssim,
                                 lambda_diff=lbd_diff, lambda_diversity=lbd_div)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

        for epoch in range(num_epochs):
            train_loss, train_metrics, last_fc_inputs_train, last_fc_outputs_train = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            val_loss, val_metrics, last_fc_inputs_val, last_fc_outputs_val = validate_epoch(model, val_loader, criterion, device)

            # Log results
            csv_writer.writerow([latent_dim, 0, keyword, train_loss, val_loss, train_metrics[0], val_metrics[0], train_metrics[1], val_metrics[1]])

            # early stopping based on val loss
            if val_loss < best_val_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_val_loss = val_loss

                best_results = [
                    latent_dim, 0, f'AE_{keyword}_lat{latent_dim}.pt',
                    train_loss, val_loss, 
                    train_metrics[0], val_metrics[0],
                    train_metrics[1], val_metrics[1]
                ]

                wait = 0
            else:
                wait += 1

            if (wait > 30):
                print(f'Early stopped on epoch: {epoch}.')

                break

        os.makedirs(f"./models/{keyword}", exist_ok=True)

        # Save model weights.
        if best_model_wts is not None:
            torch.save(best_model_wts, f'./models/{keyword}/AE_{keyword}_lat{latent_dim}.pt')
        
        return best_results