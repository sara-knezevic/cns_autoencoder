import csv
from itertools import product
import os
import numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import torch
from utils.Preprocessing import Preprocessing
import utils.utils as utils
from train import single_fold_validated_training
import optuna
from scipy.stats import zscore

def autoencoder_training():
    # Check for folders.
    utils.check_folders('models')
    utils.check_folders('results')
    # utils.check_folders('plots')

    preprocessor = Preprocessing()
    
    # Set randomness seed.
    seed_number = 42
    utils.set_seed(seed_number)

    # Load the data.
    mat_data = loadmat('./data/laufs_sleep.mat')
    assert mat_data['TS_N1'].shape == mat_data['TS_N2'].shape == mat_data['TS_N3'].shape == mat_data['TS_W'].shape

    n1_data = mat_data['TS_N1'][0][1:]
    n2_data = mat_data['TS_N2'][0][1:]
    n3_data = mat_data['TS_N3'][0][1:]
    wake_data = mat_data['TS_W'][0][1:]

    n1_data_shortened, _ = preprocessor.shorten_data(n1_data, final_length=200)
    n2_data_shortened, _ = preprocessor.shorten_data(n2_data, final_length=200)
    n3_data_shortened, _ = preprocessor.shorten_data(n3_data, final_length=200)
    wake_data_shortened, _ = preprocessor.shorten_data(wake_data, final_length=200)

    participants_per_state = {
        'N1': n1_data_shortened,
        'N2': n2_data_shortened,
        'N3': n3_data_shortened,
        'Wake': wake_data_shortened
    }

    val_data_list = []
    train_data_list = []

    for state, data in participants_per_state.items():
        val_participant = data[:3]
        val_data_list.append(val_participant)
        train_data_list.append(data[3:])

    train_data = zscore(np.concatenate(train_data_list, axis=0))
    val_data = zscore(np.concatenate(val_data_list, axis=0))

    train_data = train_data.reshape(-1, 80)
    val_data = val_data.reshape(-1, 80)

    train_tensor_data = torch.tensor(train_data, dtype=torch.float32)
    val_tensor_data = torch.tensor(val_data, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(train_tensor_data)
    val_dataset = torch.utils.data.TensorDataset(val_tensor_data)

    # !--- TRAINING AND VALIDATION ---!

    # OPTUNA SEARCH

    # def objective(trial, keyword="optuna_with_div"):
    #     lbd_mse = trial.suggest_float("lbd_mse", 0.0001, 1)
    #     lbd_corr = trial.suggest_float("lbd_corr", 0.0001, 1)
    #     lbd_diff = trial.suggest_float("lbd_diff", 0.0001, 1)
    #     lbd_div = trial.suggest_float("lbd_div", 0.0001, 1)
    #     lr = trial.suggest_float("lr", 0.0001, 0.1)
    #     latent_dim = 10

    #     with open(os.path.join("./results/", f"{keyword}_{trial.number}.csv"), mode='w', newline='') as file:
    #         writer = csv.writer(file)

    #         if file.tell() == 0:
    #             header = ["lat", "lbd", "file_name", "train_loss", "val_loss", 
    #                       "fc_diff_train", "fc_diff_val", "corr_train", "corr_val"]
    #             writer.writerow(header)

    #         best_results = single_fold_validated_training(train_dataset, val_dataset, csv_writer=writer,
    #                                                     batch_size=200, num_epochs=500, keyword=keyword,
    #                                                     lbd_mse=lbd_mse, lbd_corr=lbd_corr,
    #                                                     lbd_diff=lbd_diff, lbd_div=lbd_div,
    #                                                     lr=lr, latent_dim=latent_dim)
        
    #     return best_results[5]

    # study = optuna.create_study(direction="minimize", study_name="optuna_study_div",
    #                             storage="sqlite:///optuna_study_div.db", load_if_exists=True)
    # study.optimize(objective, n_trials=100, show_progress_bar=True)

    os.makedirs("./results/", exist_ok=True)

    lbd_mse = 1
    lbd_corr = [0.005, 0]
    lbd_diff = [0.005, 0]
    lbd_ssim = [0.005, 0]
    lbd_div = 0.1
    lr = 0.001

    for l_corr, l_diff, l_ssim in product(lbd_corr, lbd_diff, lbd_ssim):
        keyword = f"mse_corr{l_corr}_diff{l_diff}_ssim{l_ssim}_div{lbd_div}"

        with open(os.path.join("./results/", f"{keyword}.csv"), mode='w', newline='') as file:
            writer = csv.writer(file)

            if file.tell() == 0:
                header = ["lat", "lbd", "file_name", "train_loss", "val_loss", 
                        "fc_diff_train", "fc_diff_val", "corr_train", "corr_val"]
                writer.writerow(header)
            
            latent_dims = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

            for lat_dim in latent_dims:
                print(f"Training for latent dimension {lat_dim}...")

                best_results = single_fold_validated_training(train_dataset, val_dataset, csv_writer=writer,
                                                                batch_size=200, num_epochs=500, keyword=f"{keyword}",
                                                                lbd_mse=lbd_mse, lbd_corr=l_corr, lbd_ssim=l_ssim,
                                                                lbd_diff=l_diff, lbd_div=lbd_div,
                                                                lr=lr, latent_dim=lat_dim)
                
                print(f"Best results for latent dimension {lat_dim}:")
                print(f"Train loss: {best_results[3]}")
                print(f"Val loss: {best_results[4]}")
                print(f"FC diff train: {best_results[5]}")
                print(f"FC diff val: {best_results[6]}")
                print(f"Corr train: {best_results[7]}")
                print(f"Corr val: {best_results[8]}\n")
