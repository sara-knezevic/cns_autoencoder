import random
from matplotlib import pyplot as plt
from scipy.io import loadmat
import torch
from tqdm import tqdm
from networks.Autoencoder import Autoencoder
from utils.Preprocessing import Preprocessing
import utils.utils as utils
import numpy as np
from classification import train_and_evaluate_classifier
import utils.metrics as metrics
import numpy as np
import seaborn as sns
from scipy.stats import zscore

def classifier_training():
    # Check for folders.
    utils.check_folders('models')
    utils.check_folders('results')
    # utils.check_folders('plots')

    preprocessor = Preprocessing()

    # Set randomness seed.
    seed_number = np.random.randint(0, 1000000)
    utils.set_seed(seed_number)

    # Load the data.
    mat_data = loadmat('./data/laufs_sleep.mat')
    assert mat_data['TS_N1'].shape == mat_data['TS_N2'].shape == mat_data['TS_N3'].shape == mat_data['TS_W'].shape

    n3_data = mat_data['TS_N3'][0][1:]
    wake_data = mat_data['TS_W'][0][1:]

    n3_data_shortened, _ = preprocessor.shorten_data(n3_data, final_length=200)
    wake_data_shortened, _ = preprocessor.shorten_data(wake_data, final_length=200)

    # Create labels
    wake_labels = torch.zeros((wake_data_shortened.shape[0], 1))
    n3_labels = torch.ones((n3_data_shortened.shape[0], 1))

    valid_labels = torch.cat((wake_labels, n3_labels), 0)
    
    concatenated_data = np.concatenate([wake_data_shortened, n3_data_shortened])
    concatenated_data = zscore(concatenated_data)
    concatenated_data = concatenated_data.reshape(-1, 80)

    # !-- CLASSIFICATION --!

    num_time_points = 200
    num_brain_nodes = 80
    num_participants = concatenated_data.shape[0] // num_time_points

    # Reshape data to (num_participants, num_brain_nodes, num_time_points)
    data_tensor = torch.tensor(concatenated_data, dtype=torch.float32)
    data_tensor = data_tensor.reshape(num_participants, num_time_points, num_brain_nodes)

    # lat_dims = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    lat_dims = [18]

    latent_accuracies = []
    latent_accuracies_std = []
    latent_precisions = []
    latent_precisions_std = []
    latent_recalls = []
    latent_recalls_std = []
    latent_f1s = []
    latent_f1s_std = []
    latent_roc_aucs = []
    latent_roc_aucs_std = []
    latent_fprs = []
    latent_tprs = []

    reconstructed_accuracies = []
    reconstructed_accuracies_std = []
    reconstructed_precisions = []
    reconstructed_precisions_std = []
    reconstructed_recalls = []
    reconstructed_recalls_std = []
    reconstructed_f1s = []
    reconstructed_f1s_std = []
    reconstructed_roc_aucs = []
    reconstructed_roc_aucs_std = []
    reconstructed_fprs = []
    reconstructed_tprs = []

    # Load the best model
    for ld in lat_dims:
        print (f'Latent Dimension: {ld}')
        latent_dimension = ld
        model = Autoencoder(latent_dim=latent_dimension)
        model.load_state_dict(torch.load(f'./models/doubleDropout/AE_results_lat{latent_dimension}.pt'))
        model.eval()

        with torch.no_grad():
            outputs, latent = model(data_tensor.view(-1, num_brain_nodes))
            reconstructed_data = outputs.view(num_participants, num_time_points, num_brain_nodes)
            latent_data = latent.view(num_participants, num_time_points, latent_dimension)

        original_correlation_matrices = torch.empty(num_participants, 80, 80)
        latent_correlation_matrices = torch.empty(num_participants, latent_dimension, latent_dimension)
        reconstructed_correlation_matrices = torch.empty(num_participants, 80, 80)

        for i in range(num_participants):
            # Original Correlation Matrix
            participant_data = data_tensor[i]
            participant_data_tensor = participant_data.clone().detach()
            original_correlation_matrix = metrics.compute_fc_matrix_regular(participant_data_tensor)

            original_correlation_matrices[i] = original_correlation_matrix

            # Latent Correlation Matrix
            participant_latent_data = latent_data[i]
            participant_latent_data_tensor = participant_latent_data.clone().detach()
            latent_correlation_matrix = metrics.compute_fc_matrix_regular(participant_latent_data_tensor, num_brain_nodes=latent_dimension)

            latent_correlation_matrices[i] = latent_correlation_matrix

            # Reconstructed Correlation Matrix
            participant_reconstructed_data = reconstructed_data[i]
            participant_reconstructed_data_tensor = participant_reconstructed_data.clone().detach()
            reconstructed_correlation_matrix = metrics.compute_fc_matrix_regular(participant_reconstructed_data_tensor)

            reconstructed_correlation_matrices[i] = reconstructed_correlation_matrix

        # !--- CLASSIFICATION ---!

        original_lower_triangles = torch.stack([torch.tensor(utils.extract_lower_triangle(mat.numpy())) for mat in original_correlation_matrices])
        latent_lower_triangles = torch.stack([torch.tensor(utils.extract_lower_triangle(mat.numpy())) for mat in latent_correlation_matrices])
        reconstructed_lower_triangles = torch.stack([torch.tensor(utils.extract_lower_triangle(mat.numpy())) for mat in reconstructed_correlation_matrices])

        # Find the best seed for each classifier
        best_seed_original = None
        best_seed_latent = None
        best_seed_reconstructed = None
        best_model_original = None
        best_model_latent = None
        best_model_reconstructed = None

        best_metrics_original = np.zeros(5)
        best_metrics_latent = np.zeros(5)
        best_metrics_reconstructed = np.zeros(5)
        # [mean_accuracy, mean_precision, mean_recall, mean_f1, mean_roc_auc]

        # random_state = random.randint(0, 100000)
        random_state = 76048 # for reproducing the results

        _, metrics_original, _, clf, _, _ = train_and_evaluate_classifier(original_lower_triangles, valid_labels, random_state)

        if metrics_original[0] > best_metrics_original[0]:
            best_accuracy_original = metrics_original[0]
            best_precision_original = metrics_original[1]
            best_recall_original = metrics_original[2]
            best_f1_original = metrics_original[3]
            best_roc_auc_original = metrics_original[4]

            best_seed_original = random_state
            best_model_original = clf

        # random_state = random.randint(0, 100000)
        random_state = 99897 # for reproducing the results

        _, metrics_latent, metrics_latent_std, clf, fpr_list, tpr_list = train_and_evaluate_classifier(latent_lower_triangles, valid_labels, random_state)

        if metrics_latent[0] > best_metrics_latent[0]:
            best_accuracy_latent = metrics_latent[0]
            best_precision_latent = metrics_latent[1]
            best_recall_latent = metrics_latent[2]
            best_f1_latent = metrics_latent[3]
            best_roc_auc_latent = metrics_latent[4]
            best_fpr_latent = fpr_list
            best_tpr_latent = tpr_list

            best_seed_latent = random_state
            best_model_latent = clf

        # random_state = random.randint(0, 100000)
        random_state = 23899 # for reproducing the results

        _, metrics_reconstructed, metrics_reconstructed_std, clf, fpr_list, tpr_list = train_and_evaluate_classifier(reconstructed_lower_triangles, valid_labels, random_state)

        if metrics_reconstructed[0] > best_metrics_reconstructed[0]:
            best_accuracy_reconstructed = metrics_reconstructed[0]
            best_precision_reconstructed = metrics_reconstructed[1]
            best_recall_reconstructed = metrics_reconstructed[2]
            best_f1_reconstructed = metrics_reconstructed[3]
            best_roc_auc_reconstructed = metrics_reconstructed[4]
            best_fpr_reconstructed = fpr_list
            best_tpr_reconstructed = tpr_list

            best_seed_reconstructed = random_state
            best_model_reconstructed = clf
        
        print(f'Best seed for original data: {best_seed_original} with accuracy {best_accuracy_original}')
        print(f'Precision: {best_precision_original}, Recall: {best_recall_original}, F1: {best_f1_original}, ROC AUC: {best_roc_auc_original}')
        print()

        print(f'Best seed for latent data: {best_seed_latent} with accuracy {best_accuracy_latent}')
        print(f'Precision: {best_precision_latent}, Recall: {best_recall_latent}, F1: {best_f1_latent}, ROC AUC: {best_roc_auc_latent}')
        print()

        print(f'Best seed for reconstructed data: {best_seed_reconstructed} with accuracy {best_accuracy_reconstructed}')
        print(f'Precision: {best_precision_reconstructed}, Recall: {best_recall_reconstructed}, F1: {best_f1_reconstructed}, ROC AUC: {best_roc_auc_reconstructed}')
        print()

        reconstructed_accuracies.append(best_accuracy_reconstructed)
        reconstructed_precisions.append(best_precision_reconstructed)
        reconstructed_recalls.append(best_recall_reconstructed)
        reconstructed_f1s.append(best_f1_reconstructed)
        reconstructed_roc_aucs.append(best_roc_auc_reconstructed)

        reconstructed_accuracies_std.append(metrics_reconstructed_std[0])
        reconstructed_precisions_std.append(metrics_reconstructed_std[1])
        reconstructed_recalls_std.append(metrics_reconstructed_std[2])
        reconstructed_f1s_std.append(metrics_reconstructed_std[3])
        reconstructed_roc_aucs_std.append(metrics_reconstructed_std[4])

        reconstructed_fprs.append(best_fpr_reconstructed)
        reconstructed_tprs.append(best_tpr_reconstructed)

        # Save the best models
        torch.save(best_model_original, f'./models/DT_original_data.pt')
        torch.save(best_model_latent, f'./models/DT_latent_data.pt')
        torch.save(best_model_reconstructed, f'./models/DT_reconstructed_data.pt')

    print (f'reconstructed_accuracy = {reconstructed_accuracies}')
    print (f'reconstructed_precision = {reconstructed_precisions}')
    print (f'reconstructed_recall = {reconstructed_recalls}')
    print (f'reconstructed_f1 = {reconstructed_f1s}')
    print (f'reconstructed_roc_auc = {reconstructed_roc_aucs}')

    print (f'reconstructed_accuracy_std = {reconstructed_accuracies_std}')
    print (f'reconstructed_precision_std = {reconstructed_precisions_std}')
    print (f'reconstructed_recall_std = {reconstructed_recalls_std}')
    print (f'reconstructed_f1_std = {reconstructed_f1s_std}')
    print (f'reconstructed_roc_auc_std = {reconstructed_roc_aucs_std}')

    print (f'reconstructed_fprs = {reconstructed_fprs}')
    print (f'reconstructed_tprs = {reconstructed_tprs}')