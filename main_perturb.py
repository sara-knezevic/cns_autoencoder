from sklearn import metrics
import torch
import numpy as np
from scipy.io import loadmat
from utils.Preprocessing import Preprocessing
from utils.metrics import compute_fc_matrix_regular
import matplotlib.pyplot as plt
from networks.Autoencoder import Autoencoder
from scipy.stats import zscore
import utils.utils as utils

def perturb_and_classify(autoencoder, classifier, latent_data, perturbation_amount=0.1):
    num_participants, num_time_points, latent_dim = latent_data.shape
    
    changes = 0

    for participant in range(num_participants):
        original_latent = latent_data[participant].detach()  # Detach here
        with torch.no_grad():
            reconstructed_signal = autoencoder.decode(original_latent.view(-1, latent_dim))
            original_class = classifier.predict(reconstructed_signal).argmax(dim=1).detach().cpu().numpy()

        for i in range(latent_dim):
            perturbed_latent = original_latent.clone().detach()  # Detach here
            perturbation = torch.randn(num_time_points) * perturbation_amount
            perturbed_latent[:, i] += perturbation

            with torch.no_grad():
                perturbed_reconstructed_signal = autoencoder.decode(perturbed_latent.view(-1, latent_dim))
                new_class = classifier.predict(perturbed_reconstructed_signal).argmax(dim=1).detach().cpu().numpy()

            if not np.array_equal(original_class, new_class):
                changes += 1
                print(f"Participant {participant}, Feature {i}: Class changed from {original_class} to {new_class}")

    return changes
    
def perturb():
    # Load and preprocess the data as before
    preprocessor = Preprocessing()
    mat_data = loadmat('./data/laufs_sleep.mat')
    assert mat_data['TS_N1'].shape == mat_data['TS_N2'].shape == mat_data['TS_N3'].shape == mat_data['TS_W'].shape

    n3_data = mat_data['TS_N3'][0][1:]
    wake_data = mat_data['TS_W'][0][1:]

    n3_data_shortened, _ = preprocessor.shorten_data(n3_data, final_length=200)
    wake_data_shortened, _ = preprocessor.shorten_data(wake_data, final_length=200)

    wake_labels = torch.zeros((wake_data_shortened.shape[0], 1))
    n3_labels = torch.ones((n3_data_shortened.shape[0], 1))

    valid_labels = torch.cat((wake_labels, n3_labels), 0)
    concatenated_data = np.concatenate([wake_data_shortened, n3_data_shortened])
    concatenated_data = zscore(concatenated_data)
    concatenated_data = concatenated_data.reshape(-1, 80)

    num_time_points = 200
    num_brain_nodes = 80
    num_participants = concatenated_data.shape[0] // num_time_points

    data_tensor = torch.tensor(concatenated_data, dtype=torch.float32)
    data_tensor = data_tensor.reshape(num_participants, num_time_points, num_brain_nodes)

    latent_dimension = 18
    model = Autoencoder(latent_dim=latent_dimension)
    model.load_state_dict(torch.load(f'./models/doubleDropout/AE_results_lat{latent_dimension}.pt'))
    model.eval()

    with torch.no_grad():
        reconstructed, latent = model(data_tensor.view(-1, num_brain_nodes))
        latent_data = latent.view(num_participants, num_time_points, latent_dimension)

    # Load the classifier
    classifier = torch.load(f'./models/DT_reconstructed_data.pt')

    # Perturb the latent space and classify
    changes = perturb_and_classify(model, classifier, latent_data)