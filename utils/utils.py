import csv
import numpy as np
import os, random, torch, json
from torch.utils.data import DataLoader
from networks.Autoencoder import Autoencoder

def check_nan_inf(matrix, name="Matrix"):
    return print(f"{name} has NaN: {np.isnan(matrix).any()}, Inf: {np.isinf(matrix).any()}")

def save_results(results, header, file_name):
    # save dictionary to .csv file
    with open(f'./{file_name}.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for result in results:
            writer.writerow(result)

def load_best_results(file_name):
    with open(f'./{file_name}.txt', 'r') as f:
        return eval(f.read())
    
def check_folders(folder_name, location='./'):
    if not os.path.exists(f'{location}{folder_name}'):
        os.makedirs(f'{location}{folder_name}')
    return

def set_seed(seed=None, seed_torch=True):
    """
    Function that controls randomness. NumPy and random modules must be imported.

    Args:
    seed : Integer
        A non-negative integer that defines the random state. Default is `None`.
    seed_torch : Boolean
        If `True` sets the random seed for pytorch tensors, so pytorch module
        must be imported. Default is `True`.

    Returns:
    Nothing.
    """
    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)
    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')

def create_latent_dataset(model_name, dataset, encoding_dim=10):
    """
    Create a dataset from the latent space of the model.

    Parameters
    ----------
    model : str
        Trained model path.
    dataset : torch.utils.data.TensorDataset
        PyTorch tensor dataset.

    Returns
    -------
    numpy.ndarray
        Latent space data.
    """
    model = Autoencoder(input_dim=80*200, encoding_dim=encoding_dim)

    model_dict = torch.load(model_name)
    model.load_state_dict(model_dict)
    model.eval()

    latent_data = model.encode(dataset.clone().detach())

    return latent_data

def create_reconstructed_dataset(model_name, dataset, enoding_dim=10):
    """
    Create a dataset from the reconstructed space of the model.

    Parameters
    ----------
    model : str
        Trained model path.
    dataset : torch.utils.data.TensorDataset
        PyTorch tensor dataset.
    encoding_dim : int
        Encoding dimension of the model.

    Returns
    -------
    numpy.ndarray
        Reconstructed data.
    """
    model = Autoencoder(input_dim=80*200, encoding_dim=enoding_dim)

    model_dict = torch.load(model_name)
    model.load_state_dict(model_dict)
    model.eval()

    reconstructed_data = model(torch.tensor(dataset, dtype=torch.float32))

    return reconstructed_data

def extract_lower_triangle(matrix):
    return matrix[np.tril_indices(matrix.shape[0], k=-1)]