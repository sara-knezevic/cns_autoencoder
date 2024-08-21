import numpy as np
from sklearn.discriminant_analysis import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader

class Preprocessing:
    def __init__(self):
        pass

    def concatenate_data(self, n1_data, n2_data, n3_data, wake_data):
        """
        Concatenate the data from the different sleep stages into a single dataset.
        
        Parameters
        ----------
        n1_data : numpy.ndarray
            Data from sleep stage N1.
        n2_data : numpy.ndarray
            Data from sleep stage N2.
        n3_data : numpy.ndarray
            Data from sleep stage N3.
        wake_data : numpy.ndarray
            Data from wake stage.
        
        Returns
        -------
        numpy.ndarray
            Concatenated data.
        """
        return np.concatenate((n1_data, n2_data, n3_data, wake_data), axis=0)

    def print_time_steps(self, data):
        """
        Print the number of time steps in the data.
        
        Parameters
        ----------
        data : numpy.ndarray
            Data to print the number of time steps.
        """
        for index, participant_data in enumerate(data):
            num_time_steps = participant_data.shape[1]
            print(f"Participant {index + 1} has {num_time_steps} time steps")

    def pad_data(self, data, max_time_steps):
        """
        Pad the data with zeros to ensure all participants have the same number of time steps.
        
        Parameters
        ----------
        data : numpy.ndarray
            Data to pad.
        max_time_steps : int
            Maximum number of time steps across all participants.
        
        Returns
        -------
        numpy.ndarray
            Padded data.
        """
        padded_data_array = []
        for participant_data in data:
            padded_data = np.pad(participant_data, ((0, 0), (0, max_time_steps - participant_data.shape[1])), mode='constant', constant_values=0)
            padded_data_array.append(padded_data)
        return np.array(padded_data_array)

    def shorten_data(self, data, labels=None, final_length=0):
        """
        Shorten the data to a fixed length.
        
        Parameters
        ----------
        data : numpy.ndarray
            Data to shorten.
        final_length : int
            Length to shorten the data to.
        labels : numpy.ndarray
            Labels corresponding to the data.
        
        Returns
        -------
        shortened_data_array : list
            Shortened data.
        """
        if final_length == 0:
            final_length = min([participant_data.shape[1] for participant_data in data])
        
        shortened_data_array = []
        valid_labels = []

        for i, participant_data in enumerate(data):
            if participant_data.shape[1] >= final_length:
                shortened_data_array.append(participant_data[:, :final_length])

                if labels is not None:
                    valid_labels.append(labels[i])

        return np.array(shortened_data_array), np.array(valid_labels)

    def flatten_data(self, data):
        """
        Flatten the data.
        
        Parameters
        ----------
        data : numpy.ndarray
            Data to flatten.
        
        Returns
        -------
        numpy.ndarray
            Flattened data.
        """
        flattened_data = [participant_data.flatten() for participant_data in data]
        return np.vstack(flattened_data)
    
    def zscore_normalize_flattened(self, data):
        """
        Z-score normalize the data.
        
        Parameters
        ----------
        data : numpy.ndarray
            Data to normalize.
        
        Returns
        -------
        numpy.ndarray
            Z-score normalized data.
        """
        mean = data.mean(axis=0, keepdims=True)
        std = data.std(axis=0, keepdims=True)
        return (data - mean) / std
    
    def standardize_and_normalize_data(self, data):
        # Calculate mean and standard deviation for each time step
        mean_per_time_step = np.mean(data, axis=(0, 1))
        std_per_time_step = np.std(data, axis=(0, 1))
        
        # Standardize data
        standardized_data = (data - mean_per_time_step) / std_per_time_step
        
        # Normalize data to the range [0, 1]
        min_per_time_step = np.min(standardized_data, axis=(0, 1))
        max_per_time_step = np.max(standardized_data, axis=(0, 1))
        
        normalized_data = (standardized_data - min_per_time_step) / (max_per_time_step - min_per_time_step)
        
        return normalized_data

    def get_data_loaders(self, train_idx, val_idx, flattened_data, batch_size):
        train_data = flattened_data[train_idx]
        val_data = flattened_data[val_idx]

        normalize = self.normalize_within_fold(train_data)
        train_data = normalize(train_data)
        val_data = normalize(val_data)

        train_tensor = torch.tensor(train_data, dtype=torch.float32)
        val_tensor = torch.tensor(val_data, dtype=torch.float32)

        train_loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(val_tensor), batch_size=batch_size, shuffle=False)

        return train_loader, val_loader

    def create_tensor_dataset(self, data):
        """
        Create a PyTorch tensor dataset from the data.

        Parameters
        ----------
        data : list
            List of data arrays.

        Returns
        -------
        torch.utils.data.TensorDataset
            PyTorch tensor dataset.
        """
        data_tensors = [torch.tensor(participant_data).float() for participant_data in data]
        dataset = TensorDataset(torch.stack(data_tensors))
        return dataset

