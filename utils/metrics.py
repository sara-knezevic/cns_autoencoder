import numpy as np
from skimage.metrics import structural_similarity as ssim
import torch

def compute_fc_matrix_regular(data, num_brain_nodes=80, num_time_points=200):
    """
    Compute the functional connectivity matrix for the given data.

    Parameters
    ----------
    data : torch.Tensor
        Data to compute the functional connectivity matrix.
    num_brain_nodes : int
        Number of brain nodes.
    num_time_points : int
        Number of time points.
    epsilon : float
        Small value to add to prevent numerical instability.

    Returns
    -------
    torch.Tensor
        Tensor of functional connectivity matrices.
    """
    assert not torch.isnan(data).any(), "Data contains NaN values."

    data = data.clone().detach()
    data = data.reshape(1, num_brain_nodes, num_time_points)[0]
    correlation_matrix = torch.corrcoef(data)

    # Check for NaN values in the correlation matrices
    assert not torch.isnan(correlation_matrix).any(), "Correlation matrices contain NaN values."
    
    return correlation_matrix

def calculate_fc_diff(FC_list1, FC_list2):
    """
    Calculate the mean square difference between lists of FC matrices.

    Parameters
    ----------
    FC_list1 : list
        List of FC matrices.
    FC_list2 : list
        List of FC matrices.

    Returns
    -------
    float
        Average FCD.
    """
    # check for nan values in the FC matrices
    assert not torch.isnan(FC_list1).any(), "FC_list1 contains NaN values."
    assert not torch.isnan(FC_list2).any(), "FC_list2 contains NaN values."

    fcd_scores = []
    # compare each matrix at the same index and calculate MSE difference
    for FC1, FC2 in zip(FC_list1, FC_list2):
        fcd = torch.nn.functional.mse_loss(FC1, FC2)
        fcd_scores.append(fcd.item())

    # check if there are NaN values in the fcd_scores
    assert not np.isnan(fcd_scores).any(), "FCD list contains NaN values."

    return np.mean(fcd_scores)

def calculate_ssim_fc(FC_list1, FC_list2, data_range=2):
    """ 
    Calculate SSIM for lists of FC matrices.

    Parameters
    ----------
    FC_list1 : list
        List of FC matrices.
    FC_list2 : list
        List of FC matrices.
    data_range : int
        Maximum value of the data.

    Returns
    -------
    float
        Average SSIM.
    """
    # check for nan values in the FC matrices
    assert not torch.isnan(FC_list1).any(), "FC_list1 contains NaN values."
    assert not torch.isnan(FC_list2).any(), "FC_list2 contains NaN values."
    
    ssim_scores = []
    for FC1, FC2 in zip(FC_list1, FC_list2):
        # Convert to NumPy arrays if necessary
        if isinstance(FC1, torch.Tensor):
            FC1 = FC1.detach().cpu().numpy()
        if isinstance(FC2, torch.Tensor):
            FC2 = FC2.detach().cpu().numpy()
            
        ss_ix = ssim(FC1, FC2, data_range=data_range)
        ssim_scores.append(ss_ix)
        
    # check if there are NaN values in the ssim_scores
    assert not np.isnan(ssim_scores).any(), "SSIM list contains NaN values."

    return np.mean(ssim_scores)

def calculate_correlation_fc(FC_list1, FC_list2):
    """
    Calculate the average correlation coefficient between lists of FC matrices.

    Parameters
    ----------
    FC_list1 : list
        List of FC matrices.
    FC_list2 : list
        List of FC matrices.

    Returns
    -------
    float
        Average correlation coefficient.
    """
    # Check for NaN values in the FC matrices
    assert not torch.isnan(FC_list1).any(), "FC_list1 contains NaN values."
    assert not torch.isnan(FC_list2).any(), "FC_list2 contains NaN values."
    
    correlations = []
    # Calculate correlation between each pair of matrices at the same index
    for FC1, FC2 in zip(FC_list1, FC_list2):
        FC1_flat = FC1.flatten()
        FC2_flat = FC2.flatten()
        
        # Check for constant vectors
        if torch.var(FC1_flat) == 0 or torch.var(FC2_flat) == 0:
            correlation = 0.0  # Assign a default value for correlation
        else:
            correlation = np.corrcoef(FC1_flat.cpu().numpy(), FC2_flat.cpu().numpy())[0, 1]
        
        correlations.append(correlation)
    
    # Check if there are NaN values in the correlations
    assert not np.isnan(correlations).any(), "Correlation list contains NaN values."

    return np.mean(correlations)
