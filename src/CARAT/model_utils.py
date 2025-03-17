import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from statsmodels.tsa.stattools import adfuller
from torch.utils.data import Dataset, DataLoader

def scale_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Scales a PyTorch tensor to the range [0,1].
    
    Args:
        tensor (torch.Tensor): Input tensor to be scaled.
    
    Returns:
        torch.Tensor: Scaled tensor with values in the range [0,1].
    """
    min_val = tensor.min()
    max_val = tensor.max()
    
    if max_val == min_val:
        return torch.zeros_like(tensor)  # Avoid division by zero if all values are the same
    
    return (tensor - min_val) / (max_val - min_val)
    
class TimeSeriesDataset(Dataset):
    def __init__(self, dataframe, device,time_steps=10):
        """
        Args:
            dataframe (pd.DataFrame): Time-series data
            time_steps (int): Number of past time steps to consider
        """
        self.data = torch.tensor(dataframe.values, dtype=torch.float32)
        self.time_steps = time_steps
        self.device = device
    
    def __len__(self):
        return len(self.data) - self.time_steps

    def __getitem__(self, idx):
        """
        Returns:
            X: Past `time_steps` values (shape: [time_steps, num_features])
            y: Next step prediction target (shape: [num_features])
        """
        X = self.data[idx : idx + self.time_steps, :].to(self.device)
        time_context = torch.arange(self.time_steps).float().to(self.device)  # Time indexing
        
        return X, time_context

def generate_decreasing_weights(n,start=0.3):
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return np.array([1.0])  # If only one weight, it must be 1.

    weights = np.zeros(n)
    weights[0] = start  # First weight is always 0.4

    # Generate decreasing values
    remaining_weights = np.linspace(n-1, 1, n-1)  # Decreasing sequence
    remaining_weights /= remaining_weights.sum()  # Normalize to sum to 0.6
    remaining_weights *= 1-start  # Scale to sum to 0.6

    weights[1:] = remaining_weights
    return weights

def get_adjacency(cols,causal_indices,non_causal_indices,num_nodes):
    A0=torch.zeros(num_nodes,num_nodes)
    for i, row in enumerate(A0):
        for j, column in enumerate(row):
            if (j in non_causal_indices) and (i in causal_indices) & (i!=j):
                A0[i,j] = 0.5
            else:
                A0[i,j] = -1
    return A0

def replace_zero(tensor, device, small_number=1e-3):
  """
  Replaces zeros in a PyTorch tensor with a small number.

  Args:
    tensor: The input tensor.
    small_number: The small number to replace zeros with (default: 1e-8).

  Returns:
    A new tensor with zeros replaced by the small number.
  """
  return torch.where(tensor == 0, torch.tensor(small_number, dtype=tensor.dtype,device=device), tensor)

def notears_constraint(W):
    """NoTears DAG constraint: trace(exp(W * W)) - d"""
    d = W.shape[0]
    W = torch.clamp(W, -2, 2)
    expm_ww = torch.matrix_exp(W * W)
    return torch.trace(expm_ww) - d
