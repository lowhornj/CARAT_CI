import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
from CARAT.model_utils import *

def scale_tensor(tensor):
  """Scales a PyTorch tensor to the range [0, 1].

  Args:
    tensor: The input tensor.

  Returns:
    A new tensor with values scaled to [0, 1].
  """
  min_val = tensor.min()
  max_val = tensor.max()
  scaled_tensor = (tensor - min_val) / (max_val - min_val)
  return scaled_tensor

def hard_concrete(log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
    u = torch.rand_like(log_alpha)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    stretched_s = s * (zeta - gamma) + gamma
    return torch.clamp(stretched_s, 0, 1)


class TemporalCausalGraph(nn.Module):
    """
    Implements a Temporal Causal Graph (TCG) with:
    - Time-dependent adjacency matrix (instantaneous + delayed effects)
    """
    def __init__(self, num_nodes, hidden_dim, latent_dim,device,time_steps=10, prior_adj=None, instantaneous_weight=0.5,lag_weight=0.5,mixed_data=False):
        super(TemporalCausalGraph, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.device = device
        self.instantaneous_weight=instantaneous_weight
        self.lag_weight=lag_weight

        # Learnable adjacency matrices (instantaneous + delayed)
        self.edge_score_now = nn.Parameter(torch.randn(num_nodes, num_nodes, device=self.device) * 0.1)
        #self.adj_mat = nn.Parameter(torch.randn(num_nodes, num_nodes,device=self.device))
        self.edge_score_lag = nn.Parameter(torch.randn(num_nodes, num_nodes, device=self.device) * 0.1)
        self.prior_adj = prior_adj if prior_adj is not None else torch.zeros(num_nodes, num_nodes,device=self.device)

        self.x_projection2 = nn.Linear(hidden_dim,num_nodes,dtype=torch.float32,device=self.device)


    def forward(self, X_transformed, time_context):
        """ Learns causal graph over time and performs inference """
        # Compute adjacency matrices
        x = self.x_projection2(X_transformed)

        weights_schedule = generate_decreasing_weights(self.time_steps,start=0.1)
        lag_mats = []
        for i in range(0,x.shape[0]):
            if i ==0:
                lag_mats.append( replace_zero(
                    hard_concrete(self.edge_score_now * weights_schedule[i]
                                        )/3  + 
                                              (self.prior_adj )/3
                    + torch.sigmoid((torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i]
                                           ))/3
                                             ,self.device)) 
            else:
                lag_mats.append(  replace_zero(
                    hard_concrete(self.edge_score_lag* weights_schedule[i]
                                        )/3 +
                    (self.prior_adj )/3
            + torch.sigmoid((torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i]
                                   ))/3
                                             ,self.device)) 
        
        adj_now = (lag_mats[0]) 
        if x.shape[0] > 1:
            adj_lag = (torch.sum(torch.stack(lag_mats[1:]), dim=0) / len(lag_mats[1:]))
        else:
            adj_lag = (lag_mats[1]).fill_diagonal_(0)
        self.adj_mat = torch.sigmoid(adj_now * self.instantaneous_weight + adj_lag * self.lag_weight).fill_diagonal_(0)

        
        return adj_now, adj_lag
