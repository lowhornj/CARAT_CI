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
        self.mixed_data = mixed_data  # Support for categorical + continuous

        self.pos_embedding = nn.Embedding(time_steps, hidden_dim,dtype=torch.float32,device=self.device)

        # Learnable adjacency matrices (instantaneous + delayed)
        self.edge_score_now = nn.Parameter(torch.randn(num_nodes, num_nodes,device=self.device))
        #self.adj_mat = nn.Parameter(torch.randn(num_nodes, num_nodes,device=self.device))
        self.edge_score_lag = nn.Parameter(torch.randn(num_nodes, num_nodes,device=self.device))
        self.prior_adj = prior_adj if prior_adj is not None else torch.zeros(num_nodes, num_nodes,device=self.device)

        # Attention Layer (Query = final_hidden, Keys/Values = Transformer Output)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)  # Query transformation
        self.W_k = nn.Linear(hidden_dim, hidden_dim)  # Key transformation
        self.W_v = nn.Linear(hidden_dim, hidden_dim)  # Value transformation

        # Direct adjacency learning
        self.dropout = nn.Dropout(p=0.3)  # Dropout rate 30%
        """self.x_projection1 = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim, dtype=torch.float32,device=self.device),
          #  nn.ReLU(),
           # self.dropout
        )"""
        self.x_projection1 =nn.GRU(num_nodes, hidden_dim, batch_first=True,dtype=torch.float32, device=self.device,num_layers =3,dropout =0.1)
      
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, dtype=torch.float32,dropout=0.1,device=self.device),
            num_layers=4
        )
        self.x_projection2 = nn.Linear(hidden_dim,num_nodes,dtype=torch.float32,device=self.device)

    def attention_layer(self, transformer_out, final_hidden):
        """
        Compute attention scores using the GRU hidden state as the query.
        - transformer_out: [batch_size, time_steps, hidden_dim] (Keys & Values)
        - final_hidden: [batch_size, hidden_dim] (Query)
        """
        Q = self.W_q(final_hidden)  # Ensure shape: [batch_size, 1, hidden_dim]
        K = self.W_k(transformer_out)  # [batch_size, time_steps, hidden_dim]
        V = self.W_v(transformer_out)  # [batch_size, time_steps, hidden_dim]
    
        # Compute attention scores
        attn_scores = torch.bmm(Q, K.transpose(1, 2))  # [batch_size, 1, time_steps]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, 1, time_steps]
    
        # Compute weighted sum of values
        attn_output = torch.bmm(attn_weights, V).squeeze(1)  # [batch_size, hidden_dim]
    
        return attn_output, attn_weights


    def forward(self, X, time_context,Z):
        """ Learns causal graph over time and performs inference """
        # Compute adjacency matrices
        x,hidden= self.x_projection1(X)
        pos_indices = torch.arange(self.time_steps, device=X.device)  # [time_steps]
        X_permuted = x.permute(1, 0, 2)  # [time_steps, batch_size, num_nodes]
        pos_embedding = self.pos_embedding(pos_indices).unsqueeze(1) 
        
        X_transformed = self.self_attention(X_permuted + pos_embedding)

        attn_output, attn_weights = self.attention_layer(X_transformed, hidden)
        
        x = self.x_projection2(attn_output)

        weights_schedule = generate_decreasing_weights(self.time_steps,start=0.2)
        lag_mats = []
        for i in range(0,x.shape[0]):
            if i ==0:
                lag_mats.append( replace_zero(
                    hard_concrete(self.edge_score_now * weights_schedule[i]
                                        )/3  + 
                                              (self.prior_adj )/3
                    + (hard_concrete(torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i]
                                           ))/3
                                             ,self.device).fill_diagonal_(-1)) 
            else:
                lag_mats.append(  replace_zero(
                    hard_concrete(self.edge_score_lag* weights_schedule[i]
                                        )/3 +
                    (self.prior_adj )/3
            + (hard_concrete(torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i]
                                   ))/3
                                             ,self.device).fill_diagonal_(-1)) 
        
        adj_now = (lag_mats[0] )  # Amplify signal
        if x.shape[0] >1:
            adj_lag = (torch.sum(torch.stack(lag_mats[1:]), dim=0))/len(lag_mats[1:])
        else:
            adj_lag = (lag_mats[1])
        self.adj_mat = torch.sigmoid(adj_now * self.instantaneous_weight + adj_lag * self.lag_weight)

        
        return adj_now, adj_lag
