import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
from CARAT.model_utils import *
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class FourierTimeEmbedding(nn.Module):
    """Encodes time indices using sinusoidal embeddings for better temporal representation."""
    def __init__(self, embedding_dim,time_steps, max_time=100):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.max_time = max_time
        
        # Create frequency bands (log-spaced)
        self.freqs = torch.exp(torch.linspace(0, time_steps, embedding_dim//2,dtype=torch.float32)).to(device)

    def forward(self, time_indices):
        """
        Args:
            time_indices: Tensor of shape [batch_size, time_steps] containing integer time indices
        Returns:
            Embedded time representations of shape [batch_size, time_steps, embedding_dim]
        """
        time_indices = time_indices.unsqueeze(-1)  # Shape [batch_size, time_steps, 1]
        sinusoidal_in = time_indices * self.freqs  # Shape [batch_size, time_steps, embedding_dim//2]
        time_embedding = torch.cat([torch.sin(sinusoidal_in), torch.cos(sinusoidal_in)], dim=-1)
        return time_embedding  # Shape [batch_size, time_steps, embedding_dim]

class TemporalRealNVPFlow(nn.Module):
    """Time-Adaptive Normalizing Flow for Latent Confounders."""
    def __init__(self, latent_dim,input_dim):
        super().__init__()
        self.scale = nn.Linear(latent_dim // 2, latent_dim // 2,dtype=torch.float32)
        self.translate = nn.Linear(latent_dim // 2, latent_dim // 2,dtype=torch.float32)
        self.time_embedding = FourierTimeEmbedding(latent_dim,input_dim)
        self.temporal_gate = nn.GRU(latent_dim, latent_dim//2, batch_first=True,dtype=torch.float32)  # Time-aware updates
        self.time_projection = nn.Linear(input_dim,1,dtype=torch.float32)
    def forward(self, z, time_context):
        z1, z2 = z.chunk(2, dim=1)  # Split into two parts
        s = torch.sigmoid(self.scale(z1))
        t = self.translate(z1)
        z2 = s * z2 + t
        time_embed = self.time_embedding(time_context)
        # Temporal adjustment to confounders
        time_out, _ = self.temporal_gate(time_embed)
        time_out = self.time_projection(time_out.permute(0,2,1)).squeeze(2)
        z2 = z2 + time_out#.squeeze(1)  # Adjust for temporal shift
        
        return torch.cat([z1, z2], dim=1)

class TemporalCausalGraph(nn.Module):
    """
    Implements a Temporal Causal Graph (TCG) with:
    - Time-dependent adjacency matrix (instantaneous + delayed effects)
    - Adaptive normalizing flow for non-stationarity handling
    """
    def __init__(self, num_nodes, hidden_dim, latent_dim,time_steps=10, prior_adj=None, instantaneous_weight=0.5,lag_weight=0.5,mixed_data=False):
        super(TemporalCausalGraph, self).__init__()
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.instantaneous_weight=instantaneous_weight
        self.lag_weight=lag_weight
        self.mixed_data = mixed_data  # Support for categorical + continuous

        self.pos_embedding = nn.Embedding(time_steps, hidden_dim,dtype=torch.float32)

        # Learnable adjacency matrices (instantaneous + delayed)
        self.edge_score_now = nn.Parameter(torch.randn(num_nodes, num_nodes,device=device))
        #self.adj_mat = nn.Parameter(torch.randn(num_nodes, num_nodes,device=device))
        self.edge_score_lag = nn.Parameter(torch.randn(num_nodes, num_nodes,device=device))
        self.prior_adj = prior_adj if prior_adj is not None else torch.zeros(num_nodes, num_nodes,device=device)

        # Direct adjacency learning
        self.dropout = nn.Dropout(p=0.3)  # Dropout rate 30%
        self.x_projection1 = nn.Sequential(
            nn.Linear(num_nodes, hidden_dim, dtype=torch.float32),
            nn.ReLU(),
            self.dropout
        )
      
        self.self_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim, dtype=torch.float32,dropout=0.2),
            num_layers=2
        )
        self.x_projection2 = nn.Linear(hidden_dim,num_nodes,dtype=torch.float32)

        # Latent Confounders Z (Time-Aware)
        
        self.temporal_flow = TemporalRealNVPFlow(latent_dim,time_steps)

        # Temporal Graph Attention Network
        self.gnn = GATv2Conv(num_nodes, hidden_dim, heads=4, concat=True, dropout=0.2).to(torch.float32)

        # Mapping from latent space to num_nodes
        self.latent_to_nodes = nn.Linear(latent_dim, num_nodes,dtype=torch.float32)

        # Likelihood Models
        self.gaussian_likelihood = nn.Sequential(
            nn.Dropout(0.2),  # Dropout before final layer
            nn.Linear(hidden_dim * 4, self.num_nodes*2, dtype=torch.float32)
        )  # Mean, Log-Variance
        if mixed_data:
            self.categorical_likelihood = nn.Linear(hidden_dim, num_nodes,dtype=torch.float32)  # Gumbel-Softmax Output

    def forward(self, X, time_context,Z):
        """ Learns causal graph over time and performs inference """
        # Compute adjacency matrices
        x = self.x_projection1(X)
        pos_indices = torch.arange(self.time_steps, device=X.device)  # [time_steps]
        X_permuted = x.permute(1, 0, 2)  # [time_steps, batch_size, num_nodes]
        pos_embedding = self.pos_embedding(pos_indices).unsqueeze(1) 
        
        X_transformed = self.self_attention(X_permuted + pos_embedding)
        x = self.x_projection2(X_transformed)


        weights_schedule = generate_decreasing_weights(3,start=0.2)
        lag_mats = []
        for i in range(0,x.shape[0]):
            if i ==0:
                lag_mats.append( replace_zero(torch.sigmoid(self.edge_score_now* weights_schedule[i])  +self.prior_adj + F.normalize(F.gumbel_softmax(torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i],tau=100,eps=1e-16,))).fill_diagonal_(-1)) 
            else:
                lag_mats.append( replace_zero(torch.sigmoid(self.edge_score_lag* weights_schedule[i])  +self.prior_adj + F.normalize(F.gumbel_softmax(torch.einsum('bk,bj->kj', x[i,:,:], x[i,:,:]) * weights_schedule[i],tau=100,eps=1e-16,))).fill_diagonal_(-1)) 
        
        adj_now = torch.sigmoid(lag_mats[0] )  # Amplify signal
        if x.shape[0] >1:
            adj_lag = torch.sigmoid(torch.sum(torch.stack(lag_mats[1:]), dim=0))
        else:
            adj_lag = torch.sigmoid(lag_mats[1])
        self.adj_mat = adj_now * 0.3 + adj_lag * 0.7
       
        # Encode latent confounders with time-awareness
        #Z = self.latent_Z + torch.randn_like(self.latent_Z) * 0.1
        """Z = self.temporal_flow(Z.to(device), time_context.to(device))  # Apply time-adaptive normalizing flow
        Z = self.latent_to_nodes(Z)  # Map latent space to num_nodes

        # Temporal graph attention
        edge_index,edge_weights = dense_to_sparse( adj_mat)
        X_emb = self.gnn(Z, edge_index)
        X_emb = X_emb.view(X_emb.shape[0], -1)

        # Likelihood computation
        mean_logvar = self.gaussian_likelihood(X_emb)
        mean, log_var = torch.split(mean_logvar, mean_logvar.shape[-1] // 2, dim=-1)
        log_var = torch.clamp(log_var, -5, 2)  # Stabilization
        
        likelihood = Laplace(mean, torch.exp(0.5 * log_var))"""
        likelihood = 0
        
        return adj_now, adj_lag, likelihood
