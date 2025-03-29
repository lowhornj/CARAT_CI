import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
from CARAT.model_utils import *


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import dense_to_sparse
from CARAT.causal_discovery_utils import *

class TemporalCausalGraph(nn.Module):
    """
    Enhanced Temporal Causal Graph (TCG) with multiple causal discovery techniques
    and improved regularization.
    """
    def __init__(self, num_nodes, hidden_dim, latent_dim, device, time_steps=10, 
                 prior_adj=None, instantaneous_weight=0.5, lag_weight=0.5, use_attention=False):
        super(TemporalCausalGraph, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.time_steps = time_steps
        self.device = device
        self.instantaneous_weight = instantaneous_weight
        self.lag_weight = lag_weight
        self.use_attention = use_attention

        if self.use_attention:
            self.agg_nodes = 3
        else:
            self.agg_nodes = 2

        # Learnable adjacency matrices
        self.edge_score_now = nn.Parameter(torch.randn(num_nodes, num_nodes, device=self.device) * 0.1)
        self.edge_score_lag = nn.Parameter(torch.randn(num_nodes, num_nodes, device=self.device) * 0.1)
        
        # Prior adjacency matrix (if available)
        self.prior_adj = prior_adj if prior_adj is not None else torch.zeros(num_nodes, num_nodes, device=self.device)
        
        # Feature projections
        self.node_projection = nn.Linear(hidden_dim, hidden_dim, dtype=torch.float32, device=self.device)
        self.edge_projection = nn.Linear(hidden_dim * 2, 1, dtype=torch.float32, device=self.device)
        
        # Attention mechanism for edge scoring
        # Make sure embed_dim is divisible by num_heads
        self.attn_dim = max(hidden_dim, (num_nodes // 4) * 4)  # Ensure divisible by 4
        self.node_to_attn = nn.Linear(hidden_dim, self.attn_dim, device=self.device)
        self.attn_to_node = nn.Linear(self.attn_dim, num_nodes, device=self.device)
        
        self.edge_attention = nn.MultiheadAttention(
            embed_dim=self.attn_dim, 
            num_heads=4, 
            batch_first=True,
            device=self.device
        )
        
        # Graph neural network for structure learning
        self.gnn_edge_predictor = GATv2Conv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=4,
            edge_dim=1
        ).to(self.device)

        

        
        # Final edge scoring
        self.edge_scoring = nn.Sequential(
            nn.Linear(self.agg_nodes, 16, device=self.device),
            nn.LeakyReLU(),
            nn.Linear(16, 1, device=self.device),
            nn.Sigmoid()
        )
        

    def hard_concrete_sigmoid(self, log_alpha, beta=2/3, gamma=-0.1, zeta=1.1):
        """
        Improved Hard Concrete distribution for differentiable edge selection.
        Outputs values between 0 and 1 that are close to binary.
        """
        u = torch.rand_like(log_alpha)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
        stretched_s = s * (zeta - gamma) + gamma
        return torch.clamp(stretched_s, 0, 1)
  
    def correlation_based_edges(self, features):
        """
        Computes correlation-based edge scores between nodes.
        Works with both 1D and 2D feature tensors.
        """
        # Check if features is 1D (single sample) or 2D (batch)
        if features.dim() == 1:
            # For 1D features, we need a different approach
            # Since we don't have true correlations from a single sample
            # Create a sparse random matrix with values between 0.1 and 0.9
            edge_scores = torch.rand(self.num_nodes, self.num_nodes, device=self.device) * 0.8 + 0.1
            
            # Make it symmetric (for undirected connections)
            edge_scores = (edge_scores + edge_scores.t()) / 2
            
            # Add some structure based on the original feature values
            # Nodes with similar values in the feature vector will have stronger connections
            for i in range(self.num_nodes):
                for j in range(i+1, self.num_nodes):
                    # Calculate similarity between nodes based on feature values
                    # Higher weight for closer values
                    similarity_weight = 1.0 - min(1.0, abs(features[i] - features[j]) / (torch.max(features) - torch.min(features) + 1e-8))
                    # Adjust edge scores
                    edge_scores[i, j] = edge_scores[i, j] * 0.5 + similarity_weight * 0.5
                    edge_scores[j, i] = edge_scores[i, j]  # Keep symmetry
        
        else:
            # For multi-sample features, compute true cross-correlations
            # Use batch dimension for more reliable estimation
            batch_size = features.shape[0]
            corr_matrix = torch.zeros((self.num_nodes, self.num_nodes), device=self.device)
            
            # Compute correlations across the batch dimension
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:
                        # Get time series for nodes i and j
                        series_i = features[:, i]
                        series_j = features[:, j]
                        
                        # Center the series
                        series_i = series_i - series_i.mean()
                        series_j = series_j - series_j.mean()
                        
                        # Compute correlation
                        numer = torch.sum(series_i * series_j)
                        denom = torch.sqrt(torch.sum(series_i**2) * torch.sum(series_j**2) + 1e-8)
                        corr = numer / denom
                        
                        # Store absolute correlation (direction determined elsewhere)
                        corr_matrix[i, j] = abs(corr)
            
            edge_scores = corr_matrix
        
        # Remove self-loops
        eye_mask = torch.eye(self.num_nodes, device=self.device)
        edge_scores = edge_scores * (1 - eye_mask)
        
        # Ensure values are in [0, 1]
        edge_scores = torch.clamp(edge_scores, 0, 1)
        
        return edge_scores
    
    def attention_based_edges(self, features):
        """
        Uses attention mechanism to compute edge importance.
        """
        # Project node features to attention dimension
        node_feats = torch.einsum('b,nm->bm', features, self.node_to_attn.weight) + self.node_to_attn.bias
        
        # Reshape for attention
        query = node_feats.unsqueeze(0)  # [1, num_nodes, attn_dim]
        key = node_feats.unsqueeze(0)    # [1, num_nodes, attn_dim]
        value = node_feats.unsqueeze(0)  # [1, num_nodes, attn_dim]
        
        # Compute attention scores
        attn_output, attn_weights = self.edge_attention(query, key, value)
        
        # Extract attention weights
        edge_scores = attn_weights.squeeze(0)
        
        # If attention dimension is different from number of nodes, project back
        if self.attn_dim != self.num_nodes:
            # Create identity-like initialization for the projection
            edge_scores_full = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
            for i in range(min(self.num_nodes, edge_scores.shape[0])):
                for j in range(min(self.num_nodes, edge_scores.shape[1])):
                    edge_scores_full[i, j] = edge_scores[i, j]
            edge_scores = edge_scores_full
        
        # Remove self-loops
        edge_scores = edge_scores * (1 - torch.eye(self.num_nodes, device=self.device))
        
        return edge_scores
    
    
    def forward(self, X_transformed, time_context):
        """
        Enhanced forward pass with ensemble of causal discovery methods.
        """
        # Project features to node representation space
        #node_features = self.node_projection(X_transformed)  # [time_steps, batch, hidden_dim]
        
        # Process for each timestep and get decreasing weights
        weights_schedule = torch.linspace(1.0, 0.1, self.time_steps, device=self.device)
        
        # Store matrices for instantaneous and lagged effects
        now_matrices = []
        lag_matrices = []
        
        for t in range(X_transformed.shape[0]):
            # Get batch-averaged node features for this timestep
            avg_features = X_transformed[t].mean(dim=0)  # [hidden_dim]
            
            # 1. Method: Correlation-based edges
            corr_edges = self.correlation_based_edges(avg_features)
            
            # 2. Method: Attention-based edges
            if self.use_attention:
                attn_edges = self.attention_based_edges(avg_features)
            
            # 3. Method: Learnable parameters with Hard Concrete
            param_edges = self.edge_score_now if t == 0 else self.edge_score_lag
            #self.hard_concrete_sigmoid(
             
            #)
            # Ensemble the methods (stacking features for a learned combination)
            if self.use_attention:
                edge_features = torch.stack([
                    corr_edges, 
                    attn_edges,
                    param_edges
                ], dim=-1)  # [num_nodes, num_nodes, 3]
                
            else:
                edge_features = torch.stack([
                    corr_edges, 
                    param_edges
                ], dim=-1)  # [num_nodes, num_nodes, 3]
            
            # Flatten the last dimension
            edge_features_flat = edge_features.view(self.num_nodes, self.num_nodes, self.agg_nodes)
            
            # Compute final edge scores
            final_edges = torch.zeros(self.num_nodes, self.num_nodes, device=self.device)
            for i in range(self.num_nodes):
                for j in range(self.num_nodes):
                    if i != j:  # Skip self-loops
                        final_edges[i, j] = self.edge_scoring(edge_features_flat[i, j]).squeeze()
            
            # Apply prior knowledge if available
            if self.prior_adj is not None:
                prior_weight = 0.3
                final_edges = (1 - prior_weight) * final_edges + prior_weight * torch.sigmoid(self.prior_adj)
            
            # Apply time weighting
            weighted_edges = final_edges * weights_schedule[t]
            
            # Store in appropriate list
            if t == 0:
                now_matrices.append(weighted_edges)
            else:
                lag_matrices.append(weighted_edges)
        
        # Combine instantaneous effects
        adj_now = now_matrices[0].fill_diagonal_(0)
        
        # Combine lagged effects
        if len(lag_matrices) > 0:
            adj_lag = torch.stack(lag_matrices).mean(dim=0).fill_diagonal_(0)
        else:
            adj_lag = torch.zeros_like(adj_now)

        self.adj_mat = adj_lag + adj_now
        
        
        return adj_now, adj_lag
    

