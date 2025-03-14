import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
#from CARAT.components import *
from CARAT.model_utils import *
import pandas as pd
torch.backends.cudnn.benchmark = False
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes, time_steps=10, prior_adj=None,instantaneous_weight=0.5):
        super(CausalGraphVAE, self).__init__()
        self.instantaneous_weight = instantaneous_weight
        self.lag_weight = 1.0 - self.instantaneous_weight
        self.time_steps = time_steps
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.prior_adj = prior_adj
        self.causal_graph = TemporalCausalGraph(num_nodes, hidden_dim, latent_dim, time_steps, prior_adj,self.instantaneous_weight,self.lag_weight)
        self.alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=device)
        self.rho = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=device)

        # Temporal-aware Encoder and Decoder
        self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True,dtype=torch.float32, device=device)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim,dtype=torch.float32, device=device)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim,dtype=torch.float32, device=device)
        self.decoder_rnn = nn.GRU(latent_dim, hidden_dim, batch_first=True,dtype=torch.float32, device=device)
        self.decoder_fc = nn.Linear(hidden_dim, input_dim,dtype=torch.float32, device=device)

    def encode(self, X, time_context):
        X_enc, _ = self.encoder_rnn(X)
        mu, logvar = self.mu_layer(X_enc[:, -1, :]), self.logvar_layer(X_enc[:, -1, :])
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        adj_now, adj_lag, likelihood = self.causal_graph(X, time_context,Z)  # Use temporal causal graph
        return mu, logvar, adj_now, adj_lag, likelihood

    def decode(self, Z, adj_now, adj_lag):
        Z_expanded = Z.unsqueeze(1).repeat(1, self.time_steps, 1)
        X_dec, _ = self.decoder_rnn(Z_expanded)
        X_dec = self.decoder_fc(X_dec)
        return X_dec

    def forward(self, X, time_context):
        X = X.to(torch.float32)  # Convert input X to float32
        time_context = time_context.to(torch.float32)  # Convert time context to float32

        mu, logvar, adj_now, adj_lag, likelihood = self.encode(X, time_context)
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon_X = self.decode(Z, adj_now, adj_lag)
        return recon_X, mu, logvar, adj_now, adj_lag, likelihood

    def infer_causal_effect(self, X_data, T_data, target_variable, labels, non_causal_indices=[]):
        """Infers the top causal factors for a given target variable using counterfactual analysis."""
        
        try:
            target_idx = labels.index(target_variable, 0)
        except:
            return target_variable + " not in index"
    
        self.eval()
        n_samples = X_data.shape[0]
    
        with torch.no_grad():
            _, _, adj_now, adj_lag, _ = self.encode(X_data, T_data)
            adj_matrix = adj_now * self.instantaneous_weight + adj_lag * self.lag_weight
    
            # Compute edge strengths
            causal_strengths = adj_matrix[:, target_idx].cpu().detach().numpy()
            edge_strengths = {labels[i]: float(np.round(val.item(), 6)) for i, val in enumerate(causal_strengths) if i not in non_causal_indices}

            instantaneous_strengths = adj_now[:, target_idx].cpu().detach().numpy()
            instantaneous_edge_strengths = {}
            for i, val in enumerate(instantaneous_strengths):
                if i not in non_causal_indices:
                    instantaneous_edge_strengths[labels[i]] = float(np.round(val.item(),6))
            
            lagged_strengths = adj_lag[:, target_idx].cpu().detach().numpy()
            lagged_edge_strengths = {}
            for i, val in enumerate(lagged_strengths):
                if i not in non_causal_indices:
                    lagged_edge_strengths[labels[i]] = float(np.round(val.item(),6))

            
            counterfactual_results_positive = {}
            counterfactual_results_negative = {}
    
            for i in range(self.num_nodes):
                if i == target_idx or i in non_causal_indices:
                    continue
                
                label = labels[i]
    
                # Positive intervention: set cause variable to 1
                intervention_pos = X_data.clone()
                intervention_pos[:, :, i] = 1  
                mu, logvar, adj_now_int, adj_lag_int, likelihood = self.encode(intervention_pos, T_data)
                Z = mu + torch.randn_like(logvar, dtype=torch.float32, device=device) * torch.exp(0.5 * logvar)
                recon_X = self.decode(Z, adj_now_int, adj_lag_int)
    
                # Compute normalized deviation for positive intervention
                counterfactual_results_positive[label] = torch.abs((X_data[:, :, target_idx] - recon_X[:, :, target_idx]) / (X_data[:, :, target_idx] + 1e-6)).mean().item()
    
                # Negative intervention: set cause variable to -1
                intervention_neg = X_data.clone()
                intervention_neg[:, :, i] = -1
                mu, logvar, adj_now_int, adj_lag_int, likelihood = self.encode(intervention_neg, T_data)
                Z = mu + torch.randn_like(logvar, dtype=torch.float32, device=device) * torch.exp(0.5 * logvar)
                recon_X = self.decode(Z, adj_now_int, adj_lag_int)
    
                # Compute normalized deviation for negative intervention
                counterfactual_results_negative[label] = torch.abs((X_data[:, :, target_idx] - recon_X[:, :, target_idx]) / (X_data[:, :, target_idx] + 1e-6)).mean().item()
    
            # Compute composite counterfactual score
            counterfactual_rankings = {key: counterfactual_results_positive[key] + counterfactual_results_negative[key] for key in counterfactual_results_positive}
    
            # Sort by impact score
            sorted_keys = sorted(counterfactual_rankings, key=counterfactual_rankings.get, reverse=True)
            counterfactual_rankings = {key: counterfactual_rankings[key] for key in sorted_keys}
            sorted_keys = sorted(edge_strengths, key=edge_strengths.get, reverse=True)
            top_causes = {key: edge_strengths[key] for key in sorted_keys}
            sorted_keys = sorted(instantaneous_edge_strengths, key=instantaneous_edge_strengths.get, reverse=True)
            instantaneous_effects = {key: instantaneous_edge_strengths[key] for key in sorted_keys}
            sorted_keys = sorted(lagged_edge_strengths, key=lagged_edge_strengths.get, reverse=True)
            lagged_effects = {key: lagged_edge_strengths[key] for key in sorted_keys}
            
                
        return top_causes, counterfactual_rankings, instantaneous_effects, lagged_effects

    def loss_function(self, recon_X, X, mu, logvar, likelihood, adj_now, adj_lag, epoch, max_epochs, rho_max=30.0, alpha_max=15.0, lambda_prior=5.0):
        """Loss function including reconstruction, KL divergence, DAG penalty, and prior regularization"""
        recon_loss = F.mse_loss(recon_X, X, reduction='sum')
        beta = min(1.0, epoch / max_epochs)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
        # Regularization on adjacency matrix (pretrained `prior_adj` should remain mostly intact)
        #prior_loss = lambda_prior * (torch.norm(adj_now - self.prior_adj, p=1) + torch.norm(adj_lag - self.prior_adj, p=1))
    
        sparsity_loss = torch.norm(adj_now, p=1) * self.instantaneous_weight + torch.norm(adj_lag, p=1) * self.lag_weight
        h_value = (notears_constraint(adj_now) * self.instantaneous_weight + notears_constraint(adj_lag)) * self.lag_weight
    
        self.rho = min(rho_max, 1.0 + (epoch / max_epochs) ** 2 * 15.0)
        self.alpha = min(alpha_max, (epoch / max_epochs) ** 2 * 5.0)
    
        #likelihood_loss = (-likelihood.log_prob(X[:, 0, :]).mean()) + 5.0
        likelihood_loss = 0
    
        lagrangian_loss = (self.alpha * h_value + 0.5 * self.rho * (h_value ** 2)) / (self.num_nodes ** 2)


        return recon_loss , kl_loss , sparsity_loss , lagrangian_loss, likelihood_loss

    def train_model(self, dataloader, optimizer, num_epochs=100, patience=10,BATCH_SIZE=64,rho_max=30.0,alpha_max=15.0):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (X_batch, time_batch) in enumerate(dataloader):
                if X_batch.shape[0] < X_batch.shape[2]:
                    continue
                optimizer.zero_grad()
                recon_X, mu, logvar, adj_now, adj_lag, likelihood = self.forward(X_batch, time_batch)
                recon_loss , kl_loss , sparsity_loss , lagrangian_loss, likelihood_loss = self.loss_function(recon_X, X_batch, mu, logvar,likelihood, adj_now, 
                                                                                            adj_lag, epoch, num_epochs,rho_max,alpha_max)
                loss = recon_loss + kl_loss + sparsity_loss + lagrangian_loss # + likelihood_loss
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.adj_mat = scale_tensor(adj_now * 0.5 + adj_lag * 0.5)

            avg_loss = total_loss / len(dataloader)
            if epoch % 250 == 0:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
                print(f"Recon Loss ={recon_loss:.4f}, KL Loss = {kl_loss:.4f}, Sparsity Loss = {sparsity_loss:.4f}, Lagrangian Loss = {lagrangian_loss:.4f}, Likelihood Loss = {likelihood_loss:.4f}") 

            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. Last Epoch: " + str(epoch) )
                    print(f"Recon Loss ={recon_loss:.4f}, KL Loss = {kl_loss:.4f}, Sparsity Loss = {sparsity_loss:.4f}, Lagrangian Loss = {lagrangian_loss:.4f}, Likelihood Loss = {likelihood_loss:.4f}") 
                    break
