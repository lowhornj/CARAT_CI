import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GATConv
from torch_geometric.utils import dense_to_sparse
from torch.distributions import Normal, Laplace, RelaxedOneHotCategorical
from torchdiffeq import odeint  # For continuous-time normalizing flows
from CARAT.components import *
from CARAT.model_utils import *
import pandas as pd

def generate_causal_mask(seq_len, device):
    """Creates a causal mask with -inf values for masked positions."""
    mask = torch.triu(torch.full((seq_len, seq_len), float('-inf')), diagonal=1)
    return mask.to(device)


class CausalGraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_nodes, device, time_steps=10, prior_adj=None,instantaneous_weight=0.5,heads=4,dropout=0.3,use_attention=False):
        super(CausalGraphVAE, self).__init__()
        self.device = device
        self.instantaneous_weight = instantaneous_weight
        self.lag_weight = 1.0 - self.instantaneous_weight
        self.time_steps = time_steps
        self.num_nodes = num_nodes
        self.latent_dim = latent_dim
        self.prior_adj = prior_adj
        self.causal_graph = TemporalCausalGraph(num_nodes, hidden_dim, latent_dim, device=self.device,time_steps=time_steps ,prior_adj=prior_adj,instantaneous_weight=self.instantaneous_weight,lag_weight=self.lag_weight, use_attention=use_attention)
        self.alpha = torch.tensor(0.0, dtype=torch.float32, requires_grad=False, device=self.device)
        self.rho = torch.tensor(1.0, dtype=torch.float32, requires_grad=False, device=self.device)

        self.pos_embedding = nn.Embedding(time_steps, hidden_dim,dtype=torch.float32,device=self.device)

        self.enocder_projection =nn.Linear(num_nodes, hidden_dim,dtype=torch.float32, device=self.device)
        self.encoder_norm = nn.LayerNorm(hidden_dim, device=self.device)
      
        self.encoder_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim, dtype=torch.float32,dropout=0.3,device=self.device),
            num_layers=2
        )

        # Temporal-aware Encoder and Decoder
        #self.encoder_rnn = nn.GRU(input_dim, hidden_dim, batch_first=True,dtype=torch.float32, device=self.device,num_layers =1,dropout =0)
        self.mu_layer = nn.Linear(hidden_dim, latent_dim,dtype=torch.float32, device=self.device)
        self.logvar_layer = nn.Linear(hidden_dim, latent_dim,dtype=torch.float32, device=self.device)

        self.decoder_projection = nn.Linear(latent_dim, hidden_dim,dtype=torch.float32, device=self.device)
        self.decoder_norm = nn.LayerNorm(hidden_dim, device=self.device)

        self.decoder_transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=heads, dim_feedforward=hidden_dim, dtype=torch.float32,dropout=0.3,device=self.device),
            num_layers=2
        )

        self.decoder_fc = nn.Linear(hidden_dim, input_dim,dtype=torch.float32, device=self.device)

    def encode(self, X, time_context):
        X_enc = self.enocder_projection(X)
        X_enc = self.encoder_norm(X_enc)
        pos_indices = torch.arange(self.time_steps, device=X.device)
        pos_emb = self.pos_embedding(pos_indices).unsqueeze(1)
    
        X_permuted = X_enc.permute(1, 0, 2)  # Transformer expects (seq_len, batch, hidden_dim)
    
        # Generate the causal mask for the encoder
        src_mask = generate_causal_mask(self.time_steps, X.device)
    
        # Apply Transformer Encoder with causal mask
        memory = self.encoder_transformer(X_permuted + pos_emb)
    
        mu, logvar = self.mu_layer(memory[-1, :, :]), self.logvar_layer(memory[-1, :, :])
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
    
        adj_now, adj_lag = self.causal_graph(memory, time_context)
    
        return mu, logvar, adj_now, adj_lag, memory.detach()

    def decode(self, Z, adj_now, adj_lag, memory):
        Z_expanded = Z.unsqueeze(1).repeat(1, self.time_steps, 1)
        X_dec = self.decoder_projection(Z_expanded)
        X_dec = self.decoder_norm(X_dec)
        X_permuted = X_dec.permute(1, 0, 2)
    
        # Generate new positional embedding for the decoder
        pos_indices = torch.arange(self.time_steps, device=X_dec.device)
        tgt_pos_emb = self.pos_embedding(pos_indices).unsqueeze(1)
    
        # Generate causal mask
        tgt_mask = generate_causal_mask(self.time_steps, X_dec.device)
    
        # Apply Transformer Decoder
        X_transformed = self.decoder_transformer(X_permuted + tgt_pos_emb, memory, tgt_mask=tgt_mask)
    
        X_transformed = X_transformed.permute(1, 0, 2)
        x_recon = self.decoder_fc(X_transformed)
    
        return x_recon

    def forward(self, X, time_context):
        X = X.to(torch.float32)  # Convert input X to float32
        time_context = time_context.to(torch.float32)  # Convert time context to float32

        mu, logvar, adj_now, adj_lag, X_transformed = self.encode(X, time_context)
        Z = mu + torch.randn_like(logvar) * torch.exp(0.5 * logvar)
        recon_X = self.decode(Z, adj_now, adj_lag, X_transformed)
        return recon_X, mu, logvar, adj_now, adj_lag

    def counterfactual_estimation(self, X_data, T_data, target_idx, labels, non_causal_indices=[]):
    # Multi-scale interventions instead of just +1/-1
        intervention_scales = [-2.0, -1.0, -0.5, 0.5, 1.0, 2.0]
        counterfactual_effects = {}
        
        # Baseline prediction without intervention
        with torch.no_grad():
            mu_base, logvar_base, adj_now_base, adj_lag_base, memory_base = self.encode(X_data, T_data)
            Z_base = mu_base + torch.randn_like(logvar_base, dtype=torch.float32, device=self.device) * torch.exp(0.5 * logvar_base)
            recon_X_base = self.decode(Z_base, adj_now_base, adj_lag_base, memory_base)
            baseline_target = recon_X_base[:, :, target_idx]
        
        for i in range(self.num_nodes):
            if i == target_idx or i in non_causal_indices:
                continue
            
            label = labels[i]
            effects = []
            
            # Get original distribution statistics for this variable
            var_mean = X_data[:, :, i].mean().item()
            var_std = X_data[:, :, i].std().item()
            
            # For each intervention scale
            for scale in intervention_scales:
                # Create intervention that respects original distribution
                intervention = X_data.clone()
                
                # Apply intervention as a shift in standard deviations
                intervention_value = var_mean + scale * var_std
                intervention[:, :, i] = intervention_value
                
                # Run through model
                mu_int, logvar_int, adj_now_int, adj_lag_int, memory_int = self.encode(intervention, T_data)
                Z_int = mu_int + torch.randn_like(logvar_int, dtype=torch.float32, device=self.device) * torch.exp(0.5 * logvar_int)
                recon_X_int = self.decode(Z_int, adj_now_int, adj_lag_int, memory_int)
                
                # Measure both absolute and relative changes
                # 1. Absolute difference
                abs_diff = torch.abs(recon_X_int[:, :, target_idx] - baseline_target).mean().item()
                
                # 2. Relative difference (percent change)
                rel_diff = torch.abs((recon_X_int[:, :, target_idx] - baseline_target) / 
                                    (torch.abs(baseline_target) + 1e-6)).mean().item()
                
                # 3. Distribution shift (KL divergence approximation)
                if baseline_target.std() > 1e-6 and recon_X_int[:, :, target_idx].std() > 1e-6:
                    # Simple Gaussian approximation of KL divergence
                    mu1 = baseline_target.mean()
                    sigma1 = baseline_target.std()
                    mu2 = recon_X_int[:, :, target_idx].mean()
                    sigma2 = recon_X_int[:, :, target_idx].std()
                    kl_div = torch.log(sigma2/sigma1) + (sigma1**2 + (mu1 - mu2)**2)/(2*sigma2**2) - 0.5
                    kl_effect = kl_div.abs().item()
                else:
                    kl_effect = 0.0
                
                # Combine metrics
                combined_effect = (abs_diff + rel_diff + kl_effect) / 3
                effects.append(combined_effect)
            
            # Calculate effect as area under intervention curve
            total_effect = sum(effects)
            
            # Scale effect by intervention responsiveness
            # This helps distinguish direct from indirect causes
            responsiveness = torch.std(torch.tensor(effects)).item() / (torch.mean(torch.tensor(effects)).item() + 1e-6)
            weighted_effect = total_effect * (1 + responsiveness)
            
            counterfactual_effects[label] = weighted_effect
        
        return counterfactual_effects

    def infer_causal_effect(self, X_data, T_data, target_variable, labels, non_causal_indices=[],root_rank=False):
        """Infers the top causal factors for a given target variable using counterfactual analysis."""
        
        try:
            target_idx = labels.index(target_variable, 0)
        except:
            return target_variable + " not in index"
    
        self.eval()
        n_samples = X_data.shape[0]
    
        with torch.no_grad():
            _, _, adj_now, adj_lag, _ = self.encode(X_data, T_data)
            adj_matrix = (adj_now * self.instantaneous_weight + adj_lag * self.lag_weight)
    
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
  
            counterfactual_rankings = self.counterfactual_estimation( X_data, T_data, target_idx, labels, non_causal_indices=[])
               
    
            # Sort by impact score
            sorted_keys = sorted(counterfactual_rankings, key=counterfactual_rankings.get, reverse=True)
            counterfactual_rankings = {key: counterfactual_rankings[key] for key in sorted_keys}
            sorted_keys = sorted(edge_strengths, key=edge_strengths.get, reverse=True)
            top_causes = {key: edge_strengths[key] for key in sorted_keys}
            sorted_keys = sorted(instantaneous_edge_strengths, key=instantaneous_edge_strengths.get, reverse=True)
            instantaneous_effects = {key: instantaneous_edge_strengths[key] for key in sorted_keys}
            sorted_keys = sorted(lagged_edge_strengths, key=lagged_edge_strengths.get, reverse=True)
            lagged_effects = {key: lagged_edge_strengths[key] for key in sorted_keys}
 
            l, v = np.linalg.eig(adj_matrix.cpu().detach().numpy().T)
            score = (v[:,0]/np.sum(v[:,0])).real
            score=np.around(score/np.sum(score),decimals=3)
            scores_df = pd.DataFrame({'Column':labels,'RootRank':score})
            root_rank_score = scores_df.sort_values(by="RootRank", ascending=False)
            root_rank_score.set_index('Column', inplace=True)
            root_rank_score=(root_rank_score-root_rank_score.min())/(root_rank_score.max()-root_rank_score.min())

            causes_df = pd.DataFrame.from_dict(top_causes, columns=['causes'],orient='index')
            causes_df=(causes_df-causes_df.min())/(causes_df.max()-causes_df.min())
            isntantaneous_df = pd.DataFrame.from_dict(instantaneous_effects, columns=['instantaneous'],orient='index')
            isntantaneous_df=(isntantaneous_df-isntantaneous_df.min())/(isntantaneous_df.max()-isntantaneous_df.min())
            lagged_df = pd.DataFrame.from_dict(lagged_effects, columns=['lagged'],orient='index')
            lagged_df=(lagged_df-lagged_df.min())/(lagged_df.max()-lagged_df.min())
            
            counterfactual_df = pd.DataFrame.from_dict(counterfactual_rankings, columns=['counterfactuals'],orient='index')
            counterfactual_df=(counterfactual_df-counterfactual_df.min())/(counterfactual_df.max()-counterfactual_df.min())

            if root_rank:
                total_score = pd.concat([causes_df, isntantaneous_df, lagged_df, counterfactual_df,root_rank_score], axis=1)
                total_score['causal_strength'] = total_score.mean(axis=1)
                total_score=total_score.sort_values(by='causal_strength',ascending=False)
            else:
                total_score = pd.concat([causes_df, isntantaneous_df, lagged_df, counterfactual_df], axis=1)
                total_score['causal_strength'] = total_score.mean(axis=1)
                total_score=total_score.sort_values(by='causal_strength',ascending=False)
                total_score = pd.concat([total_score, root_rank_score], axis=1)
                
                
        return total_score

    def improved_acyclicity_constraint(self, adj_mat):
        """
        Implements a smoother acyclicity constraint using a power series approximation
        that's more stable than the original NOTEARS.
        """
        # Identity matrix
        m = adj_mat.shape[0]
        I = torch.eye(m, device=adj_mat.device)
        
        # Matrix exponential approximation via power series
        # We compute (I + A + A²/2! + A³/3! + ...) - I
        power_sum = torch.zeros_like(I)
        A_power = adj_mat.clone()
        factorial = 1.0
        
        # Use 5 terms in the power series (can adjust based on needs)
        for i in range(1, 6):
            factorial = factorial * i if i > 1 else 1.0
            power_sum = power_sum + A_power / factorial
            A_power = A_power @ adj_mat
        
        # Compute sum of diagonal elements for directed cycles
        cycle_measure = torch.trace(power_sum)
        
        return cycle_measure

    def loss_function(self, recon_X, X, mu, logvar, adj_now, adj_lag, epoch, max_epochs, rho_max=30.0, alpha_max=15.0, lambda_prior=5.0):
        """Loss function including reconstruction, KL divergence, DAG penalty, and prior regularization"""
        recon_loss = F.mse_loss(recon_X, X, reduction='sum')
        beta = min(1.0, epoch / (max_epochs * 0.3))  # Gradual increase in KL weight
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) * beta
    
        # Regularization on adjacency matrix (pretrained `prior_adj` should remain mostly intact)
        #prior_loss = lambda_prior * (torch.norm(adj_now - self.prior_adj, p=1) + torch.norm(adj_lag - self.prior_adj, p=1))
    
        #sparsity_loss = torch.norm(adj_now, p=1) * self.instantaneous_weight + torch.norm(adj_lag, p=1) * self.lag_weight
        #h_value = (notears_constraint(adj_now) * self.instantaneous_weight + notears_constraint(adj_lag)) * self.lag_weight
    
        #self.rho = min(rho_max, 1.0 + (epoch / max_epochs) ** 2)
        #self.alpha = min(alpha_max, (epoch / max_epochs) ** 2)

    
        #lagrangian_loss = (self.alpha * h_value + 0.5 * self.rho * (h_value ** 2)) #/ (self.num_nodes ** 2)

        h_now = self.improved_acyclicity_constraint(adj_now)
        h_lag = self.improved_acyclicity_constraint(adj_lag)
        h_value = h_now * self.instantaneous_weight + h_lag * self.lag_weight
        
        # Adaptive weights for DAG constraint
        self.rho = min(rho_max, 1.0 + (epoch / max_epochs) ** 2)
        self.alpha = min(alpha_max, (epoch / max_epochs) ** 2)
        
        # Lagrangian loss
        lagrangian_loss = self.alpha * h_value + 0.5 * self.rho * (h_value ** 2)


        return recon_loss , kl_loss, lagrangian_loss

    def train_model(self, dataloader, optimizer, scheduler=None, num_epochs=100, patience=10,BATCH_SIZE=64,rho_max=30.0,alpha_max=15.0):
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(num_epochs):
            total_loss = 0
            for batch_idx, (X_batch, time_batch) in enumerate(dataloader):
                if X_batch.shape[0] < X_batch.shape[2]:
                    continue
                optimizer.zero_grad()
                recon_X, mu, logvar, adj_now, adj_lag = self.forward(X_batch, time_batch)
                recon_loss , kl_loss , lagrangian_loss = self.loss_function(recon_X, X_batch, mu, logvar, adj_now, 
                                                                                            adj_lag, epoch, num_epochs,rho_max,alpha_max)
                loss = recon_loss + kl_loss + lagrangian_loss 
                
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            self.adj_mat = scale_tensor(adj_now * 0.5 + adj_lag * 0.5)

            avg_loss = total_loss / len(dataloader)

            if scheduler is not None:
                scheduler.step(avg_loss)
                
            if epoch % 50 == 0:
                print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")
                print(f"Recon Loss = {recon_loss:.4f}, KL Loss = {kl_loss:.4f}, Lagrangian Loss = {lagrangian_loss:.4f}") 

            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("Early stopping triggered. Last Epoch: " + str(epoch) )
                    print(f"Recon Loss = {recon_loss:.4f}, KL Loss = {kl_loss:.4f}, Lagrangian Loss = {lagrangian_loss:.4f}") 
                    break
