# Causal AutoRegressive ATtention with Automated Causal Inference  

### Praxis Research by Jeremiah Lowhorn  

This research explores **causal discovery and inference** using a **Variational Autoencoder (VAE) integrated with a graph learning module**. The VAE learns representations from observational time series data, while the graph learning component constructs both **direct and indirect adjacency matrices**, capturing both **instantaneous and lagged causal effects**.  

The model's **instantaneous adjacency** reflects direct causal relationships, while **lagged adjacency matrices** account for temporal dependencies, aggregated through a **weight decay mechanism** to enhance stability. To ensure a **Directed Acyclic Graph (DAG)** structure, the model incorporates a **NOTEARS-style optimization constraint** alongside the standard **VAE reparameterization trick** (reconstruction loss & KL divergence).  

Inference is embedded within the VAE, allowing for a **comprehensive evaluation of edge strengths** across different adjacency representations: instantaneous, lagged, combined, as well as **positive and negative counterfactual interventions**. These diverse scoring methods can be **combined** to provide a **robust root cause identification framework**, offering enhanced interpretability and precision in causal reasoning.
