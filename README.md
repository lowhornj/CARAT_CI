# CARAT_CI
Causal AutoRegressive ATtention with Automatic Causal Inference

Praxis Research by Jeremiah Lowhorn

Causal discovery & causal inference through a Variational Autoencoder & graph learning model. Observational time series data is learned via the VAE while the graph learning module learns a direct and indirect adjacency matrix. Instantaneous adjacency and lagged adjacency components are learned making the learned causal graph robust to direct and temporal causal effects. The lagged adjacency matrices are aggregated via weight decay. DAGs are optimized using a NOTEARS style loss function in additional to the VAE reparameterization trick (reconstruction & KL Divergence). Inference is built into the VAE and allows for the measurement of edge strengths between the instantaneous adjacency, lagged adjacency, combined adjacency, and a negative & positive counterfactual analysis. Scores can be combined for each of the methods for a robust root cause identification. 
