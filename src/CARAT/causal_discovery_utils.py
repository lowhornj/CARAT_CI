import torch
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats

def compute_partial_correlations(data, significance_level=0.05):
    """
    Computes partial correlation matrix and significance masks.
    
    Args:
        data: pandas DataFrame containing the variables
        significance_level: p-value threshold for significance
        
    Returns:
        partial_corr: partial correlation matrix
        significance_mask: boolean mask of significant correlations
    """
    n_vars = data.shape[1]
    n_samples = data.shape[0]
    
    # Convert to numpy for calculations
    data_np = data.values
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(data_np, rowvar=False)
    
    # Initialize partial correlation matrix
    partial_corr = np.zeros((n_vars, n_vars))
    p_values = np.zeros((n_vars, n_vars))
    
    # Compute partial correlations
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            # Get conditioning set (all other variables)
            cond_set = [k for k in range(n_vars) if k != i and k != j]
            
            if len(cond_set) > 0:
                # Compute partial correlation
                x = data_np[:, i]
                y = data_np[:, j]
                z = data_np[:, cond_set]
                
                # Residualize x and y with respect to z
                beta_x = np.linalg.lstsq(z, x, rcond=None)[0]
                beta_y = np.linalg.lstsq(z, y, rcond=None)[0]
                
                res_x = x - z @ beta_x
                res_y = y - z @ beta_y
                
                # Compute correlation between residuals
                corr, p_value = stats.pearsonr(res_x, res_y)
                
                partial_corr[i, j] = corr
                partial_corr[j, i] = corr
                p_values[i, j] = p_value
                p_values[j, i] = p_value
            else:
                # No conditioning, use regular correlation
                partial_corr[i, j] = corr_matrix[i, j]
                partial_corr[j, i] = corr_matrix[j, i]
                
                # Compute p-value for Pearson correlation
                t_stat = partial_corr[i, j] * np.sqrt((n_samples - 2) / (1 - partial_corr[i, j]**2))
                p_values[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n_samples - 2))
                p_values[j, i] = p_values[i, j]
    
    # Create significance mask
    significance_mask = p_values < significance_level
    
    return partial_corr, significance_mask

def pc_algorithm_simplified(data, significance_level=0.05):
    """
    Simplified PC algorithm for causal discovery.
    
    Args:
        data: pandas DataFrame containing the variables
        significance_level: p-value threshold for independence tests
        
    Returns:
        adjacency_matrix: estimated causal graph
    """
    n_vars = data.shape[1]
    var_names = data.columns
    
    # Step 1: Start with complete undirected graph
    graph = nx.complete_graph(n_vars)
    
    # Step 2: Edge removal based on conditional independence
    partial_corr, significance_mask = compute_partial_correlations(data, significance_level)
    
    # Remove edges where partial correlation is not significant
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            if not significance_mask[i, j]:
                if graph.has_edge(i, j):
                    graph.remove_edge(i, j)
    
    # Step 3: Edge orientation using simple rules
    # This is a simplified version - full PC has more rules
    
    # Create directed graph
    dag = nx.DiGraph()
    dag.add_nodes_from(range(n_vars))
    
    # Identify v-structures (colliders)
    for i in range(n_vars):
        for j in range(n_vars):
            if i != j and not graph.has_edge(i, j):
                # Find common neighbors
                common_neighbors = set(graph.neighbors(i)) & set(graph.neighbors(j))
                
                for k in common_neighbors:
                    # Check if i-k-j forms a v-structure (i→k←j)
                    if partial_corr[i, j] < partial_corr[i, k] and partial_corr[i, j] < partial_corr[j, k]:
                        dag.add_edge(i, k)
                        dag.add_edge(j, k)
    
    # Add remaining edges with a direction based on time-ordering heuristic
    # For time series, we assume earlier variables cause later ones
    for i, j in graph.edges():
        if not (dag.has_edge(i, j) or dag.has_edge(j, i)):
            # Add edge in direction of higher index (assuming time ordering)
            if i < j:
                dag.add_edge(i, j)
            else:
                dag.add_edge(j, i)
    
    # Convert to adjacency matrix
    adjacency_matrix = np.zeros((n_vars, n_vars))
    for i, j in dag.edges():
        adjacency_matrix[i, j] = 1
    
    return adjacency_matrix, var_names

def greedy_equivalence_search(data, score_function='bic', max_parents=3):
    """
    Greedy Equivalence Search (GES) algorithm for causal discovery.
    
    Args:
        data: pandas DataFrame containing the variables
        score_function: scoring function ('bic' or 'aic')
        max_parents: maximum number of parents for any node
        
    Returns:
        adjacency_matrix: estimated causal graph
    """
    n_vars = data.shape[1]
    var_names = data.columns
    n_samples = data.shape[0]
    
    # Define scoring function
    def compute_local_score(node, parents, data):
        if len(parents) == 0:
            # No parents, use variance of node
            variance = np.var(data.iloc[:, node])
            if score_function == 'bic':
                return -0.5 * n_samples * np.log(variance) - 0.5 * np.log(n_samples)
            else:  # aic
                return -0.5 * n_samples * np.log(variance) - 1
        else:
            # Compute regression from parents to node
            X = data.iloc[:, parents].values
            y = data.iloc[:, node].values
            
            # Add constant term
            X = np.column_stack([np.ones(n_samples), X])
            
            # Compute least squares
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            variance = np.mean(residuals**2)
            
            # Compute score
            k = len(parents) + 1  # +1 for intercept
            if score_function == 'bic':
                return -0.5 * n_samples * np.log(variance) - 0.5 * k * np.log(n_samples)
            else:  # aic
                return -0.5 * n_samples * np.log(variance) - k
    
    # Initialize empty graph
    parents = [[] for _ in range(n_vars)]
    best_score = sum(compute_local_score(i, [], data) for i in range(n_vars))
    
    # Forward phase: add edges
    improved = True
    while improved:
        improved = False
        best_addition = None
        best_addition_score = best_score
        
        # Try adding each possible edge
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j and j not in parents[i] and len(parents[i]) < max_parents:
                    # Add edge j→i
                    new_parents_i = parents[i] + [j]
                    
                    # Check for cycles
                    if not has_cycle(i, new_parents_i, parents):
                        # Compute new score
                        new_score = best_score - compute_local_score(i, parents[i], data) + compute_local_score(i, new_parents_i, data)
                        
                        if new_score > best_addition_score:
                            best_addition = (j, i)
                            best_addition_score = new_score
        
        # Add the best edge if it improves the score
        if best_addition_score > best_score:
            j, i = best_addition
            parents[i].append(j)
            best_score = best_addition_score
            improved = True
    
    # Backward phase: remove edges
    improved = True
    while improved:
        improved = False
        best_removal = None
        best_removal_score = best_score
        
        # Try removing each existing edge
        for i in range(n_vars):
            for j in parents[i]:
                # Remove edge j→i
                new_parents_i = [p for p in parents[i] if p != j]
                
                # Compute new score
                new_score = best_score - compute_local_score(i, parents[i], data) + compute_local_score(i, new_parents_i, data)
                
                if new_score > best_removal_score:
                    best_removal = (j, i)
                    best_removal_score = new_score
        
        # Remove the best edge if it improves the score
        if best_removal_score > best_score:
            j, i = best_removal
            parents[i] = [p for p in parents[i] if p != j]
            best_score = best_removal_score
            improved = True
    
    # Convert to adjacency matrix
    adjacency_matrix = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        for j in parents[i]:
            adjacency_matrix[j, i] = 1
    
    return adjacency_matrix, var_names

def has_cycle(node, new_parents, all_parents):
    """Check if adding an edge creates a cycle in the graph."""
    visited = set()
    stack = new_parents.copy()
    
    while stack:
        current = stack.pop()
        if current == node:
            return True
        if current not in visited:
            visited.add(current)
            stack.extend(all_parents[current])
    
    return False

def plot_causal_graph(adjacency_matrix, var_names=None, threshold=0.1, title="Causal Graph"):
    """
    Visualizes a causal graph from an adjacency matrix.
    
    Args:
        adjacency_matrix: numpy array or pytorch tensor
        var_names: list of variable names
        threshold: threshold for considering an edge present
        title: plot title
    """
    if isinstance(adjacency_matrix, torch.Tensor):
        adjacency_matrix = adjacency_matrix.cpu().detach().numpy()
    
    n_nodes = adjacency_matrix.shape[0]
    
    if var_names is None:
        var_names = [f"X{i}" for i in range(n_nodes)]
    
    # Create graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i, label=var_names[i])
    
    # Add edges with weights above threshold
    for i in range(n_nodes):
        for j in range(n_nodes):
            if adjacency_matrix[i, j] > threshold:
                G.add_edge(i, j, weight=float(adjacency_matrix[i, j]))
    
    # Set up the plot
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)  # for reproducible layout
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')
    
    # Draw node labels
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['label'] for i in G.nodes})
    
    # Draw edges with width proportional to weight
    edge_weights = [G[u][v]['weight'] * 3 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_weights, arrowsize=20, alpha=0.7)
    
    # Draw edge weights
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)
    
    plt.title(title)
    plt.axis('off')
    
    return plt

def ensemble_causal_discovery(data, methods=['pc', 'ges', 'notears'], threshold=0.3):
    """
    Performs ensemble causal discovery using multiple methods.
    
    Args:
        data: pandas DataFrame with variables
        methods: list of methods to use
        threshold: threshold for edge confidence
        
    Returns:
        ensemble_graph: combined causal graph
    """
    n_vars = data.shape[1]
    var_names = data.columns
    
    # Initialize ensemble adjacency matrix
    ensemble_adj = np.zeros((n_vars, n_vars))
    method_results = {}
    
    # Run each method
    for method in methods:
        if method == 'pc':
            adj, _ = pc_algorithm_simplified(data)
            method_results['pc'] = adj
            
        elif method == 'ges':
            adj, _ = greedy_equivalence_search(data)
            method_results['ges'] = adj
            
        elif method == 'notears':
            # This is a placeholder - you'd need to implement NOTEARS or use a library
            # Here we just generate a synthetic result based on correlations
            corr_matrix = np.abs(np.corrcoef(data.values, rowvar=False))
            np.fill_diagonal(corr_matrix, 0)
            adj = np.zeros_like(corr_matrix)
            # Assign direction based on simple heuristic (higher index causes lower)
            for i in range(n_vars):
                for j in range(i):
                    if corr_matrix[i, j] > 0.3:
                        adj[j, i] = corr_matrix[i, j]
            method_results['notears'] = adj
        
        # Add to ensemble
        ensemble_adj += adj
    
    # Normalize ensemble
    ensemble_adj /= len(methods)
    
    # Apply threshold
    ensemble_adj = (ensemble_adj > threshold) * ensemble_adj
    
    return ensemble_adj, var_names, method_results

def intervention_effect_analysis(model, data, target_node, intervention_nodes, 
                                intervention_values=None, num_samples=100):
    """
    Analyzes causal effects using interventional analysis.
    
    Args:
        model: trained causal model
        data: baseline data
        target_node: index of target variable
        intervention_nodes: list of node indices to intervene on
        intervention_values: list of values for interventions (if None, uses multiple values)
        num_samples: number of samples per intervention
        
    Returns:
        effects: dictionary of causal effects
    """
    if not hasattr(model, 'eval'):
        raise ValueError("Model must have an eval() method")
    
    model.eval()
    
    # Default intervention values
    if intervention_values is None:
        # Use percentiles of observed data for interventions
        intervention_values = []
        for node in intervention_nodes:
            node_data = data[:, node].cpu().detach().numpy()
            values = np.percentile(node_data, [10, 50, 90])
            intervention_values.append(values)
    
    # Store effects
    effects = {}
    
    # Baseline prediction
    with torch.no_grad():
        baseline_output = model(data, None)[0]
        baseline_target = baseline_output[:, target_node].mean().item()
    
    # Analyze each intervention
    for i, node in enumerate(intervention_nodes):
        node_effects = []
        
        # Try different intervention values
        for value in intervention_values[i]:
            # Create intervention
            intervention_data = data.clone()
            intervention_data[:, node] = value
            
            # Get model prediction
            with torch.no_grad():
                intervention_output = model(intervention_data, None)[0]
                intervention_target = intervention_output[:, target_node].mean().item()
            
            # Calculate effect
            effect = (intervention_target - baseline_target) / baseline_target
            node_effects.append((value, effect))
        
        effects[node] = node_effects
    
    return effects