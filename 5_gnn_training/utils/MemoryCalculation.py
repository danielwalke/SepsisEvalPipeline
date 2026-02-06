def calculate_loader_params(max_ram_gb, batch_size, num_workers, num_hops, neighbor_limit, num_features=7):
    if neighbor_limit == 1:
        max_nodes_per_seed = num_hops + 1
    else:
        max_nodes_per_seed = (neighbor_limit**(num_hops + 1) - 1) / (neighbor_limit - 1)
    
    max_edges_per_seed = max_nodes_per_seed * 3
    node_bytes = (num_features * 4) + 4 + 8 
    edge_bytes = 8 + 8 + 4
    subgraph_bytes = (max_nodes_per_seed * node_bytes) + (max_edges_per_seed * edge_bytes)
    total_bytes_per_seed = subgraph_bytes * 4
    total_bytes_per_batch = total_bytes_per_seed * batch_size
    max_ram_bytes = max_ram_gb * (1024**3)
    max_batches_in_ram = max_ram_bytes / total_bytes_per_batch
    safe_batches = int(max_batches_in_ram * 0.8)
    
    if num_workers > 0:
        prefetch_factor = safe_batches // num_workers
    else:
        prefetch_factor = safe_batches
    return max(2, min(prefetch_factor, 50))