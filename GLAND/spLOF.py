import numpy as np
import faiss
import matplotlib.pyplot as plt
import scanpy as sc

def filter_points_with_lof(
    features,
    spatial_coords,
    adj,
    adata=None,
    ground_truth_labels=None,
    k=13,
    filter_method='threshold',
    spLOF_threshold=1.8,
    dataset='',
    show_spatial_plot=False
):
    """
    Filters outlier points ("false neighbors") using the LOF strategy.
    Supports two filtering modes: fixed threshold and percentage.

    Args:
    - features (np.ndarray): Coordinates in the feature space.
    - spatial_coords (np.ndarray): Spatial coordinates.
    - adj (np.ndarray): The original adjacency matrix.
    - adata (anndata.AnnData, optional): AnnData object for spatial visualization.
    - ground_truth_labels (np.ndarray, optional): Label information for statistical analysis.
    - k (int): Number of neighbors for LOF calculation.
    - filter_method (str): Filtering method, either 'threshold' or 'percent'.
    - spLOF_threshold (float): The value used for filtering.
        - If filter_method='threshold', this is the LOF threshold (e.g., 1.8).
        - If filter_method='percent', this is the percentage of top LOF points to filter (e.g., 10).
    - dataset (str, optional): Dataset name, used to dynamically adjust point size in plots.
    - show_spatial_plot (bool): Whether to display the spatial visualization of filtered points.

    Returns:
    - filtered_adj (np.ndarray): The filtered adjacency matrix.
    """
    n_points = features.shape[0]
    spatial_coords_float32 = np.array(spatial_coords, dtype=np.float32)

    # Compute spatial neighbors using Faiss
    index = faiss.IndexFlatL2(spatial_coords_float32.shape[1])
    index.add(spatial_coords_float32)
    _, spatial_indices = index.search(spatial_coords_float32, k)

    # Calculate d2: average feature distance for each point to its spatial neighbors
    d2 = np.zeros(n_points)
    for i in range(n_points):
        neighbor_indices = spatial_indices[i]
        feature_distances = np.linalg.norm(features[i] - features[neighbor_indices], axis=1)
        d2[i] = np.mean(feature_distances)

    # Calculate the LOF value for each point
    lof_values = np.zeros(n_points)
    for i in range(n_points):
        neighbors = spatial_indices[i]
        lof_values[i] = np.mean([d2[i] / d2[neighbor] for neighbor in neighbors])

    # Pre-filtering statistics
    print(f"Total number of points before filtering: {n_points}")
    has_labels = ground_truth_labels is not None and len(ground_truth_labels) == n_points
    if has_labels:
        unique_labels = np.unique(ground_truth_labels)
        print("\nNumber of points per label before filtering:")
        for label in unique_labels:
            print(f"Label {label}: {np.sum(ground_truth_labels == label)} points")
        print("\nAverage LOF value per label:")
        for label in unique_labels:
            label_indices = np.where(ground_truth_labels == label)[0]
            avg_lof = np.mean(lof_values[label_indices]) if len(label_indices) > 0 else 0
            print(f"Label {label}: Average LOF = {avg_lof:.4f}")

    # Determine filter threshold
    if filter_method == 'threshold':
        threshold_value = spLOF_threshold
        filter_indices = np.where(lof_values > threshold_value)[0]
        print(f"\nFiltering using a fixed threshold: {threshold_value}")
    elif filter_method == 'percent':
        top_percent = spLOF_threshold
        threshold_value = np.percentile(lof_values, 100 - top_percent)
        filter_indices = np.where(lof_values > threshold_value)[0]
        print(f"\nFiltering top {top_percent}% by percentage, calculated threshold is: {threshold_value:.4f}")
    else:
        raise ValueError("filter_method must be either 'threshold' or 'percent'")

    # Apply the filter
    filtered_adj = adj.copy()
    filtered_adj[filter_indices, :] = 0
    # filtered_adj[:, filter_indices] = 0

    # Post-filtering statistics
    retained_indices_mask = np.any(filtered_adj != 0, axis=1)
    total_points_after = np.sum(retained_indices_mask)
    print(f"\nTotal number of points after filtering: {total_points_after}")
    print(f"Number of points filtered out: {n_points - total_points_after}")
    adata.obs["lof_score"] = lof_values
    adata.obs["lof_outlier"] = "Retained"
    adata.obs.iloc[filter_indices, adata.obs.columns.get_loc("lof_outlier")] = "Filtered"

    if has_labels:
        print("\nNumber of points per label after filtering:")
        for label in unique_labels:
            retained_mask = (ground_truth_labels == label) & retained_indices_mask
            print(f"Label {label}: {np.sum(retained_mask)} points")

    # Visualize LOF distribution
    plt.figure(figsize=(10, 6))
    plt.hist(lof_values, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(threshold_value, color='red', linestyle='--', label=f'Filter Threshold: {threshold_value:.4f}')
    plt.title('Distribution of LOF Values')
    plt.xlabel('LOF Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Optional spatial visualization
    if show_spatial_plot and adata is not None:
        match = re.match(r'^JBO(\d+)', dataset) if dataset else None
        chosen_size = 50 if match and int(match.group(1)) < 6 else (300 if match else 6)

        print("\nGenerating spatial visualization plots...")
        if has_labels and 'ground_truth' in adata.obs.columns:
            sc.pl.spatial(
                adata, img_key="hires", color=["ground_truth"],
                title=["Ground Truth"], spot_size=chosen_size, show=True
            )

        sc.pl.spatial(
            adata, img_key="hires", color="lof_outlier",
            title=["LOF Filtering Result"], spot_size=chosen_size,
            palette={"Filtered": "red", "Retained": "lightgray"}, show=True
        )

    return filtered_adj
