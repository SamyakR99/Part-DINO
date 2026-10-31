import os
import torch
import numpy as np
import time
from sklearn.cluster import MiniBatchKMeans
import torch.nn.functional as F
from tqdm import tqdm

def extract_foreground_features(model, dataset, segmenter, n_reference, target_k,
                                backbone='dinov2', patch_size=14, device='cuda', alpha=0.0, skip_gradient_merge=True, min_shard_size=10, smooth_ramp=False, selected_heads=False):
    """Extract intermediate region mean features for global K-Means."""
    from main import run_discovery_pipeline
    
    all_region_features = []
    
    # Shuffle indices to get a random subset of reference images
    indices = np.random.permutation(len(dataset))[:n_reference]
    
    print(f"Extracting region features from {n_reference} reference images...")
    
    for idx in tqdm(indices):
        sample = dataset[idx]
        img_tensor = sample['image'].unsqueeze(0).to(device)
        
        # 1. Get foreground mask
        if segmenter is not None:
            # Use per-image supercategory as SAM prompt if available (OOD discovery)
            supercat_prompt = sample.get('supercategory', None)
            if supercat_prompt is not None and hasattr(segmenter, 'segment_tensor_with_prompt'):
                from segmentation import get_sam3_prompt
                sam_prompt = get_sam3_prompt(supercat_prompt)
                fg_mask = segmenter.segment_tensor_with_prompt(
                    sample['image'], sam_prompt
                ).astype(bool)
            else:
                fg_mask = segmenter.segment_tensor(sample['image']).astype(bool)
        else:
            fg_mask = None

            
        # Run pipeline up to atomic regions only (by setting target_k_final very high)
        try:
            segments, mask, tokens_up, *_ = run_discovery_pipeline(
                model=model, 
                img_tensor=img_tensor, 
                original_img=sample['image'], 
                target_k_global=100,
                target_k_final=999,  # don't merge, keep atomic
                min_shard_size=min_shard_size,
                external_mask=fg_mask,
                alpha=alpha,
                backbone=backbone,
                patch_size=patch_size,
                skip_gradient_merge=skip_gradient_merge,
                global_centroids=None, # Ensure no global substitution yet
                smooth_ramp=smooth_ramp,
                selected_heads=selected_heads
            )
        except Exception as e:
            print(f"   -> SKIPPED (error: {e})")
            continue
            
        # Extract mean feature per atomic region efficiently
        unique_parts = [rid for rid in np.unique(segments) if rid != 0]
        if len(unique_parts) > 0:
            segments_flat = segments.ravel()
            tokens_flat = tokens_up.reshape(-1, tokens_up.shape[-1])
            
            valid = segments_flat != 0
            seg_valid = segments_flat[valid]
            tok_valid = tokens_flat[valid]
            
            max_id = seg_valid.max()
            sum_feat = np.zeros((max_id + 1, tok_valid.shape[-1]), dtype=tok_valid.dtype)
            np.add.at(sum_feat, seg_valid, tok_valid)
            counts = np.bincount(seg_valid, minlength=max_id + 1)
            
            valid_ids = np.array(unique_parts)
            mean_feats = sum_feat[valid_ids] / counts[valid_ids][:, None]
            all_region_features.extend(mean_feats)
        
    stacked_features = np.array(all_region_features)
    # L2 normalize the region means
    stacked_features = stacked_features / (np.linalg.norm(stacked_features, axis=1, keepdims=True) + 1e-8)
    
    print(f"Extracted a total of {stacked_features.shape[0]} region mean features.")
    
    # Run MiniBatchKMeans
    print(f"Running MiniBatchKMeans with K={target_k}...")
    t0 = time.time()
    kmeans = MiniBatchKMeans(n_clusters=target_k, random_state=42, n_init="auto", batch_size=4096)
    kmeans.fit(stacked_features)
    print(f"KMeans finished in {time.time()-t0:.2f}s")
    
    centroids = kmeans.cluster_centers_
    # L2 normalize centroids to match cosine similarity calculations later
    centroids = centroids / (np.linalg.norm(centroids, axis=-1, keepdims=True) + 1e-8)
    
    return centroids


def discover_global_centroids(model, dataset, segmenter, n_reference, target_k, 
                             save_path, backbone='dinov2', patch_size=14, device='cuda',
                             alpha=0.0, skip_gradient_merge=True, min_shard_size=10, smooth_ramp=False, selected_heads=False):
    if os.path.exists(save_path):
        print(f"Loading cached global centroids from {save_path}")
        return np.load(save_path)
        
    centroids = extract_foreground_features(
        model=model,
        dataset=dataset,
        segmenter=segmenter,
        n_reference=n_reference,
        target_k=target_k,
        backbone=backbone,
        patch_size=patch_size,
        device=device,
        alpha=alpha,
        skip_gradient_merge=skip_gradient_merge,
        min_shard_size=min_shard_size,
        smooth_ramp=smooth_ramp,
        selected_heads=selected_heads
    )
    
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    np.save(save_path, centroids)
    print(f"Saved global centroids to {save_path}")
    return centroids


def discover_all_supercat_centroids(model, dataset, segmenter, n_reference, target_k,
                                    save_dir, backbone='dinov2', patch_size=14, device='cuda',
                                    alpha=0.0, skip_gradient_merge=True, min_shard_size=10, smooth_ramp=False, selected_heads=False):
    """
    Discover per-supercategory centroids for PartImageNet_OOD (Strategy B).

    For each of the 11 supercategories, filters the dataset to images of that
    supercategory, runs discover_global_centroids() with target_k clusters, and
    saves/loads from <save_dir>/centroids_<supercat>.npy.

    Returns:
        centroids_dict: dict[str -> np.ndarray(target_k, D)]
    """
    os.makedirs(save_dir, exist_ok=True)

    # Group dataset indices by supercategory
    supercat_indices = {}
    for idx in range(len(dataset)):
        sc = dataset.database[idx]['supercategory']
        supercat_indices.setdefault(sc, []).append(idx)

    centroids_dict = {}
    for supercat, indices in sorted(supercat_indices.items()):
        save_path = os.path.join(save_dir, f'centroids_{supercat}.npy')

        if os.path.exists(save_path):
            print(f"[{supercat}] Loading cached centroids from {save_path}")
            centroids_dict[supercat] = np.load(save_path)
            continue

        n_ref = min(n_reference, len(indices))
        chosen = np.random.choice(indices, size=n_ref, replace=False).tolist()
        print(f"\n[{supercat}] Discovering centroids from {n_ref} images (total={len(indices)})...")

        # Build a lightweight wrapper so existing extract_foreground_features
        # can index into the sub-selected list transparently
        class SupercatSubset:
            def __init__(self, full_ds, chosen_indices):
                self._ds = full_ds
                self._idx = chosen_indices
            def __len__(self):
                return len(self._idx)
            def __getitem__(self, i):
                return self._ds[self._idx[i]]

        sub_dataset = SupercatSubset(dataset, chosen)

        centroids = extract_foreground_features(
            model=model,
            dataset=sub_dataset,
            segmenter=segmenter,
            n_reference=n_ref,
            target_k=target_k,
            backbone=backbone,
            patch_size=patch_size,
            device=device,
            alpha=alpha,
            skip_gradient_merge=skip_gradient_merge,
            min_shard_size=min_shard_size,
            smooth_ramp=smooth_ramp,
            selected_heads=selected_heads,
        )

        np.save(save_path, centroids)
        print(f"[{supercat}] Saved centroids -> {save_path}")
        centroids_dict[supercat] = centroids

    print(f"\nAll supercategory centroids ready: {list(centroids_dict.keys())}")
    return centroids_dict


def assign_regions_to_centroids(final_labels, tokens_up, centroids):

    """
    Given an image's merged parts (final_labels) and its feature map (tokens_up),
    re-assign each part ID (1..K) to the most similar global centroid (1..K).
    """
    from scipy.spatial.distance import cdist
    
    new_labels = np.zeros_like(final_labels)
    unique_parts = [p for p in np.unique(final_labels) if p != 0]
    
    if len(unique_parts) == 0:
        return new_labels
        
    # Flatten segment labels and feature map for vectorized grouping
    labels_flat = final_labels.ravel()
    tokens_flat = tokens_up.reshape(-1, tokens_up.shape[-1])
    
    valid = labels_flat != 0
    lab_valid = labels_flat[valid]
    tok_valid = tokens_flat[valid]
    
    # 1. Compute region mean features efficiently using np.add.at
    max_id = lab_valid.max()
    sum_feat = np.zeros((max_id + 1, tok_valid.shape[-1]), dtype=tok_valid.dtype)
    np.add.at(sum_feat, lab_valid, tok_valid)
    counts = np.bincount(lab_valid, minlength=max_id + 1)
    
    valid_ids = np.array(unique_parts)
    mean_feats = sum_feat[valid_ids] / counts[valid_ids][:, None]
    
    # 2. Compute distances and find best matches for all parts simultaneously
    dists = cdist(mean_feats, centroids, metric='cosine')
    best_indices = np.argmin(dists, axis=1) + 1  # 1..K index
    
    # 3. Map new global IDs over the entire 2D image directly
    map_array = np.zeros(max_id + 1, dtype=int)
    map_array[valid_ids] = best_indices
    new_labels = map_array[final_labels]
        
    return new_labels
