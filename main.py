import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
from scipy.ndimage import label as ndi_label
from skimage import graph
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, pair_confusion_matrix
import time

# Custom dataset
from dataset import PartDiscoveryDataset

# ==========================================
# 1. TOKENCUT & MASK PREPARATION
# ==========================================

def get_tokencut_data(model, img_tensor, grid_h, grid_w, backbone='dinov2', selected_heads=None):
    features = {}
    num_patches = grid_h * grid_w

    # ── DINOv3 (HuggingFace AutoModel) — uses model.layer with q/k/v_proj ─────
    if backbone == 'dinov3':
        # DINOv3: 1 CLS + 4 register tokens = 5 prefix tokens before spatial patches
        N_PREFIX    = 5
        num_heads   = model.config.num_attention_heads
        hidden_size = model.config.hidden_size

        def hook_hf(module, input, output):
            hidden_states = input[0]
            features['k'] = module.k_proj(hidden_states).detach()  # (B, N, hidden)
            features['x'] = hidden_states.detach()                  # (B, N, hidden)

        handle = model.layer[-1].attention.register_forward_hook(hook_hf)
        with torch.no_grad():
            model(pixel_values=img_tensor)
        handle.remove()

        # Skip prefix tokens; take the spatial patch tokens
        patch_tokens = features['x'][0, N_PREFIX:N_PREFIX + num_patches, :]  # (P, D)
        k_flat       = features['k'][0, N_PREFIX:N_PREFIX + num_patches, :]  # (P, hidden)
        if selected_heads is not None:
            head_dim = hidden_size // num_heads
            k_flat = k_flat.view(num_patches, num_heads, head_dim)[:, selected_heads, :]
            k_flat = k_flat.reshape(num_patches, -1)
        k_norm = F.normalize(k_flat, p=2, dim=-1)
        W = torch.mm(k_norm, k_norm.t()).cpu().numpy()
        W = W * (W > 0)
        return W, patch_tokens


    # ── DINOv1 / DINOv2 (torch.hub) — combined QKV projection ───────────────
    last_block = model.blocks[-1]
    num_heads  = last_block.attn.num_heads

    def hook(module, input, output):
        B, N, C3 = output.shape
        C = C3 // 3
        head_dim = C // num_heads
        qkv = output.reshape(B, N, 3, num_heads, head_dim).permute(2, 0, 1, 3, 4)
        features['k'] = qkv[1]

    handle = last_block.attn.qkv.register_forward_hook(hook)
    with torch.no_grad():
        if backbone in ('dinov1', 'dino_vits8'):
            out = model.get_intermediate_layers(img_tensor, n=1)[0]
            patch_tokens = out[0, 1:, :]  # skip CLS token
        else:
            # DINOv2 with registers
            forward_out = model.forward_features(img_tensor)
            patch_tokens = forward_out['x_norm_patchtokens'].squeeze(0)
    handle.remove()

    k = features['k'][0]
    if backbone in ('dinov1', 'dino_vits8'):
        k_spatial = k[1:num_patches+1, :, :]
    else:
        k_spatial = k[-num_patches:, :, :]
    if selected_heads is not None:
        k_spatial = k_spatial[:, selected_heads, :]
    k_flat = k_spatial.reshape(num_patches, -1)

    k_norm = F.normalize(k_flat, p=2, dim=-1)
    W = torch.mm(k_norm, k_norm.t()).cpu().numpy()
    W = W * (W > 0)
    return W, patch_tokens

def run_tokencut(W, grid_h, grid_w, orig_h, orig_w):
    W = (W + W.T) / 2
    d = np.sum(W, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(d + 1e-12))
    L_norm = np.eye(W.shape[0]) - D_inv_sqrt @ W @ D_inv_sqrt

    evals, evecs = np.linalg.eigh(L_norm)
    vec = evecs[:, 1]
    vec = D_inv_sqrt @ vec
    vec_2d = vec.reshape(grid_h, grid_w)

    border_pixels = np.concatenate([vec_2d[0, :], vec_2d[-1, :], vec_2d[:, 0], vec_2d[:, -1]])
    if np.median(border_pixels) > 0:
        vec_2d = -vec_2d

    vec_2d_up = cv2.resize(vec_2d, (orig_w, orig_h), interpolation=cv2.INTER_CUBIC)
    return vec_2d_up

def refine_mask(vec_2d_up, smoothed_ramp):
    fg_thresh = np.max(vec_2d_up) * 0.30
    bg_thresh = np.min(vec_2d_up) * 0.30
    markers = np.zeros_like(vec_2d_up, dtype=np.int32)
    markers[vec_2d_up > fg_thresh] = 1
    markers[vec_2d_up < bg_thresh] = 2

    labels = watershed(smoothed_ramp, markers)
    tight_mask = (labels == 1)

    kernel = np.ones((5, 5), np.uint8)
    tight_mask = cv2.erode(tight_mask.astype(np.uint8), kernel, iterations=1).astype(bool)
    return tight_mask


# ==========================================
# 2. GRADIENTS & RAG UTILITIES
# ==========================================

def calculate_feature_gradient(tokens, grid_h, grid_w):
    feat_2d = F.normalize(tokens.view(grid_h, grid_w, -1), p=2, dim=-1)
    sim_x = F.cosine_similarity(feat_2d[:, :-1, :], feat_2d[:, 1:, :], dim=-1)
    dist_x = F.pad(1.0 - sim_x, (0, 1), mode='replicate')
    sim_y = F.cosine_similarity(feat_2d[:-1, :, :], feat_2d[1:, :, :], dim=-1)
    dist_y = F.pad((1.0 - sim_y).unsqueeze(0), (0, 0, 0, 1), mode='replicate').squeeze(0)
    return torch.sqrt(dist_x**2 + dist_y**2).cpu().numpy()

def akbas_ahuja_ramp_2020(gray_img, max_width=5):
    h, w = gray_img.shape
    angles = [0, 45, 90, 135]
    weighted_mags = np.zeros((h, w))
    sum_qualities = np.zeros((h, w)) + 1e-8
    for angle in angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(gray_img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        high = maximum_filter1d(rotated, size=max_width, axis=1)
        low = minimum_filter1d(rotated, size=max_width, axis=1)
        mag = high - low
        qual = mag / max_width
        M_inv = cv2.getRotationMatrix2D((w//2, h//2), -angle, 1.0)
        weighted_mags += cv2.warpAffine(mag, M_inv, (w, h)) * cv2.warpAffine(qual, M_inv, (w, h))
        sum_qualities += cv2.warpAffine(qual, M_inv, (w, h))
    return (weighted_mags / sum_qualities)

def weight_boundary(rag, src, dst, n):
    default = {'weight': 0.0, 'count': 0}
    count_src = rag[src].get(n, default)['count']
    count_dst = rag[dst].get(n, default)['count']
    weight_src = rag[src].get(n, default)['weight']
    weight_dst = rag[dst].get(n, default)['weight']
    count = count_src + count_dst
    return {'count': count, 'weight': (count_src * weight_src + count_dst * weight_dst) / count}

def merge_boundary(rag, src, dst): pass


# ==========================================
# 3. MERGING & ABSORPTION
# ==========================================

def merge_to_k_global(initial_segments, edge_map, target_k):
    labels, _, _ = relabel_sequential(initial_segments)
    current_count = len(np.unique(labels))
    if current_count <= target_k: return labels

    # Build the RAG ONCE
    base_rag = graph.rag_boundary(labels, edge_map)

    low, high = 0.0, 1.0
    best_labels = labels
    closest_diff = float('inf')

    import copy
    for _ in range(25):
        thresh = (low + high) / 2.0
        # Deepcopy to safely prevent any internal dict mutations by scikit-image
        rag = copy.deepcopy(base_rag)

        temp_labels = graph.merge_hierarchical(labels, rag, thresh=thresh,
                                               rag_copy=False, in_place_merge=False,
                                               merge_func=merge_boundary, weight_func=weight_boundary)
        num_regions = len(np.unique(temp_labels))
        diff = abs(num_regions - target_k)
        if diff < closest_diff:
            closest_diff = diff
            best_labels = temp_labels

        if num_regions == target_k: break
        elif num_regions > target_k: low = thresh
        else: high = thresh
    return best_labels

def filter_segments_by_mask(segments, mask):
    filtered = np.copy(segments)
    filtered[~mask] = 0
    final_filtered = np.zeros_like(filtered)
    for new_id, old_id in enumerate(np.unique(filtered[filtered > 0]), start=1):
        final_filtered[filtered == old_id] = new_id
    return final_filtered

def build_semantic_rag(labels, features):
    dummy_map = np.zeros(labels.shape, dtype=float)
    rag = graph.rag_boundary(labels, dummy_map)
    for n in rag.nodes():
        mask = (labels == n)
        rag.nodes[n]['pixel count'] = np.sum(mask)
        rag.nodes[n]['total color'] = np.sum(features[mask], axis=0)
        rag.nodes[n]['mean color'] = rag.nodes[n]['total color'] / (rag.nodes[n]['pixel count'] + 1e-8)

    for u, v, d in rag.edges(data=True):
        diff = rag.nodes[u]['mean color'] - rag.nodes[v]['mean color']
        d['weight'] = np.linalg.norm(diff)
    return rag

def weight_mean_feature(rag, src, dst, n):
    diff = rag.nodes[dst]['mean color'] - rag.nodes[n]['mean color']
    return {'weight': np.linalg.norm(diff)}

def merge_mean_feature(rag, src, dst):
    rag.nodes[dst]['total color'] += rag.nodes[src]['total color']
    rag.nodes[dst]['pixel count'] += rag.nodes[src]['pixel count']
    rag.nodes[dst]['mean color'] = rag.nodes[dst]['total color'] / (rag.nodes[dst]['pixel count'] + 1e-8)

def merge_to_k_semantic(initial_segments, tokens_up, target_k_final):
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist
    from scipy.ndimage import label as ndi_label
    
    labels = initial_segments.copy()
    region_ids = [n for n in np.unique(labels) if n != 0]
    
    if len(region_ids) <= target_k_final:
        return labels
    
    # Compute mean features once
    features = np.array([
        np.mean(tokens_up[labels == rid], axis=0)
        for rid in region_ids
    ])
    
    # Cosine distance + average linkage (correct for L2-normalized features)
    dists = pdist(features, metric='cosine')
    Z = linkage(dists, method='average')
    cluster_ids = fcluster(Z, t=target_k_final, criterion='maxclust')
    
    # Remap labels
    new_labels = np.zeros_like(labels)
    for new_id, rid in zip(cluster_ids, region_ids):
        new_labels[labels == rid] = new_id
    
    # Split any non-contiguous regions created by non-adjacent merges
    final_labels = np.zeros_like(new_labels)
    current_id = 1
    for part_id in np.unique(new_labels):
        if part_id == 0:
            continue
        binary = (new_labels == part_id).astype(int)
        connected, n_components = ndi_label(binary)
        for comp in range(1, n_components + 1):
            final_labels[connected == comp] = current_id
            current_id += 1

    # If contiguity split caused > K parts, iteratively absorb the smallest ones
    # until we hit exactly target_k_final
    unique_parts = [n for n in np.unique(final_labels) if n != 0]
    while len(unique_parts) > target_k_final:
        # Find the smallest part
        sizes = {part: float((final_labels == part).sum()) for part in unique_parts}
        smallest_part = min(sizes, key=sizes.get)
        
        # We can dynamically pass 'smallest_part_size + 1' to absorb_small_regions
        # so it is guaranteed to absorb the smallest piece into its neighbor list
        min_size = sizes[smallest_part] + 1
        
        # absorb_small_regions is in global scope of main.py
        from main import absorb_small_regions  # ensure it's loaded if called internally
        final_labels = absorb_small_regions(final_labels, tokens_up, min_size=min_size)
        unique_parts = [n for n in np.unique(final_labels) if n != 0]
        
    # Re-index to be 1..K cleanly
    reindexed = np.zeros_like(final_labels)
    for i, p in enumerate(unique_parts, 1):
        reindexed[final_labels == p] = i

    return reindexed

def absorb_small_regions(segments, tokens_up, min_size=300):
    labels = segments.copy()
    tokens_flat = tokens_up.reshape(-1, tokens_up.shape[-1])
    
    dummy_map = np.zeros(labels.shape, dtype=float)
    rag = graph.rag_boundary(labels, dummy_map)
    
    labels_flat = labels.ravel()
    max_id = labels_flat.max()
    
    counts = np.bincount(labels_flat, minlength=max_id + 1)
    sum_features = np.zeros((max_id + 1, tokens_flat.shape[-1]), dtype=tokens_flat.dtype)
    np.add.at(sum_features, labels_flat, tokens_flat)
    
    for n in list(rag.nodes()):
        if n == 0: continue
        rag.nodes[n]['pixel count'] = counts[n]
        rag.nodes[n]['total color'] = sum_features[n]
        rag.nodes[n]['mean color'] = sum_features[n] / (counts[n] + 1e-8)

    while True:
        small_nodes = [n for n in rag.nodes() if n != 0 and rag.nodes[n]['pixel count'] < min_size]
        if not small_nodes:
            break

        u = min(small_nodes, key=lambda n: rag.nodes[n]['pixel count'])
        neighbors = [v for v in rag.neighbors(u) if v != 0]

        if not neighbors:
            rag.remove_node(u)
            labels[labels == u] = 0
            continue

        best_neighbor = None
        min_dist = float('inf')
        for v in neighbors:
            diff = rag.nodes[u]['mean color'] - rag.nodes[v]['mean color']
            dist = np.linalg.norm(diff)
            if dist < min_dist:
                min_dist = dist
                best_neighbor = v
                
        # Merge u into best_neighbor
        rag.nodes[best_neighbor]['total color'] += rag.nodes[u]['total color']
        rag.nodes[best_neighbor]['pixel count'] += rag.nodes[u]['pixel count']
        rag.nodes[best_neighbor]['mean color'] = rag.nodes[best_neighbor]['total color'] / (rag.nodes[best_neighbor]['pixel count'] + 1e-8)
        
        # update edges manually in RAG
        for nbr in list(rag.neighbors(u)):
            if nbr != best_neighbor:
                if not rag.has_edge(best_neighbor, nbr):
                    rag.add_edge(best_neighbor, nbr)
                    
        rag.remove_node(u)
        labels[labels == u] = best_neighbor

    final_labels = np.zeros_like(labels)
    for new_id, old_id in enumerate(np.unique(labels[labels > 0]), start=1):
        final_labels[labels == old_id] = new_id

    return final_labels


# ==========================================
# 4. PIPELINE EXECUTION
# ==========================================

def run_discovery_pipeline(model, img_tensor, original_img, alpha=0.8,
                          target_k_global=100, target_k_final=8,
                          min_shard_size=200, external_mask=None,
                          backbone='dinov2', patch_size=14,
                          skip_gradient_merge=False,
                          global_centroids=None,
                          smooth_ramp=False,
                          min_distance=2,
                          debug=False,
                          selected_heads=None):
    device = img_tensor.device
    B, C, H, W = img_tensor.shape
    grid_h, grid_w = H // patch_size, W // patch_size

    # 1. TOKENCUT
    W_mat, tokens = get_tokencut_data(model, img_tensor, grid_h, grid_w, backbone=backbone, selected_heads=selected_heads)
    vec_2d_up = run_tokencut(W_mat, grid_h, grid_w, H, W)

    # 2. FEATURE MAPS
    embed_dim = tokens.shape[-1]
    tokens_2d = tokens.view(grid_h, grid_w, embed_dim).permute(2, 0, 1).unsqueeze(0)
    tokens_up = F.interpolate(tokens_2d, size=(H, W), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).cpu().numpy()
    tokens_up = tokens_up / (np.linalg.norm(tokens_up, axis=-1, keepdims=True) + 1e-8)

    dino_ramp_raw = calculate_feature_gradient(tokens, grid_h, grid_w)
    dino_ramp_up = F.interpolate(torch.tensor(dino_ramp_raw).unsqueeze(0).unsqueeze(0),
                                size=(H, W), mode='bicubic').squeeze().numpy()
    dino_ramp_norm = (dino_ramp_up - dino_ramp_up.min()) / (dino_ramp_up.max() - dino_ramp_up.min() + 1e-8)

    # Convert tensor back to gray (or use original_img)
    img_np = original_img.cpu().numpy().transpose(1, 2, 0)
    # Undo normalization for structure diff
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    # Gray conversion
    gray_img = np.dot(img_np[...,:3], [0.2989, 0.5870, 0.1140])

    struc_ramp = akbas_ahuja_ramp_2020(gray_img)
    struc_ramp_norm = (struc_ramp - struc_ramp.min()) / (struc_ramp.max() - struc_ramp.min() + 1e-8)

    fused_ramp = (alpha * dino_ramp_norm) + ((1.0 - alpha) * struc_ramp_norm)
    fused_ramp = (fused_ramp - fused_ramp.min()) / (fused_ramp.max() - fused_ramp.min() + 1e-8)
    smoothed = gaussian_filter(fused_ramp, sigma=1.0) if smooth_ramp else fused_ramp

    # 3. COMPUTE FOREGROUND MASK FIRST
    if external_mask is not None:
        tight_mask = external_mask.astype(bool)
    else:
        tight_mask = refine_mask(vec_2d_up, smoothed)

    # 4. APPLY MASK *BEFORE* WATERSHED — ensures all regions are on the object
    #    Set gradient to max in background so no seeds are placed there
    masked_smoothed = smoothed.copy()
    masked_smoothed[~tight_mask] = 1.0  # high gradient = no seeds in background

    import time
    wm_start = time.time()

    coords = peak_local_max(1.0 - masked_smoothed, min_distance=min_distance)
    # Keep only seeds that fall inside the foreground mask
    fg_coords = coords[tight_mask[coords[:, 0], coords[:, 1]]]
    if len(fg_coords) == 0:
        # Fallback: no seeds found, return single foreground region
        final_segments = np.zeros((H, W), dtype=int)
        final_segments[tight_mask] = 1
        if debug:
            return final_segments, tight_mask, tokens_up, {
                'fused_ramp': fused_ramp,
                'smoothed_ramp': smoothed,
                'atomic_segments': final_segments,
                'robust_segments': final_segments
            }
        return final_segments, tight_mask, tokens_up, 1, 0.0

    seed_mask = np.zeros_like(smoothed, dtype=bool)
    seed_mask[tuple(fg_coords.T)] = True
    markers, _ = ndi_label(seed_mask)
    atomic_segments = watershed(masked_smoothed, markers)

    # Zero out background regions
    atomic_segments[~tight_mask] = 0

    # 5. MERGE
    robust_segments = absorb_small_regions(atomic_segments, tokens_up, min_size=min_shard_size)

    if global_centroids is not None:
        # GLOBAL K-MEANS: bypass local merges entirely and map straight to global IDs
        from global_kmeans import assign_regions_to_centroids
        final_segments = assign_regions_to_centroids(robust_segments, tokens_up, global_centroids)
    elif skip_gradient_merge:
        # DINO-only merge: go directly from atomic regions → K parts by semantic similarity
        final_segments = merge_to_k_semantic(robust_segments, tokens_up, target_k_final)
    else:
        # Original: gradient merge first, then semantic merge
        fg_fraction = tight_mask.sum() / (H * W)
        effective_k_global = max(target_k_final * 3, int(target_k_global * fg_fraction))
        global_segments = merge_to_k_global(robust_segments, masked_smoothed, effective_k_global)
        final_segments = merge_to_k_semantic(global_segments, tokens_up, target_k_final)

    wm_time = time.time() - wm_start
    n_robust = len(np.unique(robust_segments)) - (1 if 0 in robust_segments else 0)

    if debug:
        return final_segments, tight_mask, tokens_up, {
            'fused_ramp': fused_ramp,
            'smoothed_ramp': smoothed,
            'atomic_segments': atomic_segments,
            'robust_segments': robust_segments,
            'n_robust': n_robust,
            'wm_time': wm_time
        }

    return final_segments, tight_mask, tokens_up, n_robust, wm_time

# ==========================================
# 5. VISUALIZATION
# ==========================================

# 17 distinct colors for up to 16 parts + background
PART_COLORS = np.array([
    [0.0, 0.0, 0.0],       # 0: background (black)
    [0.90, 0.10, 0.10],    # 1: red
    [0.10, 0.70, 0.10],    # 2: green
    [0.20, 0.30, 0.90],    # 3: blue
    [1.00, 0.85, 0.00],    # 4: yellow
    [0.80, 0.00, 0.80],    # 5: magenta
    [0.00, 0.80, 0.80],    # 6: cyan
    [1.00, 0.50, 0.00],    # 7: orange
    [0.50, 1.00, 0.00],    # 8: lime
    [0.00, 0.50, 1.00],    # 9: sky blue
    [1.00, 0.00, 0.50],    # 10: pink
    [0.60, 0.40, 0.20],    # 11: brown
    [0.40, 0.80, 0.60],    # 12: mint
    [0.80, 0.60, 1.00],    # 13: lavender
    [1.00, 0.70, 0.70],    # 14: salmon
    [0.50, 0.50, 0.00],    # 15: olive
    [0.00, 0.40, 0.40],    # 16: teal
])

# CUB part names (15 keypoints)
CUB_PART_NAMES = [
    'back', 'beak', 'belly', 'breast', 'crown',
    'forehead', 'left_eye', 'left_leg', 'left_wing',
    'nape', 'right_eye', 'right_leg', 'right_wing',
    'tail', 'throat'
]


def save_visualization(img_tensor, segments, landmarks, save_path, alpha=0.5):
    """
    Save a 3-panel visualization:
      Panel 1: Original image
      Panel 2: Colored part segmentation overlay
      Panel 3: Overlay + GT landmarks with part names
    """
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    H, W = segments.shape

    # Build colored segmentation map
    seg_color = np.zeros((H, W, 3))
    unique_labels = np.unique(segments)
    for label_id in unique_labels:
        color_idx = label_id % len(PART_COLORS)
        seg_color[segments == label_id] = PART_COLORS[color_idx]

    # Blend: overlay = alpha * seg_color + (1-alpha) * original
    overlay = alpha * seg_color + (1 - alpha) * img_np
    overlay = np.clip(overlay, 0, 1)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Original
    axes[0].imshow(img_np)
    axes[0].set_title('Original Image', fontsize=14)
    axes[0].axis('off')

    # Panel 2: Segmentation overlay
    axes[1].imshow(overlay)
    axes[1].set_title(f'Part Segmentation ({len(unique_labels) - (1 if 0 in unique_labels else 0)} parts)', fontsize=14)
    axes[1].axis('off')

    # Panel 3: Overlay + landmarks
    axes[2].imshow(overlay)
    if landmarks is not None:
        for j in range(landmarks.shape[0]):
            x, y, vis = landmarks[j, 0], landmarks[j, 1], landmarks[j, 2] if landmarks.shape[1] > 2 else 1.0
            if vis > 0.5:
                axes[2].plot(x, y, 'o', markersize=6, markerfacecolor='white',
                           markeredgecolor='black', markeredgewidth=1.5)
                part_name = CUB_PART_NAMES[j] if j < len(CUB_PART_NAMES) else str(j)
                axes[2].annotate(part_name, (x, y), fontsize=6, color='white',
                               fontweight='bold', ha='center', va='bottom',
                               xytext=(0, -8), textcoords='offset points',
                               bbox=dict(boxstyle='round,pad=0.15', fc='black', alpha=0.6))
    axes[2].set_title('Overlay + GT Landmarks', fontsize=14)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ==========================================
# 6. NMI / ARI EVALUATION (MPAE Protocol)
# ==========================================

def adjusted_rand_score_overflow(labels_true, labels_pred):
    """ARI that handles large confusion matrices without overflow."""
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    if fn == 0 and fp == 0:
        return 1.0
    (tn, fp), (fn, tp) = (tn / 1e8, fp / 1e8), (fn / 1e8, tp / 1e8)
    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))


def segments_to_onehot_mask(segments, num_parts):
    """
    Convert integer segmentation map (H, W) -> one-hot mask (1, K+1, H, W).
    Channel 0..K-1 = foreground parts (labels 1..K in segments).
    Channel K     = background (label 0 in segments).
    This matches MPAE convention: mask[:, :-1] = foreground, mask[:, -1] = background.
    """
    H, W = segments.shape
    num_channels = num_parts + 1  # K foreground + 1 background
    mask = torch.zeros(1, num_channels, H, W, dtype=torch.float32)

    for part_id in range(1, num_parts + 1):
        mask[0, part_id - 1] = torch.from_numpy((segments == part_id).astype(np.float32))
    # Background channel (last)
    mask[0, num_parts] = torch.from_numpy((segments == 0).astype(np.float32))

    return mask


def evaluate_landmarks(model, dataset, device, target_k=16, target_k_global=100, min_shard_size=10, alpha=0.5, max_samples=0,
                 save_vis=False, vis_dir=None, prototypes=None, segmenter=None,
                 backbone='dinov2', patch_size=14, skip_gradient_merge=False,
                 global_centroids=None, dataset_name='CUB', smooth_ramp=False, min_distance=2,
                 selected_heads=None):
    """
    Evaluate part discovery using NMI, ARI, and NME.
    Generalized for any dataset with landmarks (e.g., CUB with 15, CelebA with 5).
    """
    use_prototypes = prototypes is not None
    if use_prototypes:
        print(f"Using prototype matching ({prototypes.shape[0]} prototypes)")
    if segmenter is not None:
        print(f"Using external segmenter for foreground masks")

    if save_vis and vis_dir:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"Saving visualizations to {vis_dir}")

    all_nmi_preds = []       # foreground-only predictions
    all_nmi_preds_w_bg = []  # predictions with background channel
    all_nmi_gts = []
    
    # For NME
    all_centroids = []
    all_landmarks_norm = []
    all_visibles = []

    all_n_robust = []
    all_wm_times = []

    limit = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    start_time = time.time()

    for i in range(limit):
        sample = dataset[i]
        img_tensor = sample['image'].unsqueeze(0).to(device)
        landmarks = sample['landmarks']  # (15, 3): x, y, visible

        elapsed = time.time() - start_time
        eta = (elapsed / (i + 1)) * (limit - i - 1) if i > 0 else 0
        
        # Compute external mask if segmenter is provided
        ext_mask = None
        if segmenter is not None:
            ext_mask = segmenter.segment_tensor(sample['image'])

        # --- PIPELINE RUN ---
        final_segments, _, tokens_up, n_robust, wm_time = run_discovery_pipeline(
            model=model,
            img_tensor=img_tensor,
            original_img=sample['image'],
            target_k_global=target_k_global,
            target_k_final=target_k,
            min_shard_size=min_shard_size,
            external_mask=ext_mask,
            alpha=alpha,
            backbone=backbone,
            patch_size=patch_size,
            skip_gradient_merge=skip_gradient_merge,
            global_centroids=global_centroids,
            smooth_ramp=smooth_ramp,
            min_distance=min_distance,
            selected_heads=selected_heads
        )

        all_n_robust.append(n_robust)
        all_wm_times.append(wm_time)

        # Convert segments to one-hot mask tensor (1, K+1, H, W)
        mask_w_bg = segments_to_onehot_mask(final_segments, target_k).to(device)
        mask_fg = mask_w_bg[:, :-1]  # foreground only (1, K, H, W)

        # Visibility mask
        visible = landmarks[:, 2] > 0.5  # (15,)

        # Landmark points -> normalized grid coords for grid_sample
        points = landmarks[:, :2].unsqueeze(0).unsqueeze(2).clone()  # (1, 15, 1, 2)
        H, W = final_segments.shape
        points[:, :, :, 0] /= W  # normalize x
        points[:, :, :, 1] /= H  # normalize y
        points = points * 2 - 1   # map [0,1] -> [-1,1]

        # grid_sample: (1, K+1, H, W) sampled at (1, 15, 1, 2) -> (1, K+1, 15, 1)
        pred_w_bg = F.grid_sample(mask_w_bg.float(), points.float().to(device),
                                  mode='nearest', align_corners=False)
        pred_w_bg = torch.argmax(pred_w_bg, dim=1).squeeze(2)  # (1, 15)
        pred_w_bg = pred_w_bg[0][visible].cpu().numpy()
        all_nmi_preds_w_bg.append(pred_w_bg)

        pred_fg = F.grid_sample(mask_fg.float(), points.float().to(device),
                                mode='nearest', align_corners=False)
        pred_fg = torch.argmax(pred_fg, dim=1).squeeze(2)  # (1, 15)
        pred_fg = pred_fg[0][visible].cpu().numpy()
        all_nmi_preds.append(pred_fg)

        # GT: each landmark index is its own part class
        num_lms = landmarks.shape[0]
        gt = torch.arange(num_lms)[visible].numpy()
        all_nmi_gts.append(gt)
        
        # Calculate centroids for NME
        centroids = np.full((target_k, 2), np.nan)
        for k in range(1, target_k + 1):
            ys, xs = np.where(final_segments == k)
            if len(ys) > 0:
                centroids[k-1, 0] = np.mean(xs) / W
                centroids[k-1, 1] = np.mean(ys) / H
        all_centroids.append(centroids)
        
        # Normalized landmarks for NME
        lms_norm = np.zeros((landmarks.shape[0], 2))
        lms_norm[:, 0] = landmarks[:, 0].numpy() / W
        lms_norm[:, 1] = landmarks[:, 1].numpy() / H
        all_landmarks_norm.append(lms_norm)
        
        # Visibles for NME
        all_visibles.append(visible.numpy())

        num_discovered = len(np.unique(final_segments)) - (1 if 0 in final_segments else 0)
        
        if (i+1) % 50 == 0 or i == limit - 1:
            print(f"[{i+1}/{limit}] -> {num_discovered} parts discovered, {visible.sum().item()} visible landmarks")

        # Save visualization
        if save_vis and vis_dir:
            img_name = os.path.splitext(os.path.basename(sample['image_path']))[0]
            save_path = os.path.join(vis_dir, f"{img_name}_parts.png")
            save_visualization(sample['image'], final_segments, landmarks.numpy(), save_path)

    # Aggregate and compute metrics
    all_nmi_gts = np.concatenate(all_nmi_gts, axis=0)
    all_nmi_preds = np.concatenate(all_nmi_preds, axis=0)
    all_nmi_preds_w_bg = np.concatenate(all_nmi_preds_w_bg, axis=0)

    nmi_wbg = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds_w_bg) * 100
    ari_wbg = adjusted_rand_score_overflow(all_nmi_gts, all_nmi_preds_w_bg) * 100
    nmi_fg = normalized_mutual_info_score(all_nmi_gts, all_nmi_preds) * 100
    ari_fg = adjusted_rand_score_overflow(all_nmi_gts, all_nmi_preds) * 100
    
    # Compute NME
    all_centroids = np.array(all_centroids)       # (N, K, 2)
    all_landmarks_norm = np.array(all_landmarks_norm) # (N, L, 2)
    all_visibles = np.array(all_visibles)         # (N, L)
    
    L = all_landmarks_norm.shape[1]
    K_parts = all_centroids.shape[1]
    
    dist_matrix = np.zeros((L, K_parts))
    for l in range(L):
        for k in range(K_parts):
            # Consider only images where landmark l is visible AND part k is discovered
            valid = all_visibles[:, l] & ~np.isnan(all_centroids[:, k, 0])
            if np.sum(valid) > 0:
                dists = np.sqrt(np.sum((all_centroids[valid, k, :] - all_landmarks_norm[valid, l, :])**2, axis=1))
                dist_matrix[l, k] = np.mean(dists)
            else:
                dist_matrix[l, k] = np.inf
                
    # For each ground truth landmark, pick the part centroid that minimizes the mean Euclidean distance
    min_dists = np.min(dist_matrix, axis=1) # (L,)
    valid_min_dists = min_dists[min_dists != np.inf]
    nme = (np.mean(valid_min_dists) * 100) if len(valid_min_dists) > 0 else np.nan

    avg_n_robust = np.mean(all_n_robust) if all_n_robust else 0
    avg_wm_time = np.mean(all_wm_times) if all_wm_times else 0
    avg_total_time = (time.time() - start_time) / limit

    print("\n" + "="*60)
    print(f"{dataset_name} Evaluation Results")
    print("="*60)
    print(f"  Avg Regions:  {avg_n_robust:.1f}")
    print(f"  Avg W+M Time: {avg_wm_time:.3f}s")
    print(f"  Avg Tot Time: {avg_total_time:.3f}s")
    print(f"  NME:          {nme:.2f}%")
    print(f"  NMI (w/ bg):  {nmi_wbg:.2f}%")
    print(f"  ARI (w/ bg):  {ari_wbg:.2f}%")
    print(f"  FG_NMI:       {nmi_fg:.2f}%")
    print(f"  FG_ARI:       {ari_fg:.2f}%")
    print("="*60)
    print(f"  Total time: {time.time() - start_time:.1f}s")

    return nmi_wbg, nme, nmi_fg, ari_fg, avg_n_robust, avg_wm_time, avg_total_time


# ==========================================
# 7. PARTIMAGENET OOD EVALUATION
# ==========================================


def predict_supercat_from_centroids(img_tensor, model, centroids_dict,
                                    backbone='dinov2', patch_size=14, device='cuda'):
    """
    Classify an image to one of the 11 PartImageNet supercategories by comparing
    its CLS-token embedding to the mean of each supercategory's centroids.

    Returns:
        predicted_supercat (str): e.g. 'Bird'
    """
    with torch.no_grad():
        if backbone == 'dinov2':
            feat = model.forward_features(img_tensor.to(device))
            cls_token = feat['x_norm_clstoken'][0].cpu().numpy()  # (D,)
        else:
            # dino_vits8 / dinov1: get CLS from last layer output
            out = model(img_tensor.to(device))  # (1, D)
            cls_token = out[0].cpu().numpy()

    # L2 normalise
    cls_token = cls_token / (np.linalg.norm(cls_token) + 1e-8)

    # Compare to mean of each supercategory's centroids (cosine distance)
    best_sc, best_dist = None, float('inf')
    for sc, cents in centroids_dict.items():
        mean_cent = cents.mean(axis=0)
        mean_cent = mean_cent / (np.linalg.norm(mean_cent) + 1e-8)
        dist = float(np.dot(cls_token, mean_cent))  # cosine similarity
        if best_sc is None or dist > best_dist:
            best_sc, best_dist = sc, dist
    return best_sc


def save_ood_debug(img_tensor, vec_2d_up, tight_mask, atomic_segs,
                   final_segments, gt_mask_tensor, save_path, target_k=8):
    """
    Save a multi-panel debug figure for one PartImageNet_OOD image:
      1. Original image
      2. TokenCut eigenvector (foreground score)
      3. Foreground mask (TokenCut)
      4. Atomic watershed regions
      5. Final discovered parts (K=target_k)
      6. GT part segmentation (argmax of GT mask channels)
    """
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img_np = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img_np = np.clip(std * img_np + mean, 0, 1)
    H, W   = img_np.shape[:2]

    def color_seg(seg):
        out = np.zeros((seg.shape[0], seg.shape[1], 3))
        for lid in np.unique(seg):
            c = PART_COLORS[lid % len(PART_COLORS)]
            out[seg == lid] = c
        return out

    # GT: argmax over (num_parts+1) channels; last channel = background weight 0.1
    C = gt_mask_tensor.shape[0]
    bg_ch = torch.full((1, H, W), 0.1)
    gt_with_bg = torch.cat([gt_mask_tensor.cpu(), bg_ch], dim=0)  # (C+1, H, W)
    gt_resized = torch.nn.functional.interpolate(
        gt_with_bg.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)
    gt_label = torch.argmax(gt_resized, dim=0).numpy()  # (H, W), 0..C (C=background)

    panels = [
        (img_np,                           'Original'),
        (plt.cm.RdYlGn(                    # type: ignore
            (vec_2d_up - vec_2d_up.min()) /
            (vec_2d_up.max() - vec_2d_up.min() + 1e-8))[:, :, :3],
                                           'TokenCut eigenvector'),
        (np.stack([tight_mask.astype(np.float32)] * 3, axis=-1),
                                           'Foreground mask'),
        (color_seg(atomic_segs),            'Atomic regions'),
        (0.5 * color_seg(final_segments) + 0.5 * img_np,
                                           f'Discovered parts (K={target_k})'),
        (color_seg(gt_label),              f'GT parts ({C} classes)'),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(np.clip(panel, 0, 1))
        ax.set_title(title, fontsize=9)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()


def evaluate_partimagenet_ood(model, dataset, device,
                              centroids_dict,
                              target_k=8,
                              target_k_global=100,
                              min_shard_size=10,
                              alpha=0.5,
                              backbone='dinov2',
                              patch_size=14,
                              skip_gradient_merge=False,
                              max_samples=0,
                              n_debug=5,
                              debug_dir=None,
                              segmenter=None,
                              smooth_ramp=False,
                              min_distance=2,
                              selected_heads=None):
    """
    Pixel-level NMI / ARI evaluation for PartImageNet_OOD following the
    Ref_MPAE validate_PartImage protocol:

      - Predicted: argmax of the discovered-part one-hot map -> per-pixel label
      - GT: argmax of (GT mask channels + low-weight background) -> per-pixel label
      - Concatenate ALL pixels across ALL images, then compute global NMI + ARI.

    Per-image routing (Strategy B):
      sample['supercategory'] -> centroids_dict[supercat] (K centroids)
    """
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug figures will be saved to: {debug_dir}")

    limit = len(dataset) if max_samples <= 0 else min(max_samples, len(dataset))
    print(f"Evaluating {limit} images (K={target_k}, backbone={backbone})")

    all_pred_pixels = []
    all_gt_pixels   = []
    start_time = time.time()

    for i in range(limit):
        sample    = dataset[i]
        img_tensor = sample['image'].unsqueeze(0).to(device)
        supercat   = sample['supercategory']
        gt_mask    = sample['mask']          # (num_parts, H_orig, W_orig) float32
        H, W       = img_tensor.shape[-2:]

        # Route to the right centroid set
        # --- If SAM3 available: predict supercat from centroids, use as SAM prompt ---
        if segmenter is not None and hasattr(segmenter, 'segment_tensor_with_prompt'):
            predicted_supercat = predict_supercat_from_centroids(
                img_tensor, model, centroids_dict,
                backbone=backbone, patch_size=patch_size, device=device
            )
            from segmentation import get_sam3_prompt
            sam_prompt = get_sam3_prompt(predicted_supercat)
            fg_mask = segmenter.segment_tensor_with_prompt(
                sample['image'], sam_prompt
            ).astype(bool)
        else:
            predicted_supercat = sample['supercategory']  # GT for TokenCut (known)
            fg_mask = None    # TokenCut handled inside pipeline

        centroids = centroids_dict.get(predicted_supercat, None)
        if centroids is None:
            print(f"  [{i+1}] SKIP — no centroids for supercat '{supercat}'")
            continue

        # ---- Run pipeline (debug mode for first n_debug images) ----
        want_debug = (debug_dir is not None) and (i < n_debug)
        try:
            result = run_discovery_pipeline(
                model=model,
                img_tensor=img_tensor,
                original_img=sample['image'],
                target_k_global=target_k_global,
                target_k_final=target_k,
                min_shard_size=min_shard_size,
                external_mask=fg_mask,       # SAM mask if available, else TokenCut
                alpha=alpha,
                backbone=backbone,
                patch_size=patch_size,
                skip_gradient_merge=skip_gradient_merge,
                global_centroids=centroids,
                smooth_ramp=smooth_ramp,
                min_distance=min_distance,
                debug=want_debug,
                selected_heads=selected_heads
            )

            if want_debug:
                final_segments, tight_mask, tokens_up, dbg = result
                img_name = os.path.splitext(os.path.basename(sample['image_path']))[0]
                debug_path = os.path.join(debug_dir, f"{i:03d}_{img_name}_debug.png")
                save_ood_debug(
                    img_tensor=sample['image'],
                    vec_2d_up=dbg.get('smoothed_ramp', np.zeros((H, W))),
                    tight_mask=tight_mask,
                    atomic_segs=dbg.get('atomic_segments', np.zeros((H, W), dtype=int)),
                    final_segments=final_segments,
                    gt_mask_tensor=gt_mask,
                    save_path=debug_path,
                    target_k=target_k,
                )
                print(f"  [{i+1}] Debug fig -> {debug_path}")
            else:
                final_segments, tight_mask, tokens_up, _, _ = result

        except Exception as e:
            print(f"  [{i+1}] SKIP — pipeline error: {e}")
            continue

        # ---- Convert prediction to per-pixel label ----
        # segments: 0=background, 1..K=parts
        pred_pixels = final_segments.ravel()   # (H*W,) int

        # ---- Convert GT mask to per-pixel label ----
        # gt_mask: (num_parts, orig_H, orig_W) — resize to pipeline resolution
        num_parts  = gt_mask.shape[0]
        bg_ch      = torch.full((1, gt_mask.shape[1], gt_mask.shape[2]), 0.1)
        gt_with_bg = torch.cat([gt_mask.cpu(), bg_ch], dim=0)  # (num_parts+1, H_orig, W_orig)
        gt_resized = torch.nn.functional.interpolate(
            gt_with_bg.unsqueeze(0).float(),
            size=(H, W),
            mode='nearest',
        ).squeeze(0)                           # (num_parts+1, H, W)
        gt_pixels = torch.argmax(gt_resized, dim=0).numpy().ravel()  # (H*W,) int

        all_pred_pixels.append(pred_pixels)
        all_gt_pixels.append(gt_pixels)

        if (i + 1) % 20 == 0 or i == limit - 1:
            elapsed = time.time() - start_time
            eta     = (elapsed / (i + 1)) * (limit - i - 1)
            n_disc  = len(np.unique(final_segments)) - (1 if 0 in final_segments else 0)
            sam_str = f'SAM({predicted_supercat})' if segmenter is not None else f'TC({supercat})'
            print(f"  [{i+1}/{limit}] {sam_str:22s}  "
                  f"gt={supercat:12s}  parts={n_disc}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s")

    # ---- Aggregate and compute metrics ----
    all_pred = np.concatenate(all_pred_pixels)
    all_gt   = np.concatenate(all_gt_pixels)

    from sklearn.metrics import normalized_mutual_info_score
    nmi = normalized_mutual_info_score(all_gt, all_pred) * 100
    ari = adjusted_rand_score_overflow(all_gt, all_pred) * 100

    print("\n" + "=" * 60)
    print("PartImageNet_OOD Evaluation Results")
    print("=" * 60)
    print(f"  Evaluated images : {len(all_pred_pixels)}")
    print(f"  K (parts)        : {target_k}")
    print(f"  NMI              : {nmi:.2f}%")
    print(f"  ARI              : {ari:.2f}%")
    print(f"  Total time       : {time.time() - start_time:.1f}s")
    print("=" * 60)
    return nmi, ari


# ==========================================
# 6. MAIN ENTRY POINT
# ==========================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['CelebA', 'CUB', 'PartImageNet', 'PartImageNet_OOD', 'PartImageNet_Seg'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--mode', type=str, default='evaluate', choices=['discover', 'evaluate'])
    parser.add_argument('--centroid_dir', type=str, default='./centroids_ood',
                        help='Directory to save/load per-supercategory centroids (PartImageNet_OOD)')
    parser.add_argument('--n_debug', type=int, default=5,
                        help='Number of images to save debug visualizations for (0 to disable)')
    parser.add_argument('--backbone', type=str, default='dinov2',
                        choices=['dinov1', 'dinov2', 'dino_vits8', 'dinov3'],
                        help='Feature backbone: dinov1 (dino_vitb16), dinov2 (dinov2_vitb14_reg), '
                             'dino_vits8 (dino_vits8), dinov3 (facebook/dinov3-vitb16-pretrain-lvd1689m)')
    parser.add_argument('--target_k', type=int, default=16, help='Number of foreground parts (K)')
    parser.add_argument('--img_size', type=int, default=448, help='Optional resize size for images')
    parser.add_argument('--min_shard_size', type=int, default=10, help='Min segment size in px (lower = finer parts)')
    parser.add_argument('--alpha', type=float, default=0.5, help='Blend: alpha*DINOv2_edges + (1-alpha)*structural_edges')
    parser.add_argument('--skip_gradient_merge', action='store_true',
                        help='Skip merge_to_k_global (gradient merge) and use DINO-only merge')
    parser.add_argument('--smooth_ramp', action='store_true',
                        help='Apply Gaussian smoothing to fused ramp before watershed (default: off)')
    parser.add_argument('--min_distance', type=int, default=2,
                        help='Minimum pixel distance between watershed seeds (default: 2). Higher = fewer, larger initial regions.')
    parser.add_argument('--max_samples', type=int, default=0, help='Max samples (0 for all)')
    parser.add_argument('--save_vis', action='store_true', help='Save visualization images')
    parser.add_argument('--vis_dir', type=str, default='./vis_output', help='Directory to save visualizations')
    # Prototype matching args
    parser.add_argument('--use_prototypes', action='store_true', help='Enable prototype-based consistent labeling')
    parser.add_argument('--n_reference', type=int, default=50, help='Number of reference images for prototype discovery')
    parser.add_argument('--proto_path', type=str, default='./prototypes.npy', help='Path to save/load prototypes')
    # SAM3 args
    parser.add_argument('--use_sam3', action='store_true', help='Use SAM3 for foreground segmentation instead of TokenCut')
    parser.add_argument('--sam3_prompt', type=str, default='bird', help='Text prompt for SAM3')
    parser.add_argument('--hf_token', type=str, default=None, help='HuggingFace token for SAM3 access')
    parser.add_argument('--selected_heads', type=int, nargs='+', default=None,
                        help='Attention head indices to use for TokenCut affinity (e.g. --selected_heads 0 1 2 3 6 8 9 10). Uses all heads if omitted.')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t_load = time.time()
    if args.backbone == 'dino_vits8':
        print(f"Loading DINO ViT-S/8 on {device}...")
        model = torch.hub.load('facebookresearch/dino', 'dino_vits8').to(device).eval()
        patch_size = 8
    elif args.backbone == 'dinov1':
        print(f"Loading DINOv1 (dino_vitb16) on {device}...")
        model = torch.hub.load('facebookresearch/dino', 'dino_vitb16').to(device).eval()
        patch_size = 16
    elif args.backbone == 'dinov3':
        print(f"Loading DINOv3 (facebook/dinov3-vitb16-pretrain-lvd1689m) on {device}...")
        from transformers import AutoImageProcessor, AutoModel
        _dinov3_processor = AutoImageProcessor.from_pretrained(
            'facebook/dinov3-vitb16-pretrain-lvd1689m')
        model = AutoModel.from_pretrained(
            'facebook/dinov3-vitb16-pretrain-lvd1689m').to(device).eval()
        # Store the processor on the model object so downstream code can access it
        model._hf_processor = _dinov3_processor
        patch_size = 16
    else:
        print(f"Loading DINOv2 (dinov2_vitb14_reg) on {device}...")
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg').to(device).eval()
        patch_size = 14
    print(f"DINO loaded in {time.time()-t_load:.1f}s")

    # Initialize SAM3 segmenter if requested
    segmenter = None
    if args.use_sam3:
        # Automatically adjust prompt for CelebA if user left it as default 'bird'
        prompt = args.sam3_prompt
        if args.dataset == 'CelebA' and prompt == 'bird':
            prompt = 'head, face'
            print(f"Auto-adjusting SAM3 prompt to '{prompt}' for CelebA dataset.")
            
        from segmentation import SAM3Segmenter
        segmenter = SAM3Segmenter(device=device, text_prompt=prompt,
                                  hf_token=args.hf_token)

    print(f"Initializing {args.dataset} dataset from {args.data_root} (split={args.split})...")
    if args.dataset == 'CelebA':
        from dataset import CelebA_Dataset_MPAE
        dataset = CelebA_Dataset_MPAE(root=args.data_root, is_train=(args.split == 'train'), Image_size=args.img_size)
    elif args.dataset == 'PartImageNet_OOD':
        from dataset import PartImageNet_OOD_Dataset
        dataset = PartImageNet_OOD_Dataset(root=args.data_root, split=args.split, img_size=args.img_size)
    elif args.dataset == 'PartImageNet_Seg':
        from dataset import PartImageNet_Seg_Dataset
        dataset = PartImageNet_Seg_Dataset(root=args.data_root, split=args.split, img_size=args.img_size)
    else:
        dataset = PartDiscoveryDataset(root=args.data_root, dataset_name=args.dataset,
                                      split=args.split, img_size=args.img_size)
    print(f"Total samples: {len(dataset)}")

    # Discover prototypes if requested
    global_centroids = None
    if args.use_prototypes:
        # Use train split for centroid discovery
        if args.dataset == 'CelebA':
            from dataset import CelebA_Dataset_MPAE
            train_dataset = CelebA_Dataset_MPAE(root=args.data_root, is_train=(args.split == 'train'), Image_size=args.img_size)
        else:
            train_dataset = PartDiscoveryDataset(root=args.data_root, dataset_name=args.dataset,
                                                split=args.split, img_size=args.img_size)
        print(f"Train set for centroids: {len(train_dataset)} images")

        from global_kmeans import discover_global_centroids
        global_centroids = discover_global_centroids(
            model=model,
            dataset=train_dataset,
            segmenter=segmenter,
            n_reference=args.n_reference,
            target_k=args.target_k,
            save_path=args.proto_path,
            backbone=args.backbone,
            patch_size=patch_size,
            device=device,
            alpha=args.alpha,
            skip_gradient_merge=args.skip_gradient_merge,
            min_shard_size=args.min_shard_size,
            smooth_ramp=args.smooth_ramp,
            selected_heads=args.selected_heads
        )


    if args.mode == 'evaluate':
        if args.dataset in ['CUB', 'CelebA']:
            evaluate_landmarks(model, dataset, device, target_k=args.target_k,
                         target_k_global=100, min_shard_size=args.min_shard_size,
                         alpha=args.alpha, backbone=args.backbone, patch_size=patch_size,
                         segmenter=segmenter, skip_gradient_merge=args.skip_gradient_merge,
                         max_samples=args.max_samples, global_centroids=global_centroids,
                         dataset_name=args.dataset, smooth_ramp=args.smooth_ramp,
                         min_distance=args.min_distance,
                         selected_heads=args.selected_heads)
        elif args.dataset in ('PartImageNet_OOD', 'PartImageNet_Seg'):
            # Build train split for centroid discovery, route test set to eval
            from global_kmeans import discover_all_supercat_centroids
            if args.dataset == 'PartImageNet_OOD':
                from dataset import PartImageNet_OOD_Dataset
                train_dataset_ood = PartImageNet_OOD_Dataset(
                    root=args.data_root, split=args.split, img_size=args.img_size
                )
            else:
                from dataset import PartImageNet_Seg_Dataset
                train_dataset_ood = PartImageNet_Seg_Dataset(
                    root=args.data_root, split=args.split, img_size=args.img_size
                )
            print(f"{args.dataset} train set for centroid discovery: {len(train_dataset_ood)} images")
            centroids_dict = discover_all_supercat_centroids(
                model=model,
                dataset=train_dataset_ood,
                segmenter=segmenter,
                n_reference=args.n_reference,
                target_k=args.target_k,
                save_dir=args.centroid_dir,
                backbone=args.backbone,
                patch_size=patch_size,
                device=device,
                alpha=args.alpha,
                skip_gradient_merge=args.skip_gradient_merge,
                min_shard_size=args.min_shard_size,
                smooth_ramp=args.smooth_ramp,
                selected_heads=args.selected_heads
            )

            debug_subdir = 'debug_ood' if args.dataset == 'PartImageNet_OOD' else 'debug_seg'
            debug_dir = os.path.join(args.vis_dir, debug_subdir) if args.n_debug > 0 else None
            evaluate_partimagenet_ood(
                model=model,
                dataset=dataset,
                device=device,
                centroids_dict=centroids_dict,
                target_k=args.target_k,
                target_k_global=100,
                min_shard_size=args.min_shard_size,
                alpha=args.alpha,
                backbone=args.backbone,
                patch_size=patch_size,
                skip_gradient_merge=args.skip_gradient_merge,
                max_samples=args.max_samples,
                n_debug=args.n_debug,
                debug_dir=debug_dir,
                segmenter=segmenter,   # SAM3 if --use_sam3, else None (TokenCut)
                smooth_ramp=args.smooth_ramp,
                min_distance=args.min_distance,
                selected_heads=args.selected_heads
            )
        else:
            print(f"Evaluation not yet implemented for {args.dataset}")
    else:
        # Discovery-only mode (original behavior)
        limit = len(dataset) if args.max_samples <= 0 else args.max_samples
        for i in range(limit):
            sample = dataset[i]
            img_tensor = sample['image'].unsqueeze(0).to(device)
            print(f"Processing image {i+1}/{limit}: {sample['image_path']}")
            try:
                final_segments, mask, _, _, _ = run_discovery_pipeline(
                    model=model,
                    img_tensor=img_tensor,
                    original_img=sample['image'],
                    target_k_global=100,
                    target_k_final=args.target_k,
                    min_shard_size=args.min_shard_size,
                    alpha=args.alpha,
                    backbone=args.backbone,
                    patch_size=patch_size,
                    skip_gradient_merge=args.skip_gradient_merge,
                    smooth_ramp=args.smooth_ramp,
                    selected_heads=args.selected_heads
                )
                print(f"  -> Discovered {len(np.unique(final_segments)) - 1} parts.")
            except Exception as e:
                print(f"  -> Failed: {e}")

if __name__ == "__main__":
    main()
