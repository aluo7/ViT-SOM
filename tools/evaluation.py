# evaluation.py

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import umap
import math
import torch.nn.functional as F

from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from torchvision.utils import make_grid
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, precision_recall_fscore_support

def evaluate_clustering(model, config, dataloader):
    '''
    Evaluates clustering performance (Purity, NMI) using the model's native BMU assignments.
    '''
    model.eval()
    y_preds = []
    y_trues = []

    start_time = time.time()
    model_arch = config['hyperparameters']['model_arch']

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)

            if model_arch == 'vit_som':
                num_channels = config['data']['num_channels']
                input_size = config['data']['input_size']
                _, _, _, _, bmu_indices = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch in ['desom']:
                _, _, _, bmu_indices = model(x.view(x.size(0), -1))

            y_preds.extend(bmu_indices.detach().cpu().numpy())
            y_trues.extend(y.detach().cpu().numpy())

    y_trues_np = np.array(y_trues).flatten()
    y_preds_np = np.array(y_preds)

    purity = calculate_purity(y_trues_np, y_preds_np)
    nmi = normalized_mutual_info_score(y_trues_np, y_preds_np)
    inference_time = time.time() - start_time

    print(f'Purity: {purity:.3f}, NMI: {nmi:.3f}, Inference Time: {inference_time:.3f}')
    return purity, nmi, inference_time

def evaluate_kmeans(model, config, dataloader):
    '''
    Evaluated clustering performance by applying K-Means on the latent feature embeddings.
    '''
    model.eval()
    encoded = []
    y_trues = []

    start_time = time.time()
    model_arch = config['hyperparameters']['model_arch']

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)
        
            if model_arch  == 'vit_som':
                num_channels = config['data']['num_channels']
                input_size = config['data']['input_size']
                _, x_encoded, _, _, _ = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch == 'desom':
                _, x_encoded, _, _ = model(x.view(x.size(0), -1))

            encoded.extend(x_encoded.cpu().detach().numpy())
            y_trues.extend(y.cpu().detach().numpy())

    encoded = np.array(encoded).reshape(len(encoded), -1)
    y_trues = np.array(y_trues)

    kmeans = KMeans(n_clusters=len(np.unique(y_trues)), random_state=0, n_init=10)
    y_preds = kmeans.fit_predict(encoded)
    
    purity_cluster = calculate_purity(y_trues, y_preds)
    nmi_cluster = normalized_mutual_info_score(y_trues, y_preds)
    inference_time = time.time() - start_time
    
    print(f'Purity (KMeans): {purity_cluster:.3f}, NMI (KMeans): {nmi_cluster:.3f}, Inference Time: {inference_time:.3f}')
    return purity_cluster, nmi_cluster, inference_time

def evaluate_classification(model, config, dataloader):
    '''
    Evaluates classification metrics (Accuracy, Precision, Recall, F1) using the classification head.
    '''
    model.eval()
    y_preds = []
    y_trues = []

    start_time = time.time()
    model_arch = config['hyperparameters']['model_arch']

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)

            if model_arch in ['vit', 'deit', 'mobile_vit', 'swin']:
                cls_logits = model(x)
            elif model_arch == 'vit_som':
                num_channels = config['data']['num_channels']
                input_size = config['data']['input_size']
                _, _, cls_logits, _, _ = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch == 'desom':
                cls_logits, _, _, _ = model(x.view(x.size(0), -1))

            y_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()

            y_preds.extend(y_pred)
            y_trues.extend(y.cpu().numpy())

    accuracy = accuracy_score(y_trues, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=np.nan)
    inference_time = time.time() - start_time

    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, Inference Time: {inference_time:.3f}')
    return accuracy, precision, recall, f1, inference_time

def calculate_purity(y_trues, y_preds):
    '''
    Computes clustering purity by aligning predicted clusters with true labels via majority voting.
    '''
    if isinstance(y_trues, torch.Tensor):
        y_trues = y_trues.numpy()
    if isinstance(y_preds, torch.Tensor):
        y_preds = y_preds.numpy()
    
    y_trues = y_trues.astype(np.int64)
    assert y_preds.size == y_trues.size, f'y_preds ({y_preds.size}) and y_trues ({y_trues.size}) must be the same size'
    
    D = max(y_preds.max(), y_trues.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_preds.size):
        w[y_preds[i], y_trues[i]] += 1
    
    label_mapping = w.argmax(axis=1)
    y_pred_voted = y_preds.copy()
    for i in range(y_preds.size):
        y_pred_voted[i] = label_mapping[y_preds[i]]
    return accuracy_score(y_pred_voted, y_trues)

def visualize_decoded_prototypes(model, config, output_dir='experiments/plots', return_decoded=False):
    '''
    Decodes SOM prototypes back into image space using the decoder and saves the grid visualization.
    '''
    model.eval()

    model_arch = config['hyperparameters']['model_arch']
    use_reduced = config['hyperparameters']['som']['use_reduced']

    if model_arch != 'vit_som' or use_reduced:
        print("Visualization supported only for vit_som with use_reduced=False.")
        return None
    
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        prototypes = model.som_layer.prototypes.detach().cpu()  # [n_prototypes, flat_dim]
        num_channels = config['data']['num_channels']
        input_size = config['data']['input_size']
        patch_size = config['hyperparameters']['vit']['patch_size']
        num_patches = (input_size // patch_size) ** 2
        embed_dim = config['hyperparameters']['vit']['emb_dim']

        if prototypes.shape[1] != num_patches * embed_dim:
            raise ValueError("Prototype dimensions mismatch for decoding.")
        
        decoded_prototypes = np.zeros((prototypes.shape[0], num_channels, input_size, input_size))

        for i, proto in enumerate(prototypes):
            decoded = decode_prototype(model.vit, proto, num_patches, embed_dim, model.device)
            decoded_prototypes[i] = decoded.cpu().numpy().squeeze(0)  # [C, H, W]

    # plot    
    nrows, ncols = model.som_layer.map_size
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        img = decoded_prototypes[i]

        if num_channels == 1:  # Grayscale
            img = img.squeeze(0)  # [1, H, W] -> [H, W]

        elif num_channels == 3:  # RGB
            img = img.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
            
        ax.imshow(img, cmap='gray' if num_channels == 1 else None)
        ax.axis('off')

    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    save_path = os.path.join(output_dir, f'{model_arch}_epoch_{model.current_epoch}_decoded_prototypes.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved decoded prototypes visualization to {output_dir}")

    if return_decoded:
        return decoded_prototypes  # [n_prototypes, C, H, W]
    
def decode_prototype(vit, prototype, num_patches, embed_dim, device):
    '''
    Helper function to reconstruct a single prototype vector into an image using the ViT decoder.
    '''
    prototype = prototype.to(device)
    patches = prototype.reshape(1, num_patches, embed_dim)

    cls_placeholder = torch.zeros(1, 1, embed_dim, device=device)
    x = torch.cat((cls_placeholder, patches), dim=1)
    
    decoded_patches, _ = vit.forward_decoder(x, return_attn=False)  # extract decoded_patches from tuple
    recon_img = vit.unpatchify(decoded_patches)

    return recon_img  # [1, C, H, W]

def visualize_label_heatmap(model, config, dataloader, output_dir='experiments/plots'):
    '''
    Generates and saves a heatmap showing the distribution of ground truth labels across the SOM grid.
    '''
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    model_arch = config['hyperparameters']['model_arch']
    
    y_preds = []
    y_trues = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)

            if model_arch == 'vit_som':
                num_channels = config['data']['num_channels']
                input_size = config['data']['input_size']
                _, _, _, _, bmu_indices = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch == 'desom':
                _, _, _, bmu_indices = model(x.view(x.size(0), -1))

            y_preds.extend(bmu_indices.cpu().numpy())
            y_trues.extend(y.cpu().numpy())

    y_preds = np.array(y_preds)
    y_trues = np.array(y_trues)

    map_size = model.som_layer.map_size
    heatmap = np.zeros(map_size, dtype=int)

    for bmu_index, label in zip(y_preds, y_trues):
        bmu_row, bmu_col = divmod(bmu_index, map_size[1])
        heatmap[bmu_row, bmu_col] = label

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(heatmap, annot=True, fmt="d", cmap="viridis")
    plt.savefig(os.path.join(output_dir, f'{model_arch}_epoch_{model.current_epoch}_label_heatmap.png'))
    plt.close(fig)

    print(f"Saved label heatmap visualization to {output_dir}")

def visualize_umap_progression(model, config, dataloader, epoch=0, output_dir='experiments/plots/vit_som/umap'):
    '''
    Generates a UMAP scatter plot of the latent representations, colored by class label.
    '''
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    encoded_points = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(model.device), y.to(model.device)
            latent = model.get_latent_representation(x).cpu()
            encoded_points.append(latent)
            all_labels.append(y.cpu())

    encoded_points = torch.cat(encoded_points).numpy()
    all_labels = torch.cat(all_labels).numpy()

    plt.figure(figsize=(10, 8), dpi=300)
    
    plt.axis('off')
    
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine',
        random_state=42,
        n_jobs=-1
    )
    embedding = reducer.fit_transform(encoded_points)
    
    scatter = plt.scatter(
        embedding[:, 0], 
        embedding[:, 1],
        c=all_labels,
        cmap='tab10',
        s=3,
        alpha=0.7,
        edgecolor='none',
        rasterized=True
    )
    
    cbar = plt.colorbar(scatter, ticks=range(10), drawedges=True)
    cbar.set_ticklabels([str(i) for i in range(10)])
    cbar.ax.tick_params(labelsize=10, width=0.5)
    cbar.outline.set_linewidth(0.5)
    
    plt.savefig(
        os.path.join(output_dir, f'som_umap_epoch_{epoch}.png'),
        bbox_inches='tight',
        pad_inches=0,
        transparent=False,
        dpi=400
    )
    plt.close()
