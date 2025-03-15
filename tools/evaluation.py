import torch
import numpy as np
import time

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score, accuracy_score, precision_recall_fscore_support

def evaluate_clustering(model, config, dataloader):
    model.eval()
    y_preds = []
    y_trues = []

    start_time = time.time()
    model_arch = config.hparams.model_arch

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)

            if model_arch == 'vit_som':
                num_channels = config.data.num_channels
                input_size = config.data.input_size
                _, _, _, _, bmu_indices = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch in ['desom']:
                _, _, _, bmu_indices = model(x.view(x.size(0), -1))
            else:
                _, bmu_indices = model(x.view(x.size(0), -1))

            y_preds.extend(bmu_indices.detach().cpu().numpy())
            y_trues.extend(y.detach().cpu().numpy())

    purity = calculate_purity(np.array(y_trues), np.array(y_preds))
    nmi = normalized_mutual_info_score(np.array(y_trues), np.array(y_preds))

    print(f'Purity: {purity:.3f}, NMI: {nmi:.3f}, Inference Time: {(time.time() - start_time):.3f}')
    return purity, nmi

def evaluate_kmeans(model, config, dataloader):
    model.eval()
    encoded = []
    y_trues = []

    start_time = time.time()
    model_arch = config.hparams.model_arch

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)
        
            if model_arch  == 'vit_som':
                num_channels = config.data.num_channels
                input_size = config.data.input_size
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
    
    print(f'Purity (KMeans): {purity_cluster:.3f}, NMI (KMeans): {nmi_cluster:.3f}, Inference Time: {(time.time() - start_time):.3f}')
    return purity_cluster, nmi_cluster

def evaluate_classification(model, config, dataloader):
    model.eval()
    y_preds = []
    y_trues = []

    start_time = time.time()
    model_arch = config.hparams.model_arch

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(model.device)
            y = y.to(model.device)

            if model_arch == 'vit':
                cls_logits = model(x)
            elif model_arch == 'vit_som':
                num_channels = config.data.num_channels
                input_size = config.data.input_size
                cls_logits, _, _, _, _ = model(x.reshape(-1, num_channels, input_size, input_size))
            elif model_arch == 'desom':
                cls_logits, _, _, _ = model(x.view(x.size(0), -1))

            y_pred = torch.argmax(cls_logits, dim=1).cpu().numpy()

            y_preds.extend(y_pred)
            y_trues.extend(y.cpu().numpy())

    accuracy = accuracy_score(y_trues, y_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_trues, y_preds, average='macro', zero_division=np.nan)

    print(f'Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1-score: {f1:.3f}, Inference Time: {(time.time() - start_time):.3f}')
    return accuracy, precision, recall, f1

def calculate_purity(y_trues, y_preds):
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
