import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np
import os

from data.data import get_dataloaders
from models.vit_som import ViTSOM
from tools.evaluation import evaluate_clustering, evaluate_kmeans
from tools.utils import load_config

pl.seed_everything(0)


def main():
    """
    Main function to execute the training and evaluation pipeline
    """
    print(torch.cuda.is_available())
    config = load_config('configs/vit_som/vit_som.yaml')

    purity_scores = []
    nmi_scores = []
    purity_cluster_scores = []
    nmi_cluster_scores = []

    dataset_name = config['data']['dataset']
    train_loader, val_loader = get_dataloaders(
        config['data']['dataset'],
        batch_size=config['hyperparameters']['batch_size'],
        num_workers=config['data']['num_workers'],
        use_validation=config['data']['num_classes'] > 0
    )

    print(f'Training ViT SOM...')
    model = ViTSOM.load_from_checkpoint("experiments/logs/vit_som/test/checkpoints.ckpt")
    model.eval()

    torch.cuda.empty_cache()

    purity, nmi, inference_time = evaluate_clustering(model, config, train_loader)
    purity_scores.append(purity)
    nmi_scores.append(nmi)

    purity_cluster, nmi_cluster, inference_time = evaluate_kmeans(model, config, train_loader)
    purity_cluster_scores.append(purity_cluster)
    nmi_cluster_scores.append(nmi_cluster)


    results = {
        'Purity Mean': np.mean(purity_scores),
        'Purity Std': np.std(purity_scores),
        'NMI Mean': np.mean(nmi_scores),
        'NMI Std': np.std(nmi_scores),
        'Purity Cluster Mean': np.mean(purity_cluster_scores),
        'Purity Cluster Std': np.std(purity_cluster_scores),
        'NMI Cluster Mean': np.mean(nmi_cluster_scores),
        'NMI Cluster Std': np.std(nmi_cluster_scores)
    }
    
    print(f'Aggregated results across runs for {dataset_name}:')
    for key, value in results.items():
        print(f"{key}: {value:.4f}")

if __name__ == "__main__":
    main()
