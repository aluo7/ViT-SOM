"""
Benchmarking script for DESOM model

@author Alan Luo
@version 1.0
"""

import argparse
import torch
import pytorch_lightning as pl
import numpy as np
import shutil
import os
import time

from models.desom import DESOM
from data.data import get_dataloaders
from tools.evaluation import evaluate_clustering, evaluate_classification
from tools.utils import load_config

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

def clear_model_states(directory):
    """
    Clear all model states in the specified directory
    """
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main():
    """
    Main function to execute the training and evaluation pipeline for ViTSOM
    """
    pl.seed_everything(0)

    parser = argparse.ArgumentParser(description="Benchmarking script for DESOM model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)

    print(torch.cuda.is_available())
    accelerator = os.getenv('ACCELERATOR')
    devices = os.getenv('DEVICES')

    classification = config.data.num_classes > 0

    n_runs = 10
    purity_scores = []
    nmi_scores = []
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    model_states_dir = 'experiments/states/vit_som'

    for run in range(n_runs):
        dataset_name = config.data.dataset
        print(f'Starting run {run+1} for {dataset_name}...')
        
        clear_model_states(model_states_dir)

        train_loader, val_loader, test_loader = get_dataloaders(
            config=config,
            batch_size=config.hparams.batch_size,
            num_workers=config.data.num_workers,
            use_validation=classification
        )
        
        print(f'Training DESOM...')
        model = DESOM(config=config)
        
        logger = TensorBoardLogger('experiments/logs', name='desom')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        final_model = ModelCheckpoint(dirpath='experiments/states/desom', filename=f'desom_{dataset_name}_final', save_last=True)
        
        trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                            devices=devices, max_epochs=config.hparams.total_epochs, 
                            enable_progress_bar=True,
                            callbacks=[lr_monitor, final_model]
        )
        
        trainer.fit(model, train_loader, val_loader)

        torch.cuda.empty_cache()

        best_model = DESOM.load_from_checkpoint(os.path.join(final_model.dirpath, final_model.filename + '.ckpt'))

        if classification:
            accuracy, recall, precision, f1 = evaluate_classification(best_model, config, test_loader)
            accuracy_scores.append(accuracy)
            precision_scores.append(precision)
            recall_scores.append(recall)
            f1_scores.append(f1)

            results = {
                'Accuracy Mean, Std': (np.mean(accuracy_scores), np.std(accuracy_scores)),
                'Precision Mean, Std': (np.mean(precision_scores), np.std(precision_scores)),
                'Recall Mean, Std': (np.mean(recall_scores), np.std(recall_scores)),
                'F1 Mean, Std': (np.mean(f1_scores), np.std(f1_scores))
            }
        else:
            purity, nmi = evaluate_clustering(best_model, config, train_loader)
            purity_scores.append(purity)
            nmi_scores.append(nmi)

            results = {
                'Purity Mean, Std': (np.mean(purity_scores), np.std(purity_scores)),
                'NMI Mean, Std': (np.mean(nmi_scores), np.std(nmi_scores))
            }

        time.sleep(10)

        for filename in os.listdir('experiments/states/desom'):
            file_path = os.path.join('experiments/states/desom', filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)

    print(f'Aggregated results across runs for {dataset_name}:')
    for key, value in results.items():
        print(f'{key}: {value[0]:.3f}, {value[1]:.3f}')

if __name__ == '__main__':
    main()
