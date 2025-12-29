# train_vit_som.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import numpy as np
import shutil
import os
import argparse
import time

from models.vit_som import ViTSOM
from data.data import get_dataloaders
from tools.evaluation import evaluate_classification, evaluate_clustering
from tools.utils import load_config

def clear_directory(directory):
    '''
    Clear all model states in the specified directory
    '''
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main():
    '''
    Main function to execute the training and evaluation pipeline for ViTSOM
    '''
    parser = argparse.ArgumentParser(description="Benchmarking script for ViT SOM model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(0)
    
    # --- Environment Setup ---
    hp = config['hyperparameters']
    data_hp = config['data']
    use_validation = data_hp['num_classes'] > 0

    print(torch.cuda.is_available())
    accelerator = os.getenv('ACCELERATOR', 'gpu')
    devices = os.getenv('DEVICES', 1)
    dataset_name = data_hp['dataset']
    model_states_dir = 'experiments/states/vit_som'

    n_runs = 5
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'purity': [], 'nmi': [], 'run_duration': [], 'inference_time': []}

    for run in range(n_runs):
        print(f'Starting run {run+1} for {dataset_name}...')
        start_time = time.time()

        clear_directory(model_states_dir)

        train_loader, val_loader, test_loader = get_dataloaders(
            dataset_name=data_hp['dataset'],
            batch_size=hp['batch_size'],
            num_workers=data_hp['num_workers'],
            use_validation=use_validation,
            horizontal_flip=data_hp['augment']['horizontal_flip'],
            randaug_n=data_hp['augment']['randaug_n'],
            resize_scale=tuple(data_hp['augment']['resize_scale']),
            resize_ratio=tuple(data_hp['augment']['resize_ratio']),
            reprob=data_hp['augment']['reprob'],
            remode=data_hp['augment']['remode'],
            recount=data_hp['augment']['recount'],
            autoaugment=data_hp['augment']['autoaugment'],
            input_size=data_hp['input_size'],
            num_channels=data_hp['num_channels']
        )

        print('Training ViTSOM...')
        model = ViTSOM(config=config)

        logger = TensorBoardLogger('experiments/logs', name=f'vit_som/{dataset_name}', log_graph=True)
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        if use_validation:
            checkpoint = ModelCheckpoint(monitor=f'val/accuracy', mode='max', dirpath=model_states_dir, filename=f'vit_som_{dataset_name}_best', save_top_k=1)
        else:
            checkpoint = ModelCheckpoint(dirpath=model_states_dir, filename=f'vit_som_{dataset_name}_final', save_last=True)

        trainer = pl.Trainer(accelerator=accelerator, logger=logger, 
                            devices=devices, max_epochs=hp['total_epochs'], 
                            enable_progress_bar=True,
                            callbacks=[lr_monitor, checkpoint],
                            log_every_n_steps=50
        )
        
        trainer.fit(model, train_loader, val_loader)
        torch.cuda.empty_cache()

        end_time = time.time()
        run_duration = end_time - start_time
        print(f'Run {run+1} duration: {run_duration:.2f} seconds')
        
        # final_model = ViTSOM.load_from_checkpoint(checkpoint.best_model_path, config=config)

        if use_validation:
            # final_model = ViTSOM.load_from_checkpoint(checkpoint.best_model_path, config=config)
            accuracy, precision, recall, f1, inference_time = evaluate_classification(model, config, test_loader)

            all_metrics['accuracy'].append(accuracy)
            all_metrics['precision'].append(precision)
            all_metrics['recall'].append(recall)
            all_metrics['f1'].append(f1)
        else:
            final_model = ViTSOM.load_from_checkpoint(checkpoint.last_model_path, config=config)
            purity, nmi, inference_time = evaluate_clustering(final_model, config, train_loader)

            all_metrics['purity'].append(purity)
            all_metrics['nmi'].append(nmi)

        all_metrics['run_duration'].append(run_duration)
        all_metrics['inference_time'].append(inference_time)
    
    if n_runs > 1:
        print(f"\n--- Aggregated Results Across {n_runs} Runs for {dataset_name} ---")
        for key, scores in all_metrics.items():
            if scores:
                mean_score = np.mean(scores)
                std_score = np.std(scores)

                if key == 'run_duration' or key == 'inference_time':
                    print(f'Avg {key.capitalize()} (Std): {mean_score:.2f}s ({std_score:.2f}s)')
                else:
                    print(f'{key.capitalize()} Mean (Std): {mean_score:.4f} ({std_score:.4f})')

if __name__ == '__main__':
    main()
