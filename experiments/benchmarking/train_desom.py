# train_desom.py

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

import numpy as np
import shutil
import os
import time
import argparse

from models.desom import DESOM
from data.data import get_dataloaders
from tools.evaluation import evaluate_clustering, evaluate_kmeans, evaluate_classification, visualize_decoded_prototypes
from tools.utils import load_config

def clear_directory(directory):
    '''
    Deletes the contents of a directory if it exists, then recreates it.
    '''
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

def main():
    '''
    Main function to execute the training and evaluation pipeline for DeiT.
    '''
    parser = argparse.ArgumentParser(description="Benchmarking script for DESOM model")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()

    config = load_config(args.config)
    pl.seed_everything(0)
    
    print(torch.cuda.is_available())
    accelerator = os.getenv('ACCELERATOR')
    devices = os.getenv('DEVICES')

    dataset_name = config['data']['dataset']
    model_states_dir = 'experiments/states/desom'

    n_runs = 5
    all_metrics = {'purity': [], 'nmi': [], 'run_duration': [], 'inference_time': []}

    # benchmarking loop
    for run in range(n_runs):
        print(f'Starting run {run+1} for {dataset_name}...')
        start_time = time.time()

        clear_directory(model_states_dir)
        
        train_loader, val_loader, _ = get_dataloaders(
            dataset_name=config['data']['dataset'],
            batch_size=config['hyperparameters']['batch_size'],
            num_workers=config['data']['num_workers'],
            use_validation=config['data']['num_classes'] > 0,
            horizontal_flip=config['data']['augment']['horizontal_flip'],
            randaug_n=config['data']['augment']['randaug_n'],
            resize_scale=tuple(config['data']['augment']['resize_scale']),
            resize_ratio=tuple(config['data']['augment']['resize_ratio']),
            reprob=config['data']['augment']['reprob'],
            remode=config['data']['augment']['remode'],
            recount=config['data']['augment']['recount'],
            autoaugment=config['data']['augment']['autoaugment'],
            input_size=config['data']['input_size'],
            num_channels=config['data']['num_channels']
        )
        
        print(f'Training DESOM...')
        model = DESOM(config=config)
        
        logger = TensorBoardLogger('experiments/logs', name='desom')
        lr_monitor = LearningRateMonitor(logging_interval='epoch')

        final_model = ModelCheckpoint(dirpath='experiments/states/desom', filename=f'desom_{dataset_name}_final', save_last=True)
        
        trainer = pl.Trainer(accelerator=accelerator, logger=logger, 
                             devices=devices, max_epochs=config['hyperparameters']['total_epochs'], 
                             enable_progress_bar=True,
                             callbacks=[lr_monitor, final_model]
        )
        
        trainer.fit(model, train_loader, val_loader)

        torch.cuda.empty_cache()
        
        end_time = time.time()
        run_duration = end_time - start_time
        print(f'Run {run+1} duration: {run_duration:.2f} seconds')

        best_model = DESOM.load_from_checkpoint(os.path.join(final_model.dirpath, final_model.filename + '.ckpt'))

        purity, nmi, inference_time = evaluate_clustering(best_model, config, train_loader)
        all_metrics['purity'].append(purity)
        all_metrics['nmi'].append(nmi)

        all_metrics['run_duration'].append(run_duration)
        all_metrics['inference_time'].append(inference_time)


    # aggregate and print results
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
