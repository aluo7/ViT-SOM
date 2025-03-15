import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
import os
import numpy as np

from data.data import get_dataloaders, extract_data_from_loader
from models.som import TestSOM 
from tools.utils import load_config

pl.seed_everything(0)

def main():
    print(f"GPU available: {torch.cuda.is_available()}")
    accelerator = os.getenv('ACCELERATOR', 'gpu')
    devices = int(os.getenv('DEVICES', 1))

    config = load_config('configs/som.yaml')

    train_loader, val_loader = get_dataloaders(config['data']['dataset'],
                           batch_size=config['hyperparameters']['batch_size'],
                           num_workers=config['data']['num_workers'])
    
    model = TestSOM(map_size=config['hyperparameters']['map_size'],
                    input_dim=config['hyperparameters']['encoder_dims'][0],
                    Tmax=config['hyperparameters']['Tmax'],
                    Tmin=config['hyperparameters']['Tmin'],
                    lr_max=config['hyperparameters']['lr_max'],
                    lr_min=config['hyperparameters']['lr_min'],
                    total_epochs=config['hyperparameters']['total_epochs'])
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    logger = TensorBoardLogger("logs", name="test_som")
    
    trainer = pl.Trainer(accelerator=accelerator, logger=logger,
                         devices=devices, max_epochs=config['hyperparameters']['total_epochs'],
                         enable_progress_bar=True,
                         callbacks=[lr_monitor])
    
    trainer.fit(model, train_loader, val_loader)

    X_train, y_train = extract_data_from_loader(train_loader)
    n_clusters = len(np.unique(y_train))

    quantization_error = model.compute_quantization_error(val_loader)
    topographic_error = model.compute_topographic_error(val_loader)
    print(f"Quantization Error: {quantization_error}")
    print(f"Topographic Error: {topographic_error}")

    accuracy, nmi, pur = model.evaluate_with_kmeans(train_loader, val_loader, n_classes=n_clusters)
    print(f"Clustering Accuracy: {accuracy}\nPurity: {pur}\nNMI: {nmi}")


if __name__ == "__main__":
    main()
