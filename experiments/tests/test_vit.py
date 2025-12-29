# Test Results: [{'test_loss': 0.07560201734304428, 'test_acc': 0.9782000184059143}] - MNIST, 10 epochs
# Test Results: [{'test_loss': 0.37634986639022827, 'test_acc': 0.8589000105857849}] - Fashion-MNIST, 10 epochs
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from data.data import get_dataloaders
from models.vit import VisionTransformer
from tools.utils import load_config

pl.seed_everything(42)

def main():
    config = load_config('configs//vit/vit.yaml')
    print(f"CUDA Available: {torch.cuda.is_available()}")
    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
    devices = 1 if torch.cuda.is_available() else None

    train_loader, test_loader = get_dataloaders(
        dataset_name=config['data']['dataset'],
        batch_size=config['hyperparameters']['batch_size'],
        num_workers=config['data']['num_workers'],
        use_validation=config['data']['num_classes'] > 0
    )
    
    model = VisionTransformer(
        config=config, n_classes=config['data']['n_classes'],
        image_size=config['data']['image_size'][2],
        patch_size=config['hyperparameters']['vit']['patch_size']
    )

    logger = TensorBoardLogger("logs", name="vit")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(
        max_epochs=config['hyperparameters']['total_epochs'],
        accelerator=accelerator,
        devices=devices,
        logger=logger,
        callbacks=[lr_monitor]
    )

    trainer.fit(model, train_loader, test_loader)

    result = trainer.test(dataloaders=test_loader, model=model)
    print(f"Test Results: {result}")

if __name__ == "__main__":
    main()
