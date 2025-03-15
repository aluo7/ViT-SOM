"""
Implementation of DESOM in PyTorch Lightning

@author Alan Luo
@version 1.0
"""

import torch
import math
from torch import nn
import torchvision
import pytorch_lightning as pl

from models.ae import Autoencoder
from models.som_layer import SOMLayer
from tools.evaluation import evaluate_clustering, evaluate_kmeans, evaluate_classification

class DESOM(pl.LightningModule):
    """
    Deep Embedded Self-Organizing Map (DESOM) model integration

    Attributes:
        autoencoder (Autoencoder): Autoencoder for feature extraction
        som_layer (SOMLayer): The SOM layer for topology-preserving mappings

    """
    def __init__(self, config):
        """
        Initializes the DESOM model

        Args:
            config (dict): Python dictionary config read from `desom.yaml`
            train_loader (iterable): PyTorch train set DataLoader
        """
        super(DESOM, self).__init__()
        torch.set_float32_matmul_precision('medium')
        self.config = config

        self.classification = self.config.data.num_classes > 0

        self.autoencoder = Autoencoder(config)
        self.som_layer = SOMLayer(config)
        self.classifier = nn.Linear(self.config.hparams.ae.encoder_dims[-1], self.config.data.num_classes) if self.classification else nn.Identity()

        self.recon_loss_fn = nn.L1Loss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters(self.config.to_dict())
        self.register_buffer('iteration', torch.tensor(0))

    def forward(self, x):
        """
        Process input through both the autoencoder and SOM layer

        Args:
            x (torch.Tensor): The input tensor

        Returns:
            tuple: The decoded output from the autoencoder and distances from the SOM layer
        """
        x_encoded = self.autoencoder.encoder(x)
        distances, bmu_indices = self.som_layer(x_encoded)
        cls_logits = self.classifier(x_encoded) if self.classification else None
        return cls_logits, x_encoded, distances, bmu_indices

    def training_step(self, batch, batch_idx):
        """
        Performs a single step of training

        Args:
            batch (tuple): A batch from the DataLoader
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The loss for the batch
        """
        x, _ = batch
        x_flattened = x.view(x.size(0), -1)
        cls_logits, x_encoded, distances, bmu_indices = self.forward(x_flattened)

        self.update()

        total_loss = self.compute_and_log_losses(
            batch=batch,
            x_encoded=x_encoded,
            distances=distances,
            bmu_indices=bmu_indices,
            cls_logits=cls_logits,
            prefix='train'
        )

        pred_img = self.autoencoder.decoder(x_encoded)
        self.log_images(x=x, pred_img=pred_img, bmu_indices=bmu_indices, distances=distances, prefix='train')

        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """
        Performs a single step of validation

        Args:
            batch (tuple): A batch from the DataLoader
            batch_idx (int): The index of the batch
        """
        x, y = batch
        x_flattened = x.view(x.size(0), -1)
        cls_logits, x_encoded, distances, bmu_indices = self.forward(x_flattened)

        total_loss = self.compute_and_log_losses(
            batch=batch,
            x_encoded=x_encoded,
            distances=distances,
            bmu_indices=bmu_indices,
            cls_logits=cls_logits,
            prefix='val'
        )

        pred_img = self.autoencoder.decoder(x_encoded)
        self.log_images(x=x, pred_img=pred_img, bmu_indices=bmu_indices, distances=distances, prefix='val')
        
        return total_loss
    
    def configure_optimizers(self):
        if self.config.hparams.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                            lr=self.config.hparams.optimizer.lr, 
                            betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))

            return optimizer
        elif self.config.hparams.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.config.hparams.optimizer.lr * self.config.hparams.batch_size / 256, 
                                        betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2),
                                        weight_decay=self.weight_decay)

            lr_func = lambda epoch: min((epoch + 1) / (self.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.config.hparams.total_epochs * math.pi) + 1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

            return [optimizer], [scheduler]

    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 10 == 0:
            if self.classification:
                accuracy, precision, recall, f1 = evaluate_classification(self, self.config, self.trainer.val_dataloaders)
                
                self.log_dict({
                    'classification/accuracy': accuracy,
                    'classification/precision': precision,
                    'classification/recall': recall,
                    'classification/f1': f1
                }, on_epoch=True)
            else:
                purity, nmi = evaluate_clustering(self, self.config, self.trainer.train_dataloader)
                purity_kmeans, nmi_kmeans = evaluate_kmeans(self, self.config, self.trainer.train_dataloader)

                self.log_dict({
                    'clustering/purity': purity,
                    'clustering/nmi': nmi,
                    'clustering/purity_kmeans': purity_kmeans,
                    'clustering/nmi_kmeans': nmi_kmeans,
                }, on_epoch=True)

            # for name, params in self.named_parameters():
            #     self.logger.experiment.add_histogram(f'parameters/{name}', params, self.current_epoch)

            # visualize_decoded_prototypes(self, self.config, output_dir='experiments/plots/desom/decoded_prototypes')
            # visualize_label_heatmap(self, self.config, self.trainer.train_dataloader, output_dir='experiments/plots/desom/label_heatmap')

    def update(self):
        self.som_layer.update_temperature(self.iteration)
        self.iteration += 1
        
    def compute_and_log_losses(self, batch, x_encoded, distances, bmu_indices, cls_logits, prefix):
        x, y = batch
        x_flattened = x.view(x.size(0), -1)

        weights = self.som_layer.compute_weights(bmu_indices)
        som_loss = self.som_layer.som_loss(weights, distances)

        recon_loss = self.recon_loss_fn(self.autoencoder.decoder(x_encoded), x_flattened)

        if self.classification:
            cls_loss = self.cls_loss_fn(cls_logits, y)
            total_loss = cls_loss + self.config.hparams.gamma * (som_loss + recon_loss)
            self.log(f'{prefix}/cls_loss', cls_loss, on_step=True)
        else:
            total_loss = recon_loss + self.config.hparams.gamma * som_loss

        self.log_dict({
            f'{prefix}/recon_loss': recon_loss,
            f'{prefix}/som_loss': som_loss,
        }, on_step=True)
        self.log(f'{prefix}/total_loss', total_loss, prog_bar=True)

        return total_loss
    
    def log_images(self, x, pred_img, bmu_indices, distances, prefix):
        grid_x = torchvision.utils.make_grid(x[:10])
        self.logger.experiment.add_image(f'{prefix}/x', grid_x, 0)

        grid_pred_img = torchvision.utils.make_grid(pred_img[:10])
        self.logger.experiment.add_image(f'{prefix}/pred_img', grid_pred_img, 0)

        grid_bmu_indices = torchvision.utils.make_grid(torch.unsqueeze(bmu_indices[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_bmu_indices', grid_bmu_indices, 0)

        grid_distances = torchvision.utils.make_grid(torch.unsqueeze(distances[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_distances', grid_distances, 0)
