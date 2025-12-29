# desom.py

import torch
import math
from torch import nn
import torchvision
import pytorch_lightning as pl

from models.ae import Autoencoder
from models.som_layer import SOMLayer
from tools.evaluation import evaluate_clustering, evaluate_kmeans, evaluate_classification, visualize_decoded_prototypes, visualize_label_heatmap

class DESOM(pl.LightningModule):
    '''
    Deep Embedded Self-Organizing Map (DESOM) model integration with Autoencoder and SOM.
    '''
    def __init__(self, config):
        '''
        Initializes the DESOM model.
        '''
        super(DESOM, self).__init__()
        torch.set_float32_matmul_precision('medium')
        self.config = config

        self.total_epochs = config['hyperparameters']['total_epochs']
        self.batch_size = config['hyperparameters']['batch_size']
        self.gamma = config['hyperparameters']['gamma']

        self.encoder_dims = config['hyperparameters']['ae']['encoder_dims']

        self.opt_type = config['hyperparameters']['optimizer']['type']
        self.opt_lr = config['hyperparameters']['optimizer']['lr']
        self.beta_1 = config['hyperparameters']['optimizer']['beta_1']
        self.beta_2 = config['hyperparameters']['optimizer']['beta_2']

        self.num_classes = config['data']['num_classes']

        self.classification = self.num_classes > 0

        self.autoencoder = Autoencoder(config)
        self.som_layer = SOMLayer(config)
        self.classifier = nn.Linear(self.encoder_dims[-1], self.num_classes) if self.classification else nn.Identity()

        self.recon_loss_fn = nn.L1Loss()
        self.cls_loss_fn = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        self.register_buffer('iteration', torch.tensor(0))

    def forward(self, x):
        x_encoded = self.autoencoder.encoder(x)
        distances, bmu_indices = self.som_layer(x_encoded)
        cls_logits = self.classifier(x_encoded) if self.classification else None
        return cls_logits, x_encoded, distances, bmu_indices

    def training_step(self, batch, batch_idx):
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
        '''
        Configures the optimizer and scheduler, applying and cosine scheduler.
        '''
        if self.opt_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                            lr=self.opt_lr, 
                            betas=(self.beta_1, self.beta_2))

            return optimizer
        elif self.opt_type == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.opt_lr * self.batch_size / 256, 
                                        betas=(self.beta_1, self.beta_2),
                                        weight_decay=self.weight_decay)

            lr_func = lambda epoch: min((epoch + 1) / (self.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.total_epochs * math.pi) + 1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

            return [optimizer], [scheduler]
        
    def update(self):
        '''
        Temperature update per iteration.
        '''
        self.som_layer.update_temperature(self.iteration)
        self.iteration += 1

    def on_train_end(self):
        '''
        End of training hook to capture memory usage
        '''
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / 1e9  # convert bytes to gigabytes
        
        print(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")
        
    def compute_and_log_losses(self, batch, x_encoded, distances, bmu_indices, cls_logits, prefix):
        '''
        Compute losses for both cls/clustering cases and log to TensorBoard.
        '''
        x, y = batch
        x_flattened = x.view(x.size(0), -1)

        weights = self.som_layer.compute_weights(bmu_indices)
        som_loss = self.som_layer.som_loss(weights, distances)

        recon_loss = self.recon_loss_fn(self.autoencoder.decoder(x_encoded), x_flattened)

        if self.classification:
            cls_loss = self.cls_loss_fn(cls_logits, y)
            total_loss = cls_loss + self.gamma * (som_loss + recon_loss)
            self.log(f'{prefix}/cls_loss', cls_loss, on_step=True)
        else:
            total_loss = recon_loss + self.gamma * som_loss

        self.log_dict({
            f'{prefix}/recon_loss': recon_loss,
            f'{prefix}/som_loss': som_loss,
        }, on_step=True)
        self.log(f'{prefix}/total_loss', total_loss, prog_bar=True)

        return total_loss
    
    def log_images(self, x, pred_img, bmu_indices, distances, prefix):
        '''
        Log images of SOM grid to TensorBoard.
        '''
        grid_x = torchvision.utils.make_grid(x[:10])
        self.logger.experiment.add_image(f'{prefix}/x', grid_x, 0)

        grid_pred_img = torchvision.utils.make_grid(pred_img[:10])
        self.logger.experiment.add_image(f'{prefix}/pred_img', grid_pred_img, 0)

        grid_bmu_indices = torchvision.utils.make_grid(torch.unsqueeze(bmu_indices[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_bmu_indices', grid_bmu_indices, 0)

        grid_distances = torchvision.utils.make_grid(torch.unsqueeze(distances[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_distances', grid_distances, 0)
