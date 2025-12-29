# som_layer.py

import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

class SOMLayer(pl.LightningModule):
    """
    Self-Organizing Map (SOM) Layer implementation.
    """
    def __init__(self, config):
        """
        Initializes the SOMLayer with the specified grid size and latent dimension.
        """
        super(SOMLayer, self).__init__()

        hp = config['hyperparameters']
        self.model_arch = hp['model_arch']

        som_hp = hp['som']
        vit_hp = hp['vit'] if self.model_arch == 'vit_som' else None
        data_hp = config['data']

        self.total_epochs = hp['total_epochs']
        self.batch_size = hp['batch_size']

        self.map_size = som_hp['map_size']
        self.Tmax = som_hp['Tmax']
        self.Tmin = som_hp['Tmin']
        self.topology = som_hp['topology']
        self.distance_fcn = som_hp['distance_fcn']
        self.n_prototypes = int(np.prod(self.map_size))

        self.use_reduced = som_hp['use_reduced'] if self.model_arch == 'vit_som' else False
        latent_dim = vit_hp['emb_dim'] if self.model_arch == 'vit_som' else config['hyperparameters']['ae']['encoder_dims'][-1]
        if not self.use_reduced and self.model_arch == 'vit_som':
            num_patches = (data_hp['input_size'] // vit_hp['patch_size']) ** 2
            latent_dim *= num_patches
        self.latent_dim = latent_dim

        self.current_temperature = self.Tmax

        if self.distance_fcn == 'cosine':
            # normalized for cosine similarity
            self.prototypes = nn.Parameter(
                torch.nn.functional.normalize(
                    torch.rand(self.n_prototypes, self.latent_dim),
                    p=2, dim=1
                )
            )
        else:
            # standard initialization for Euclidean/Manhattan
            self.prototypes = nn.Parameter(
                torch.rand(self.n_prototypes, self.latent_dim)
            )

        self.create_grid_positions()

    def create_grid_positions(self):
        if self.topology == 'square':
            grid_y, grid_x = torch.meshgrid(
                torch.arange(self.map_size[0]),
                torch.arange(self.map_size[1]),
                indexing='ij'
            )
            positions = torch.stack([grid_y, grid_x], dim=-1).view(-1, 2).float()
        elif self.topology == 'hexa':
            rows, cols = self.map_size
            positions = torch.zeros(self.n_prototypes, 2)
            for i in range(self.n_prototypes):
                row = i // cols
                col = i % cols
                positions[i, 0] = col
                positions[i, 1] = row * np.sqrt(3) / 2
                if row % 2 == 1:
                    positions[i, 0] += 0.5
        else:
            raise ValueError(f"Unsupported topology: {self.topology}")

        self.register_buffer('grid_positions', positions)

    def forward(self, x):
        if x.dim() > 2:
            x = x.flatten(start_dim=1)
        
        distances = self.compute_distances(x)
        bmu_indices = torch.argmin(distances, dim=1)
        return distances, bmu_indices
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_flattened = x.view(x.size(0), -1)
        distances, bmu_indices = self.forward(x_flattened)
        
        self.update_temperature()

        weights = self.compute_weights(bmu_indices)
        loss = self.som_loss(weights, distances)
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=self.opt_lr, 
                                    betas=(self.beta_1, self.beta_2))
        
        return optimizer

    def compute_distances(self, x):
        '''
        Distance formula function for BMU computation. Support for manhattan, euclidean, and cosine.
        '''
        if self.distance_fcn == 'manhattan':
            distances = torch.cdist(x, self.prototypes, p=1)
        elif self.distance_fcn == 'euclidean':
            distances = torch.cdist(x, self.prototypes, p=2)
        elif self.distance_fcn == 'cosine':
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            prototypes_norm = torch.nn.functional.normalize(self.prototypes, p=2, dim=1)
            distances = 1 - torch.mm(x_norm, prototypes_norm.T)
        else:
            raise ValueError(f"Unsupported distance function: {self.distance_fcn}")
        return distances

    def update_temperature(self, iteration):
        '''
        Iteratively update temperature based on cosine scheduler.
        '''
        total_iterations = (len(self.trainer.train_dataloader.dataset) / self.batch_size) * self.total_epochs
        self.current_temperature = self.Tmax * (self.Tmin / self.Tmax) ** (iteration / (total_iterations - 1))

    def index_to_position(self, indices):
        return torch.stack((indices // self.map_size[1], indices % self.map_size[1]), dim=1).float()

    def som_loss(self, weights, distances):
        """
        Calculates SOM loss as the mean of weighted distances.
        """
        weighted_distances = weights * distances
        return torch.mean(weighted_distances)

    def compute_weights(self, bmu_indices):
        """
        Computes the weights for SOM loss based on distances to BMU positions.
        """
        bmu_grid_positions = self.grid_positions[bmu_indices]
        distances_to_bmu = torch.norm(self.grid_positions.unsqueeze(0) - bmu_grid_positions.unsqueeze(1), dim=2)
        weights = torch.exp(-distances_to_bmu ** 2 / (2 * self.current_temperature ** 2))

        return weights
    