"""
Implementation of SOM Layer in PyTorch Lightning

@author Alan Luo
@version 1.0
"""
import torch
from torch import nn
import pytorch_lightning as pl
import numpy as np

from tools.evaluation import evaluate_clustering

class SOMLayer(pl.LightningModule):
    """
    Self-Organizing Map (SOM) Layer implementation

    Attributes:
        map_size (tuple): The dimensions of the SOM grid
        latent_dim (int): The dimensionality of the input feature space
        n_prototypes (int): The total number of prototypes (grid size) in the SOM
        prototypes (Tensor): Learnable parameters representing the prototypes of the SOM

    Methods:
        forward(x): Calculates the distances between input features and prototypes and identifies the BMU
    """
    def __init__(self, config):
        """
        Initializes the SOMLayer with the specified grid size and latent dimension

        Args:
            map_size (tuple): Dimensions of the SOM grid
            latent_dim (int): Dimensionality of the input feature space
        """
        super(SOMLayer, self).__init__()

        self.config = config

        self.use_reduced = self.config.hparams.som.use_reduced if self.config.hparams.model_arch == 'vit_som' else False
        ld = self.config.hparams.vit.emb_dim if self.config.hparams.model_arch == 'vit_som' else self.config.hparams.ae.encoder_dims[-1]
        self.latent_dim = ld if self.use_reduced or self.config.hparams.model_arch in ['desom', 'som'] else ld * ((self.config.data.input_size // self.config.hparams.vit.patch_size) ** 2 + 1)

        self.current_temperature = self.config.hparams.som.Tmax

        if self.config.hparams.som.distance_fcn == 'cosine':
            # normalized for cosine similarity
            self.prototypes = nn.Parameter(
                torch.nn.functional.normalize(
                    torch.rand(int(np.prod(self.config.hparams.som.map_size)), self.latent_dim),
                    p=2, dim=1
                )
            )
        else:
            # standard initialization for Euclidean/Manhattan
            self.prototypes = nn.Parameter(
                torch.rand(int(np.prod(self.config.hparams.som.map_size)), self.latent_dim)
            )

    def forward(self, x):
        """
        Forward pass of the SOM layer

        Args:
            x (Tensor): The input tensor

        Returns:
            Tuple[Tensor, Tensor]: Pairwise distances to prototypes and indices of BMUs
        """
        distances = self.compute_distances(x.reshape(x.size(0), -1))
        bmu_indices = torch.argmin(distances, dim=1)
        return distances, bmu_indices
    
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_flattened = x.view(x.size(0), -1)
        distances, bmu_indices = self.forward(x_flattened)
        
        self.update_temperature(self.global_step)

        weights = self.compute_weights(bmu_indices)
        loss = self.som_loss(weights, distances)
        self.log('train_loss', loss)
        
        return loss
    
    def configure_optimizers(self):
        '''
        Configures the optimizer for the DESOM model

        Returns:
            torch.optim.Optimizer: The configured optimizer
        '''
        optimizer = torch.optim.AdamW(self.parameters(), 
                                    lr=self.config.hparams.optimizer.lr, 
                                    betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))
        
        return optimizer

    def compute_distances(self, x):
        if self.config.hparams.som.distance_fcn == 'manhattan':
            distances = torch.cdist(x, self.prototypes, p=1)
        elif self.config.hparams.som.distance_fcn == 'euclidean':
            distances = torch.cdist(x, self.prototypes, p=2)
        elif self.config.hparams.som.distance_fcn == 'cosine':
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            prototypes_norm = torch.nn.functional.normalize(self.prototypes, p=2, dim=1)
            distances = 1 - torch.mm(x_norm, prototypes_norm.T)
        else:
            raise ValueError(f"Unsupported distance function: {self.config.hparams.som.distance_fcn}")
        return distances

    def update_temperature(self, iteration):
        total_iterations = (len(self.trainer.train_dataloader.dataset) / self.config.hparams.batch_size) * self.config.hparams.total_epochs
        self.current_temperature = self.config.hparams.som.Tmax * (self.config.hparams.som.Tmin / self.config.hparams.som.Tmax) ** (iteration / (total_iterations - 1))

    def index_to_position(self, indices):
        return torch.stack((indices // self.config.hparams.som.map_size[1], indices % self.config.hparams.som.map_size[1]), dim=1).float()

    def som_loss(self, weights, distances):
        """
        Calculates SOM loss as the mean of weighted distances

        Args:
            weights (Tensor): Tensor of weights
            distances (Tensor): Tensor of distances

        Returns:
            Tensor: Mean of weighted distances
        """
        weighted_distances = weights * distances
        return torch.mean(weighted_distances)
    
    def on_train_epoch_end(self):
        if ((self.current_epoch + 1) % 10 == 0):
            purity, nmi = evaluate_clustering(self, self.config, self.trainer.train_dataloader)

            self.log_dict({
                'clustering/purity': purity,
                'clustering/nmi': nmi
            }, on_epoch=True)

    def compute_weights(self, bmu_indices):
        """
        Computes the weights for SOM loss based on distances to BMU positions

        Args:
            grid_width (int): Width of the SOM grid
            bmu_indices (Tensor): Tensor of Best Matching Unit (BMU) indices
            temperature (float): Temperature parameter for weight computation
            device (str): The device to perform calculations on. Defaults to 'cuda'

        Returns:
            Tensor: Computed weights based on distances to BMU positions
        """
        grid_positions = torch.stack(torch.meshgrid(torch.arange(self.config.hparams.som.map_size[0], device=self.device),
                                                    torch.arange(self.config.hparams.som.map_size[1], device=self.device)), -1).view(-1, 2).float()

        bmu_grid_positions = grid_positions[bmu_indices]
        distances_to_bmu = torch.norm(grid_positions.unsqueeze(0) - bmu_grid_positions.unsqueeze(1), dim=2)
        weights = torch.exp(-distances_to_bmu ** 2 / (2 * self.current_temperature ** 2))

        return weights
    