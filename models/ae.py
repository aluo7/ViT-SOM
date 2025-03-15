"""
Implementation of DESOM Autoencoder in PyTorch Lightning

@author Alan Luo
@version 1.0
"""

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.init import xavier_uniform_
from torch.nn import BatchNorm1d

class Autoencoder(pl.LightningModule):
    """
    Fully-connected symmetric autoencoder model.

    Attributes:
        encoder (Sequential): Encoder
        decoder (Sequential): Decoder
        
    """
    def __init__(self, config):
        """
        Initializes the Autoencoder model.

        Args:
            encoder_dims (list): Dimensions for the encoder and decoder layers
            act (callable, optional): Activation function for hidden layers. Defaults to nn.ReLU()
            output_act (callable, optional): Activation function for the output layer. Defaults to nn.Sigmoid()
        """
        super(Autoencoder, self).__init__()

        self.config = config

        self.act = nn.ReLU() if self.config.hparams.ae.act == 'relu' else nn.Identity()
        self.output_act = nn.Identity()

        input_dim = torch.prod(torch.tensor([self.config.data.num_channels, self.config.data.input_size, self.config.data.input_size])).item()
        self.config.hparams.ae.encoder_dims = [input_dim] + self.config.hparams.ae.encoder_dims

        self.encoder = self.build_layers(self.config.hparams.ae.encoder_dims, self.act, initialize=True, batchnorm=self.config.hparams.ae.act, is_encoder=True)
        decoder_dims = list(reversed(self.config.hparams.ae.encoder_dims))
        self.decoder = self.build_layers(decoder_dims, self.act, initialize=True, batchnorm=self.config.hparams.ae.act, is_encoder=False)

        self.classification = config.data.num_classes > 0

        self.recon_loss_fn = torch.nn.L1Loss()
        self.cls_loss_fn = torch.nn.CrossEntropyLoss()

    def build_layers(self, dims, act, initialize=True, batchnorm=False, is_encoder=True):
        """
        Constructs Autoencoder symmetric layers.

        Args:
            dims (list): A list of integers specifying the dimensions of each layer
            act (callable): Activation function for all layers except the output layer
            output_act (callable, optional): Activation function for the output layer. Defaults to None
            initialize (bool): Whether to apply Xavier (Glorot) uniform initialization
            batchnorm (bool): Indicates whether to include batch normalization layers

        Returns:
            nn.Sequential: A sequence of layers comprising the neural network
        """
        layers = []
        num_layers = len(dims) - 1
        for i in range(num_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            if initialize:
                xavier_uniform_(layer.weight)
            layers.append(layer)

            if batchnorm and i < num_layers - 1:
                layers.append(BatchNorm1d(dims[i + 1]))

            if i < num_layers - 1:
                layers.append(act)
            elif not is_encoder:
                layers.append(self.output_act)

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass of the autoencoder

        Args:
            x (Tensor): The input tensor

        Returns:
            Tensor: The reconstructed input tensor
        """
        x = x.view(x.size(0), -1)
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step

        Args:
            batch (tuple): The batch of input data and labels
            batch_idx (int): The index of the batch

        Returns:
            torch.Tensor: The computed loss for the batch
        """
        x, _ = batch
        x_decoded = self.forward(x)

        if self.classification:
            loss = self.cls_loss_fn(x_decoded, x)
            self.log('cls_loss', loss)
        else:
            loss = self.recon_loss_fn(x_decoded, x)
            self.log('recon_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_decoded = self.forward(x)
        
        if self.classification:
            loss = self.cls_loss_fn(x_decoded, x)
            self.log('cls_loss', loss)
        else:
            loss = self.recon_loss_fn(x_decoded, x)
            self.log('recon_loss', loss)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                    lr=self.config.hparams.optimizer.lr, 
                                    betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))
        return optimizer
