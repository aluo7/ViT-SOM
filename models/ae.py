# ae.py

import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn.init import xavier_uniform_
from torch.nn import BatchNorm1d

class Autoencoder(pl.LightningModule):
    '''
    Fully-connected symmetric autoencoder model.
    '''
    def __init__(self, config):
        '''
        Initializes the Autoencoder
        '''
        super(Autoencoder, self).__init__()

        self.lr = config['hyperparameters']['optimizer']['lr']
        self.beta_1 = config['hyperparameters']['optimizer']['beta_1']
        self.beta_2 = config['hyperparameters']['optimizer']['beta_2']

        self.encoder_dims = config['hyperparameters']['ae']['encoder_dims']
        self.act = nn.ReLU() if config['hyperparameters']['ae']['act'] == 'relu' else nn.Identity()
        self.output_act = nn.Identity()
        self.batch_norm = config['hyperparameters']['ae']['batch_norm']

        input_dim = torch.prod(torch.tensor([config['data']['num_channels'], config['data']['input_size'], config['data']['input_size']])).item()
        self.encoder_dims = [input_dim] + self.encoder_dims

        self.encoder = self.build_layers(self.encoder_dims, self.act, initialize=True, batchnorm=self.batch_norm, is_encoder=True)
        decoder_dims = list(reversed(self.encoder_dims))
        self.decoder = self.build_layers(decoder_dims, self.act, initialize=True, batchnorm=self.batch_norm, is_encoder=False)

        self.classification = config['data']['num_classes'] > 0

        self.recon_loss_fn = torch.nn.L1Loss()
        self.cls_loss_fn = torch.nn.CrossEntropyLoss()

    def build_layers(self, dims, act, initialize=True, batchnorm=False, is_encoder=True):
        '''
        Constructs Autoencoder symmetric layers
        '''
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
        x = x.view(x.size(0), -1)
        x_encoded = self.encoder(x)
        x_decoded = self.decoder(x_encoded)
        return x_decoded

    def training_step(self, batch, batch_idx):
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
                                    lr=self.lr, 
                                    betas=(self.beta_1, self.beta_2))
        return optimizer
