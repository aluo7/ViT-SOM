# vit_som.py

import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import math

from functools import partial
from tkinter import Y

from models.vit import ViTAutoencoder
from models.som_layer import SOMLayer
from tools.evaluation import evaluate_clustering, evaluate_classification
from tools.utils import param_groups_lrd

class ViTSOM(pl.LightningModule):
    '''
    Vision Transformer Self-Organizing Map (ViT-SOM) model integration with ViT and SOM layer.
    '''
    def __init__(self, config):
        super().__init__()
        torch.set_float32_matmul_precision('medium')

        self.config = config
        self.save_hyperparameters(config)

        hp = config['hyperparameters']
        vit_hp = hp['vit']
        opt_hp = hp['optimizer']
        data_hp = config['data']
        som_hp = hp['som']

        self.gamma = hp['gamma']
        self.use_reduced = som_hp['use_reduced']
        self.classification = data_hp['num_classes'] > 0
        print(f'Eval on CLS task: {self.classification}')

        self.vit = ViTAutoencoder(  # init vit
            img_size=data_hp['input_size'],
            patch_size=vit_hp['patch_size'],
            in_chans=data_hp['num_channels'],
            embed_dim=vit_hp['emb_dim'],
            depth=vit_hp['depth'],
            num_heads=vit_hp['heads'],
            decoder_embed_dim=vit_hp['dec_emb_dim'],
            decoder_depth=vit_hp['dec_depth'],
            decoder_num_heads=vit_hp['heads'],
            mlp_ratio=vit_hp['mlp_ratio'],
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            norm_pix_loss=False
        )

        self.som_layer = SOMLayer(config)  # init SOM layer

        # if cls, construct cls head
        if self.classification:
            self.cls_head = nn.Linear(vit_hp['emb_dim'], data_hp['num_classes'])
            torch.nn.init.normal_(self.cls_head.weight, std=0.02)

        self.recon_loss_fn = nn.L1Loss()
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=opt_hp['smoothing'])

        self.save_hyperparameters()
        self.register_buffer('iteration', torch.tensor(0))

    def forward(self, x):
        cls_token, patches, recon_img = self.vit(x)  # vit forward pass
        
        if self.use_reduced:
            som_input = cls_token
        else:
            som_input = patches.flatten(start_dim=1)
        
        distances, bmu_indices = self.som_layer(som_input)  # som forward pass

        logits = self.cls_head(cls_token) if self.classification else None
        return cls_token, recon_img, logits, distances, bmu_indices

    def training_step(self, batch, batch_idx):
        x, y = batch
        cls_token, recon_img, logits, distances, bmu_indices = self.forward(x)

        self.som_layer.update_temperature(self.iteration)
        weights = self.som_layer.compute_weights(bmu_indices)
        som_loss = self.som_layer.som_loss(weights, distances)

        # ramp up for gamma (prioritize feature extractor learning for early stages)
        ramp_up_end_step = self.trainer.estimated_stepping_batches // 2 
        current_gamma = self.config['hyperparameters']['gamma'] * min(1.0, self.iteration.item() / ramp_up_end_step)
        self.log('hp/gamma', current_gamma, on_step=True, on_epoch=False)

        # compute and log losses
        if self.classification:
            y = y.view(-1)
            cls_loss = self.cls_loss_fn(logits, y)
            total_loss = cls_loss + current_gamma * som_loss
            self.log_dict({'train/cls_loss': cls_loss, 'train/som_loss': som_loss, 'train/total_loss': total_loss}, on_step=True, on_epoch=False)
        else:
            recon_loss = self.recon_loss_fn(recon_img, x)
            total_loss = recon_loss + current_gamma * som_loss
            self.log_dict({'train/recon_loss': recon_loss, 'train/som_loss': som_loss, 'train/total_loss': total_loss}, on_step=True, on_epoch=False)

        self.iteration += 1
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        cls_token, recon_img, logits, distances, bmu_indices = self.forward(x)

        weights = self.som_layer.compute_weights(bmu_indices)
        som_loss = self.som_layer.som_loss(weights, distances)

        if self.classification:
            y = y.view(-1)
            cls_loss = self.cls_loss_fn(logits, y)
            total_loss = cls_loss + self.gamma * som_loss
            acc = (logits.argmax(dim=-1) == y).float().mean()
            self.log_dict({'val/cls_loss': cls_loss, 'val/som_loss': som_loss, 'val/total_loss': total_loss, 'val/accuracy': acc}, on_step=True, on_epoch=True)
        else:
            recon_loss = self.recon_loss_fn(recon_img, x)
            total_loss = recon_loss + self.gamma * som_loss
            self.log_dict({'val/recon_loss': recon_loss, 'val/som_loss': som_loss, 'val/total_loss': total_loss}, on_step=True, on_epoch=True)

        return total_loss

    def configure_optimizers(self):
        '''
        Configures the optimizer and scheduler, applying layer-wise rate decay and cosine scheduler.
        '''
        hp = self.config['hyperparameters']
        opt_hp = hp['optimizer']

        param_groups = param_groups_lrd(
            self.vit,
            weight_decay=opt_hp['weight_decay'],
            layer_decay=opt_hp['layer_decay']
        )

        other_params = list(self.som_layer.parameters())
        if self.classification:
            other_params.extend(list(self.cls_head.parameters()))
        
        param_groups.append({"params": other_params})

        if opt_hp['type'] == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups, 
                lr=opt_hp['lr'] * hp['batch_size'] / 256, 
                betas=(opt_hp['beta_1'], opt_hp['beta_2'])
            )
        elif opt_hp['type'] == 'adam':
            optimizer = torch.optim.Adam(
                param_groups, 
                lr=opt_hp['lr'] * hp['batch_size'] / 256, 
                betas=(opt_hp['beta_1'], opt_hp['beta_2'])
            )

        if opt_hp['scheduler'] == 'cosine_annealing':
            lr_func = lambda epoch: max(opt_hp['min_lr'], min((epoch + 1) / (opt_hp['warmup_epochs'] + 1e-8), 0.5 * (math.cos(epoch / hp['total_epochs'] * math.pi) + 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        return [optimizer], [scheduler]

    def on_train_end(self):
        '''
        End of training hook to capture memory usage
        '''
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / 1e9  # convert bytes to gigabytes
        
        print(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")
    
    def get_latent_representation(self, x):
        """
        Returns flattened patches WITH CLS token when needed (for UMAP vis)
        """
        with torch.no_grad():
            cls_token, patches, _, _ = self.vit(x, return_attns=False)
            
            if self.use_reduced:
                return cls_token  # [B, 1, D]
            else:
                if not self.config['hyperparameters']['vit']['global_pool']:
                    return patches.flatten(start_dim=1)  # [B, (num_patches+1)*D]
                else:
                    return patches.flatten(start_dim=1)  # [B, num_patches*D]