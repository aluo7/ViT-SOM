import torch
import pytorch_lightning as pl
import math
import numpy as np
import timm

from torch import nn
from functools import partial

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.layers import trunc_normal_
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models.vision_transformer import VisionTransformer

from tools.evaluation import evaluate_classification
from tools.utils import param_groups_lrd, initialize_mixup, get_2d_sincos_pos_embed

class MAE_Encoder(torch.nn.Module):
    def __init__(self, transformer : VisionTransformer):
        super().__init__()

        self.transformer = transformer
        
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.transformer.embed_dim))
        pos_embedding = get_2d_sincos_pos_embed(
            embed_dim=self.transformer.embed_dim,
            grid_size=self.transformer.patch_embed.grid_size[0],
            cls_token=not self.transformer.global_pool  # only add CLS pos emb if using CLS token
        )

        pos_embedding_tensor = torch.tensor(pos_embedding, dtype=torch.float32).unsqueeze(0)

        self.pos_embed = torch.nn.Parameter(
            pos_embedding_tensor,
            requires_grad=False  # fixed embeddings
        )

        self.layer_norm = torch.nn.LayerNorm(self.transformer.embed_dim)
        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)

    def forward(self, x):
        token, patches = self.transformer.forward_features(x)
        return token, patches

class MAE_Decoder(torch.nn.Module):
    def __init__(self, transformer, num_channels, input_size, patch_size, emb_dim):
        super().__init__()

        self.transformer = transformer

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        pos_embedding = get_2d_sincos_pos_embed(
            embed_dim=self.transformer.embed_dim,
            grid_size=self.transformer.patch_embed.grid_size[0],
            cls_token=not self.transformer.global_pool  # only add CLS pos emb if using CLS token
        )

        pos_embedding_tensor = torch.tensor(pos_embedding, dtype=torch.float32).unsqueeze(0)

        self.pos_embed = torch.nn.Parameter(
            pos_embedding_tensor,
            requires_grad=False  # fixed embeddings
        )

        self.head = torch.nn.Linear(emb_dim, num_channels * patch_size ** 2)
        self.patch2img = Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=input_size // patch_size, w=input_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)

    def forward(self, features):
        features = features[:, 1:, :] if not self.transformer.global_pool else features  # remove the first token
        features += self.pos_embed[:, :features.size(1), :]

        for blk in self.transformer.blocks:
            features = blk(features)
        
        features = self.transformer.norm(features)
        patches = self.head(features)
        img = self.patch2img(patches)
        
        return img

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer with support for global average pooling and cls token/GAP accessible at the 0th index.
    """
    def __init__(self, config, global_pool=False, **kwargs):
        super().__init__(**kwargs)

        self.config = config

        self.config.hparams.vit.global_pool = global_pool

        pos_embed = get_2d_sincos_pos_embed(
            self.embed_dim, 
            self.patch_embed.grid_size[0],
            cls_token = not self.config.hparams.vit.global_pool  # only add CLS pos emb if using CLS token
        )

        self.pos_embed = nn.Parameter(
            torch.tensor(pos_embed).float().unsqueeze(0),
            requires_grad=False
        )

        if not self.config.hparams.vit.global_pool:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        else:
            self.cls_token = None  # disable CLS token
            self.fc_norm = nn.LayerNorm(self.embed_dim)

    def forward_features(self, x):
        x = self.patch_embed(x)  # [B, num_patches, embed_dim]
        B, N, D = x.shape

        if not self.config.hparams.vit.global_pool:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)  # [B, 1+N, D]

        x += self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.config.hparams.vit.global_pool:
            pooled = x[:, self.has_class_token:].mean(dim=1)
            pooled = self.fc_norm(pooled)
            return pooled, x
        else:
            return x[:, 0], x
    
    def forward_head(self, x):
        return self.head(x)

class MAE_ViT(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        torch.set_float32_matmul_precision('medium')

        self.config = config
        self.checkpoint_dir = 'experiments/states/'

        self.config.data.num_channels = self.config.data.num_channels
        self.config.data.input_size = self.config.data.input_size
        self.config.data.num_classes = self.config.data.num_classes

        self.encoder = MAE_Encoder(
            transformer=VisionTransformer(
                config=config, global_pool=self.config.hparams.vit.global_pool, in_chans=self.config.data.num_channels, 
                img_size=self.config.data.input_size, patch_size=self.config.hparams.vit.patch_size, 
                embed_dim=self.config.hparams.vit.emb_dim, depth=self.config.hparams.vit.enc_depth, 
                num_heads=self.config.hparams.vit.heads, mlp_ratio=self.config.hparams.vit.mlp_ratio, 
                qkv_bias=self.config.hparams.vit.qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            )
        )
        
        self.decoder = MAE_Decoder(
            transformer=VisionTransformer(
                config=config, global_pool=self.config.hparams.vit.global_pool, in_chans=self.config.data.num_channels, 
                img_size=self.config.data.input_size, patch_size=self.config.hparams.vit.patch_size,
                embed_dim=self.config.hparams.vit.emb_dim, depth=self.config.hparams.vit.dec_depth, 
                num_heads=self.config.hparams.vit.heads, mlp_ratio=self.config.hparams.vit.mlp_ratio, 
                qkv_bias=self.config.hparams.vit.qkv_bias, norm_layer=partial(nn.LayerNorm, eps=1e-6)
            ),
            num_channels=self.config.data.num_channels,
            input_size=self.config.data.input_size,
            patch_size=self.config.hparams.vit.patch_size,
            emb_dim=self.config.hparams.vit.emb_dim
        )

        self.recon_loss_fn = nn.L1Loss()

        self.save_hyperparameters(config.to_dict())

    def forward(self, x):
        _, patches = self.encoder(x)
        pred_img = self.decoder(patches)
        return pred_img

    def training_step(self, batch, batch_idx):
        x, _ = batch
        pred_img = self(x)

        recon_loss = self.recon_loss_fn(pred_img, x)
        
        self.log('train/recon_loss', recon_loss, on_step=True, on_epoch=False, prog_bar=True)

        return recon_loss

    def configure_optimizers(self):
        param_groups = param_groups_lrd(self.encoder.transformer, weight_decay=self.config.hparams.optimizer.weight_decay, layer_decay=self.config.hparams.optimizer.layer_decay)

        if self.config.hparams.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(param_groups, 
                            lr=self.config.hparams.optimizer.lr, 
                            betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))
        elif self.config.hparams.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, 
                                        lr=self.config.hparams.optimizer.lr * self.config.hparams.batch_size / 256, 
                                        betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))

        if self.config.hparams.optimizer.scheduler == 'cosine_annealing':
            lr_func = lambda epoch: min((epoch + 1) / (self.config.hparams.optimizer.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.config.hparams.total_epochs * math.pi) + 1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        return [optimizer], [scheduler]

class ViT_Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.mae = MAE_ViT(config)
        self.encoder = self.mae.encoder
        self.decoder = self.mae.decoder

        self.cls_head = torch.nn.Linear(self.config.hparams.vit.emb_dim, self.config.data.num_classes)

        self.config.data.augment.mixup_alpha_fn = initialize_mixup(config) if self.config.data.augment.mixup_alpha > 0 else None
        
        if self.config.data.augment.mixup_alpha_fn:
            self.train_loss_fn = SoftTargetCrossEntropy()
        elif self.config.hparams.optimizer.smoothing > 0:
            self.train_loss_fn = LabelSmoothingCrossEntropy(smoothing=self.config.hparams.optimizer.smoothing)
        else:
            self.train_loss_fn = nn.CrossEntropyLoss()

        self.val_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        token, _ = self.encoder(x)
        cls_logits = self.cls_head(token)
        return cls_logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        if self.config.data.augment.mixup_alpha_fn is not None:
            x, y = self.config.data.augment.mixup_alpha_fn(x, y)

        cls_logits = self(x)

        cls_loss = self.train_loss_fn(cls_logits, y)
        self.log('train/cls_loss', cls_loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return cls_loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        cls_logits = self(x)

        cls_loss = self.val_loss_fn(cls_logits, y)
        self.log('val/cls_loss', cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        
        return cls_loss

    def configure_optimizers(self):
        param_groups = param_groups_lrd(self.encoder.transformer, weight_decay=self.config.hparams.optimizer.weight_decay, layer_decay=self.config.hparams.optimizer.layer_decay)

        if self.config.hparams.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(param_groups, 
                            lr=self.config.hparams.optimizer.lr, 
                            betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))
        elif self.config.hparams.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, 
                                        lr=self.config.hparams.optimizer.lr * self.config.hparams.batch_size / 256, 
                                        betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))

        if self.config.hparams.optimizer.scheduler == 'cosine_annealing':
            lr_func = lambda epoch: max(self.config.hparams.optimizer.min_lr, min((epoch + 1) / (self.config.hparams.optimizer.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.config.hparams.total_epochs * math.pi) + 1)))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)

        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self):
        if (self.current_epoch + 1) % 50 == 0 and self.current_epoch != (self.config.hparams.total_epochs - 1):
            accuracy, precision, recall, f1 = evaluate_classification(self, self.config, self.trainer.train_dataloader)

            self.log_dict({
                'classification/accuracy': accuracy,
                'classification/precision': precision,
                'classification/recall': recall,
                'classification/f1': f1
            }, on_epoch=True)

            for name, params in self.named_parameters():
                self.logger.experiment.add_histogram(f'parameters/{name}', params, self.current_epoch)
