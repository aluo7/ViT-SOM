# vit.py

# Adapted from https://github.com/facebookresearch/mae/blob/main/models_mae.py.

import torch
import pytorch_lightning as pl
from torch import nn

import math
from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from tools.utils import param_groups_lrd, get_2d_sincos_pos_embed
from tools.evaluation import evaluate_classification

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, return_attn=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn_out = self.attn_drop(attn)

        x = (attn_out @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attn:
            return x, attn  # [B, heads, N, N]
        return x, None

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop)
        )

    def forward(self, x, return_attn=False):
        x_attn, attn = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x, attn

class ViTAutoencoder(nn.Module):
    """
    Vision Transformer Autoencoder (uses MAE arch for feature extraction)
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, depth=12, num_heads=12,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        
        # encoder
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # decoder
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for _ in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        # init and freeze pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # init patch_embed like nn.Linear
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # init cls token
        torch.nn.init.normal_(self.cls_token, std=.02)

        # init other layers
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        Converts an image to a sequence of patches.
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        c = imgs.shape[1]
        x = imgs.reshape(shape=(imgs.shape[0], c, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * c))
        return x

    def unpatchify(self, x):
        """
        Converts a sequence of patches back to an image.
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]

        c = x.shape[2] // (p * p)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward_features(self, x, return_attns=False):
        """
        Forward pass through encoder for classification or feature extraction.
        """
        # embed patches
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        # prepend class token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attns = [] if return_attns else None

        # transformer blocks
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attns)
            if return_attns and attn is not None:
                attns.append(attn)
        x = self.norm(x)
        
        if return_attns:
            return x[:, 0], attns
        return x[:, 0], None
    
    
    def forward_decoder(self, x, return_attn=False):
        """
        Forward pass through encoder for self-supervised signal.
        """
        decoded = self.decoder_embed(x)
        decoded = decoded + self.decoder_pos_embed
        attns = [] if return_attn else None
        for blk in self.decoder_blocks:
            if return_attn:
                decoded, attn = blk(decoded, return_attn=True)
                if attn is not None:
                    attns.append(attn)
            else:
                decoded = blk(decoded, return_attn=False)
        decoded = self.decoder_norm(decoded)
        decoded_patches = self.decoder_pred(decoded)[:, 1:, :]
        if return_attn:
            return decoded_patches, attns
        return decoded_patches, None

    def forward(self, x, return_attns=False):
        """
        Forward pass as a standard, non-masked autoencoder.
        """
        # encoder forward pass
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attns = [] if return_attns else None
        for blk in self.blocks:
            x, attn = blk(x, return_attn=return_attns)  # Ensure blk returns (x, attn)
            if return_attns and attn is not None:
                attns.append(attn)
        x = self.norm(x)
        
        cls_token_out = x[:, 0]
        patch_tokens_out = x[:, 1:]

        # project to decoder dimension and add position embeddings
        decoded = self.decoder_embed(x)
        decoded = decoded + self.decoder_pos_embed

        # apply decoder transformer blocks
        for blk in self.decoder_blocks:
            decoded, _ = blk(decoded, return_attn=False)  # unpack only the tensor, ignore attn
        decoded = self.decoder_norm(decoded)

        # predict pixels and remove the CLS token prediction
        decoded_patches = self.decoder_pred(decoded)[:, 1:, :]
        
        recon_img = self.unpatchify(decoded_patches)
        
        if return_attns:
            return cls_token_out, patch_tokens_out, recon_img, attns
        return cls_token_out, patch_tokens_out, recon_img


class ViTClassifier(pl.LightningModule):
    """
    Classification module for ViT.
    """
    def __init__(self, config):
        super().__init__()
        torch.set_float32_matmul_precision('medium')

        self.save_hyperparameters(config)
        self.config = config

        hp = config['hyperparameters']
        data_hp = config['data']
        vit_hp = hp['vit']
        opt_hp = hp['optimizer']

        # ViTAutoencoder
        self.model = ViTAutoencoder(
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

        # initialize cls head
        self.cls_head = nn.Linear(vit_hp['emb_dim'], data_hp['num_classes'])
        torch.nn.init.normal_(self.cls_head.weight, std=0.02)

        self.cls_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        cls_token, _ = self.model.forward_features(x)
        logits = self.cls_head(cls_token)
        return logits

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        cls_loss = self.cls_loss(logits, y)
        self.log('train/cls_loss', cls_loss, on_step=True, on_epoch=False, prog_bar=True)
        return cls_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        cls_loss = self.cls_loss(logits, y)
        self.log('val/cls_loss', cls_loss, on_step=False, on_epoch=True, prog_bar=True)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val/accuracy', acc, on_step=False, on_epoch=True, prog_bar=True)
        return cls_loss

    def configure_optimizers(self):
        '''
        Configures the optimizer and scheduler, applying layer-wise rate decay and cosine scheduler.
        '''
        hp = self.config['hyperparameters']
        opt_hp = hp['optimizer']

        param_groups = param_groups_lrd(
            self.model,
            weight_decay=opt_hp['weight_decay'],
            layer_decay=opt_hp['layer_decay']
        )

        param_groups.append({"params": self.cls_head.parameters()})  # add the cls head params as a separate group

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