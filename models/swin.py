# swin.py

import torch
import pytorch_lightning as pl
from torch import nn
import timm
import math

class Swin(pl.LightningModule):
    """
    Swin Transformer module.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        hp = config['hyperparameters']
        swin_hp = hp['swin']
        opt_hp = hp['optimizer']
        data_hp = config['data']

        self.model = timm.create_model(
            'swin_tiny_patch4_window7_224',
            pretrained=False,
            num_classes=data_hp['num_classes'],
            img_size=data_hp['input_size'],
            patch_size=swin_hp['patch_size'],
            window_size=swin_hp['window_size'],
            embed_dim=swin_hp['embed_dim'],
            depths=swin_hp['depths'],
            num_heads=swin_hp['num_heads'],
        )

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=opt_hp.get('smoothing', 0.0))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train/cls_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        val_loss = self.loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        self.log('val/cls_loss', val_loss, on_epoch=True)
        self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        '''
        Configures the optimizer and scheduler, applying cosine decay with linear warmup.
        '''
        hp = self.config['hyperparameters']
        opt_hp = hp['optimizer']

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_hp['lr'],
            betas=(opt_hp.get('beta_1', 0.9), opt_hp.get('beta_2', 0.999)),
            weight_decay=opt_hp.get('weight_decay', 0.05)
        )

        if opt_hp.get('scheduler') == 'cosine_annealing':
            total_epochs = hp['total_epochs']
            warmup_epochs = opt_hp.get('warmup_epochs', 10)
            min_lr = opt_hp.get('min_lr', 1e-6)

            def lr_lambda(epoch):
                if epoch < warmup_epochs:
                    return float(epoch) / float(max(1, warmup_epochs))
                
                progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
                
                cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
                
                min_lr_ratio = min_lr / opt_hp['lr']
                scaled_lr = (1.0 - min_lr_ratio) * cosine_decay + min_lr_ratio
                return scaled_lr
                
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
        
        return optimizer

    def on_train_end(self):
        '''
        End of training hook to capture memory usage
        '''
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / 1e9  # convert bytes to gigabytes
        
        print(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")