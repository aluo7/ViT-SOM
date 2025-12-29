# mobile_vit.py

import torch
import pytorch_lightning as pl
from torch import nn
import timm

class MobileViT(pl.LightningModule):
    """
    MobileViT module.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        hp = config['hyperparameters']
        opt_hp = hp['optimizer']
        data_hp = config['data']
        model_name = 'mobilevit_s'

        print(f"Creating timm model: {model_name}")
        self.model = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=data_hp['num_classes'],
            img_size=data_hp['input_size']
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
        hp = self.config['hyperparameters']
        opt_hp = hp['optimizer']

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=opt_hp['lr'],
            betas=(opt_hp.get('beta_1', 0.9), opt_hp.get('beta_2', 0.999)),
            weight_decay=opt_hp.get('weight_decay', 0.05)
        )

        if opt_hp.get('scheduler') == 'cosine_annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=hp['total_epochs']
            )
            return [optimizer], [scheduler]
        
        return optimizer

    def on_train_end(self):
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / 1e9  # convert bytes to gigabytes
        
        print(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")