# deit.py

import torch
import pytorch_lightning as pl
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights

from vit_pytorch.distill import DistillableViT, DistillWrapper

class DeiT(pl.LightningModule):
    """
    DeiT using knowledge distillation from a teacher model.
    """
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        hp = config['hyperparameters']
        vit_hp = hp['vit']
        opt_hp = hp['optimizer']
        data_hp = config['data']
        distill_hp = hp.get('distillation', {})

        print("Initializing pre-trained ResNet-50 teacher...")
        teacher = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        
        num_ftrs = teacher.fc.in_features
        teacher.fc = nn.Linear(num_ftrs, data_hp['num_classes'])
        
        for param in teacher.parameters():
            param.requires_grad = False

        student = DistillableViT(
            image_size=data_hp['input_size'],
            patch_size=vit_hp['patch_size'],
            num_classes=data_hp['num_classes'],
            dim=vit_hp['emb_dim'],
            depth=vit_hp['depth'],
            heads=vit_hp['heads'],
            mlp_dim=vit_hp['emb_dim'] * int(vit_hp['mlp_ratio']),
            dropout=vit_hp.get('proj_drop', 0.1),
            emb_dropout=vit_hp.get('attn_drop', 0.1)
        )

        self.distiller = DistillWrapper(
            student=student,
            teacher=teacher,
            temperature=distill_hp.get('temperature', 3.0),
            alpha=distill_hp.get('alpha', 0.5),
            hard=distill_hp.get('hard', False)
        )
        
        self.val_loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.distiller.student(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.distiller(x, y)
        self.log('train/distill_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        
        val_loss = self.val_loss_fn(logits, y)
        acc = (logits.argmax(dim=-1) == y).float().mean()
        
        self.log('val/cls_loss', val_loss, on_epoch=True)
        self.log('val/accuracy', acc, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        '''
        Configures the optimizer (with distillation) and scheduler, applying cosine decay.
        '''
        hp = self.config['hyperparameters']
        opt_hp = hp['optimizer']

        optimizer = torch.optim.AdamW(
            self.distiller.student.parameters(),
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
        '''
        End of training hook to capture memory usage
        '''
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_gb = peak_memory_bytes / 1e9  # convert bytes to gigabytes
        
        print(f"Peak GPU memory usage: {peak_memory_gb:.4f} GB")