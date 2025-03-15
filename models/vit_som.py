"""
Implementation of ViT-SOM model in PyTorch Lightning

@author Alan Luo
@version 1.0
"""
import torch
import torch.nn as nn
import torchvision
import pytorch_lightning as pl
import math

from models.vit import ViT_Classifier
from models.som_layer import SOMLayer
from tools.evaluation import evaluate_clustering, evaluate_classification
from tools.utils import param_groups_lrd

class ViTSOM(pl.LightningModule):
    def __init__(self, config):
        super(ViTSOM, self).__init__()
        torch.set_float32_matmul_precision('medium')

        self.config = config

        self.num_patches = ((self.config.data.input_size // self.config.hparams.vit.patch_size) ** 2) + 1
        self.total_iterations = self.config.hparams.batch_size * self.config.hparams.total_epochs

        self.vit_classifier = ViT_Classifier(config=config)

        self.som_layer = SOMLayer(config)

        self.recon_loss_fn = nn.L1Loss()
        self.cls_loss_fn = nn.CrossEntropyLoss(label_smoothing=self.config.hparams.optimizer.smoothing)

        self.classification = self.config.data.num_classes > 0
        print(f'Eval on CLS task: {self.classification}')

        self.save_hyperparameters(config.to_dict())
        self.register_buffer('iteration', torch.tensor(0))

    def forward(self, x):
        token, patches = self.vit_classifier.encoder(x)
        cls_logits = self.vit_classifier(x) if self.classification else None

        som_input = token if self.config.hparams.som.use_reduced else patches # [B, N, D]

        pred_img = self.vit_classifier.decoder(patches) if not self.classification else None
        distances, bmu_indices = self.som_layer(som_input)

        return cls_logits, patches, pred_img, distances, bmu_indices

    def training_step(self, batch, batch_idx):
        x, _ = batch
        cls_logits, x_encoded, pred_img, distances, bmu_indices = self.forward(x)

        self.update()

        total_loss = self.compute_and_log_losses(
            batch=batch,
            pred_img=pred_img,
            distances=distances,
            bmu_indices=bmu_indices,
            class_logits=cls_logits,
            prefix='train'
        )

        if not self.classification:
            self.log_images(x=x, pred_img=pred_img, bmu_indices=bmu_indices, distances=distances, prefix='train')

        return total_loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        cls_logits, _, pred_img, distances, bmu_indices = self.forward(x)

        total_loss = self.compute_and_log_losses(
            batch=batch,
            pred_img=pred_img,
            distances=distances,
            bmu_indices=bmu_indices,
            class_logits=cls_logits,
            prefix='val'
        )

        return total_loss

    def configure_optimizers(self):
        if self.config.hparams.optimizer.layer_decay != 1 and self.config.hparams.optimizer.weight_decay != 0:
            param_groups = self._create_param_groups()
        else:
            param_groups = list(self.parameters())

        if self.config.hparams.optimizer.type == 'adam':
            optimizer = torch.optim.Adam(param_groups, lr=self.config.hparams.optimizer.lr, betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))
        elif self.config.hparams.optimizer.type == 'adamw':
            optimizer = torch.optim.AdamW(param_groups, lr=self.config.hparams.optimizer.lr * self.config.hparams.batch_size / 256, betas=(self.config.hparams.optimizer.beta_1, self.config.hparams.optimizer.beta_2))

        if self.config.hparams.optimizer.scheduler == 'cosine_annealing':
            lr_func = lambda epoch: min((epoch + 1) / (self.config.hparams.optimizer.warmup_epochs + 1e-8), 0.5 * (math.cos(epoch / self.config.hparams.total_epochs * math.pi) + 1))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
            return [optimizer], [scheduler]

        return [optimizer], []
    
    def _create_param_groups(self):
        transformer_params = param_groups_lrd(
            self.vit_classifier.encoder.transformer,
            weight_decay=self.config.hparams.optimizer.weight_decay,
            layer_decay=self.config.hparams.optimizer.layer_decay
        )

        transformer_param_ids = set()
        for group in transformer_params:
            for p in group["params"]:
                transformer_param_ids.add(id(p))

        others = []
        som_no_decay = []
        for name, param in self.named_parameters():
            if id(param) in transformer_param_ids:
                continue

            if 'som_layer.prototypes' in name:
                som_no_decay.append(param)
            else:
                others.append(param)

        transformer_params += [
            {'params': others, 'weight_decay': self.config.hparams.optimizer.weight_decay, 'lr_scale': 1.0},
            {'params': som_no_decay, 'weight_decay': 0.0, 'lr_scale': 10.0},
        ]
        return transformer_params

    def on_train_epoch_end(self):
        if ((self.current_epoch + 1) % 25 == 0):
            if self.classification:
                accuracy, precision, recall, f1 = evaluate_classification(self, self.config, self.trainer.val_dataloaders)

                self.log_dict({
                    'classification/accuracy': accuracy,
                    'classification/precision': precision,
                    'classification/recall': recall,
                    'classification/f1': f1
                }, on_epoch=True)
            else:
                purity, nmi = evaluate_clustering(self, self.config, self.trainer.train_dataloader)

                visualize_decoded_prototypes(self, self.config, output_dir='experiments/plots/vit_som/decoded_prototypes')

                self.log_dict({
                    'clustering/purity': purity,
                    'clustering/nmi': nmi
                }, on_epoch=True)

            for name, params in self.named_parameters():
                if params.numel() > 0:
                    self.logger.experiment.add_histogram(f'parameters/{name}', params, self.current_epoch)

        self.log('hparams/temperature', self.som_layer.current_temperature)

    def update(self):
        self.som_layer.update_temperature(self.iteration)
        self.iteration += 1

    def compute_and_log_losses(self, batch, pred_img, distances, bmu_indices, class_logits, prefix):
        x, y = batch

        weights = self.som_layer.compute_weights(bmu_indices)
        som_loss = self.som_layer.som_loss(weights, distances)

        if self.classification:
            cls_loss = self.cls_loss_fn(class_logits, y)
            total_loss = cls_loss + self.config.hparams.gamma * som_loss
            self.log(f'{prefix}/cls_loss', cls_loss, on_step=True)
        else:
            recon_loss = self.recon_loss_fn(pred_img, x)
            total_loss = recon_loss + self.config.hparams.gamma * som_loss
            self.log(f'{prefix}/recon_loss', recon_loss, on_step=True, on_epoch=False)

        self.log_dict({
            f'{prefix}/som_loss': som_loss
        }, on_step=True, on_epoch=False)
        self.log(f'{prefix}/total_loss', total_loss, on_step=True, prog_bar=True)

        return total_loss

    def log_images(self, x, pred_img, bmu_indices, distances, prefix):
        grid_x = torchvision.utils.make_grid(x[:10])
        self.logger.experiment.add_image(f'{prefix}/x', grid_x, 0)

        grid_pred_img = torchvision.utils.make_grid(pred_img[:10])
        self.logger.experiment.add_image(f'{prefix}/pred_img', grid_pred_img, 0)

        grid_bmu_indices = torchvision.utils.make_grid(torch.unsqueeze(bmu_indices[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_bmu_indices', grid_bmu_indices, 0)

        grid_distances = torchvision.utils.make_grid(torch.unsqueeze(distances[:10], 0))
        self.logger.experiment.add_image(f'{prefix}/grid_distances', grid_distances, 0)
    
    def get_latent_representation(self, x):
        """Returns flattened patches WITH CLS token when needed"""
        with torch.no_grad():
            token, patches = self.vit_classifier.encoder(x)
            
            if self.config.hparams.som.use_reduced:
                return token  # [B, 1, D]
            else:
                if not self.config.hparams.vit.global_pool:
                    return patches.flatten(start_dim=1)  # [B, (num_patches+1)*D]
                else:
                    # No CLS token, use all patches
                    return patches.flatten(start_dim=1)  # [B, num_patches*D]