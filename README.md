## DESOM Pytorch Lightning Reimplementation

### **Environment**

Python → `3.8.5`

Pytorch → `'2.2.1+cu121'`

Pytorch Lightning → `2.2.1`

### **How to run**

1. build container → `make build`
2. execute container and run experiment → `make run accelerator=cuda devices=1`
3. run corresponding experiments `make train model=vit_som dataset=cifar-10`
4. tensorboard → `tensorboard --logdir=logs --bind_all`

### Repo Structure

`/configs` → yaml configuration files detailing hyperparams, data configs, etc

`/data` → builds dataloaders and applies augmentation

`/models` → holds model files (`desom.py`, `vit_som.py`, etc.)

`/tools` → contains utility scripts for evaluation metric gathering and other tools

## Experiments