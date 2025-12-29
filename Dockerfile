FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

RUN pip install pytorch-lightning lightning[extra] vim vit-pytorch torchvision scikit-learn tensorboard h5py einops timm seaborn SciencePlots umap-learn --ignore-installed
RUN pip install --upgrade psutil

RUN apt-get update && apt-get install -y \
    build-essential \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    cm-super \
    --no-install-recommends

WORKDIR /app
COPY . /app
