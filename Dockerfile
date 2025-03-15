FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# Upgrade pip and setuptools
RUN pip install --upgrade pip
RUN pip install --upgrade setuptools

# Install Python dependencies
RUN pip install pytorch-lightning lightning[extra] torchvision scikit-learn tensorboard h5py einops timm seaborn SciencePlots umap-learn --ignore-installed
RUN pip install --upgrade psutil

# Update apt-get and install necessary system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-fonts-extra \
    dvipng \
    cm-super \
    --no-install-recommends

# Set working directory and copy application files
WORKDIR /app
COPY . /app
