## Description

This ML project works with graph neural networks and gives a pipeline for creating and transforming dataset, training and testing.

## Installation

```bash
conda create --name gcn
conda activate gcn
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
pip install -r requirements.txt
```

### DGL with GPU

If you want to use GPU, install newer version of dgl library:

```bash
# old version of dgl without gpu - 0.4.3.post2
# dgl version with gpu
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c dglteam dgl-cuda10.2

```

## Run application

Create `.env` and `.flaskenv` files based on `.example` files, and then run command `flask run`. Optionally, you can build docker image and run with `docker-compose up -d`.
