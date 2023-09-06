## Description

This ML project works with graph neural networks and gives a pipeline for creating and transforming dataset, training and testing. Input dataset are documents split into images (one page is one image) and words with text and coordinates on the page/image. These documents are structured (bills, receipts etc), which means that they will always have the same form and we can expect that text will always appear on the same positions on the page. For each image, graph is generated using `networkx` and `dgl` libraries, where each node represents a word on the page, and has features such as text, number of alphabetic and numerical characters, coordinates, neighbors etc. Each word/node has its own label that is the important information that needs to be extracted, so it is a node-level classification. Neural network that is trained has convolutional layers that gather information about node and node's neighbors. We use `dgl` library for deep learning on graphs in combination with `PyTorch`.

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
