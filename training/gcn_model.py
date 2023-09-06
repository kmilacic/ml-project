import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import numpy as np
import time
import datetime
from torch.utils.tensorboard import SummaryWriter
from predictions.predictor import Predictor
from evaluation.score import Score
import os
import glob
from shared_utils.utils import init_logger
from training.model import Model
import mlflow
from collections.abc import Mapping

logger = init_logger('./logs.txt')

def log(key, value, d):
    if isinstance(value, Mapping):
        for k, v in value.items():
            log(k, v, d[key])
    else:
        mlflow.log_param(key, value)

def log_mlflow_param(config_file):
    for key, value in config_file.items():
        log(key, value, config_file)

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes, config_file):
        super(GCN, self).__init__()
        self.in_feats = in_feats
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        number_of_layers = config_file['node_classification']['number_of_layers']
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(GraphConv(in_feats, hidden_size))

        for i in range(0, number_of_layers-2):
           self.conv_layers.append(GraphConv(hidden_size, hidden_size))

        self.conv_layers.append(GraphConv(hidden_size, num_classes))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.conv_layers):
            h = layer(g, h)
            if i != len(self.conv_layers)-1:
                h = torch.relu(h)
        return h

class GCNTrainer:
    def __init__(self, labels_map, path, hidden_size, config_file, save_checkpoint_steps = 10, model_prefix = '',
                    checkpoint = None, continue_from_epoch = 0):
        super(GCNTrainer, self).__init__()

        self.config_file = config_file
        self.save_checkpoint_steps = save_checkpoint_steps
        self.labels_map = labels_map
        self.inverted_labels_map = {v: k for k,v in labels_map.items()}
        self.model_prefix = model_prefix
        self.learning_rate = self.config_file['node_classification']['learning_rate']
        self.path = path
        self.hidden_size = hidden_size
        self.device = torch.device('cpu')

        if checkpoint:
            self.model, self.optimizer, continue_from_epoch, self.loss_func = load_gcn_model(checkpoint)
            self.continue_from_epoch = continue_from_epoch + 1

        else:
            self.continue_from_epoch = continue_from_epoch
            self.loss_func = nn.CrossEntropyLoss()
            number_of_classes = len(self.labels_map.keys())
            self.model = GCN(hidden_size, hidden_size, number_of_classes, self.config_file)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        self.model.to(self.device)

    def train(self, trainset):
        logger.info('GCN training started')

        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("hidden_size", self.hidden_size)
        mlflow.log_param('epochs', self.config_file['node_classification']['epochs'])
        mlflow.log_param('type', self.config_file['node_classification']['name'])
        log_mlflow_param(self.config_file)
        
        number_of_epochs = self.config_file['node_classification']['epochs']
        writer = SummaryWriter()
        self.model.train()

        epoch_losses = []
        all_logits = []

        start = time.time()
        epoch_start = start
        for epoch in range(self.continue_from_epoch, number_of_epochs):
            epoch_loss = 0

            # (dgl_graph, labels, df, f, label)
            iter = 0
            for iter, sample in enumerate(trainset):
                inputs = sample.graph.ndata['feat'] #torch.tensor(sample.graph.ndata['feat'].tolist())
                logits = self.model(sample.graph, inputs)
                all_logits.append(logits.detach())
                loss = self.loss_func(logits, sample.node_labels) # torch.tensor(sample.node_labels, device=self.device))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # epoch_loss += loss.detach().item()
                epoch_loss += loss
            epoch_loss /= (iter + 1)
            mlflow.log_metric("epoch_loss", epoch_loss.item())
            epoch_end = time.time()
            epoch_time = datetime.timedelta(seconds = epoch_end - epoch_start)
            remaining_time = datetime.timedelta( seconds = (epoch_end - start) / (epoch + 1) * number_of_epochs - (epoch_end - start))
            logger.info('Epoch {}, loss {:.4f}, e {}, r {}'.format(epoch, epoch_loss, epoch_time, remaining_time))

            epoch_start = epoch_end
            epoch_losses.append(epoch_loss)
            writer.add_scalar(f"{self.model_prefix} Loss/train", epoch_loss, epoch)

            if ((epoch + 1) % self.save_checkpoint_steps == 0):
                self.save(epoch)

        writer.flush()
        writer.close()

    def save(self, epoch):
        torch.save({
                'config_file': self.config_file,
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'hidden_size': self.model.hidden_size,
                'number_of_classes': self.model.num_classes,
                }, os.path.join(self.path, self.model_prefix + '_epoch_' + str(epoch) + '_'  + str(int(time.time())) + '.pt'))

    def validate(self, dataset, master_labels):
        y_true = [label for x in dataset for label in x.node_labels.tolist()]
        scores = []

        files = sorted(glob.glob(os.path.join(self.path, self.model_prefix + '*.pt')))
        files.sort(key=os.path.getmtime)

        for f in files:
            if not f.endswith('.pt'):
                continue

            checkpoint = torch.load(f)

            model = Model(checkpoint, load_gcn_model)
            model.load_model()

            node_predictions = Predictor.predict_nodes(model, self.labels_map, self.inverted_labels_map, dataset, master_labels=master_labels)

            y_pred = [x.prediction for x in node_predictions.predictions]

            s = Score(self.labels_map)
            res = s.evaluate(y_true[:len(y_pred)], y_pred, skip_labels=['O'])
            f1_score = round(res['report']['weighted avg']['f1-score'], 4)
            scores.append(f1_score)
            logger.info('{} f1_score: {:.4f}'.format(f, f1_score))

        index = np.argmax(scores)
        logger.info('Best model: {} {:.4f}'.format(files[index], scores[index]))
        mlflow.log_metric('best_epoch', int(files[index].split('_epoch_')[1].split('_')[0]))
        mlflow.log_metric("best_f1_score", scores[index])
        checkpoint = torch.load(files[index])

        for f in files[:len(files)-1]:
            os.remove(f)

        return Model(checkpoint, load_gcn_model)

def load_gcn_model(checkpoint):
    hidden_size = checkpoint['hidden_size']
    number_of_classes = checkpoint['number_of_classes']

    net = GCN(hidden_size, hidden_size, number_of_classes, checkpoint['config_file'])
    net.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cpu')
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=checkpoint['config_file']['node_classification']['learning_rate'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    loss_func = nn.CrossEntropyLoss()

    return net, optimizer, epoch, loss_func

