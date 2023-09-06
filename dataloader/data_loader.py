import networkx as nx
import dgl
import torch


class DataSample:
  def __init__(self, filename, table_label=None, train=True):
    self.filename = filename
    self.table_label = table_label
    self.train = train
    self.device = torch.device('cpu')

  def set_dataframe(self, df):
    self.dataframe = df

  def set_graph(self, graph):
    self.graph = graph

  def set_table_label(self, table_label):
    self.table_label = table_label

  def extract_node_labels(self, labels_map):
    if self.train:
      self.node_labels = torch.tensor([labels_map[key.strip().split('_:_')[-1].split('_id_')[-1]] for key in self.dataframe['Label'].tolist()], device=self.device)
    else:
      self.node_labels = None

  def create_graph(self, df):
    g = nx.DiGraph(df)
    nx_G = g.to_undirected()
    dgl_graph = dgl.from_networkx(nx_G).to(self.device)
    print(self.features.shape, dgl_graph.number_of_nodes())
    dgl_graph.ndata['feat'] = torch.from_numpy(self.features).float().to(self.device)

    self.graph = dgl_graph

  def create_new_dataframe(self, df_data, img, geom_features):
    pass

class Dataset:
  def __init__(self, files_dict, master_labels, img_dir, features_dict=None, train=True):    
    self.train = train
    self.master_labels = master_labels
    self.img_dir = img_dir
    self.features_dict = features_dict
    self.no_of_features = None
    self.initialize_dataset(files_dict)
    

  def initialize_dataset(self, files_dict):
    self.read_dataset()
    self.numberOfPages = len(self.dataset)
    self.numberOfDocs = len(files_dict)      

  def get_dataset(self):
    return self.dataset

  def set_dataset(self, dataset):
    self.dataset = dataset

  def read_dataset(self):
    pass