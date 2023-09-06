class Prediction:
  def __init__(self, prediction, confidence, filename, features, labels_map, table_label = None, page_label = None):
    super(Prediction, self).__init__()

    self.prediction = prediction
    self.confidence = confidence
    self.true_label = features.get('Label', None)
    self.filename = filename
    self.table_label = table_label
    self.page_label = page_label
    self.features = features
    
class NodePredictions:
  def __init__(self, labels_map, master_labels, predictions):
    super(NodePredictions, self).__init__()
    self.predictions = predictions
    self.labels_map = labels_map
    self.master_labels = master_labels
    self.page_index = 0

  def add_to_predictions_list(self, datasample, predictions, confidence, table_label, page_label):
    pass