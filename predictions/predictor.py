import numpy as np
from shared_utils.visualisation import draw
from scipy.special import softmax
from predictor import NodePredictions

class Predictor:
	def __init__(self, master_labels, table_models, is_test = False):
		super(Predictor, self).__init__()
		self.master_labels = master_labels
		self.table_models = table_models
		self.inverted_labels_map = {v: k for k,v in self.master_labels['labels_map'].items()}
		self.inverted_table_labels_map = {v: k for k,v in self.master_labels['table_labels_map'].items()}
		self.is_test = is_test

	@staticmethod
	def predict(model, labels_map, inverted_labels_map, testset, table_label = None,
		  page_label=None, draw_imgs=True, graph_imgs_folder='./images', postprocessing_f = None, dtree_fix=False,
		  master_labels=None):
		model.model.eval()

		node_predictions = NodePredictions(labels_map, master_labels, [])

		for element in testset:
			logits = model.model(element.graph, element.graph.ndata['feat'])

			predictions = [i for i in np.argmax(logits.tolist(), axis=1)]
			confidence = [c for c in np.array(softmax(logits.tolist(), axis=1)).max(axis=1)]

			if draw_imgs and graph_imgs_folder != '':
				draw(element, predictions, labels_map, element.filename, path=graph_imgs_folder,
				 intent_labels=master_labels['all_labels_map']['intents'], pred=False)
				if element.node_labels is not None:
					# draw true labels
					draw(element, element.node_labels, labels_map, element.filename+'_true', 
					 intent_labels=master_labels['all_labels_map']['intents'], path=graph_imgs_folder, pred=False)

			node_predictions.add_to_predictions_list(element, predictions, confidence, table_label, page_label)

		return node_predictions
