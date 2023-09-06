from sklearn.metrics import classification_report, confusion_matrix

class Score:
    def __init__(self, labels_map):
        super(Score, self).__init__()
        self.labels_map = labels_map
        self.inverted_labels_map = {v: k for k,v in labels_map.items()}

    def run_evaluation(self, page_nodes_predictions):
        reports = []

        files_predictions = page_nodes_predictions.group_by_file()
        
        for _, node_predictions in files_predictions.items():
            y_true = []
            y_pred = []

            for prediction in node_predictions.predictions:
                print(prediction.true_label, prediction.prediction)
                y_true.append(prediction.true_label)
                y_pred.append(prediction.prediction)

            report = self.evaluate(y_true, y_pred, skip_labels=['O'])
            reports.append(report)

        avg_report = self.average_report(reports)
        matrix, mismatch = self.get_confusion_matrix(page_nodes_predictions)
        histogram = self.get_histogram(page_nodes_predictions)

        return avg_report, matrix, mismatch, histogram

    def get_confusion_matrix(self, page_nodes_predictions):
        y_true = []
        y_pred = []
        mismatch = 0

        for prediction in page_nodes_predictions.predictions:
            y_true.append(prediction.true_label)
            y_pred.append(prediction.prediction)

            if prediction.true_label != prediction.prediction:
                mismatch = mismatch + 1
            
        return self.conf_matrix(y_true, y_pred), mismatch

    def conf_matrix(self, y_true, y_pred):
        labels = list(self.labels_map.values())
        matrix = confusion_matrix(y_true, y_pred, labels=labels)

        return {
            'matrix': matrix,
            'labels': list(self.labels_map.keys())
        }

    def get_histogram(self, page_nodes_predictions):
        y_true = [self.inverted_labels_map[x.true_label] for x in page_nodes_predictions.predictions]
        y_pred = [self.inverted_labels_map[x.prediction] for x in page_nodes_predictions.predictions]
        confidence = [x.confidence for x in page_nodes_predictions.predictions]

        return list(zip(y_true, y_pred, confidence))

    def evaluate(self, y_true, y_pred, skip_labels):
        labels_map = {k:v for k,v in self.labels_map.items() if k not in skip_labels}
        acc = [1 if p==i else 0 for p, i in zip(y_true, y_pred)]
        acc = sum(acc)/len(acc) if len(acc) > 0 else 0

        report = classification_report(y_true, y_pred, output_dict=True,
            target_names = list(labels_map.keys()), labels=list(labels_map.values()))
        return {
            'accuracy': acc,
            'report': report
        }

    @staticmethod
    def average_report(reports):
        accuracy = sum(d['accuracy'] for d in reports) / len(reports)
        f1_score = sum(d['report']['weighted avg']['f1-score'] for d in reports) / len(reports)
        precision = sum(d['report']['weighted avg']['precision'] for d in reports) / len(reports)

        return {
			"accuracy": accuracy,
			"f1_score": f1_score,
			"precision": precision
		}
