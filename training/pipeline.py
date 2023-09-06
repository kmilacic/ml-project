import os
import mlflow
import time
import torch
from dataloader.data_loader import Dataset
from training.model import Model
from gcn_model import GCNTrainer
from predictions.predictor import Predictor
from shared_utils.utils import split_train_test_datasets
from evaluation.score import Score

def run_training(trainset, validation_set, path, config_file):
    m = GCNTrainer(trainset.master_labels['labels_map'], path, 
            hidden_size = trainset.no_of_features,
            config_file = config_file)
    m.train(trainset.dataset)
    model = m.validate(validation_set.dataset, trainset.master_labels)

    model_name = save_model(path, model, None, trainset.master_labels, config_file)

    return model_name

def run_predictions(analysis_dataset, table_models):
    predictor = Predictor(analysis_dataset.master_labels, table_models)
    analysis_predictions = predictor.predict(analysis_dataset)

    return analysis_predictions

def run_test(test_dataset, table_models):
    predictor = Predictor(test_dataset.master_labels, table_models, is_test = True)
    
    node_predictions = predictor.predict(test_dataset)

    score = Score(test_dataset.master_labels['labels_map'])
    avg_report, conf_matrix, mismatch, histogram = score.run_evaluation(node_predictions)

    return avg_report, conf_matrix, mismatch, histogram

def save_model(path, table_models, master_labels, config_file):
    model_name = str(int(time.time())) + '.pt'
    model = {
        'table_models': table_models,
        'master_labels': master_labels,
        'config_file': config_file
    }

    torch.save(model, os.path.join(path, model_name))

    return model_name

def load_model(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu') )

    table_models = Model.get_models(checkpoint['table_models'])
    master_labels = checkpoint['master_labels']
    config_file = checkpoint['config_file']

    return table_models, master_labels, config_file

def map_labels(labels):
    pass

def training_pipeline(files_dict, labels, path, features_dict, config_file):    
    start_time = time.time()

    img_dir = os.path.join('/uploads', 'img/')
    master_labels = map_labels(labels)

    trainset, testset = split_train_test_datasets(files_dict)

    train_dataset = Dataset(trainset, master_labels, img_dir, features_dict=features_dict)
    test_dataset = Dataset(testset, master_labels, img_dir, features_dict=features_dict)

    mlflow.log_param('training_samples', train_dataset.numberOfSamples)
    mlflow.log_param('test_samples', test_dataset.numberOfSamples)
    mlflow.log_param('training_pages', train_dataset.numberOfPages)
    mlflow.log_param('test_pages', test_dataset.numberOfPages)

    model_name = run_training(train_dataset, test_dataset, path, config_file)
    model_path = os.path.join(path, model_name)
    table_models, _, _ = load_model(model_path)
    avg_report, conf_matrix, mismatch, histogram = run_test(test_dataset, table_models)

    training_time = time.time() - start_time
    mlflow.log_param('training_time', training_time)
    mlflow.log_metric('accuracy', avg_report['accuracy'])
    mlflow.log_metric('f1_score', avg_report['f1_score'])
    mlflow.log_metric('precision', avg_report['precision'])

    report = {
        "modelFile": model_name,
        "trainingSamples": train_dataset.numberOfSamples,
        "testSamples": test_dataset.numberOfSamples,
        "trainDocs": train_dataset.numberOfDocs,
        "testDocs": test_dataset.numberOfDocs,
        "confMatrix": conf_matrix,
        "histogram": histogram,
        "mismatch": mismatch,
        "intent": avg_report,
        "trainingTime": training_time,
        "entity": {}
    }

    return report

def analysis_pipeline(files_dict, path, model_name, features_dict):
    img_dir = os.path.join('/uploads', 'img/')
    table_models, master_labels, _ = load_model(path, model_name)
    analysis_dataset = Dataset(files_dict, master_labels, img_dir, features_dict=features_dict, train=False)

    return run_predictions(analysis_dataset, table_models)

