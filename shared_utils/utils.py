import logging
from sklearn.model_selection import train_test_split

def split_train_test_datasets(files_dict, test_size=0.1):
    trainset, testset = train_test_split(list(files_dict.keys()), test_size=test_size, random_state=42)

    trainset = {k:v for k,v in files_dict.items() if k in trainset}
    testset = {k:v for k,v in files_dict.items() if k in testset}

    return trainset, testset

def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger
