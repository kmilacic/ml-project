


class Model:

    def __init__(self, checkpoint, load_f):
        self.model = None
        self.checkpoint = checkpoint
        self.load_f = load_f
    
    def load_model(self):
        self.model = self.load_f(self.checkpoint)[0]

    @staticmethod
    def get_model(checkpoint):
        if checkpoint is None:
            return None
        _model = checkpoint
        _model.load_model()
        return _model

    @staticmethod
    def get_models(models_dict):
        if models_dict is None:
            return None
        gcn_models = {}

        for key, _model in models_dict.items():
            _model.load_model()
            gcn_models[key] = _model  # load_gcn_model(model_checkpoint)[0]

        return gcn_models