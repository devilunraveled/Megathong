from transformers import AutoModel, AutoTokenizer

class Model :
    def __init__(self, name):
        self.name = name

    def getResponse(self, *args, **kwargs):
        raise NotImplementedError


class LlamaModel(Model):
    def __init__(self, name, *args, **kwargs):
        super().__init__(name)
        self._model = AutoModel.from_pretrained(name, *args, **kwargs).to('cuda')
        self._tokenizer = AutoTokenizer.from_pretrained(name)
    
    @property
    def model(self):
        return self._model

    @property
    def tokenizer(self):
        return self._tokenizer

    def getResponse(self, *args, **kwargs):
        return self._model(*args, **kwargs)

    def eval(self):
        self._model.eval()
