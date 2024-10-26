from clsf.model import LlamaModel
from config import Model

if __name__ == "__main__":
    model = LlamaModel(Model.name, **Model.config)
    print(model.getResponse("I am depressed."))
