import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from .utils import configure_lora
from peft import prepare_model_for_kbit_training

class Model(torch.nn.Module) :
    def __init__(self, modelName, modificationType = None, *args, **kwargs) :
        super().__init__()
        quantizationConfig = BitsAndBytesConfig(load_in_8bit=True)
        # self._outputClasses = outputClasses
        self._model = AutoModelForCausalLM.from_pretrained(modelName, quantization_config=quantizationConfig)
        
        # self._model.lm_head = torch.nn.Linear(self._model.config.hidden_size, len(self._outputClasses))
        self._tokenizer = AutoTokenizer.from_pretrained(modelName)
        
        if modificationType :
            self._modify_model(modificationType, *args, **kwargs)
        
        # self._modify_tokenizer()
        for name, _ in self._model.named_modules():
            print(name)

    @property
    def model(self) :
        return self._model

    @property
    def tokenizer(self) :
        return self._tokenizer

    @property
    def outputLabels(self) :
        return self._outputLabels
    
    def _modify_tokenizer(self):
        self._tokenizer.pad_token = self._tokenizer.eos_token
    
    def _modify_model(self, modificationType, *args, **kwargs) :
        self._model = prepare_model_for_kbit_training(self._model)
        if modificationType == 'lora' :
            self._model = configure_lora(self._model, *args, **kwargs)
        elif modificationType == 'freeze' :
            for param in self._model.parameters() :
                param.requires_grad = False

            for param in self._model.cls.parameters() :
                param.requires_grad = True
        else :
            raise NotImplementedError

    def forward(self, *args, **kwargs) :
        return self.model(*args, **kwargs)

