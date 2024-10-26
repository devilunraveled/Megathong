from transformers import BitsAndBytesConfig

class Model:
    name = "kingabzpro/Llama-3.1-8B-Instruct-Mental-Health-Classification"

    config = {
        'return_dict': True,
        'low_cpu_mem_usage': True,
        'trust_remote_code': True,
        'quantization_config': BitsAndBytesConfig(load_in_4bit=True),
    }
