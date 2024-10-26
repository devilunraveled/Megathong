from peft import get_peft_model, LoraConfig

def configure_lora(module, useLimitedDecomps):
    taskType = 'CAUSAL_LM'
    
    targetModules = [ moduleName for moduleName, _ in module.named_modules() if moduleName != '' and moduleName.endswith('.k') or moduleName.endswith('.q') ]

    arguments = {
        'r': 8,
        'lora_alpha': 16,
        'lora_dropout': 0.05,
        'task_type': taskType,
        'bias': 'none',
    }

    if useLimitedDecomps:
        arguments['target_modules'] = targetModules

    peft_config = LoraConfig(**arguments)
    return get_peft_model(module, peft_config)
