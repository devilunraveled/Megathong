from clsf.model import LlamaModel
from config import Model
from transformers import AutoTokenizer,AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch
import pandas as pd
from clsf.config import RedditData 

if __name__ == "__main__":
    # model = LlamaModel(Model.name, **Model.config)
    # model.eval()
    # print(model.getResponse("I am depressed."))
    import sys

    model_id = "kingabzpro/Llama-3.1-8B-Instruct-Mental-Health-Classification"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
            model_id,
            return_dict=True,
            low_cpu_mem_usage=True,
            # torch_dtype=torch.float16,
            device_map="cpu",
            trust_remote_code=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
            )
    )
    model.config.pad_id = model.config.eos_token_id
    model.eval()

    text = "I'm trapped in a storm of emotions that I can't control, and it feels like no one understands the chaos inside me"
    
    # Load dataset
    dataset = pd.read_csv("../data/reddit_covid/divorce_pre_features_tfidf_256.csv")

    # Sample a batch of text inputs
    batch_size = int(sys.argv[1])  # Set your desired batch size
    texts = dataset.sample(batch_size)["post"].values[:2]

    print(texts)

    # Define the prompt template for each text
    prompts = [f"Classify the text into {','.join(RedditData.labels)} and return the answer as the corresponding mental health disorder label.\ntext: {text}\nlabel:" for text in texts]

    # Initialize the pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
    )
    
    exit()

    # Perform inference on the batch
    outputs = pipe(prompts, max_new_tokens=2, do_sample=True, temperature=0.1)

    # Extract and print the labels for each text in the batch
    # for i, output in enumerate(outputs):
    label = outputs[0]["generated_text"].split("label: ")[-1].strip()
        # print(f"Text {i+1}: {label}")

    with open("./outputs.txt", "w") as f:
        for i, output in enumerate(outputs):
            label = output[0]["generated_text"].split("label: ")[-1].strip()
            f.write(f"Text {i+1}: {label}\n")
