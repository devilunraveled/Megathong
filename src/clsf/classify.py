from transformers import RobertaTokenizer, RobertaForSequenceClassification
# from src.clsf.config import RedditData
import torch

class RedditData:
    labels : list = ['addiction', 'anxiety', 'bipolar', 'depression', 'ptsd', 'adhd', 'suicidewatch', 'neutral']

model_name = "./peft/fine-tuned-roberta"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=len(RedditData.labels))

def classify(text):
    model.eval()
    input_ids = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)['input_ids']
    outputs = model(input_ids)
    logits = outputs.logits
    return torch.softmax(logits, dim=1).tolist()[0]