import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import numpy as np
from config import Paths
from alive_progress import alive_bar

# Define your CustomDataset class (same as before)
class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, classes) -> None:
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.classes = classes
    
    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        datapoint = self.tokenizer(row['post'], return_tensors="pt", padding='max_length', truncation=True, max_length=512)

        # Convert subreddit to index
        datapoint['labels'] = torch.tensor(self.classes.index(row['subreddit']))
        datapoint['input_ids'] = datapoint['input_ids'].squeeze(0)
        datapoint['attention_mask'] = datapoint['attention_mask'].squeeze(0)

        return datapoint

# Load your evaluation data into a DataFrame (replace with your actual data)
eval_df = pd.read_csv(f"{Paths.data}/labelListTrainData.csv").sample(frac=0.1)  # Sample a fraction of the data
classes = sorted(list(set(eval_df['subreddit'])))  # Unique subreddits

# Load the fine-tuned model and tokenizer
model_name = "fine-tuned-roberta"  # Path to your saved model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained(model_name)
# model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=len(classes))

# Create the evaluation dataset and dataloader
eval_dataset = CustomDataset(dataframe=eval_df, tokenizer=tokenizer, classes=classes)
eval_dataloader = DataLoader(eval_dataset, batch_size=32)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Evaluation loop
model.eval()  # Set the model to evaluation mode

all_labels = []
all_preds = []

with torch.no_grad():
    with alive_bar(len(eval_dataloader)) as bar:
        for batch in eval_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get predicted class indices
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store predictions and true labels
            all_preds.extend(preds)
            all_labels.extend(batch['labels'].cpu().numpy())
            bar()
# Convert lists to numpy arrays for metric calculations
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)

# Compute metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("Confusion Matrix:")
print(conf_matrix)
