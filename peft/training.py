import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import AdamW
from alive_progress import alive_bar
from config import Paths

# Define your CustomDataset class (as you provided)
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

# Load your data into a DataFrame (replace with your actual data)
df = pd.read_csv(f"{Paths.data}/labelListTrainData.csv")
classes = sorted(list(df['subreddit'].unique()))  # Unique subreddits

print("Classes : ", classes)

exit()
# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=len(classes))

# Create the dataset and dataloader
dataset = CustomDataset(dataframe=df, tokenizer=tokenizer, classes=classes)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Set up training parameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=1e-5)

# Training loop
num_epochs = 0  # Set the number of epochs

for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    numSteps = 0
    with alive_bar(len(dataloader)) as bar:
        for batch in dataloader:
            optimizer.zero_grad()  # Clear previous gradients
            
            # Move inputs to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            
            loss = outputs.loss  # Get the loss from outputs
            total_loss += loss.item()
            
            runningAvg = total_loss / (numSteps + 1)
            numSteps += 1

            bar.text(f"Loss: {runningAvg:.4f}")

            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            bar()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

# Save the fine-tuned model (optional)
model.save_pretrained("fine-tuned-roberta")

print("Training complete!")


# # from accelerate import Accelerator
# import pandas as pd
# from config import Paths, ModelConfig
# from torch.utils.data import Dataset
# from src.model import Model
# from transformers import Trainer, TrainingArguments
# # from transformers import AdamW, get_linear_schedule_with_warmup
# import torch 

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

# class CustomDataset(Dataset):
#     def __init__(self, dataframe, tokenizer, classes) -> None:
#         super().__init__()
#         self.dataframe = dataframe
#         self.tokenizer = tokenizer
#         self.classes = classes
#     
#     def __len__(self) -> int:
#         return len(self.dataframe)

#     def __getitem__(self, index):
#         row = self.dataframe.iloc[index]
#         datapoint = self.tokenizer(row['post'],return_tensors="pt", padding='max_length', truncation=True, max_length=512)

#         datapoint['labels'] = torch.tensor(self.classes.index(row['subreddit']))
#         datapoint['input_ids'] = datapoint['input_ids'].squeeze(0)
#         datapoint['attention_mask'] = datapoint['attention_mask'].squeeze(0)

#         return datapoint

# def train(model, trainData, evalData, trainingArguments) :
#     # accelerator = Accelerator()
#     # accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = trainingArguments['per_device_train_batch_size']
#     # optimizer = AdamW(model.parameters(), lr=trainingArguments['learning_rate'])
#     # scheduler = get_linear_schedule_with_warmup(optimizer, 
#     #                                             num_warmup_steps=trainingArguments['warmup_steps'],
#     #                                             num_training_steps=len(trainData) * trainingArguments['num_train_epochs'])
#     # model, trainData, evalData = accelerator.prepare(model, trainData, evalData)

#     trainer = Trainer(model         = model, 
#                       train_dataset = trainData, 
#                       eval_dataset  = evalData, 
#                       args          = TrainingArguments(**trainingArguments))
#     trainer.train()

# if __name__ == "__main__" :
#     data = pd.read_csv(f"{Paths.data}/labelListData.csv")
#     toatalDataLen = len(data)

#     trainData = data[:int(toatalDataLen*0.8)]
#     testData = data[int(toatalDataLen*0.8):int(toatalDataLen*0.9)]
#     evalData = data[int(toatalDataLen*0.9):]
#     
#     outputClasses = sorted(list(data['subreddit'].unique()))

#     model = Model(modelName = ModelConfig.name, outputClasses = outputClasses, modificationType = 'freeze').to("cuda")
#     tokenizer = model.tokenizer
#     
#     print("Model Loaded")

#     trainDataset = CustomDataset(trainData, tokenizer, outputClasses)
#     testDataset = CustomDataset(testData, tokenizer, outputClasses)
#     evalDataset = CustomDataset(evalData, tokenizer, outputClasses)
#     
#     print("Dataset Created")

#     training_arguments = {
#         'output_dir': Paths.model,
#         'num_train_epochs': 4,
#         'per_device_train_batch_size': 8,
#         'per_device_eval_batch_size': 4,
#         'eval_strategy': 'epoch',
#         'logging_dir': Paths.logs,
#         'logging_steps': 100,
#         'learning_rate': 1e-5,
#         # 'fp8': True,
#         'warmup_steps': 100,
#         'weight_decay': 0.01,
#         'gradient_accumulation_steps': 2,
#         'save_safetensors' : False
#     }

#     train(model, trainDataset, evalDataset, trainingArguments = training_arguments)
