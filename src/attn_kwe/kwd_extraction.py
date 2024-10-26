from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from nltk.corpus import stopwords

# Load pre-trained model and tokenizer
model_name = "mental/mental-bert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Input sentence
sentence = "I am getting the feeling of being tired down, beaten up, can't get the energy to do something for myself, I am feeling very low, and I am afraid its only going to get worse. This is Depression"
sentence = "I am afraid my exam today went horribly bad and I think my next exam is gonna be bad too. I don't know how my parents are gonna react to that. This is Anxiety"
sentence = "I am positively surprised that this went well, I think that I am going to be fine. I am well and I am happy. This is Normal"
# remove stopwords
stop_words = set(stopwords.words('english'))
sentence = ' '.join([word for word in sentence.split() if word not in stop_words])

# Tokenize input
inputs = tokenizer(sentence, return_tensors="pt")

# Forward pass to get attention scores
with torch.no_grad():
    outputs = model(**inputs)

# Extract attention scores
attention_scores = outputs.attentions  # This is a list of attention matrices for each layer

print(attention_scores[-1].shape)

# Analyze attention for the last token "Depression"
last_token_index = inputs['input_ids'][0].tolist().index(tokenizer.encode(sentence.split()[-1])[1])  # Get index of "Depression"

print(last_token_index)
layer_attention_scores = [layer[last_token_index] for layer in attention_scores[-1][0][-1]]  # Last layer's scores

token_attention_weights : list[tuple[str, float]] = []

# Display attention scores
for i, score in enumerate(layer_attention_scores):
    token = tokenizer.decode(inputs['input_ids'][0][i])
    attn_score = score.item()
    token_attention_weights.append((token, attn_score))
    print(f"Attention from token `{tokenizer.decode(inputs['input_ids'][0][i])}` to `Depression`: {score.item()}")

sorted_weights = sorted(token_attention_weights, key=lambda x: x[1], reverse=True)

for token, weight in sorted_weights:
    print(f"{token}: {weight}")
