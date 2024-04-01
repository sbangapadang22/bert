from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import fetch_20newsgroups

# Implement the `evaluate` function
def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    total_correct = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            predictions = outputs.logits.argmax(dim=-1)
            total_correct += (predictions == labels).sum().item()
            
    avg_loss = total_loss / len(data_loader)
    accuracy = total_correct / len(data_loader.dataset)
    
    return avg_loss, accuracy

# Load and preprocess data for sentiment analysis
# load imdb dataset
imdb_data = pd.read_csv('/desktop/bert/pyf/myenv/imdb_dataset.csv') # load the dataset
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(list(imdb_data['text']), truncation=True, padding=True, max_length=512)
imdb_data['label'] = imdb_data['label'].map({'positive': 1, 'negative': 0})
train_texts, val_texts, train_labels, val_labels = train_test_split(
    encodings['input_ids'], imdb_data['label'].values, test_size=0.2, random_state=42
)

# Convert data to PyTorch tensors
train_inputs = torch.tensor(train_texts)
train_labels = torch.tensor(train_labels)
val_inputs = torch.tensor(val_texts)
val_labels = torch.tensor(val_labels)

# Create PyTorch datasets
train_dataset = TensorDataset(train_inputs, train_labels)
val_dataset = TensorDataset(val_inputs, val_labels)

# Create data loaders
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Load pre-trained BERT model and add a classification layer
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Define the loss function and optimizer
from transformers import AdamW

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=2e-5)

# Train the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

epochs = 3

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        
    val_loss, val_acc = evaluate(model, val_loader, device)
    print(f"Epoch {epoch+1} - Train Loss: {total_loss/len(train_loader)}, Val Loss: {val_loss}, Val Acc: {val_acc}")

