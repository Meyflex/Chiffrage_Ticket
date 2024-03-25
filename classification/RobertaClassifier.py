import os
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd

class RobertaClassifier:
    def __init__(self, model_save_path, batch_size=16, epochs=3):
        self.model_name = 'roberta-base'
        self.model_save_path = model_save_path
        
        self.batch_size = batch_size
        self.epochs = epochs

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        self.tokenizer = RobertaTokenizer.from_pretrained(self.model_name)
        self.model = RobertaForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=2e-5)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train(self, train_texts, train_labels):
        train_encodings = self.tokenizer(list(train_texts), truncation=True, padding=True, max_length=256, return_tensors='pt')

        train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], torch.tensor(train_labels))

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.model.to(self.device)

        if os.path.exists(self.model_save_path):
            print("Loading saved model...")
            checkpoint = torch.load(self.model_save_path)
            self.model.load_state_dict(checkpoint['model_state_dict'],strict=False)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Model loaded successfully.")
        else:
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0
                for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{self.epochs}'):
                    input_ids, attention_mask, labels = [t.to(self.device) for t in batch]

                    self.optimizer.zero_grad()
                    outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()

                average_loss = total_loss / len(train_loader)
                print(f'Training Loss: {average_loss}')

            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict()
            }, self.model_save_path)
            print("Model saved successfully.")

    def evaluate(self, test_texts, test_labels):
        self.model.eval() 
        
        test_encodings = self.tokenizer(list(test_texts), truncation=True, padding=True, max_length=256, return_tensors='pt')
        test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [t.to(self.device) for t in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        return all_labels, all_predictions

    def save_incorrect_predictions(self, test_texts, all_labels, all_predictions,file_name='./incorrect_predictions.csv'):
        incorrect_reviews = []
        incorrect_predictions = []
        correct_labels = []

        for i, (pred, actual) in enumerate(zip(all_predictions, all_labels)):
            if pred != actual:
                incorrect_reviews.append(test_texts[i])
                incorrect_predictions.append(pred)
                correct_labels.append(actual)

        df = pd.DataFrame({
            'Review': incorrect_reviews,
            'Predicted Label': incorrect_predictions,
            'Actual Label': correct_labels
        })

        df.to_csv(file_name, index=False)

