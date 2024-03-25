import os
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPTNeoModel
from tqdm import tqdm
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import torch.distributed as dist
import torch.multiprocessing as mp
import sys
import os
import torch
from torch import nn, optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPTNeoModel
from tqdm import tqdm
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from sklearn.metrics import classification_report, confusion_matrix

class GPTNeoClassifier(nn.Module):
    def __init__(self, model_save_path='./gpt_neo_2_7b.pth', batch_size=16, epochs=6, num_labels=3):
        super(GPTNeoClassifier, self).__init__()
        self.model_name = 'EleutherAI/gpt-neo-2.7B'
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.epochs = epochs
        self.num_labels = num_labels
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPTNeoModel.from_pretrained(self.model_name)
        self.model = nn.DataParallel(self.model)  # Enable Data Parallelism

        self.classifier_head = nn.Sequential(
            nn.Linear(self.model.module.config.hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_labels)
        )

        self.to(self.device)
        self.optimizer = optim.AdamW(self.parameters(), lr=2e-5)
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()  # For AMP

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        logits = self.classifier_head(outputs.last_hidden_state[:, 0, :])
        loss = self.criterion(logits, labels) if labels is not None else None
        return loss, logits

    def train_model(self, train_texts, train_labels):
        if os.path.exists(self.model_save_path):
            print("Model already exists. Loading the model...")
            self.load_state_dict(torch.load(self.model_save_path, map_location=self.device))
            print("Model loaded successfully.")
            return
            
        train_encodings = self.tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
        train_dataset = TensorDataset(train_encodings['input_ids'].to(self.device), train_encodings['attention_mask'].to(self.device), torch.tensor(train_labels).to(self.device))
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.epochs}'):
                input_ids, attention_mask, labels = batch

                self.optimizer.zero_grad()
                with autocast():  # AMP context
                    loss, _ = self(input_ids, attention_mask, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item()

            print(f'Epoch {epoch+1} Loss: {total_loss / len(train_loader)}')
        torch.save(self.state_dict(), self.model_save_path)
        print('Training complete.')

    def evaluate_model(self, test_texts, test_labels):
        # Prepare data
        test_encodings = self.tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
        test_dataset = TensorDataset(test_encodings['input_ids'].to(self.device), test_encodings['attention_mask'].to(self.device), torch.tensor(test_labels).to(self.device))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
    
        # Evaluation loop
        self.model.eval()
        all_predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc='Evaluating'):
                input_ids, attention_mask, labels = batch
                _, logits = self(input_ids, attention_mask)
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
    
        # Calculate metrics
        metrics = classification_report(test_labels, all_predictions, output_dict=True)
        cm = confusion_matrix(test_labels, all_predictions)
    
        # Save to PDF including metrics and confusion matrix
        self.save_results_to_pdf(test_texts, test_labels, all_predictions, metrics, cm)
        
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
        
    def save_results_to_pdf(self, texts, true_labels, predictions, metrics, cm):
        c = canvas.Canvas("evaluation_results.pdf", pagesize=letter)
        text = c.beginText(40, 750)
        text.setFont("Helvetica", 10)
        
        # Adding metrics and confusion matrix to PDF
        text.textLine("Evaluation Metrics:")
        for label, metric in metrics.items():
            if isinstance(metric, dict):  # Skip the overall average metrics to focus on per-class metrics
                text.textLine(f"Class {label}: Precision: {metric['precision']:.2f}, Recall: {metric['recall']:.2f}, F1-Score: {metric['f1-score']:.2f}")
        text.textLine("\nConfusion Matrix:")
        text.textLine(str(cm))
        text.textLine("\n\nDetailed Results:")
        
        # Detailed results
        for i, (txt, true_label, pred) in enumerate(zip(texts, true_labels, predictions), 1):
            text.textLine(f"{i}. Text: {txt[:100]}... True Label: {true_label}, Prediction: {pred}")
            if i % 25 == 0:  # Adjust the number based on your page layout
                c.drawText(text)
                c.showPage()
                text = c.beginText(40, 750)
        
        c.drawText(text)
        c.save()
        print("Evaluation results saved to evaluation_results.pdf")



