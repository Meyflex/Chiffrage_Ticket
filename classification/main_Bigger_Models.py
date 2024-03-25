import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import CountVectorizer
from BertClassifier import BertClassifier
from RobertaClassifier import RobertaClassifier
from GPTj import GPTNeoClassifier

from FunctionUtil import *

from tqdm import tqdm
import os
import pandas as pd
import re

# Function to clean and preprocess dataset
def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'\{.*?\}', ' ', text)  # Remove content within {}
    text = re.sub(r'\[.*?\]', ' ', text)  # Remove content within []
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    text = text.lower().strip()  # Convert to lowercase and strip whitespaces
    return text
def categorize_chiffrage(value):
    if value <= 3:
        return 0
    elif value <= 8:
        return 1
    else:
        return 2
def preprocess_dataset(dataset):
    dataset = dataset.dropna(subset=['Chiffrage TMA JH'])

    # Ensure 'Chiffrage TMA JH' column exists and convert its values
    if 'Chiffrage TMA JH' in dataset.columns:
        dataset['Chiffrage TMA JH'] = dataset['Chiffrage TMA JH'].str.replace(',', '.').astype(float)
    else:
        raise ValueError("Column 'Chiffrage TMA JH' not found in dataset")

    # Categorize 'Chiffrage TMA JH' values
    dataset['Chiffrage Catégorie'] = dataset['Chiffrage TMA JH'].apply(categorize_chiffrage)

    # Combine specified columns into 'entry', ensuring no NaN values, and clean the text
    dataset['entry'] = dataset[['Projet', 'Type de ticket', 'Priorité', 'Description US', 'Critères d’acceptation']].apply(
        lambda x: ' '.join(x.dropna().astype(str)), axis=1
    ).apply(clean_text)

    return dataset[['entry', 'Chiffrage Catégorie']]

# Apply the function

    
def main():
    dataset = pd.read_csv("dataset.csv")
    dataset_processed = preprocess_dataset(dataset)
    
    entry = dataset_processed["entry"].tolist()
    output = dataset_processed["Chiffrage Catégorie"].tolist()
    
    train_texts, test_texts, train_labels, test_labels = train_test_split(entry, output, test_size=0.3)
    
    classifier_GPT = GPTNeoClassifier('./models/GPTJ_model_oversample.pth')
    
    # Train your model
    classifier_GPT.train_model(train_texts, train_labels)

    # Evaluate your model
    classifier_GPT.evaluate_model(test_texts, test_labels)

if __name__ == "__main__":
    main()
