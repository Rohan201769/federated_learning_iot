# client/data_processor.py
import os
import json
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random

class TextDataProcessor:
    def __init__(self, data_dir, max_vocab_size=10000, max_sequence_length=250):
        self.data_dir = data_dir
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.tokenizer = None
        self.class_names = []
        
        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Initialize tokenizer
        self._initialize_tokenizer()
    
    def _initialize_tokenizer(self):
        """Initialize and fit the tokenizer on available data"""
        all_texts = []
        
        # Load all available text data
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt'):
                with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as file:
                    all_texts.append(file.read())
        
        # Initialize from existing data or create empty tokenizer
        if all_texts:
            self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
            self.tokenizer.fit_on_texts(all_texts)
        else:
            self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        
        # Load class names if available
        class_names_path = os.path.join(self.data_dir, 'class_names.json')
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as file:
                self.class_names = json.load(file)
        else:
            # Default class names for sentiment analysis
            self.class_names = ['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive']
            with open(class_names_path, 'w') as file:
                json.dump(self.class_names, file)
    
    def preprocess_text(self, text):
        """Preprocess a single text input"""
        # Basic cleaning
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=self.max_sequence_length)
        
        # Return with the proper shape (batch_size, sequence_length)
        return padded  # Now returns array with shape (1, max_sequence_length)
        
    def load_data(self):
        """Load all available data"""
        texts = []
        labels = []
        
        # Look for labeled data files (format: class_id_*.txt)
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.txt') and filename[0].isdigit():
                class_id = int(filename[0])
                
                with open(os.path.join(self.data_dir, filename), 'r', encoding='utf-8') as file:
                    content = file.read()
                    
                    # Split into separate examples if the file contains multiple
                    examples = content.split('\n\n')
                    for example in examples:
                        if example.strip():
                            texts.append(example.strip())
                            labels.append(class_id)
        
        # Convert to sequences
        if not texts:
            return np.array([]), np.array([])
        
        # Tokenize all texts
        sequences = self.tokenizer.texts_to_sequences(texts)
        # Pad sequences
        padded_sequences = pad_sequences(sequences, maxlen=self.max_sequence_length)
        
        return padded_sequences, np.array(labels)
        
    def get_training_data(self, validation_split=0.2):
        """Get training data with validation split"""
        X, y = self.load_data()
        
        # If we have no data, return empty arrays
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - validation_split))
        
        train_indices = indices[:split_idx]
        
        return X[train_indices], y[train_indices]
    
    def get_validation_data(self, validation_split=0.2):
        """Get validation data"""
        X, y = self.load_data()
        
        # If we have no data, return empty arrays
        if len(X) == 0:
            return np.array([]), np.array([])
        
        # Use fixed seed for reproducibility (same as in get_training_data)
        np.random.seed(42)
        indices = np.random.permutation(len(X))
        split_idx = int(len(X) * (1 - validation_split))
        
        val_indices = indices[split_idx:]
        
        return X[val_indices], y[val_indices]
    
    def get_class_names(self):
        """Return list of class names"""
        return self.class_names
    
    def add_new_data(self, text, class_id):
        """Add new labeled data"""
        # Save to appropriate file
        filename = f"{class_id}_{len(os.listdir(self.data_dir))}.txt"
        with open(os.path.join(self.data_dir, filename), 'w', encoding='utf-8') as file:
            file.write(text)
        
        # Update tokenizer
        self.tokenizer.fit_on_texts([text])
    
    def add_data_from_web(self, source_url, class_id=None):
        """Add data from web source (simplified)"""
        # This is a placeholder - in a real implementation, you would:
        # 1. Fetch data from the URL
        # 2. Extract text content
        # 3. Assign a class if not provided
        # 4. Save to a file
        
        # For demo purposes, we'll just simulate new data
        import requests
        from bs4 import BeautifulSoup
        
        try:
            response = requests.get(source_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract text (simplified)
            paragraphs = soup.find_all('p')
            text = '\n'.join([p.get_text() for p in paragraphs])
            
            # If no class_id provided, randomly assign one for demo
            if class_id is None:
                class_id = random.randint(0, len(self.class_names) - 1)
            
            # Save the data
            self.add_new_data(text, class_id)
            return True
        except Exception as e:
            print(f"Error fetching data from {source_url}: {e}")
            return False