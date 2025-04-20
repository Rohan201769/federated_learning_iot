# models/text_classifier.py
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.optimizers import Adam

def create_model(vocab_size=10000, embedding_dim=16, max_sequence_length=250, num_classes=5):
    """Create a simple text classification model suitable for Raspberry Pi"""
    model = Sequential([
        # Use efficient embedding dimension
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        
        # Global pooling is more efficient than LSTM/GRU for resource-constrained devices
        GlobalAveragePooling1D(),
        
        # Small dense layers
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    # Compile the model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Alternative lightweight model if the above is too resource-intensive
def create_tiny_model(vocab_size=5000, embedding_dim=8, max_sequence_length=100, num_classes=5):
    """Create an extremely lightweight model for very constrained devices"""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
        GlobalAveragePooling1D(),
        Dense(16, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model