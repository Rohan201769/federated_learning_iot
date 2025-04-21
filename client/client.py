# client/client.py
import os
import json
import time
import uuid
import numpy as np
import requests
import tensorflow as tf
from models.text_classifier import create_model
from client.data_processor import TextDataProcessor

class FederatedClient:
    def __init__(self, server_url, data_source_path, client_id=None):
        # Unique identifier for this client
        self.client_id = client_id or str(uuid.uuid4())[:8]
        self.server_url = server_url
        
        # Initialize local model
        self.local_model = create_model()
        
        # Setup data processor
        self.data_processor = TextDataProcessor(data_source_path)
        
        # Training config
        self.local_epochs = 3
        self.batch_size = 32
        
        print(f"Client {self.client_id} initialized")
    
    def get_global_model(self):
        """Fetch the latest global model from the server"""
        try:
            response = requests.get(f"{self.server_url}/get_model")
            if response.status_code == 200:
                data = response.json()
                
                # Convert lists back to numpy arrays
                weights = [np.array(w) for w in data['weights']]
                
                # Update local model
                self.local_model.set_weights(weights)
                return data['round']
            else:
                print(f"Error fetching model: {response.text}")
                return None
        except Exception as e:
            print(f"Error connecting to server: {e}")
            return None
    
    def train_local_model(self):
        """Train the local model on client data"""
        # Get training data
        X_train, y_train = self.data_processor.get_training_data()
        
        # Train the model
        history = self.local_model.fit(
            X_train, y_train,
            epochs=self.local_epochs,
            batch_size=self.batch_size,
            verbose=1
        )
        
        # Evaluate model on local validation data
        X_val, y_val = self.data_processor.get_validation_data()
        eval_results = self.local_model.evaluate(X_val, y_val, verbose=0)
        
        # Return metrics from last epoch
        metrics = {}
        for metric_name, metric_value in zip(self.local_model.metrics_names, eval_results):
            metrics[metric_name] = float(metric_value)
        
        for key, value in history.history.items():
            if key not in metrics:
                metrics[key] = float(value[-1])
                
        return metrics
    
    def submit_model_update(self):
        """Send local model updates to the server"""
        print(f"Client {self.client_id} preparing model update...")
        
        weights = self.local_model.get_weights()
        print(f"Got weights, length: {len(weights)}")
        
        # Convert weights to lists for JSON serialization with explicit type checking
        weights_as_lists = []
        for i, w in enumerate(weights):
            try:
                if isinstance(w, np.ndarray):
                    weights_as_lists.append(w.tolist())
                else:
                    weights_as_lists.append(w)
                print(f"Processed weight {i}, type: {type(w)}")
            except Exception as e:
                print(f"Error converting weight {i}: {e}")
                # If we can't convert, use an empty array as placeholder
                weights_as_lists.append([])
        
        print(f"Weights converted to lists")
        
        metrics = self.train_local_model()
        print(f"Got metrics: {metrics}")
        
        payload = {
            'client_id': self.client_id,
            'weights': weights_as_lists,
            'metrics': metrics
        }
        
        print(f"Sending payload to server...")
        try:
            print(f"Submitting to URL: {self.server_url}/submit_update")
            response = requests.post(
                f"{self.server_url}/submit_update",
                json=payload
            )
            print(f"Response status code: {response.status_code}")
            print(f"Response content: {response.text[:100]}...")  # Print first 100 chars of response
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error submitting update: {response.text}")
                return None
        except Exception as e:
            print(f"Exception in submit_update: {e}")
            print(f"Exception type: {type(e)}")
            return None
    
    def run_training_cycle(self):
        """Run a complete federated training cycle"""
        # Get the latest global model
        round_num = self.get_global_model()
        if round_num is None:
            print("Couldn't fetch global model. Using local initialization.")
        else:
            print(f"Retrieved global model from round {round_num}")
        
        # Train on local data
        print(f"Client {self.client_id} training on local data...")
        
        # Submit update
        response = self.submit_model_update()
        if response:
            print(f"Update submitted successfully for round {response.get('round')}")
        else:
            print("Failed to submit update")
    
    def run_continuous(self, interval=300):
        """Run continuous training cycles with specified interval in seconds"""
        while True:
            try:
                self.run_training_cycle()
                print(f"Waiting {interval} seconds before next cycle...")
                time.sleep(interval)
            except KeyboardInterrupt:
                print("Client training stopped by user")
                break
            except Exception as e:
                print(f"Error in training cycle: {e}")
                time.sleep(10)  # Wait a bit and try again
    
    def classify_text(self, text):
        """Use the current model to classify text"""
        processed_text = self.data_processor.preprocess_text(text)
        prediction = self.local_model.predict(processed_text)[0]
        predicted_class = np.argmax(prediction)
        
        class_names = self.data_processor.get_class_names()
        confidence = float(prediction[predicted_class])
        
        return {
            'class_name': class_names[predicted_class],
            'class_id': int(predicted_class),
            'confidence': confidence,
            'probabilities': {
                class_names[i]: float(p) for i, p in enumerate(prediction)
            }
        }

if __name__ == '__main__':
    # Example usage
    client = FederatedClient(
        server_url="http://localhost:5000",
        data_source_path="./data/client1"
    )
    client.run_continuous(interval=60)  # Run every minute