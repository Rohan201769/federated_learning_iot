# server/server.py
import os
import json
import time
import numpy as np
from flask import Flask, request, jsonify
from threading import Thread
import tensorflow as tf
from models.text_classifier import create_model

app = Flask(__name__)

class FederatedServer:
    # In server/server.py

    def __init__(self, model_path='./server/global_model'):
        self.model_path = model_path
        self.global_model = create_model()
        
        # No need to do model(sample_input) for now
        
        self.client_updates = {}
        self.clients_ready = set()
        self.round_number = 0
        self.metrics_history = []
        self.is_training = False
        # Skip initial save for now
        
    def save_global_model(self):
        """Save the global model to disk"""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.global_model.save_weights(self.model_path+'.weights.h5')
        
    def load_global_model(self):
        """Load the global model from disk if it exists"""
        if os.path.exists(self.model_path + '.weights.h5'):
            self.global_model.load_weights(self.model_path + '.weights.h5')
            
    def aggregate_models(self):
        """Federated averaging of client model updates"""
        if not self.client_updates:
            return
        
        # Get all weights
        weights_list = [update["weights"] for update in self.client_updates.values()]
        metrics = [update["metrics"] for update in self.client_updates.values()]
        
        # Simple averaging of weights
        average_weights = []
        for weights_list_tuple in zip(*weights_list):
            average_weights.append(
                np.array([np.array(w) for w in weights_list_tuple]).mean(axis=0)
            )
        
        # Update global model with new weights
        self.global_model.set_weights(average_weights)
        self.save_global_model()
        
        # Average metrics for tracking
        avg_metrics = {}
        for key in metrics[0].keys():
            avg_metrics[key] = sum(m[key] for m in metrics) / len(metrics)
        
        # Store metrics history
        self.metrics_history.append({
            "round": self.round_number,
            "timestamp": time.time(),
            "metrics": avg_metrics,
            "num_clients": len(self.client_updates)
        })
        
        # Clear updates for next round
        self.client_updates = {}
        self.clients_ready = set()
        self.round_number += 1
        print(f"Completed federated round {self.round_number}")
        
    def start_training_round(self):
        """Start a new federated training round"""
        self.is_training = True
        print(f"Starting federated round {self.round_number}")
        time.sleep(5)  # Give time for clients to connect
        self.aggregate_models()
        self.is_training = False
        print(f"Completed federated round {self.round_number}")

# Server API routes
server = FederatedServer()

@app.route('/get_model', methods=['GET'])
def get_model():
    """Endpoint for clients to download the latest global model"""
    if not os.path.exists(server.model_path + '.weights.h5'):
        return jsonify({'error': 'No model available yet'}), 404
    
    # Get model weights as list of numpy arrays
    model_weights = server.global_model.get_weights()
    
    # Convert numpy arrays to lists for JSON serialization
    weights_as_lists = [w.tolist() for w in model_weights]
    
    return jsonify({
        'round': server.round_number,
        'weights': weights_as_lists
    })

@app.route('/submit_update', methods=['POST'])
def submit_update():
    """Endpoint for clients to submit their model updates"""
    try:
        print("Received update submission")
        data = request.json
        if not data:
            print("No JSON data received")
            return jsonify({'error': 'No data received'}), 400
            
        print(f"Got data with keys: {data.keys()}")
        
        client_id = data.get('client_id')
        if not client_id:
            print("No client ID in data")
            return jsonify({'error': 'No client ID'}), 400
            
        weights = data.get('weights')
        if not weights:
            print("No weights in data")
            return jsonify({'error': 'No weights'}), 400
            
        metrics = data.get('metrics')
        if not metrics:
            print("No metrics in data")
            return jsonify({'error': 'No metrics'}), 400
        
        # Convert lists back to numpy arrays
        weights_as_np = []
        for w in weights:
            try:
                weights_as_np.append(np.array(w))
            except Exception as e:
                print(f"Error converting weight to numpy: {e}")
                weights_as_np.append(np.array([]))
        
        # Store the update
        server.client_updates[client_id] = {
            "weights": weights_as_np,
            "metrics": metrics
        }
        
        # Mark this client as ready for next round
        server.clients_ready.add(client_id)
        
        print(f"Successfully processed update from client {client_id}")
        
        return jsonify({'status': 'success', 'round': server.round_number})
    except Exception as e:
        print(f"Exception in submit_update route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/get_status', methods=['GET'])
def get_status():
    """Return the current training status"""
    return jsonify({
        'round': server.round_number,
        'is_training': server.is_training,
        'clients_ready': list(server.clients_ready),
        'updates_received': len(server.client_updates),
    })

@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """Return the metrics history"""
    return jsonify({
        'metrics_history': server.metrics_history
    })

@app.route('/start_round', methods=['POST'])
def start_round():
    """Manually trigger a new training round"""
    if server.is_training:
        return jsonify({'status': 'error', 'message': 'Training already in progress'}), 400
    
    Thread(target=server.start_training_round).start()
    return jsonify({'status': 'success', 'message': 'Started new training round'})

def run_server(host='0.0.0.0', port=5000):
    """Run the federated learning server"""
    server.load_global_model()
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_server()