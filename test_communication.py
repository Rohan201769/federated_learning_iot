# test_communication.py
import requests
import numpy as np
import json

def test_submit_update():
    """Test sending a model update to the server"""
    # Create dummy weights (just some random arrays)
    weights = [
        np.random.rand(10, 10).tolist(),
        np.random.rand(10).tolist()
    ]
    
    # Create dummy metrics
    metrics = {
        'accuracy': 0.8,
        'loss': 0.3
    }
    
    # Create payload
    payload = {
        'client_id': 'test_client',
        'weights': weights,
        'metrics': metrics
    }
    
    # Test JSON serialization
    try:
        json_str = json.dumps(payload)
        print("JSON serialization successful")
    except Exception as e:
        print(f"JSON serialization failed: {e}")
        return
    
    # Send to server
    try:
        response = requests.post(
            "http://localhost:5000/submit_update",
            json=payload
        )
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    test_submit_update()