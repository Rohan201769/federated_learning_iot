# test_server.py
import requests
import json

def test_server_endpoints():
    """Test basic server endpoints"""
    
    base_url = "http://localhost:5000"
    
    # Test status endpoint
    print("Testing GET /get_status...")
    try:
        response = requests.get(f"{base_url}/get_status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test metrics endpoint
    print("\nTesting GET /get_metrics...")
    try:
        response = requests.get(f"{base_url}/get_metrics")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text[:100]}...")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test simple update with minimal data
    print("\nTesting POST /submit_update with minimal data...")
    try:
        payload = {
            "client_id": "test_client",
            "weights": [[0.1, 0.2], [0.3, 0.4]],
            "metrics": {"accuracy": 0.5}
        }
        response = requests.post(
            f"{base_url}/submit_update", 
            json=payload
        )
        print(f"Status: {response.status_code}")
        print(f"Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_server_endpoints()