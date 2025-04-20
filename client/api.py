# client/api.py
from flask import Flask, request, jsonify
from client.client import FederatedClient

# Initialize Flask app
app = Flask(__name__)

# Global client instance
client = None

@app.route('/classify', methods=['POST'])
def classify_text():
    """Endpoint to classify text using the local model"""
    if not client:
        return jsonify({'error': 'Client not initialized'}), 500
    
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    result = client.classify_text(text)
    
    return jsonify(result)

@app.route('/status', methods=['GET'])
def get_status():
    """Return the client's status"""
    if not client:
        return jsonify({'status': 'not_initialized'}), 500
    
    return jsonify({
        'client_id': client.client_id,
        'status': 'active',
        'server_url': client.server_url
    })

def run_client_api(client_instance, host='0.0.0.0', port=5001):
    """Run the client API server"""
    global client
    client = client_instance
    app.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    # Example: This would normally be called from main.py
    from client.client import FederatedClient
    
    client = FederatedClient(
        server_url="http://localhost:5000",
        data_source_path="./data/client1"
    )
    
    run_client_api(client)