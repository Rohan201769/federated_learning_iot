# main.py
import os
import argparse
import threading
import time
import json
import random
from server.server import run_server
from client.client import FederatedClient
from client.api import run_client_api
from dashboard.app import run_dashboard

def setup_sample_data(data_dir, num_samples=50):
    """Create sample text data for initial testing"""
    client_dir = os.path.dirname(data_dir)
    os.makedirs(client_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)
    
    # Define class names for sentiment analysis
    class_names = ['Negative', 'Somewhat Negative', 'Neutral', 'Somewhat Positive', 'Positive']
    
    # Save class names
    with open(os.path.join(data_dir, 'class_names.json'), 'w') as file:
        json.dump(class_names, file)
    
    # Sample texts for each class
    sample_texts = {
        0: [  # Negative
            "This product is terrible and completely useless.",
            "I had a horrible experience with customer service.",
            "The quality is extremely poor and not worth the money.",
            "I regret buying this item, it's a complete waste.",
            "The service was awful and I will never return.",
            "This is the worst restaurant I've ever been to.",
            "The movie was boring and a waste of time.",
            "I'm very disappointed with this purchase.",
            "The app keeps crashing and is unusable.",
            "The food was cold and tasteless."
        ],
        1: [  # Somewhat Negative
            "The product isn't as good as I expected.",
            "There are several issues that need improvement.",
            "I'm a bit disappointed with the features.",
            "The service was below average.",
            "It's not terrible but I wouldn't recommend it.",
            "The quality could be better for the price.",
            "It's somewhat functional but has many bugs.",
            "The product works but doesn't meet expectations.",
            "The experience was underwhelming.",
            "There's room for improvement in many areas."
        ],
        2: [  # Neutral
            "The product does what it's supposed to do.",
            "It's an average experience, nothing special.",
            "The service was standard, neither good nor bad.",
            "It works as expected but doesn't stand out.",
            "The quality is acceptable for the price.",
            "I have mixed feelings about this product.",
            "Some features are good, others need work.",
            "It's okay for occasional use.",
            "The experience was neither impressive nor disappointing.",
            "It meets basic requirements but that's all."
        ],
        3: [  # Somewhat Positive
            "The product is pretty good overall.",
            "I'm generally satisfied with my purchase.",
            "The service was better than I expected.",
            "It has some nice features that I appreciate.",
            "The quality is decent for the price.",
            "I would probably recommend it to others.",
            "The experience was pleasant overall.",
            "It works well most of the time.",
            "I like most aspects of this product.",
            "The staff was friendly and helpful."
        ],
        4: [  # Positive
            "This product is excellent and exceeded my expectations!",
            "I had an amazing experience with the service.",
            "The quality is outstanding and worth every penny.",
            "I absolutely love this purchase and highly recommend it.",
            "The customer support was exceptional and solved my issue immediately.",
            "This is the best app I've used in a long time.",
            "The features are incredible and very user-friendly.",
            "I'm extremely satisfied with everything about this product.",
            "The performance is flawless and reliable.",
            "This has made a significant positive impact on my daily life."
        ]
    }
    
    # Create sample files
    for class_id, texts in sample_texts.items():
        # Create multiple files per class to simulate different data sources
        for i in range(3):
            # Select random subset of texts
            selected_texts = random.sample(texts, min(3, len(texts)))
            
            filename = f"{class_id}_sample_{i}.txt"
            with open(os.path.join(data_dir, filename), 'w', encoding='utf-8') as file:
                file.write('\n\n'.join(selected_texts))
    
    print(f"Created sample data in {data_dir}")

def run_in_thread(target, args=()):
    """Run a function in a separate thread"""
    thread = threading.Thread(target=target, args=args)
    thread.daemon = True
    thread.start()
    return thread

def main():
    """Main function to run the federated learning system"""
    parser = argparse.ArgumentParser(description='Run Federated Learning System')
    parser.add_argument('--mode', choices=['server', 'client', 'dashboard', 'all'], default='all',
                        help='Component to run (default: all)')
    parser.add_argument('--server-host', default='localhost', help='Server hostname')
    parser.add_argument('--server-port', type=int, default=5000, help='Server port')
    parser.add_argument('--client-id', help='Client ID (generated if not provided)')
    parser.add_argument('--client-port', type=int, default=5001, help='Client API port')
    parser.add_argument('--dashboard-port', type=int, default=8080, help='Dashboard port')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    parser.add_argument('--setup-data', action='store_true', help='Create sample data')
    
    args = parser.parse_args()
    
    # Create data directories
    os.makedirs(args.data_dir, exist_ok=True)
    
    # Setup sample data if requested
    # In main.py, ensure these directories exist
    if args.setup_data:
        print("Setting up sample data directories...")
        os.makedirs(os.path.join(args.data_dir, 'client1'), exist_ok=True)
        os.makedirs(os.path.join(args.data_dir, 'client2'), exist_ok=True)
        setup_sample_data(os.path.join(args.data_dir, 'client1'))
        setup_sample_data(os.path.join(args.data_dir, 'client2'))
    
    threads = []
    
    # Determine server URL
    server_url = f"http://{args.server_host}:{args.server_port}"
    
    # Start components based on mode
    if args.mode in ['server', 'all']:
        print(f"Starting server on port {args.server_port}...")
        threads.append(run_in_thread(run_server, args=(args.server_host, args.server_port)))
    
    # Wait a moment for server to start if we're starting clients too
    if args.mode in ['client', 'all']:
        if args.mode == 'all':
            time.sleep(2)  # Give server time to start
        
        # Start clients
        client1_data_dir = os.path.join(args.data_dir, 'client1')
        client1 = FederatedClient(
            server_url=server_url,
            data_source_path=client1_data_dir,
            client_id=args.client_id or "client1"
        )
        
        print(f"Starting client API on port {args.client_port}...")
        threads.append(run_in_thread(run_client_api, args=(client1, args.server_host, args.client_port)))
        
        print(f"Starting client training thread...")
        threads.append(run_in_thread(client1.run_continuous, args=(60,)))  # Run every minute
        
        # If running 'all' mode, start a second client too
        if args.mode == 'all':
            client2_data_dir = os.path.join(args.data_dir, 'client2')
            client2 = FederatedClient(
                server_url=server_url,
                data_source_path=client2_data_dir,
                client_id="client2"
            )
            
            print(f"Starting second client training thread...")
            threads.append(run_in_thread(client2.run_continuous, args=(70,)))  # Slight offset
    
    if args.mode in ['dashboard', 'all']:
        if args.mode == 'all':
            time.sleep(3)  # Give other components time to start
        
        print(f"Starting dashboard on port {args.dashboard_port}...")
        run_dashboard(args.server_host, args.dashboard_port)  # This will block
    
    # If we're not running the dashboard, wait for threads
    for thread in threads:
        thread.join()

if __name__ == '__main__':
    main()