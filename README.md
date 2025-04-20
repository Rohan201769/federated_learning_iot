# Federated Learning NLP System for Raspberry Pi

This project implements a federated learning system for NLP classification tasks, designed to be lightweight and portable to Raspberry Pi devices.

## Features

- Federated learning server that aggregates model updates
- Client nodes that train locally on private data
- Web-based dashboard for visualization and monitoring
- Privacy-preserving NLP text classification
- Designed for resource-constrained devices like Raspberry Pi

## System Requirements

- Python 3.8+ (tested on Python 3.9)
- For PC/Mac testing: 4GB RAM minimum
- For Raspberry Pi: Raspberry Pi 4 with 4GB RAM recommended

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/federated-learning-rpi.git
   cd federated-learning-rpi
   ```

2. Create a virtual environment and install dependencies:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up sample data:
   ```
   python main.py --setup-data
   ```

## Usage

### Run the entire system locally (for testing)

```
python main.py --mode all
```

This will start:
- A federated learning server on port 5000
- Two client instances training on sample data
- A client API on port 5001
- A web dashboard on port 8080

### Run components separately

1. Run just the server:
   ```
   python main.py --mode server
   ```

2. Run a client:
   ```
   python main.py --mode client --client-id client1
   ```

3. Run the dashboard:
   ```
   python main.py --mode dashboard
   ```

### Access the Dashboard

Open your browser and navigate to:
```
http://localhost:8080
```

## Raspberry Pi Deployment

1. Clone this repository on each Raspberry Pi

2. On the server Pi:
   ```
   python main.py --mode server --server-host 0.0.0.0
   ```

3. On client Pi devices:
   ```
   python main.py --mode client --server-host <SERVER_PI_IP> --client-id <UNIQUE_ID>
   ```

4. On a device to host the dashboard:
   ```
   python main.py --mode dashboard --server-host <SERVER_PI_IP>
   ```

## Project Structure

- `server/`: Contains the federated server implementation
- `client/`: Contains the client node implementation
- `models/`: NLP model architecture
- `dashboard/`: Web-based visualization dashboard
- `data/`: Sample and collected data
- `main.py`: Main entry point for the application

## Customization

### Adding Custom Data Sources

1. Create a new directory in `data/`
2. Add text files with format: `<class_id>_<name>.txt`
3. Update class names in `class_names.json`

### Modifying the Model

Edit `models/text_classifier.py` to change the model architecture, but keep in mind resource constraints of Raspberry Pi.

## Demonstration Ideas

- **Live Data Processing**: Connect RSS feeds or web scrapers to clients
- **User Interaction**: Have audience submit text via the dashboard
- **Visualization**: Show model improvement over federated rounds
- **Privacy Demo**: Highlight that raw text never leaves its origin device