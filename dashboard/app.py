# dashboard/app.py
import os
import json
import time
import requests
from flask import Flask, render_template, request, jsonify
import threading

dashboard = Flask(__name__)

# Configure the dashboard app
dashboard.config['SERVER_URL'] = 'http://localhost:5000'
dashboard.config['CLIENT_URL'] = 'http://localhost:5001'

@dashboard.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@dashboard.route('/api/server_status')
def server_status():
    """Get status from the federated server"""
    try:
        response = requests.get(f"{dashboard.config['SERVER_URL']}/get_status")
        return response.json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard.route('/api/metrics')
def get_metrics():
    """Get training metrics"""
    try:
        response = requests.get(f"{dashboard.config['SERVER_URL']}/get_metrics")
        return response.json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard.route('/api/start_round', methods=['POST'])
def start_round():
    """Trigger a new training round"""
    try:
        response = requests.post(f"{dashboard.config['SERVER_URL']}/start_round")
        return response.json()
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@dashboard.route('/api/classify', methods=['POST'])
def classify_text():
    """Send text to a client for classification"""
    text = request.json.get('text', '')
    
    try:
        response = requests.post(
            f"{dashboard.config['CLIENT_URL']}/classify",
            json={'text': text}
        )
        return response.json()
    except Exception as e:
        return jsonify({'error': str(e), 'message': 'Failed to connect to client'}), 500

def create_templates_directory():
    """Create templates directory and dashboard HTML file"""
    os.makedirs('dashboard/templates', exist_ok=True)
    
    dashboard_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Federated Learning Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f7fa;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                background-color: #4a6fa5;
                color: white;
                padding: 15px;
                border-radius: 5px;
                margin-bottom: 20px;
                text-align: center;
            }
            .card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 20px;
            }
            .flex-container {
                display: flex;
                gap: 20px;
                flex-wrap: wrap;
            }
            .flex-item {
                flex: 1;
                min-width: 300px;
            }
            button {
                background-color: #4a6fa5;
                color: white;
                border: none;
                padding: 10px 15px;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #3a5a80;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            table th, table td {
                padding: 8px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            canvas {
                max-width: 100%;
            }
            .status-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                border-radius: 50%;
                margin-right: 5px;
            }
            .status-active {
                background-color: #4caf50;
            }
            .status-inactive {
                background-color: #f44336;
            }
            textarea {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                resize: vertical;
                height: 100px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Federated Learning NLP Dashboard</h1>
            </div>
            
            <div class="flex-container">
                <div class="flex-item">
                    <div class="card">
                        <h2>System Status</h2>
                        <div id="statusContainer">
                            <p>Current Round: <span id="currentRound">-</span></p>
                            <p>Training Status: <span id="trainingStatus">-</span></p>
                            <p>Clients Ready: <span id="clientsReady">-</span></p>
                            <p>Updates Received: <span id="updatesReceived">-</span></p>
                            <button id="startRoundBtn">Start New Round</button>
                        </div>
                    </div>
                    
                    <div class="card">
                        <h2>Client Nodes</h2>
                        <div id="clientsContainer">
                            <table id="clientsTable">
                                <thead>
                                    <tr>
                                        <th>Client ID</th>
                                        <th>Status</th>
                                        <th>Last Update</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    <!-- Will be populated by JavaScript -->
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
                
                <div class="flex-item">
                    <div class="card">
                        <h2>Training Metrics</h2>
                        <canvas id="accuracyChart"></canvas>
                        <canvas id="lossChart"></canvas>
                    </div>
                    
                    <div class="card">
                        <h2>Text Classification</h2>
                        <textarea id="textInput" placeholder="Enter text to classify..."></textarea>
                        <button id="classifyBtn">Classify Text</button>
                        <div id="classificationResult" style="margin-top: 10px;">
                            <!-- Results will appear here -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <!-- Include Chart.js for graphs -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        
        <script>
            // Charts
            let accuracyChart, lossChart;
            
            // Initialize charts
            function initCharts() {
                const accuracyCtx = document.getElementById('accuracyChart').getContext('2d');
                accuracyChart = new Chart(accuracyCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Accuracy',
                            data: [],
                            borderColor: '#4a6fa5',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true,
                        scales: {
                            y: {
                                beginAtZero: false,
                                min: 0,
                                max: 1
                            }
                        }
                    }
                });
                
                const lossCtx = document.getElementById('lossChart').getContext('2d');
                lossChart = new Chart(lossCtx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Loss',
                            data: [],
                            borderColor: '#f44336',
                            tension: 0.1
                        }]
                    },
                    options: {
                        responsive: true
                    }
                });
            }
            
            // Update status information
            async function updateStatus() {
                try {
                    const response = await fetch('/api/server_status');
                    const data = await response.json();
                    
                    document.getElementById('currentRound').textContent = data.round;
                    document.getElementById('trainingStatus').textContent = data.is_training ? 'Training' : 'Idle';
                    document.getElementById('clientsReady').textContent = data.clients_ready.length;
                    document.getElementById('updatesReceived').textContent = data.updates_received;
                    
                    // Update clients table
                    const tbody = document.querySelector('#clientsTable tbody');
                    tbody.innerHTML = '';
                    
                    data.clients_ready.forEach((clientId) => {
                        const row = document.createElement('tr');
                        row.innerHTML = `
                            <td>${clientId}</td>
                            <td><span class="status-indicator status-active"></span> Active</td>
                            <td>Just now</td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                } catch (error) {
                    console.error('Error updating status:', error);
                }
            }
            
            // Update metrics charts
            async function updateMetrics() {
                try {
                    const response = await fetch('/api/metrics');
                    const data = await response.json();
                    
                    if (data.metrics_history && data.metrics_history.length > 0) {
                        const rounds = data.metrics_
                    if (data.metrics_history && data.metrics_history.length > 0) {
                        const rounds = data.metrics_history.map(m => `Round ${m.round}`);
                        const accuracies = data.metrics_history.map(m => m.metrics.accuracy || 0);
                        const losses = data.metrics_history.map(m => m.metrics.loss || 0);
                        
                        // Update accuracy chart
                        accuracyChart.data.labels = rounds;
                        accuracyChart.data.datasets[0].data = accuracies;
                        accuracyChart.update();
                        
                        // Update loss chart
                        lossChart.data.labels = rounds;
                        lossChart.data.datasets[0].data = losses;
                        lossChart.update();
                    }
                } catch (error) {
                    console.error('Error updating metrics:', error);
                }
            }
            
            // Classify text
            async function classifyText() {
                const text = document.getElementById('textInput').value.trim();
                
                if (!text) {
                    alert('Please enter some text to classify');
                    return;
                }
                
                try {
                    document.getElementById('classificationResult').innerHTML = 'Processing...';
                    
                    const response = await fetch('/api/classify', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ text })
                    });
                    
                    const result = await response.json();
                    
                    if (result.error) {
                        document.getElementById('classificationResult').innerHTML = 
                            `<p style="color: red;">Error: ${result.message || result.error}</p>`;
                        return;
                    }
                    
                    // Create a bar chart visualization for the probabilities
                    const classes = Object.keys(result.probabilities);
                    const probabilities = Object.values(result.probabilities);
                    
                    let resultHTML = `
                        <p>Classified as: <strong>${result.class_name}</strong> (${(result.confidence * 100).toFixed(2)}% confidence)</p>
                        <div style="margin-top: 15px;">
                    `;
                    
                    classes.forEach((className, index) => {
                        const percent = (probabilities[index] * 100).toFixed(1);
                        const isMax = className === result.class_name;
                        
                        resultHTML += `
                            <div style="margin-bottom: 8px;">
                                <div style="display: flex; align-items: center;">
                                    <div style="width: 120px; text-align: right; padding-right: 10px;">${className}</div>
                                    <div style="flex-grow: 1;">
                                        <div style="background-color: ${isMax ? '#4a6fa5' : '#a0c0e0'}; height: 20px; width: ${percent}%;"></div>
                                    </div>
                                    <div style="width: 50px; padding-left: 10px;">${percent}%</div>
                                </div>
                            </div>
                        `;
                    });
                    
                    resultHTML += '</div>';
                    document.getElementById('classificationResult').innerHTML = resultHTML;
                    
                } catch (error) {
                    console.error('Error classifying text:', error);
                    document.getElementById('classificationResult').innerHTML = 
                        `<p style="color: red;">Error connecting to the classification service</p>`;
                }
            }
            
            // Start a new training round
            async function startRound() {
                try {
                    const button = document.getElementById('startRoundBtn');
                    button.disabled = true;
                    button.textContent = 'Starting...';
                    
                    await fetch('/api/start_round', { method: 'POST' });
                    
                    // Re-enable after a moment
                    setTimeout(() => {
                        button.disabled = false;
                        button.textContent = 'Start New Round';
                    }, 2000);
                    
                } catch (error) {
                    console.error('Error starting round:', error);
                    document.getElementById('startRoundBtn').disabled = false;
                    document.getElementById('startRoundBtn').textContent = 'Start New Round';
                }
            }
            
            // Initialize the dashboard
            function initDashboard() {
                // Initialize charts
                initCharts();
                
                // Set up event listeners
                document.getElementById('startRoundBtn').addEventListener('click', startRound);
                document.getElementById('classifyBtn').addEventListener('click', classifyText);
                
                // Update status and metrics initially and then periodically
                updateStatus();
                updateMetrics();
                
                setInterval(updateStatus, 5000);  // Update status every 5 seconds
                setInterval(updateMetrics, 10000);  // Update metrics every 10 seconds
            }
            
            // Start when page loads
            window.addEventListener('DOMContentLoaded', initDashboard);
        </script>
    </body>
    </html>
    """
    
    with open('dashboard/templates/dashboard.html', 'w') as file:
        file.write(dashboard_html)

def run_dashboard(host='0.0.0.0', port=8080):
    """Run the dashboard application"""
    # Create HTML template
    create_templates_directory()
    
    # Run Flask app
    dashboard.run(host=host, port=port, debug=False)

if __name__ == '__main__':
    run_dashboard()