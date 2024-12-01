# app.py
from flask import Flask, render_template, request, jsonify
import numpy as np
import torch
import torch.nn as nn
import time
import json
import io
import base64
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

app = Flask(__name__)

# [Previous Neural Network Classes and Functions remain the same]
class ManualNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward(self, X):
        self.X = X
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        delta2 = (output - y) * self.sigmoid_derivative(output)
        dW2 = np.dot(self.a1.T, delta2)
        db2 = np.sum(delta2, axis=0, keepdims=True)
        
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = np.dot(X.T, delta1)
        db1 = np.sum(delta1, axis=0, keepdims=True)
        
        learning_rate = 0.1
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
    
    def train(self, X, y, epochs):
        losses = []
        for epoch in range(epochs):
            output = self.forward(X)
            loss = np.mean(-y * np.log(output + 1e-15) - (1 - y) * np.log(1 - output + 1e-15))
            losses.append(loss)
            self.backward(X, y, output)
        return losses

class AutomatedNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AutomatedNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.sigmoid1 = nn.Sigmoid()
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.sigmoid2 = nn.Sigmoid()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.sigmoid1(x)
        x = self.layer2(x)
        x = self.sigmoid2(x)
        return x

def create_plot(results):
    fig = Figure(figsize=(15, 5))
    
    # Plot training losses
    ax1 = fig.add_subplot(131)
    ax1.plot(results['manual_losses'], label='Manual', color='blue', alpha=0.7)
    ax1.plot(results['torch_losses'], label='PyTorch', color='red', alpha=0.7)
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot time comparison
    ax2 = fig.add_subplot(132)
    times = [results['manual_time'], results['torch_time']]
    ax2.bar(['Manual', 'PyTorch'], times, color=['blue', 'red'], alpha=0.7)
    ax2.set_title('Training Time Comparison')
    ax2.set_ylabel('Seconds')
    ax2.grid(True)
    
    # Plot final loss comparison
    ax3 = fig.add_subplot(133)
    final_losses = [results['manual_losses'][-1], results['torch_losses'][-1]]
    ax3.bar(['Manual', 'PyTorch'], final_losses, color=['blue', 'red'], alpha=0.7)
    ax3.set_title('Final Loss Comparison')
    ax3.set_ylabel('Loss')
    ax3.grid(True)
    
    fig.tight_layout()
    
    # Convert plot to base64 string
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plot_url = base64.b64encode(buf.getvalue()).decode('utf8')
    
    return plot_url

def compare_implementations(X, y, input_size, hidden_size, output_size, epochs):
    # Manual Implementation
    manual_start = time.time()
    manual_nn = ManualNeuralNetwork(input_size, hidden_size, output_size)
    manual_losses = manual_nn.train(X, y, epochs)
    manual_time = time.time() - manual_start
    
    # PyTorch Implementation
    torch_start = time.time()
    X_torch = torch.FloatTensor(X)
    y_torch = torch.FloatTensor(y)
    
    model = AutomatedNeuralNetwork(input_size, hidden_size, output_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    torch_losses = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_torch)
        loss = criterion(outputs, y_torch)
        loss.backward()
        optimizer.step()
        torch_losses.append(loss.item())
    
    torch_time = time.time() - torch_start
    
    return {
        'manual_time': manual_time,
        'manual_losses': manual_losses,
        'torch_time': torch_time,
        'torch_losses': torch_losses
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    data = request.get_json()
    
    n_samples = int(data.get('n_samples', 1000))
    input_size = int(data.get('input_size', 10))
    hidden_size = int(data.get('hidden_size', 5))
    output_size = int(data.get('output_size', 1))
    epochs = int(data.get('epochs', 100))
    
    # Generate sample data
    np.random.seed(42)
    torch.manual_seed(42)
    X = np.random.randn(n_samples, input_size)
    y = np.random.randint(2, size=(n_samples, output_size)).astype(float)
    
    # Run comparison
    results = compare_implementations(
        X=X,
        y=y,
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        epochs=epochs
    )
    
    # Create plot
    plot_url = create_plot(results)
    
    # Prepare results
    response = {
        'manual_time': float(results['manual_time']),
        'torch_time': float(results['torch_time']),
        'manual_final_loss': float(results['manual_losses'][-1]),
        'torch_final_loss': float(results['torch_losses'][-1]),
        'plot': plot_url
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)