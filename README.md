# Neural Network Performance Analyzer

## Overview
This project provides a comprehensive comparison between manual (NumPy) and automated (PyTorch) neural network implementations. It offers a web interface to explore performance differences in training neural networks across various configurations.

## Key Features
- Compare manual NumPy and PyTorch neural network implementations
- Configurable network parameters:
  - Number of samples
  - Input size
  - Hidden layer size
  - Output size
  - Number of epochs
- Visual performance metrics
- Training time comparison
- Loss function analysis

## Technologies Used
- Python
- Flask
- NumPy
- PyTorch
- Matplotlib
- HTML/JavaScript

## Prerequisites
- Python 3.8+
- pip
- Virtual environment recommended

## Installation

1. Clone the repository
```bash
git clone https://github.com/kasalaAbhinav/Neural-Network-Performance-Analyzer.git
cd Neural-Network-Performance-Analyzer
```

2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

## Running the Application
```bash
python app.py
```
Navigate to `http://localhost:5000` in your web browser

## Project Structure
- `app.py`: Main Flask application
- `index.html`: Web interface
- `requirements.txt`: Project dependencies

## How It Works
The application generates random neural network training data and compares two implementations:
1. Manual Implementation: Custom NumPy-based neural network
2. PyTorch Implementation: Automated neural network using PyTorch modules

## Visualization
The application generates three plots:
- Training Loss Over Time
- Training Time Comparison
- Final Loss Comparison

## Contributing
1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request


## Contact
[Your Name/Contact Information]
