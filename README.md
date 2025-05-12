# 🧠 stormmind.ch AI

This repository contains the machine learning and deep learning components for [stormmind.ch](https://github.com/stormmind-ch), a Bachelor's thesis project focused on forecasting storm-related damage in Switzerland using historical weather and geospatial data.

---


## 🧩 Components

### 📦 Datasets
Modular dataset classes for structured weather and storm damage data, including:
- Clustered region datasets
- Binary label preprocessing
- Inclusion of previous years’ data for temporal context
- Utility functions for grouping, feature extraction, and visualization

### 🧪 Experiments
Jupyter notebooks exploring different model architectures and setups:
- Feedforward neural networks (FNN)
- LSTM-based sequence models
- Attention-based Transformers

### 🧠 Models
Modular PyTorch implementations of:
- FNN (baseline model)
- Transformer (sequence-to-one forecasting)

### 🛠️ Training & Validation
- Scalable training loop for various experiments
- Integrated support for threshold-based classification
- Cross-validation support

### 🔧 Utilities
- Model initialization tools
- Experiment tracking with Weights & Biases

