# F1-Embed: Deep Learning for Formula 1 Telemetry Analysis

**F1-Embed** is a Transformer-based deep learning system that generates compact, high-dimensional vector representations (embeddings) of complete Formula 1 racing laps using raw telemetry data streams.

The system encodes driving patterns, vehicle dynamics, and performance characteristics from temporal telemetry sequences, enabling applications such as driver identification, team classification, lap similarity analysis, and performance prediction.

> **Conceptual Framework**: Similar to Word2Vec for natural language processing, F1-Embed learns contextual representations from sequential telemetry data including speed, throttle, brake pressure, gear selection, and aerodynamic systems.

## 📋 Project Overview

### Objective

Develop a machine learning pipeline that transforms raw Formula 1 telemetry data into meaningful vector embeddings for downstream analytical tasks.

### Key Applications

- **Driver Style Classification**: Identify unique driving patterns and techniques
- **Performance Prediction**: Predict lap times based on telemetry patterns
- **Similarity Search**: Find comparable laps across different sessions/drivers
- **Team Strategy Analysis**: Analyze constructor-specific approaches and setups

## 🚧 Current Development Status

**Phase**: Data Preprocessing Complete ✅  
**Next Phase**: Model Architecture Implementation 🚧

All Formula 1 telemetry data has been successfully extracted, cleaned, and structured for machine learning model training. The preprocessing pipeline is production-ready and the dataset is prepared for neural network training.

## 📊 Data Processing Pipeline

### Data Sources & Collection

- **Primary API**: FastF1 telemetry data extraction
- **Coverage**: Multiple F1 seasons(2022, 2023, 2024) with comprehensive session data from Silverstone, Monaco, Spa, Suzuka
- **Data Types**: Practice, Qualifying, and Race session telemetry
- **Quality Assurance**: Automated validation and outlier detection

#### Temporal Sequence Processing

Each racing lap is converted into a standardized temporal sequence:

```python
# Sequence standardization using numpy interpolation
Sequence Length: 100 time steps (fixed)
Resampling Method: numpy.interp() for uniform temporal distribution
```

#### Core Telemetry Features (6 dimensions per time step)

1. **RPM**: Engine revolutions per minute [0-15,000+]
2. **Speed**: Vehicle velocity in km/h [0-350+]
3. **Throttle**: Accelerator pedal position [0-100%]
4. **Brake**: Brake pedal pressure [True/False]
5. **Gear**: Transmission gear selection [1-8]
6. **DRS**: Drag Reduction System activation [0-9]

#### Contextual Metadata Encoding

Categorical variables processed through one-hot encoding:

- **Driver Identity**: Individual driver classification
- **Tire Compound**: Soft/Medium/Hard compound selection
- **Constructor/Team**: Vehicle manufacturer and team affiliation
- **Circuit**: Track-specific characteristics and layout
- **Session Type**: Practice/Qualifying/Race session context

#### Target Variable Preparation

- **Lap Time**: Continuous target variable in seconds (float precision)
- **Normalization**: Statistical scaling for model optimization
- **Task Compatibility**: Configured for both regression and classification objectives

### Final Data Structure

The preprocessing pipeline outputs three structured arrays:

```python
# Telemetry Time Series
X_seq: numpy.ndarray, shape (N, 100, 6)
├── N: Total number of processed laps
├── 100: Fixed temporal sequence length
└── 6: Telemetry feature dimensions

# Contextual Metadata
X_context: numpy.ndarray, shape (N, ctx_dims)
├── N: Total number of processed laps
└── ctx_dims: One-hot encoded categorical feature dimensions

# Performance Targets
y: numpy.ndarray, shape (N,)
└── Lap times in seconds (continuous float values)
```

## 🎯 Machine Learning Readiness

The preprocessed dataset supports multiple neural network architectures:

### Sequence Models

- **LSTM/GRU Networks**: For temporal pattern recognition
- **Transformer Encoders**: For attention-based sequence modeling
- **Hybrid Architectures**: Combining sequence and contextual information

### Learning Objectives

- **Primary Task**: Lap time regression
- **Secondary Tasks**: Driver classification, team identification
- **Multi-task Learning**: Joint optimization across multiple objectives

## 🛠️ Technical Infrastructure

### Core Dependencies

- **Data Processing**: FastF1, pandas, numpy
- **Deep Learning**: PyTorch 2.7.1
- **Feature Engineering**: scikit-learn
- **Development Environment**: Jupyter Lab, IPython
- **Visualization**: matplotlib
- **Web Framework**: Flask/FastAPI (deployment-ready)

### System Requirements

- **Python**: 3.8+ (recommended 3.10+)
- **Memory**: 8GB+ RAM (16GB+ recommended for large datasets)
- **Storage**: 10GB+ available space

## 🚀 Quick Start Guide

### Installation & Setup

1. **Repository Setup**

   ```bash
   git clone https://github.com/aayushxtech/f1-embed.git
   cd f1-embed
   ```

2. **Environment Configuration**

   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

## 🤝 Contributing

This project follows standard open-source contribution practices:

- Fork the repository for feature development
- Create feature branches with descriptive names
- Submit pull requests with comprehensive documentation
- Follow established code style and testing protocols

---
  
**Last Updated**: July 2025 | **Version**: 1.0.0-alpha
