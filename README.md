# F1-Embed: Deep Learning for Formula 1 Telemetry Analysis

**F1-Embed** is a Transformer-based deep learning system that generates compact, high-dimensional vector representations (embeddings) of complete Formula 1 racing laps using raw telemetry data streams.

The system encodes driving patterns, vehicle dynamics, and performance characteristics from temporal telemetry sequences, enabling applications such as driver identification, team classification, lap similarity analysis, and performance prediction.

> **Conceptual Framework**: Similar to Word2Vec for natural language processing, F1-Embed learns contextual representations from sequential telemetry data including speed, throttle, brake pressure, gear selection, and aerodynamic systems.

## 🚧 Current Development Status

**Phase**: Model Training Complete ✅ | **Current Phase**: FastAPI Deployment & Enhancement 🚀

The F1-Embed Transformer model has been successfully implemented, trained, and deployed as a FastAPI web service. Initial results show strong performance with an R² score of 0.8513 for lap time prediction, demonstrating the model's ability to learn meaningful patterns from F1 telemetry data.

## 📋 Project Overview

### Objective

Develop a machine learning pipeline that transforms raw Formula 1 telemetry data into meaningful vector embeddings for downstream analytical tasks and real-time inference via API.

## 📊 Data Processing Pipeline

### Data Sources & Collection

- **Primary API**: FastF1 telemetry data extraction
- **Drivers**: VER, LEC, SAI (3 selected drivers)
- **Years**: 2022, 2023, 2024 (3 seasons)
- **Session Types**: FP1, FP2, FP3, Q, R (Practice, Qualifying, Race)
- **Tracks**: Silverstone, Monaco, Spa, Suzuka (4 circuits)
- **Weather Integration**: Real-time weather data merged with lap data
- **Quality Assurance**: Automated validation and accurate lap filtering

### Feature Engineering

#### Temporal Sequence Processing

Each racing lap is converted into a standardized temporal sequence:

```python
# Sequence standardization using numpy interpolation
Sequence Length: 100 time steps (fixed)
Resampling Method: numpy.interp() for uniform temporal distribution
```

#### Core Telemetry Features (6 dimensions per time step)

1. **RPM**: Engine revolutions per minute [~9,000-15,000+ RPM]
2. **Speed**: Vehicle velocity in km/h [~180-350+ km/h during racing]
3. **nGear**: Transmission gear selection [1-8 gears]
4. **Throttle**: Accelerator pedal position [0-100%]
5. **Brake**: Brake pedal pressure [0/1]
6. **DRS**: Drag Reduction System activation [0-9]

> **Note**: The original `Time_x` feature is used for temporal resampling via `numpy.interp()` but is not included in the final feature vector. Each lap is resampled to exactly 100 time steps using linear interpolation.

**Feature Processing Details:**

- **Resampling Method**: `numpy.interp()` for uniform temporal distribution
- **Sequence Length**: Fixed at 100 time steps per lap
- **Data Source**: Extracted from F1 telemetry CSV using pandas groupby operations
- **Normalization**: Applied post-resampling using StandardScaler

#### Contextual Metadata Encoding

Categorical variables processed through one-hot encoding:

- **Driver Identity**: Individual driver classification (VER, LEC, SAI)
- **Tire Compound**: Soft/Medium/Hard compound selection
- **Constructor/Team**: Vehicle manufacturer and team affiliation
- **Circuit**: Track-specific characteristics and layout (Silverstone, Monaco, Spa, Suzuka)
- **Session Type**: Practice/Qualifying/Race session context
- **Track Status**: Racing conditions and safety car periods

#### Data Preprocessing & Normalization

- **Sequence Data**: StandardScaler applied to telemetry features
- **Context Data**: Selective standardization of continuous variables (LapNumber, Year, TyreLife)
- **Target Variable**: Lap times converted from timedelta to seconds and normalized

### Final Data Structure

```python
# Telemetry Time Series
X_seq: numpy.ndarray, shape (N, 100, 6)
├── N: Total number of processed laps (1,770)
├── 100: Fixed temporal sequence length
└── 6: Telemetry feature dimensions [RPM, Speed, nGear, Throttle, Brake, DRS]

# Contextual Metadata
X_context: numpy.ndarray, shape (N, 22)
├── N: Total number of processed laps
└── 22: One-hot encoded categorical features + standardized continuous variables

# Performance Targets
y: numpy.ndarray, shape (N,)
└── Lap times in seconds (normalized for training)
```

## 🤖 Model Architecture

### F1Embedder: Transformer-Based Architecture

The core model implements a hybrid architecture combining sequence modeling with contextual information:

```model-architecture
┌─────────────────────────────────────────────────────────────────┐
│                    F1-Embed Architecture                        │
└─────────────────────────────────────────────────────────────────┘

Input Data:
┌──────────────────────┐    ┌──────────────────────────────────────┐
│    X_context         │    │            X_seq                     │
│   (N, 22)            │    │         (N, 100, 6)                  │
│                      │    │                                      │
│ • Driver             │    │ • RPM      • Throttle                │
│ • Team               │    │ • Speed    • Brake                   │
│ • Compound           │    │ • nGear    • DRS                     │
│ • Circuit            │    │                                      │
│ • Session Type       │    │ Time Steps: 100                      │
└──────────────────────┘    └──────────────────────────────────────┘
         │                                   │
         │                                   │
         ▼                                   ▼
┌──────────────────────┐    ┌──────────────────────────────────────┐
│   Context MLP        │    │      Embedding Layer                 │
│                      │    │     Linear(6 → 64)                   │
│ Linear(22 → 64)      │    │                                      │
│ ReLU                 │    │                                      │
│ Linear(64 → 64)      │    │                                      │
└──────────────────────┘    └──────────────────────────────────────┘
         │                                   │
         │                                   ▼
         │                  ┌──────────────────────────────────────┐
         │                  │    Positional Encoding               │
         │                  │       (sinusoidal)                   │
         │                  └──────────────────────────────────────┘
         │                                   │
         │                                   ▼
         │                  ┌──────────────────────────────────────┐
         │                  │    Transformer Encoder               │
         │                  │                                      │
         │                  │ ┌────────────────────────────────┐   │
         │                  │ │  Multi-Head Attention (4 heads)│   │
         │                  │ │       + Layer Norm             │   │
         │                  │ └────────────────────────────────┘   │
         │                  │ ┌────────────────────────────────┐   │
         │                  │ │   Feed Forward (64→2048→64)    │   │
         │                  │ │       + Layer Norm             │   │
         │                  │ └────────────────────────────────┘   │
         │                  │                                      │
         │                  │        ×2 Layers                     │
         │                  └──────────────────────────────────────┘
         │                                   │
         │                                   ▼
         │                  ┌──────────────────────────────────────┐
         │                  │      Global Average Pooling          │
         │                  │        (100, 64) → (64)              │
         │                  └──────────────────────────────────────┘
         │                                   │
         │                                   │
         └──────────────┬────────────────────┘
                        │
                        ▼
           ┌──────────────────────────────────────┐
           │          Fusion Layer                │
           │     Concatenate: [64 + 64] = 128     │
           └──────────────────────────────────────┘
                        │
                        ▼
           ┌──────────────────────────────────────┐
           │         Regressor Head               │
           │                                      │
           │      Linear(128 → 64)                │
           │      ReLU                            │
           │      Linear(64 → 1)                  │
           └──────────────────────────────────────┘
                        │
                        ▼
           ┌──────────────────────────────────────┐
           │         Output                       │
           │    Predicted Lap Time                │
           │      (seconds)                       │
           └──────────────────────────────────────┘
```

#### Components

1. **Embedding Layer**: Linear projection of telemetry features to d_model dimensions
2. **Positional Encoding**: Sinusoidal position embeddings for temporal awareness
3. **Transformer Encoder**: Multi-head self-attention for sequence modeling
4. **Context MLP**: Multi-layer perceptron for contextual metadata processing
5. **Fusion & Regression**: Combined embeddings for final prediction

#### Architecture Specifications

```python
F1Embedder(
  embedding_layer: Linear(6 → 64)
  pos_encoder: PositionalEncoding(d_model=64, max_len=5000)
  transformer: TransformerEncoder(
    layers: 2x TransformerEncoderLayer(
      self_attn: MultiheadAttention(heads=4, d_model=64)
      feedforward: Linear(64 → 2048 → 64)
      normalization: LayerNorm + Dropout(0.1)
    )
  )
  context_mlp: Sequential(
    Linear(22 → 64) → ReLU → Linear(64 → 64)
  )
  regressor: Sequential(
    Linear(128 → 64) → ReLU → Linear(64 → 1)
  )
)
```

## 📈 Training & Performance Results

### Training Configuration

- **Optimizer**: Adam (lr=1e-3)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Train/Test Split**: 80/20 (1,416 training, 354 test samples)
- **Device**: CPU-based training
- **Epochs**: Single epoch with early convergence

### Model Performance

```performance
Test MSE: 12.5119
Test MAE: 2.6715
R² Score: 0.8513
```

### Key Achievements

- **Excellent Predictive Accuracy**: R² of 0.8513 indicates strong lap time prediction capability
- **Robust Generalization**: Low test error demonstrates effective learning of telemetry patterns
- **Fast Training**: Single-epoch convergence on CPU hardware
- **Model Persistence**: Trained model state dict saved as [`f1-embed-model.pt`](f1-embed-model.pt)

## 🚀 FastAPI Deployment

### Web Service Features

The model is deployed as a FastAPI web service with the following endpoints:

#### `/embed` - Lap Time Prediction

- **Method**: POST
- **Rate Limit**: 10 requests/minute
- **Input Format**:

  ```json
  {
    "X_seq": [[100 x 6 array]], // Telemetry sequence
    "X_context": [22-element array] // Context features
  }
  ```

- **Output**:

  ```json
  {
    "embedded_lap_time": 85.342,
    "status": "success"
  }
  ```

#### `/health` - Health Check

- **Method**: GET
- **Purpose**: API status and model loading verification

#### `/model_info` - Model Specifications

- **Method**: GET
- **Purpose**: Model architecture and input format details

### Deployment Configuration

- **Rate Limiting**: 10 requests per minute per IP
- **Error Handling**: Comprehensive validation and exception handling
- **Input Validation**: Automatic shape checking for telemetry and context data

## 🛠️ Technical Infrastructure

### Core Dependencies

- **Data Processing**: FastF1, pandas, numpy
- **Deep Learning**: PyTorch 2.7.1
- **Feature Engineering**: scikit-learn (StandardScaler)
- **Training Utilities**: tqdm for progress tracking
- **Evaluation**: scikit-learn metrics
- **Web Framework**: FastAPI with rate limiting (slowapi)
- **Development Environment**: Jupyter Lab, IPython

### System Requirements

- **Python**: 3.8+ (tested on 3.13.5)
- **Memory**: 8GB+ RAM (16GB+ recommended for larger datasets)
- **Storage**: 10GB+ available space
- **GPU**: Optional (model trains efficiently on CPU)

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

3. **Data Collection & Processing**

   ```bash
   # Extract F1 telemetry data
   jupyter lab data-extraction.ipynb
   
   # Process and prepare features
   jupyter lab data-preparation.ipynb
   ```

4. **Model Training**

   ```bash
   # Train the F1-Embed model
   jupyter lab model-preparation.ipynb
   ```

5. **API Deployment**

   ```bash
   # Start FastAPI server
   uvicorn main:app --reload
   ```

## 📖 Usage Examples

### Loading Trained Model

```python
from model import load_model
import numpy as np

# Load the trained model using proper loading function
model = load_model("f1-embed-model.pt")

# Load test data
X_seq = np.load('X_seq.npy')
X_context = np.load('X_context.npy')

# Make predictions
from model import get_embedding
predictions = get_embedding(model, X_seq[:5], X_context[:5])
print(f"Predicted lap times: {predictions}")
```

### API Usage

```python
import requests
import numpy as np

# Load sample data
X_seq = np.load('X_seq.npy')[0].tolist()  # Shape: (100, 6)
X_context = np.load('X_context.npy')[0].tolist()  # Shape: (22,)

# Make API request
response = requests.post("http://localhost:8000/embed", json={
    "X_seq": X_seq,
    "X_context": X_context
})

result = response.json()
print(f"Predicted lap time: {result['embedded_lap_time']} seconds")
```

### Extracting Embeddings

```python
# Extract embeddings for similarity analysis
def extract_embeddings(model, X_seq, X_context):
    model.eval()
    with torch.no_grad():
        x_seq = model.embedding_layer(X_seq)
        x_seq = model.pos_encoder(x_seq)
        z_seq = model.transformer(x_seq)
        embeddings = z_seq.mean(dim=1)  # Sequence embeddings
    return embeddings.numpy()
```

## 📊 Performance Analysis

### Model Strengths

- **Excellent Predictive Power**: R² = 0.8513 indicates strong lap time prediction
- **Efficient Architecture**: Lightweight design suitable for real-time inference
- **Robust Feature Learning**: Successfully captures telemetry patterns from 3 top drivers
- **Fast Convergence**: Single-epoch training demonstrates effective optimization
- **Production Ready**: FastAPI deployment with rate limiting and error handling

### Dataset Characteristics

- **Driver Coverage**: VER, LEC, SAI (representative of top-tier performance)
- **Temporal Coverage**: 3 seasons (2022-2024) capturing recent regulation changes
- **Circuit Diversity**: 4 distinct tracks with varying characteristics(Japan, Monaco, Spain, Britain)
- **Session Variety**: All session types from practice to race conditions

### Areas for Improvement

- **Multi-task Learning**: Extend to driver/team classification
- **Embedding Quality**: Implement similarity search validation
- **Dataset Expansion**: Include more years and circuits

## 🤝 Contributing

This project follows standard open-source contribution practices:

- Fork the repository for feature development
- Create feature branches with descriptive names
- Submit pull requests with comprehensive documentation
- Follow established code style and testing protocols

---

**Project Status**: Production Deployment ✅ | **Current Phase**: Performance Optimization & Feature Enhancement 🚧  
**Last Updated**: July 2025  
**Model Performance**: R² = 0.8513 | MAE = 2.67s  
**API Status**: Live with rate limiting (10 req/min)
