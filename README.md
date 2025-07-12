# F1-Embed

**F1-Embed** is a Transformer-based deep learning model that learns compact, high-dimensional vector representations (embeddings) of complete Formula 1 laps using raw telemetry data.

Each embedding encodes the driving style, car behavior, and performance characteristics from a lap — enabling driver identification, team classification, similarity search, and lap style analysis.

> Think of it as "Word2Vec for racing laps" — but built from telemetry sequences like speed, throttle, brake, gear, and more.

## 🚧 Current Status

The project is currently in the **data preprocessing phase**. All F1 telemetry data has been successfully processed and prepared for model training. The preprocessing pipeline is complete and ready for the next phase: model architecture implementation and training.

## 📊 Preprocessing Pipeline

### ✅ Completed Features

The preprocessing pipeline has been fully implemented with the following capabilities:

#### **Data Collection & Cleaning**

- Raw F1 telemetry data extraction using FastF1 API
- Data validation and quality checks
- Missing value handling and outlier detection

#### **Sequence Processing**

- **Fixed-length sequences**: Each lap resampled to exactly 100 time steps using `numpy.interp`
- **Key telemetry features**: 6 core features per time step
  - RPM (engine revolutions per minute)
  - Speed (km/h)
  - Throttle (0-100%)
  - Brake (0-100%)
  - Gear (1-8)
  - DRS (Drag Reduction System: 0/1)

#### **Contextual Metadata Encoding**

- One-hot encoding for categorical features:
  - Driver identification
  - Tire compound (Soft, Medium, Hard, etc.)
  - Team/Constructor
  - Track/Circuit information
  - Session type (Practice, Qualifying, Race)
- Contextual features attached per lap for enhanced model learning

#### **Target Variable Processing**

- Lap times extracted and normalized
- Converted to float seconds for regression tasks
- Ready for both classification and regression modeling

### 📈 Final Data Structure

The preprocessed data is structured as follows:

```python
# Input sequences (telemetry time series)
X_seq: shape (N, 100, 6)
# Where:
# - N = number of laps
# - 100 = fixed sequence length (time steps)
# - 6 = telemetry features [RPM, Speed, Throttle, Brake, Gear, DRS]

# Contextual metadata (one-hot encoded)
X_context: shape (N, ctx_dims)
# Where:
# - N = number of laps  
# - ctx_dims = total dimensions after one-hot encoding

# Target variable (lap times)
y: shape (N,)
# Lap times in seconds (float)
```

### 🎯 Ready for Model Training

The data is now fully prepared for:

- **LSTM/GRU networks** for sequence modeling
- **Transformer encoders** for attention-based learning
- **Hybrid architectures** combining sequence and contextual data
- **Multi-task learning** (lap time prediction + driver/team classification)

## 🔄 Next Steps

1. **Model Architecture Design**
   - Implement Transformer encoder for sequence processing
   - Design fusion layer for combining sequence and contextual features
   - Define multi-task learning objectives

2. **Training Pipeline**
   - Set up training/validation splits
   - Implement training loops with proper metrics
   - Add model checkpointing and early stopping

3. **Evaluation & Analysis**
   - Lap time prediction accuracy
   - Embedding quality assessment
   - Driver/team classification performance
   - Similarity search capabilities

## 🛠️ Technical Stack

- **Data Processing**: FastF1, pandas, numpy
- **Deep Learning**: PyTorch (planned)
- **Preprocessing**: scikit-learn for encoding
- **Visualization**: matplotlib, seaborn (planned)

---

*Project Status:* Preprocessing Complete ✅ | *Next Phase*: Model Implementation 🚧
