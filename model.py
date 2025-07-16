import torch
from model_arch import F1Embedder
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model(path="f1-embed-model.pt", device="cpu"):
    """
    Load the F1Embedder model with comprehensive error handling and debugging

    Args:
        path (str): Path to the model file
        device (str): Device to load the model on ('cpu' or 'cuda')

    Returns:
        F1Embedder: Loaded model in evaluation mode, or None if loading fails
    """
    try:
        logger.info(f"🔄 Starting model loading process...")
        logger.info(f"Model path: {path}")
        logger.info(f"Target device: {device}")
        logger.info(f"Current working directory: {os.getcwd()}")

        # Check if model file exists
        if not os.path.exists(path):
            logger.error(f"❌ Model file not found: {path}")
            logger.error(f"Available files: {os.listdir('.')}")
            return None

        file_size = os.path.getsize(path)
        logger.info(f"✅ Model file found, size: {file_size} bytes")

        # Initialize model architecture
        logger.info("🏗️ Initializing model architecture...")
        model = F1Embedder(
            telemetry_dim=6,
            context_dim=22,
            d_model=64,
            nhead=4,
            num_layers=2
        )
        logger.info("✅ Model architecture initialized successfully")

        # Load state dict with error handling
        logger.info("📥 Loading model state dict...")
        try:
            state_dict = torch.load(
                path, map_location=device, weights_only=True)
            logger.info("✅ State dict loaded successfully")
        except Exception as load_error:
            logger.error(f"❌ Failed to load state dict: {load_error}")
            # Try alternative loading method
            try:
                logger.info("🔄 Attempting alternative loading method...")
                state_dict = torch.load(path, map_location=device)
                logger.info("✅ Alternative loading method successful")
            except Exception as alt_error:
                logger.error(f"❌ Alternative loading also failed: {alt_error}")
                return None

        # Load state dict into model
        logger.info("🔗 Loading state dict into model...")
        try:
            model.load_state_dict(state_dict)
            logger.info("✅ State dict loaded into model successfully")
        except Exception as state_error:
            logger.error(
                f"❌ Failed to load state dict into model: {state_error}")
            logger.error("This usually indicates architecture mismatch")
            return None

        # Set model to evaluation mode
        model.eval()
        logger.info("✅ Model set to evaluation mode")

        # Validate model with dummy input
        logger.info("🧪 Validating model with dummy input...")
        try:
            dummy_seq = torch.randn(1, 100, 6)
            dummy_context = torch.randn(1, 22)

            with torch.no_grad():
                output = model(dummy_seq, dummy_context)
                logger.info(
                    f"✅ Model validation successful, output shape: {output.shape}")
        except Exception as val_error:
            logger.error(f"❌ Model validation failed: {val_error}")
            return None

        logger.info("🎉 Model loaded successfully!")
        return model

    except ImportError as import_error:
        logger.error(f"❌ Import error: {import_error}")
        logger.error(
            "Check if model_arch.py is available and contains F1Embedder class")
        return None
    except Exception as e:
        logger.error(f"❌ Unexpected error during model loading: {e}")
        logger.error(f"Error type: {type(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return None


def get_embedding(model, X_seq, X_context):
    """
    Get embeddings (lap time predictions) from the model

    Args:
        model: Loaded F1Embedder model
        X_seq: Telemetry sequence data (batch_size, 100, 6)
        X_context: Context features (batch_size, 22)

    Returns:
        list: Predicted lap times
    """
    if model is None:
        logger.error("❌ Cannot get embedding: model is None")
        return None

    try:
        with torch.no_grad():
            X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
            X_context_tensor = torch.tensor(X_context, dtype=torch.float32)

            logger.debug(
                f"Input shapes - X_seq: {X_seq_tensor.shape}, X_context: {X_context_tensor.shape}")

            embedding = model(X_seq_tensor, X_context_tensor)
            result = embedding.numpy().tolist()

            logger.debug(f"Prediction successful, output: {result}")
            return result

    except Exception as e:
        logger.error(f"❌ Error during embedding generation: {e}")
        return None
