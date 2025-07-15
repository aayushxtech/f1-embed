import torch
from model_arch import F1Embedder


def load_model(path="f1-embed-model.pt", device="cpu"):
    model = F1Embedder(
        telemetry_dim=6,
        context_dim=22,
        d_model=64,
        nhead=4,
        num_layers=2
    )

    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict)

    model.eval()
    return model


def get_embedding(model, X_seq, X_context):
    with torch.no_grad():
        X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32)
        X_context_tensor = torch.tensor(X_context, dtype=torch.float32)
        embedding = model(X_seq_tensor, X_context_tensor)
        return embedding.numpy().tolist()
