"""Model registry — get_model(config) → nn.Module outputting (B, 512) embeddings."""

from models.resnet50 import ResNet50
from models.iresnet50 import IResNet50
from models.mobilefacenet import MobileFaceNet
from models.ghostfacenet import GhostFaceNet
from models.edgeface_s import EdgeFaceS
from models.swin_tiny import SwinTiny
from models.facelivtv2_s import FaceLiVTv2S


MODEL_REGISTRY = {
    "resnet50": ResNet50,
    "iresnet50": IResNet50,
    "mobilefacenet": MobileFaceNet,
    "ghostfacenet": GhostFaceNet,
    "edgeface_s": EdgeFaceS,
    "swin_tiny": SwinTiny,
    "facelivtv2_s": FaceLiVTv2S,
}


def get_model(config):
    """Create model from config dict.

    Args:
        config: dict with at least config['model']['name'] and config['model']['embedding_dim']

    Returns:
        nn.Module: model that maps (B, 3, 112, 112) → (B, embedding_dim)
    """
    name = config["model"]["name"]
    embedding_dim = config["model"]["embedding_dim"]

    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[name](embedding_dim=embedding_dim)
