import torch

from xplique.attributions import GradientInput, Saliency
from xplique.attributions.base import WhiteBoxExplainer
from xplique.wrappers import TorchWrapper


def get_wrapped_model(model: torch.nn.Module, device: str) -> TorchWrapper:
    """
    Wraps the model with the TorchWrapper class.

    Args:
    -----
        model (torch.nn.Module): Model to be wrapped.
        device (str): Device to be used.

    Returns:
    --------
        wrapped_model (TorchWrapper): Wrapped model.
    """
    wrapped_model = TorchWrapper(model, device, is_channel_first=True)
    return wrapped_model


# def get_saliency_ma
