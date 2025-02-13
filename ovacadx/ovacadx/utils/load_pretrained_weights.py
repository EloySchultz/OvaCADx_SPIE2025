import os

import torch
import torch.nn as nn


def load_pretrained_weights_dino(model: nn.Module, path: str = None) -> nn.Module:
    """
    Load pretrained weights from a checkpoint file.

    Args:
        model: Model to load the weights into.
        path: Path to the checkpoint file.

    Returns:
        Model with loaded weights.
    """
    if path is None:
        return model

    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint file not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    state_dict = {k.replace('teacher_backbone.', ''): v for k, v in state_dict.items() if 'teacher_backbone' in k}
    msg = model.load_state_dict(state_dict, strict=False)

    print(f'Loaded pretrained weights from {path} with message: {msg}')

    return model


def load_pretrained_weights_byol(model: nn.Module, path: str = None) -> nn.Module:
    """
    Load pretrained weights from a checkpoint file.

    Args:
        model: Model to load the weights into.
        path: Path to the checkpoint file.

    Returns:
        Model with loaded weights.
    """
    if path is None:
        return model

    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint file not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    state_dict = {k.replace('backbone_momentum.', ''): v for k, v in state_dict.items() if 'backbone_momentum' in k}
    msg = model.load_state_dict(state_dict, strict=False)

    print(f'Loaded pretrained weights from {path} with message: {msg}')

    return model


def load_pretrained_weights_finetuned(model: nn.Module, path: str = None) -> nn.Module:
    """
    Load pretrained weights from a checkpoint file.

    Args:
        model: Model to load the weights into.
        path: Path to the checkpoint file.

    Returns:
        Model with loaded weights.
    """
    if path is None:
        return model

    if not os.path.exists(path):
        raise FileNotFoundError(f'Checkpoint file not found: {path}')

    checkpoint = torch.load(path, map_location='cpu')
    state_dict = checkpoint['state_dict']

    state_dict = {k.replace('model.', ''): v for k, v in state_dict.items() if 'model' in k}
    msg = model.load_state_dict(state_dict, strict=False)

    print(f'Loaded pretrained weights from {path} with message: {msg}')

    return model
