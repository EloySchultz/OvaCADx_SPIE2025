from typing import Dict

from torch.optim import Optimizer


def get_optimizer_stats(optimizer: Optimizer) -> Dict[str, float]:
    """
    Extract learning rates and momentum values from each parameter group of the optimizer.

    Args:
        optimizer (Optimizer): A PyTorch optimizer.

    Returns:
        Dictionary with formatted keys and values for learning rates and momentum.
    """
    stats_dict = {}
    for group_idx, group in enumerate(optimizer.param_groups):
        lr_key = f"optimizer/{optimizer.__class__.__name__}/lr"
        momentum_key = f"optimizer/{optimizer.__class__.__name__}/momentum"

        # Add group index to the key if there are multiple parameter groups
        if len(optimizer.param_groups) > 1:
            lr_key += f"/group{group_idx+1}"
            momentum_key += f"/group{group_idx+1}"

        # Extracting learning rate
        stats_dict[lr_key] = group["lr"]

        # Extracting momentum or betas[0] if available
        if "momentum" in group:
            stats_dict[momentum_key] = group["momentum"]
        if "betas" in group:
            stats_dict[momentum_key] = group["betas"][0]

    return stats_dict
