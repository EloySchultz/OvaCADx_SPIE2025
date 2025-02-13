from typing import Any, List


def ensure_list(input: Any) -> List:
    """Wrap the input into a list if it is not a list. If it is a None, return an empty list.

    Args:
        input (Any): Input to wrap into a list.

    Returns:
        Output list.
    """
    if isinstance(input, list):
        return input
    if isinstance(input, tuple):
        return list(input)
    if input is None:
        return []
    return [input]
