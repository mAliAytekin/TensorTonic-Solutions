import torch

def create_tensor(method, shape, value=0.0):
    """
    Returns:  list
    """
    if method == "zeros":
        return torch.zeros(shape).tolist()
    elif method == "ones":
        return torch.ones(shape).tolist()
    elif method == "full":
        return torch.full(shape,value).tolist()
    elif method == "arange":
        return torch.arange(shape).tolist()
    elif method == "linespace":
        return torch.linspace(shape,value).tolist()
    return torch.eye(shape[0]).tolist()
    
    