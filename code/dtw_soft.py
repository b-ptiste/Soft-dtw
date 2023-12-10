import torch

def soft_min(list_a, gamma):
    """Softmin function.

    Args:
        list_a (list): list of values
        gamma (float): gamma parameter

    Returns:
        float: softmin value
    """
    assert gamma >= 0, "gamma must be greater than or equal to 0"
    
    # transform list_a to numpy array
    list_a = torch.Tensor(list_a)
    
    if gamma == 0:
        _min = torch.min(list_a)
    else:
        z = -list_a / gamma
        log_sum = max(z) + torch.log(torch.sum(torch.exp(z - max(z))))
        _min = -gamma * log_sum
    return _min


def soft_dtw(x, y, gamma=1.0):
    """Soft Dynamic Time Warping.

    Args:
        x (list): time series 1
        y (list): time series 2
        gamma (float, optional): gamma parameter. Defaults to 1.0.

    Returns:
        float: soft-DTW distance
    """
    # initialize DP matrix
    n = len(x)
    m = len(y)
    R = torch.zeros((n + 1, m + 1))
    