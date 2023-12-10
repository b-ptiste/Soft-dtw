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
        log_sum = torch.max(z) + torch.log(torch.sum(torch.exp(z - max(z))))
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
    R[0, 1:] = float('inf')
    R[1:, 0] = float('inf')
    R[0, 0] = 0.0
    
    for j in range(1, m + 1):
        for i in range(1, n + 1):
            # calculate distance
            cost = (x[i - 1] - y[j - 1])**2
            
            # calculate minimum
            _min = soft_min([R[i - 1, j], R[i, j - 1], R[i - 1, j - 1]], gamma)
            
            # update cell
            R[i, j] = cost + _min
            
    return R[-1, -1], R
    
    