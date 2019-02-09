def normalize(tensor, mean, std):
    """
    Args:
        tensor (Tensor): Tensor to normalize

    Returns:
        :param tensor:
        :param std:
        :param mean:
    """
    tensor.sub_(mean).div_(std)

    return tensor
