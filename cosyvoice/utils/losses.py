import torch


def CCCLoss(x, y, eps=1e-8):
    # x, y: [N], where N is the number of words in the current sample
    if x.numel() < 2:
        # fewer than 2 samples: variance/correlation is undefined,
        # return None so the caller falls back to another loss
        return None

    vx = x - torch.mean(x)
    vy = y - torch.mean(y)

    # Pearson correlation term
    rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + eps)

    x_std = torch.std(x)
    y_std = torch.std(y)
    x_mean = torch.mean(x)
    y_mean = torch.mean(y)

    # CCC formula
    sb = 2 * rho * x_std * y_std
    cb = x_std ** 2 + y_std ** 2 + (x_mean - y_mean) ** 2

    ccc = sb / (cb + eps)

    return 1.0 - ccc
