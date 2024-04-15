import torch
import numpy as np


def mixup(size, alpha):
    rn_indices = torch.randperm(size)
    lambd = np.random.beta(alpha, alpha, size).astype(np.float32)
    lambd = np.concatenate([lambd[:, None], 1 - lambd[:, None]], 1).max(1)
    lam = torch.FloatTensor(lambd)
    # data = data * lam + data2 * (1 - lam)
    # targets = targets * lam + targets2 * (1 - lam)
    return rn_indices, lam

# in training_step:
# if self.mixup_alpha:
#     rn_indices, lam = mixup(batch_size, self.mixup_alpha)
#     lam = lam.to(x.device)
#     x = x * lam.reshape(batch_size, 1, 1, 1) + x[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))
#     # applying same mixup config also to teacher spectrograms
#     x_teacher = x_teacher * lam.reshape(batch_size, 1, 1, 1) + \
#                 x_teacher[rn_indices] * (1. - lam.reshape(batch_size, 1, 1, 1))

#     y_hat = self.forward(x)

#     samples_loss = (F.cross_entropy(y_hat, y, reduction="none") * lam.reshape(batch_size) +
#                     F.cross_entropy(y_hat, y[rn_indices], reduction="none") * (1. - lam.reshape(batch_size)))