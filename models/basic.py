import torch
import torch.nn as nn
from murenn import MuReNNDirect
from .mixstyle import MixStyle


def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MuReNNClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        J1 = config["J1"]
        Q1 = config["alpha"]
        T1 = config["beta"]
        C =  config["beta"]*config["n"]                  # config["beta"]
        J2 = config["J2"]
        Q2 = config["alpha"]*config["m"]      #config["beta"]
        T2 = 1
        mixstyle_p = config["mixstyle_p"]
        mixstyle_alpha = config["mixstyle_alpha"]


        self.l1 = MuReNNDirect(
            J=J1,
            Q=Q1,
            T=T1,
            in_channels=1,
        )

        self.l2 = nn.Sequential(
            nn.Conv1d(
                in_channels=Q1*J1,
                out_channels=C,
                kernel_size=1,
                padding="same",
                bias=False, 
            ),
            MuReNNDirect(
                J=J2,
                Q=Q2,
                T=T2,
                in_channels=C,
            ),
        )

        self.l3 = nn.Sequential(
            nn.Linear(
                in_features=C*Q2,
                out_features=10,
            ),
            nn.Softmax(dim=1),
        )

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)

        self.apply(initialize_weights)


    def forward(self, x):
        x = self.l1(x)
        x = self.mixstyle(x)
        B, C, Q, J, _ = x.shape
        x = x.view(B, C * Q * J, -1)
        x = self.l2(x)
        x = torch.sum(x, dim=(4, 3))
        x = x.view(B, -1)
        x = self.l3(x)
        return x

def get_model(
        J1=8,
        J2=4,
        alpha=42,
        beta=8,
        m=5,
        n=1,
        mixstyle_p=0.5,
        mixstyle_alpha=0.2,
):
    config = {
        "alpha": alpha,
        "beta": beta, 
        "m": m,
        "n": n,
        "J1": J1,
        "J2": J2,
        "mixstyle_p": mixstyle_p,
        "mixstyle_alpha": mixstyle_alpha,
    }
    model = MuReNNClassifier(config)
    return model

if __name__ == "__main__":
    config = {
        "alpha": 42,
        "beta": 8, 
        "J1": 8,
        "J2": 4,
        "m": 1,
        "n": 1,
        "mixstyle_p": 0.5,
        "mixstyle_alpha": 0.2,
    }
    x = torch.zeros(1, 1, 2**14)
    m = MuReNNClassifier(config)
    m.eval()
    y = m(x)
    print(y.shape)