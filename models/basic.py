import torch
import torch.nn as nn
from murenn import MuReNNDirect, DTCWT
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


class Basic(nn.Module):
    def __init__(self, config):
        super().__init__()
        J1 = config["J1"]
        Q1 = config["alpha"]
        T1 = config["beta"]
        C =  config["beta"]*config["n"]
        J2 = config["J2"]
        Q2 = config["alpha"]*config["m"]
        T2 = 1
        mixstyle_p = config["mixstyle_p"]
        mixstyle_alpha = config["mixstyle_alpha"]
        self.skip_lp = config["skip_lp"]


        self.layer1 = MuReNNDirect(
            J=J1,
            Q=Q1,
            T=T1,
            in_channels=1,
        )

        self.dirac = nn.Conv1d(
            in_channels=Q1*J1,
            out_channels=C,
            kernel_size=1,
            padding="same",
            bias=False,           
        )

        if not self.skip_lp:
            self.phi2 = DTCWT(
                J=J2+1,
                padding_mode="symmetric",
                skip_hps=True,
                normalize=True,
            )
            self.layer2 = MuReNNDirect(
                J=J2,
                Q=Q2,
                T=T2,
                in_channels=C,
            )
            self.fc = nn.Linear(
                in_features=C*(Q2*J2+1), # This is too big
                out_features=10,
            )

        else:
            self.layer2 = MuReNNDirect(
                J=J2,
                Q=Q2,
                T=T2,
                in_channels=C,
            )
            self.fc = nn.Linear(
                in_features=C*Q2*J2, # This is too big
                out_features=10,
            )

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.apply(initialize_weights)


    def forward(self, x):
        x = self.layer1(x)
        x = self.mixstyle(x)
        B, C, Q, J, T = x.shape
        x = x.view(B, C * Q * J, T)
        x = self.dirac(x)
        x_psis = self.layer2(x) #B, C, Q2, J2, T
        B, C, Q, J, T = x_psis.shape
        y = x_psis.view(B, C * Q * J, T) # B, C*Q2*J2, T
        if not self.skip_lp:
            x_phi, _ = self.phi2(x) #B, C, T 
            y = torch.cat((y, x_phi), 1) # B, C*(Q2*J2+1), T
        y = torch.sum(y, dim=-1)
        logits = self.fc(y)
        return logits


def get_model(
        J1=8,
        J2=4,
        alpha=10,
        beta=30,
        m=5,
        n=1,
        mixstyle_p=0.5,
        mixstyle_alpha=0.2,
        skip_lp=True,
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
        "skip_lp": skip_lp,
    }
    m = Basic(config)
    return m

if __name__ == "__main__":
    x = torch.zeros(1, 1, 2**14)
    m = get_model()
    m.eval()
    y = m(x)
    print(y.shape)
