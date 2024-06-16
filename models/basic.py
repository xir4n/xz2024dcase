import torch
import torch.nn as nn
from murenn import MuReNNDirect, DTCWT
from murenn.dtcwt.nn import Conv1D_MuReNN, Strided_MuReNN, Dilated_MuReNN
from murenn.dtcwt.utils import fix_length
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


class MuReNN(nn.Module):
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
        self.ln1 = nn.GroupNorm(1,J1*Q1)
        self.bn1 = nn.BatchNorm1d(J1*Q1)
        self.gn1 = nn.GroupNorm(J1,J1*Q1)
        
        
        self.mix_channel = nn.Conv1d(
            in_channels=Q1*J1,
            out_channels=C,
            kernel_size=1,
            padding="same",
            bias=False,           
        )
        self.ln2 = nn.GroupNorm(1,C*J2*Q2)
        self.bn2 = nn.BatchNorm1d(C*J2*Q2)
        self.gn2 = nn.GroupNorm(J2,C*J2*Q2)

        if not self.skip_lp:
            self.phi2 = DTCWT(
                J=J2+1,
                padding_mode="symmetric",
                skip_hps=True,
                normalize=True,
            )
            self.conv1d_lp = nn.Conv1d(
                in_channels=C,
                out_channels=C*Q2,
                groups=C,
                kernel_size=T1,
                padding="same",
                bias=False,                       
            )

            self.layer2 = MuReNNDirect(
                J=J2,
                Q=Q2,
                T=T2,
                in_channels=C,
            )
            self.fc = nn.Linear(
                in_features=C*Q2*(J2+1),
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
                in_features=C*Q2*J2,
                out_features=10,
                bias=False,
                
            )

        self.mixstyle = MixStyle(p=mixstyle_p, alpha=mixstyle_alpha)
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.layer1(x)
        x = self.mixstyle(x)
        B, C, Q, J, T = x.shape
        x = x.view(B, C * Q * J, T)
        # x = self.gn1(x)
        x = self.mix_channel(x)
        x_psis = self.layer2(x) #B, C, Q2, J2, T
        B, C, Q, J, T = x_psis.shape
        y = x_psis.view(B, C * Q * J, T) # B, C*Q2*J2, T
        # y = self.gn2(y)
        if not self.skip_lp:
            x_phi, _ = self.phi2(x) #B, C, T 
            x_phi = self.conv1d_lp(x_phi)
            y = torch.cat((y, x_phi), 1) # B, C*(Q2*J2+1), T
        y = torch.sum(y, dim=-1)
        y = self.fc(y)
        return y


class MurennV(nn.Module):
    def __init__(self, config):
        super().__init__()
        J1 = config["J1"]
        Q1 = config["alpha"]
        T1 = config["beta"]
        kwargs_1 = dict(J=J1, Q=Q1, T=T1, in_channels=1)
#Conv1D_MuReNN, Strided_MuReNN, Dilated_MuReNN
        if config["model"] == "conv1d":
            self.layer1 = Conv1D_MuReNN(**kwargs_1)
        if config["model"] == "stride":
            self.layer1 = Strided_MuReNN(**kwargs_1)
        if config["model"] == "dilate":
            self.layer1 = Dilated_MuReNN(**kwargs_1)

        self.fc = nn.Linear(
            in_features=Q1*J1,
            out_features=10,
            bias=False,
            
        )
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.layer1(x) #channel : J*C_in*Q
        x = torch.sum(x, dim=-1)
        x = self.fc(x)
        return x

class MuReL(nn.Module):
    def __init__(self, config):
        super().__init__()
        J1 = config["J1"]
        Q1 = config["alpha"]
        T1 = config["beta"]

        self.layer1 = MuReNNDirect(
                J=J1,
                Q=Q1,
                T=T1,
                in_channels=1,
        )
        
        self.fc = nn.Linear(
            in_features=J1*Q1*J1,
            out_features=10,
            bias=False,
            
        )
        self.apply(initialize_weights)

    def forward(self, x):
        x = self.layer1(x)
        B, C, Q, J, T = x.shape
        x = x.view(B, C * Q * J, T)
        x = torch.sum(x, dim=-1)
        x = self.fc(x)
        return x

def get_model(
        J1=8,
        J2=4,
        alpha=1,
        beta=8,
        m=5,
        n=1,
        mixstyle_p=0.5,
        mixstyle_alpha=0.2,
        skip_lp=False,
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
    return MuReNN(config)

def get_layer(
        J1=8,
        alpha=1,
        beta=8,
):
    config = {
        "alpha": alpha,
        "beta": beta, 
        "J1": J1,
    }
    return MuReNN(config)


def get_model_v(
        J1=8,
        alpha=1,
        beta=8,
        model="conv1d"
):
    config = {
        "alpha": alpha,
        "beta": beta, 
        "J1": J1,
        "model": model,
    }
    return MurennV(config)

if __name__ == "__main__":
    x = torch.zeros(1, 1, 2**14)
    m = get_model_v()
    m.eval()
    y = m(x)
    print(y.shape)
