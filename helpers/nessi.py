# Complexity Calculator for PyTorch models aligned with:
# https://github.com/AlbertoAncilotto/NeSsi/blob/main/nessi.py
# we only copy the complexity calculation for torch models from NeSsi to avoid
# including an additional tensorflow dependency in this code base

import torchinfo

MAX_PARAMS = 32000
MAX_MACC = 30e6


def get_torch_size(model, input_size):
    model_profile = torchinfo.summary(model, input_size=input_size)
    return model_profile.total_mult_adds, model_profile.total_params

def validate(macc, params):
    print('Model statistics:')
    print('MACC:\t \t %.3f' %  (macc/1e6), 'M')
    print('Memory:\t \t %.3f' %  (params/1e3), 'K\n')
    if macc>MAX_MACC:
        print('[Warning] Multiply accumulate count', macc, 'is more than the allowed maximum of', int(MAX_MACC))
    if params>MAX_PARAMS:
        print('[Warning] parameter count', params, 'is more than the allowed maximum of', int(MAX_PARAMS))