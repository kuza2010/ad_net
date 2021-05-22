from torch import nn

from ADNet import ADNet


def init_weights(module):
    if type(module) == nn.Conv2d:
        if module.weight.requires_grad:
            nn.init.kaiming_normal_(module.weight.data, mode='fan_in', nonlinearity='relu')

    if type(module) == nn.Linear:
        nn.init.normal_(module.weight.data, mean=0, std=0.01)
        nn.init.constant_(module.bias.data, val=0)


def build(m_device):
    model = ADNet()
    model.apply(init_weights)
    model.to(m_device.device)

    return model
