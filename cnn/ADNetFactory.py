from torch import nn

from cnn.ADNet import ADNet


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        if module.weight.requires_grad:
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


def build(m_device):
    model = ADNet()
    model.apply(init_weights)
    model.to(m_device.device)

    return model
