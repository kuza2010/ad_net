from torch import nn

from ADNet import ADNet


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        if module.weight.requires_grad:
            print('init Conv2d')
            nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                print('init Conv2d bias')
                nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        print('init BatchNorm2d')
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.Linear):
        print('init Linear')
        nn.init.xavier_uniform_(module.weight)
        nn.init.constant_(module.bias, 0)


def build(m_device):
    model = ADNet()
    model.apply(init_weights)
    model.to(m_device.device)

    return model
