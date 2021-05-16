import glob
import os

import torch
from torch import optim
from torch.utils.data import DataLoader

from ADDataset import ADDataset


def get_data_loaders(t_ds, v_ds, test_ds, batch_size,
                     train_kwargs, validation_kwargs=None, test_kwargs=None):
    if test_kwargs is None:
        test_kwargs = {}
    if validation_kwargs is None:
        validation_kwargs = {}

    train_dataloader = DataLoader(t_ds, batch_size=batch_size, shuffle=True, **train_kwargs)
    test_dataloader = DataLoader(test_ds, batch_size=batch_size, shuffle=True, **test_kwargs)
    validation_dataloader = DataLoader(v_ds, batch_size=batch_size, shuffle=False, **validation_kwargs)

    return test_dataloader, validation_dataloader, train_dataloader


def get_data_set(m_seed, train_transforms, test_transforms, validation_transforms):
    test_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                              'test', m_seed, test_transforms)
    train_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                               'train', m_seed, train_transforms)
    validation_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                                    'validation', m_seed, validation_transforms)

    return test_data_set, validation_data_set, train_data_set


def get_learning_scheduler(optimizer, milestones, gamma):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)


def get_optimizer(model, weight_decay, learning_rate, momentum):
    params_wd, params_rest = [], []
    params = model.parameters()

    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [
        {'params': params_wd, 'weight_decay': weight_decay},
        {'params': params_rest}
    ]
    optimizer = optim.SGD(param_groups, lr=learning_rate, momentum=momentum)
    return optimizer


def load_state(model, optimizer, path):
    epoch = 0

    if os.path.exists(path) and len(os.listdir(path)) >= 1:
        try:
            state_to_load_path = max(glob.glob(path + '*'), key=os.path.getctime)
            print(f'Load state from `{os.path.abspath(state_to_load_path)}`')

            state = torch.load(state_to_load_path)
            model_state = state['model_state']
            optim_state = state['optim_state']
            epoch = state['epoch']

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
        except EOFError as e:
            epoch = 0
            print(f'Can not load state due to: {e}')
    else:
        print('No saved state detected...')

    print(f'Start epoch is: {epoch}')
    return epoch
