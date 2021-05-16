import glob
import os
import random
import time

# numpy
import numpy as np
# torch
import torch
from torch import optim, nn
from torch.utils.data import DataLoader

# network and periphery
from torchvision.transforms import transforms

import ADNetFactory
from ADDataset import ADDataset
# constants
from ADTrainTransformation import ADTrainTransformation, ToTensor
from AverageMeter import AverageMeter
from DeviceProvider import DeviceProvider
from ad_net_constants import DATA_LOADER_BATCH_SIZE, WEIGHT_DECAY, \
    LEARNING_RATE, MOMENTUM, DECAY_EPOCH, LR_GAMMA

LOAD_STATE_RELATIVE_PATH = './model/load/'
SAVE_STATE_RELATIVE_PATH = './model/save/'


def _main_decorator(func):
    def seed_wrapper():
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def device_wrapper():
        print(f'Using ***{device.device_info}*** device.')

    def wrapper():
        seed_wrapper()

        print('<===============     Start   ===============>\n')
        device_wrapper()
        func()
        print('\n<===============      End     ===============>')

    return wrapper


def _provide_data_loader_kwargs():
    return {'num_workers': 1, 'pin_memory': True}


def _provide_optim(model, wd, lr, momentum, de, lrg):
    params_wd, params_rest = [], []
    params = model.parameters()

    for param_item in params:
        if param_item.requires_grad:
            (params_wd if param_item.dim() != 1 else params_rest).append(param_item)

    param_groups = [
        {'params': params_wd, 'weight_decay': wd},
        {'params': params_rest}
    ]
    optimizer = optim.SGD(param_groups, lr=lr, momentum=momentum)
    learning_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=de, gamma=lrg)
    return [optimizer, learning_scheduler]


def _load_state_if_exist(model, optimizer):
    epoch = 0

    if os.path.exists(LOAD_STATE_RELATIVE_PATH) and len(os.listdir(LOAD_STATE_RELATIVE_PATH)) >= 1:
        try:
            state_to_load_path = max(glob.glob(LOAD_STATE_RELATIVE_PATH + '*'), key=os.path.getctime)
            print(f'Load state from `{os.path.abspath(state_to_load_path)}`')

            state = torch.load(state_to_load_path)
            model_state = state['model_state']
            optim_state = state['optim_state']
            epoch = state['epoch']

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
        except EOFError:
            epoch = 0
            print('Can not load state for some reason. Start learning from baby step.')
    else:
        print('No model and optim detected. Start learning from baby step.')

    return epoch


def _create_data_set(m_seed):
    train_data_transformation = transforms.Compose([ADTrainTransformation(), ToTensor()])
    evaluate_data_transformation = transforms.Compose([ToTensor()])

    test_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                              'test', m_seed, evaluate_data_transformation)
    train_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                               'train', m_seed, train_data_transformation)
    validation_data_set = ADDataset('./data/bossbase/bossbase_v.0.93/boss_256_0.4',
                                    'validation', m_seed, evaluate_data_transformation)

    return test_data_set, validation_data_set, train_data_set


def _create_data_loaders(t_ds, v_ds, test_ds,
                         train_kwargs, validation_kwargs=None, test_kwargs=None):
    if test_kwargs is None:
        test_kwargs = {}
    if validation_kwargs is None:
        validation_kwargs = {}
    test_dataloader = DataLoader(test_ds, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=True, **test_kwargs)
    train_dataloader = DataLoader(t_ds, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=True, **train_kwargs)
    validation_dataloader = DataLoader(v_ds, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=True, **validation_kwargs)

    return test_dataloader, validation_dataloader, train_dataloader


def train(model, train_loader, optimizer, epoch):
    model.train()
    time_meter = AverageMeter()
    train_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    # train_loaders 7200 pair [cover:stego]
    # 450 iter (each 16 pair [cover:stego])
    #                        [0,1]
    for batch_num, entry in enumerate(train_loader):
        batch_time = time.time()
        images, target = entry['images'], entry['label']

        shape = list(images.size())  # images.size() = [16, 2, 1, 256, 256]
        images = images.reshape(shape[0] * shape[1], *shape[2:])  # images.size() = [32, 1, 256, 256]
        target = target.reshape(-1)

        images = images.to(device.device)
        target = target.to(device.device)

        # clear gradients
        optimizer.zero_grad()
        # forward propagation
        output = model(images)
        # find the loss
        loss = criterion(output, target)
        # compute gradient
        loss.backward()
        # update weights
        optimizer.step()
        # compute loss
        train_loss.update(loss.item(), images.size(0))

        time_meter.update(time.time() - batch_time)
        print(f'Epoch: [{epoch + 1}][{batch_num + 1}/{len(train_loader)}]\t'
              f'Mini-batch time {time_meter.val:.3f} ({time_meter.avg:.3f})\t'
              f'Loss {train_loss.val:.4f} ({train_loss.avg:.4f})\t')


def evaluate(model, m_device, data_loader, epoch, optimizer, best_acc, PARAMS_PATH):
    model.eval()
    return 100.0


@_main_decorator
def main():
    # create dataset
    test_data_set, validation_data_set, train_data_set = _create_data_set(seed)
    # create data loaders
    test_dataloader, validation_dataloader, train_dataloader = _create_data_loaders(train_data_set,
                                                                                    validation_data_set,
                                                                                    test_data_set,
                                                                                    _provide_data_loader_kwargs())

    # create model
    model = ADNetFactory.build(device)
    # create optimizer
    optimizer, scheduler = _provide_optim(model,
                                          WEIGHT_DECAY,
                                          LEARNING_RATE,
                                          MOMENTUM,
                                          DECAY_EPOCH,
                                          LR_GAMMA)
    best_accuracy = 0.0
    start_epoch = _load_state_if_exist(model, optim)
    print(f'Start epoch is: {start_epoch}')

    for epoch in range(start_epoch, 1):
        # train
        train(model, train_dataloader, optimizer, epoch)
        # validate
        best_accuracy = evaluate(model, device, validation_dataloader, epoch, optimizer, best_accuracy, '')

        scheduler.step()

    # !!!!!!!!!!!!!!!!!!!!!!!!!
    # print(model.info((1, 256, 256)))


if __name__ == '__main__':
    device = DeviceProvider()
    seed = random.randint(0, 20051997)

    main()
