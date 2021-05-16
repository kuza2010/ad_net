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
from torchvision.transforms import transforms

# network and periphery
import ADNetFactory
import utils
from ADDataset import ADDataset
# constants
from ADTrainTransformation import ADTrainTransformation, ToTensor
from AverageMeter import AverageMeter, LossHistory, AccuracyHistory
from DeviceProvider import DeviceProvider
from ad_net_constants import DATA_LOADER_BATCH_SIZE, WEIGHT_DECAY, \
    LEARNING_RATE, MOMENTUM, DECAY_EPOCH, LR_GAMMA, EPOCH

MODEL_STATE_RELATIVE_PATH = './model/state/'
MODEL_TRAIN_HISTORY = './model/history/'


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


def _provide_optim(model, wd, lr, momentum):
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
    return optimizer


def _provide_learning_scheduler(optimizer, de, lrg):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=de, gamma=lrg)


def _load_state_if_exist(model, optimizer):
    epoch = 0

    if os.path.exists(MODEL_STATE_RELATIVE_PATH) and len(os.listdir(MODEL_STATE_RELATIVE_PATH)) >= 1:
        try:
            state_to_load_path = max(glob.glob(MODEL_STATE_RELATIVE_PATH + '*'), key=os.path.getctime)
            print(f'Load state from `{os.path.abspath(state_to_load_path)}`')

            state = torch.load(state_to_load_path)
            model_state = state['model_state']
            optim_state = state['optim_state']
            epoch = state['epoch']

            model.load_state_dict(model_state)
            optimizer.load_state_dict(optim_state)
        except EOFError:
            epoch = 0
            print('Can not load state for some reason.')
    else:
        print('No model and optim detected.')

    print(f'Start epoch is: {epoch}')
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
    validation_dataloader = DataLoader(v_ds, batch_size=DATA_LOADER_BATCH_SIZE, shuffle=False, **validation_kwargs)

    return test_dataloader, validation_dataloader, train_dataloader


def train_model(model, train_loader, optimizer, epoch, tlh):
    _time_meter = AverageMeter()
    _train_loss = AverageMeter()
    criterion = nn.CrossEntropyLoss()
    model.train()

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

        _train_loss.update(loss.item(), images.size(0))
        _time_meter.update(time.time() - batch_time)
        print(f'Epoch: [{epoch + 1}][{batch_num + 1}/{len(train_loader)}]\t'
              f'Mini-batch time {_time_meter.val:.3f} avg({_time_meter.avg:.3f})\t'
              f'Loss {_train_loss.val:.4f} ({_train_loss.avg:.4f})\t')

        if batch_num % 50 == 0:
            tlh.append(_train_loss.avg, batch_num)


def evaluate_model(model, m_device, eval_loader, is_validation=False,
                   epoch=None, optimizer=None, best_acc=None, vah=None):
    correct = 0.0
    model.eval()

    # do evaluation
    with torch.no_grad():
        for entry in eval_loader:
            images, target = entry['images'], entry['label']

            shape = list(images.size())
            images = images.reshape(shape[0] * shape[1], *shape[2:])
            target = target.reshape(-1)

            images = images.to(m_device.device)
            target = target.to(m_device.device)

            # forward propagation
            output = model(images)
            # get predictions from the maximum value
            prediction = output.max(1, keepdim=True)
            prediction_tensor = prediction[1]
            correct += prediction_tensor.eq(target.view_as(prediction_tensor)).sum().item()

    accuracy = correct / (len(eval_loader.dataset) * 2)
    if is_validation:
        # append to history
        vah.append(epoch, accuracy)
        if accuracy > best_acc:
            utils.save_model(MODEL_STATE_RELATIVE_PATH, model, optimizer, epoch)

    print('-' * 25)
    print(f'Epoch {epoch}\t'
          f'Eval accuracy: {accuracy:.4f}\t'
          f'Best accuracy: {best_acc}')
    print('-' * 25)

    return best_acc


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
    optimizer = _provide_optim(model, WEIGHT_DECAY, LEARNING_RATE, MOMENTUM)
    # load prev network state
    start_epoch = _load_state_if_exist(model, optim)
    # add learning scheduler
    scheduler = _provide_learning_scheduler(optimizer, DECAY_EPOCH, LR_GAMMA)

    best_accuracy = 0.0
    train_loss_history = LossHistory()
    validation_accuracy_history = AccuracyHistory()

    for epoch in range(start_epoch, EPOCH):
        # train
        train_model(model, train_dataloader, optimizer, epoch, train_loss_history)
        # validate
        best_accuracy = evaluate_model(model=model,
                                       m_device=device,
                                       eval_loader=validation_dataloader,
                                       is_validation=True,
                                       epoch=epoch,
                                       optimizer=optimizer,
                                       best_acc=best_accuracy,
                                       vah=validation_accuracy_history)
        scheduler.step()

    # load best network parameter to test
    _load_state_if_exist(model, optimizer)
    evaluate_model(model, device, test_dataloader, False)
    # save history to file
    utils.save_history(validation_accuracy_history, train_loss_history)
    # print(model.info((1, 256, 256)))


if __name__ == '__main__':
    device = DeviceProvider()
    seed = random.randint(0, 20051997)

    main()
